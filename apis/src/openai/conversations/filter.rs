// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! [`OpenaiConversationsFilter`] handles all `/v1/conversations`
//! endpoints locally via `FilterAction::Reject`, backed by an owner-scoped
//! conversation store resolved from the per-listener registry the serving
//! runtime provisions.
//!
//! The `openai_operation` filter must run earlier in the same chain. Its typed
//! match is the sole runtime authority for Conversations dispatch.

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BoundUpstreamBodyOutcome, FilterAction, FilterError, HttpFilter, HttpFilterContext, Rejection,
    body::{BodyAccess, BodyMode, MAX_JSON_BODY_BYTES},
    parse_filter_config,
};
use serde_json::Value;
use tracing::{debug, trace, warn};

use super::{
    CONVERSATIONS_STORE_NAME,
    config::{ConversationsConfig, validate_config},
    handlers,
    routes::{APPLICATION_PROTOCOL, ConversationOperation, match_route},
};
use crate::{
    is_event_stream_content_type,
    openai::{
        operation_classifier::OpenAiOperationMatch,
        responses::{
            bound_body_outcome, buffered_parsed_json_bytes_upper_bound,
            error::responses_error_rejection,
            state::{ResponsesState, retained_json_bytes, retained_json_values_bytes},
            store::{
                PersistedResponseForConversation, ResponseRound, request_persistence_armed, response_round,
                store_response_header_skipped,
            },
        },
    },
    operation::Transport,
    service::conversations::build_item_records,
    state_owner::{StateOwner, require_state_owner},
    store::{OwnerScopedResponseStore, ResponseStoreRegistry, StoreError},
};

/// Error exposed when append-back would exceed the request-wide allowance.
const APPEND_BUDGET_MESSAGE: &str =
    "agentic retained payload exceeded openai_agentic_loop.max_retained_bytes during conversation append";

/// Separate aggregate admission failure from a durable backend failure.
enum AppendError {
    /// A request-wide retained owner would exceed its admitted allowance.
    Budget,
    /// The backend rejected an otherwise admitted write.
    Store(FilterError),
}

// -----------------------------------------------------------------------------
// OpenaiConversationsFilter
// -----------------------------------------------------------------------------

/// Handles all `/v1/conversations` endpoints locally.
///
/// All matched requests are served from the owner-scoped store and never
/// forwarded upstream. Unmatched paths pass through as `Continue`. The filter
/// keeps only request-scoped state: it resolves the store from the per-request
/// registry the serving runtime provisions, and takes an owner-bound handle.
/// `openai_operation` must precede this filter in the same chain.
/// For a managed `POST /v1/responses` with `conversation`, completed JSON and
/// SSE responses append the request input and final output items to the local
/// Conversation. Streaming append-back reads the canonical terminal response
/// state without buffering SSE. With the default fail-closed policy, it persists
/// before `response.completed` is released; `failure_mode: open` opts out of that
/// guarantee. Incomplete or failed streams do not append a turn. The provider
/// owns history on a direct OpenAI passthrough route.
///
/// # YAML
///
/// ```yaml
/// - filter: openai_operation
/// - filter: openai_conversations
///   backend: postgres
///   database_url: postgres://praxis:password@db.example.com/praxis
///   conversations_table: conversations
///   items_table: conversation_items
///   allow_private_database_url: true
/// ```
pub struct OpenaiConversationsFilter;

/// Per-request state used when another filter forces request-body pre-read
/// before this filter's header hook has run.
#[derive(Default)]
struct ConversationRequestState {
    /// Whether this filter's `on_request` hook has run for the request.
    request_filters_ran: bool,

    /// Full body captured by an early pre-read pass.
    deferred_body: Option<Bytes>,
}

/// Per-request response-phase state that controls whether append-back
/// should run during `on_response_body`.
struct ConversationResponseState {
    /// The response attempt whose hook selected this append. Request extensions
    /// outlive intermediate IRR responses, including skipped later rounds.
    round: ResponseRound,
    /// A deferred streaming terminal is followed by a separate EOS callback.
    /// Even failure-mode-open paths must not retry a possibly committed write.
    append_attempted: bool,
    /// Owner captured before response body buffering is armed.
    append_owner: Option<StateOwner>,
    /// The response uses the composed SSE terminal instead of buffered JSON.
    streaming: bool,
}

/// A configured response store must write its record before conversation
/// append-back, regardless of the two filters' order in the pipeline.
fn awaiting_response_store(ctx: &HttpFilterContext<'_>) -> bool {
    request_persistence_armed(ctx)
        && !store_response_header_skipped(ctx)
        && ctx.extensions.get::<PersistedResponseForConversation>().is_none()
}

/// Finish a header append after a later response-store hook persisted the
/// canonical response. A store hook that runs first leaves the ordinary
/// Conversations response hook to do this work instead. Ordinary backend
/// errors follow the Store hook's failure mode; request-wide budget errors
/// remain explicit terminal actions regardless of failure mode.
pub(crate) async fn append_after_store_response(ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
    if !conversation_response_selected(ctx) {
        return Ok(FilterAction::Continue);
    }
    OpenaiConversationsFilter.on_response(ctx).await
}

/// Finish a body append after the response-store terminal hook persisted its
/// record and replay events, before the terminal body is released. Ordinary
/// backend errors follow the Store hook's failure mode.
pub(crate) fn append_after_store_body(
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    end_of_stream: bool,
) -> Result<FilterAction, FilterError> {
    if !conversation_body_selected(ctx) {
        return Ok(FilterAction::Continue);
    }
    OpenaiConversationsFilter.on_response_body(ctx, body, end_of_stream)
}

/// A deferred append is eligible only when Conversations ran on this response.
pub(crate) fn conversation_response_selected(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.extensions
        .get::<ConversationResponseState>()
        .is_some_and(|state| state.round == response_round(ctx))
}

/// An outer SSE header is selected once and remains live while inner agentic
/// continuations advance their own iteration and detach IRR's iteration state.
fn conversation_body_selected(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.extensions
        .get::<ConversationResponseState>()
        .is_some_and(|state| state.streaming || state.round == response_round(ctx))
}

/// Owner captured on the request path before inference begins.
struct CapturedAppendOwner(StateOwner);

/// Capture the append-back owner once for the lifetime of the exchange.
fn capture_append_owner(ctx: &mut HttpFilterContext<'_>) -> Result<(), FilterAction> {
    if !should_append_back(ctx) || ctx.extensions.get::<CapturedAppendOwner>().is_some() {
        return Ok(());
    }
    ctx.extensions
        .insert(CapturedAppendOwner(require_state_owner(ctx)?.clone()));
    Ok(())
}

/// Capture the immutable owner after managed-provider validation has published
/// the canonical conversation ID.
///
/// Header hooks run before the bound-body phase, so append-back is not eligible
/// when the Conversations filter first sees a Responses request. The validator
/// calls this helper after publishing the canonical ID; provider-owned traffic
/// skips both filters and therefore never arms local append-back.
pub(crate) fn capture_validated_append_owner(ctx: &mut HttpFilterContext<'_>) {
    if !should_append_back(ctx) || ctx.extensions.get::<CapturedAppendOwner>().is_some() {
        return;
    }
    if let Some(owner) = ctx.extensions.get::<StateOwner>().cloned() {
        ctx.extensions.insert(CapturedAppendOwner(owner));
    }
}

/// Resolve the owner-scoped conversations store from the per-request registry.
///
/// Mirrors the response-store filter: the store is provisioned into the registry
/// on the serving runtime, and the filter takes an owner-bound handle at request
/// time. `None` when no registry is installed or the
/// store is not provisioned.
fn resolve_store(ctx: &HttpFilterContext<'_>, owner: &StateOwner) -> Option<OwnerScopedResponseStore> {
    ctx.extensions
        .get::<ResponseStoreRegistry>()
        .and_then(|registry| registry.get_scoped(CONVERSATIONS_STORE_NAME, owner))
}

impl OpenaiConversationsFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// The config is validated here so a malformed or unknown-backend config
    /// fails at pipeline construction. The serving-runtime provisioner opens the
    /// backend and registers it; the filter only resolves it at request time.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: ConversationsConfig = parse_filter_config("openai_conversations", config)?;
        validate_config(&cfg)?;
        Ok(Box::new(Self))
    }

    /// Resolve the owner-scoped store, or the fail-closed action to return.
    ///
    /// The owner is server-set from trusted request context; the returned store
    /// binds every store operation to it. A missing owner yields the auth action;
    /// an unprovisioned store yields a 500 rejection.
    fn scoped_store(ctx: &HttpFilterContext<'_>) -> Result<OwnerScopedResponseStore, FilterAction> {
        let owner = require_state_owner(ctx)?;
        resolve_store(ctx, owner).ok_or_else(|| FilterAction::Reject(reject_store_unavailable()))
    }

    /// Mark the request phase complete and return any body captured earlier.
    fn mark_request_filters_ran(ctx: &mut HttpFilterContext<'_>) -> Option<Bytes> {
        ctx.current_filter_id?;
        let mut state = ctx
            .remove_filter_state::<ConversationRequestState>()
            .unwrap_or_default();
        state.request_filters_ran = true;
        let deferred_body = state.deferred_body.take();
        ctx.insert_filter_state(state);
        deferred_body
    }

    /// Whether it is safe for the body hook to mutate the local store.
    fn request_filters_ran(ctx: &HttpFilterContext<'_>) -> bool {
        ctx.current_filter_id.is_none()
            || ctx
                .get_filter_state::<ConversationRequestState>()
                .is_some_and(|state| state.request_filters_ran)
    }

    /// Store a complete request body for handling once `on_request` runs.
    fn defer_body_until_request_filters(ctx: &mut HttpFilterContext<'_>, body: Option<&Bytes>) -> FilterAction {
        let mut state = ctx
            .remove_filter_state::<ConversationRequestState>()
            .unwrap_or_default();
        // The body hook only lends this chunk, while dispatch happens after
        // the request-header phase. `Bytes::clone` retains the shared buffer
        // across that ownership boundary without copying its payload.
        state.deferred_body = Some(body.cloned().unwrap_or_default());
        ctx.insert_filter_state(state);
        FilterAction::Release
    }

    /// Drop a body captured during pre-read when header classification shows
    /// that this filter will not handle the request locally.
    fn discard_request_state(ctx: &mut HttpFilterContext<'_>) {
        drop(ctx.remove_filter_state::<ConversationRequestState>());
    }

    /// Recover a matched parameter from the immutable original request path.
    fn path_parameter<'a>(
        ctx: &'a HttpFilterContext<'_>,
        matched: &OpenAiOperationMatch,
        name: &str,
    ) -> Option<&'a str> {
        matched.path_parameters.get(ctx.request.uri.path(), name)
    }

    /// Resolve and validate the Conversations operation from generic classifier state.
    fn matched_operation(
        ctx: &HttpFilterContext<'_>,
    ) -> Result<Option<(OpenAiOperationMatch, ConversationOperation)>, FilterError> {
        let Some(matched) = ctx.extensions.get::<OpenAiOperationMatch>().copied() else {
            return Ok(None);
        };
        if matched.application_protocol != APPLICATION_PROTOCOL {
            return Ok(None);
        }
        let operation = ConversationOperation::from_operation_id(matched.operation_id).ok_or_else(|| {
            FilterError::from(format!(
                "openai_conversations: unknown operation ID {:?}",
                matched.operation_id
            ))
        })?;
        Self::validate_operation_match(matched, operation)?;
        Ok(Some((matched, operation)))
    }

    /// Validate that generic classifier metadata describes the registry operation.
    fn validate_operation_match(
        matched: OpenAiOperationMatch,
        operation: ConversationOperation,
    ) -> Result<(), FilterError> {
        let expected_body = operation.request_body();
        if matched.application_protocol != operation.application_protocol() {
            return Err(FilterError::from(format!(
                "openai_conversations: operation {operation:?} belongs to a different generic protocol"
            )));
        }
        if matched.operation_id != operation.operation_id() || matched.transport != Transport::Http {
            return Err(FilterError::from(format!(
                "openai_conversations: inconsistent generic identity for operation {operation:?}"
            )));
        }
        if matched.request_body != expected_body {
            return Err(FilterError::from(format!(
                "openai_conversations: impossible operation/body combination for {operation:?}: {:?}",
                matched.request_body
            )));
        }
        Ok(())
    }

    /// Dispatch a matched body to the appropriate local handler.
    async fn handle_body_operation(
        ctx: &HttpFilterContext<'_>,
        store: &OwnerScopedResponseStore,
        matched: OpenAiOperationMatch,
        operation: ConversationOperation,
        body: &[u8],
    ) -> Result<FilterAction, FilterError> {
        match operation {
            ConversationOperation::CreateConversation => handlers::handle_create_conversation(ctx, store, body).await,
            ConversationOperation::UpdateConversation => {
                let id = Self::path_parameter(ctx, &matched, "conversation_id")
                    .ok_or_else(|| FilterError::from("openai_conversations: matched update route missing id"))?;
                handlers::handle_update_conversation(store, id, body).await
            },
            ConversationOperation::CreateConversationItems => {
                let id = Self::path_parameter(ctx, &matched, "conversation_id")
                    .ok_or_else(|| FilterError::from("openai_conversations: matched item create route missing id"))?;
                handlers::handle_create_items(ctx, store, id, body).await
            },
            ConversationOperation::GetConversation
            | ConversationOperation::DeleteConversation
            | ConversationOperation::ListConversationItems
            | ConversationOperation::GetConversationItem
            | ConversationOperation::DeleteConversationItem => Err(FilterError::from(format!(
                "openai_conversations: body dispatch called for bodyless operation {operation:?}"
            ))),
        }
    }

    /// Arm request-body buffering for a body-carrying operation and, when an
    /// earlier filter has already pre-read the body, dispatch it immediately.
    async fn begin_body_operation(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        matched: OpenAiOperationMatch,
        operation: ConversationOperation,
    ) -> Result<FilterAction, FilterError> {
        ctx.set_request_body_mode(BodyMode::StreamBuffer {
            max_bytes: Some(MAX_JSON_BODY_BYTES),
        });
        let Some(body) = Self::mark_request_filters_ran(ctx) else {
            return Ok(FilterAction::Continue);
        };
        let store = match Self::scoped_store(ctx) {
            Ok(store) => store,
            Err(action) => return Ok(action),
        };
        Box::pin(Self::handle_body_operation(ctx, &store, matched, operation, &body)).await
    }

    /// Dispatch a bodyless conversation operation to its local handler.
    #[expect(clippy::too_many_lines, reason = "one arm per bodyless endpoint")]
    async fn dispatch_read_operation(
        &self,
        ctx: &HttpFilterContext<'_>,
        matched: OpenAiOperationMatch,
        operation: ConversationOperation,
    ) -> Result<FilterAction, FilterError> {
        let store = match Self::scoped_store(ctx) {
            Ok(store) => store,
            Err(action) => return Ok(action),
        };
        match operation {
            ConversationOperation::GetConversation => {
                let id = Self::path_parameter(ctx, &matched, "conversation_id")
                    .ok_or_else(|| FilterError::from("openai_conversations: matched get route missing id"))?;
                handlers::handle_get_conversation(&store, id).await
            },
            ConversationOperation::ListConversationItems => {
                let id = Self::path_parameter(ctx, &matched, "conversation_id")
                    .ok_or_else(|| FilterError::from("openai_conversations: matched list route missing id"))?;
                handlers::handle_list_items(ctx, &store, id).await
            },
            ConversationOperation::GetConversationItem => {
                let id = Self::path_parameter(ctx, &matched, "conversation_id")
                    .ok_or_else(|| FilterError::from("openai_conversations: matched get item route missing id"))?;
                let item_id = Self::path_parameter(ctx, &matched, "item_id")
                    .ok_or_else(|| FilterError::from("openai_conversations: matched get item route missing item id"))?;
                handlers::handle_get_item(ctx, &store, id, item_id).await
            },
            ConversationOperation::DeleteConversation => {
                let id = Self::path_parameter(ctx, &matched, "conversation_id")
                    .ok_or_else(|| FilterError::from("openai_conversations: matched delete route missing id"))?;
                handlers::handle_delete_conversation(&store, id).await
            },
            ConversationOperation::DeleteConversationItem => {
                let id = Self::path_parameter(ctx, &matched, "conversation_id")
                    .ok_or_else(|| FilterError::from("openai_conversations: matched delete item route missing id"))?;
                let item_id = Self::path_parameter(ctx, &matched, "item_id").ok_or_else(|| {
                    FilterError::from("openai_conversations: matched delete item route missing item id")
                })?;
                handlers::handle_delete_item(&store, id, item_id).await
            },
            ConversationOperation::CreateConversation
            | ConversationOperation::UpdateConversation
            | ConversationOperation::CreateConversationItems => Err(FilterError::from(format!(
                "openai_conversations: bodyless dispatch called for body operation {operation:?}"
            ))),
        }
    }

    /// Persist conversation items synchronously using `block_in_place`.
    ///
    /// The store is resolved for the captured append owner, so the handle is
    /// bound to the same owner the exchange authenticated as.
    fn append_items_blocking(
        owner: &StateOwner,
        conversation_id: &str,
        ctx: &HttpFilterContext<'_>,
        items: Vec<Value>,
        body_bytes: usize,
    ) -> Result<(), AppendError> {
        let store = resolve_store(ctx, owner).ok_or_else(|| {
            AppendError::Store(FilterError::from(
                "openai_conversations: store unavailable for append-back",
            ))
        })?;

        let handle = tokio::runtime::Handle::current();
        tokio::task::block_in_place(|| handle.block_on(persist_items(&store, conversation_id, ctx, items, body_bytes)))
    }
}

// -----------------------------------------------------------------------------
// HttpFilter Implementation
// -----------------------------------------------------------------------------

#[async_trait]
impl HttpFilter for OpenaiConversationsFilter {
    fn name(&self) -> &'static str {
        "openai_conversations"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn bound_upstream_request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn request_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(MAX_JSON_BODY_BYTES),
        }
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::Stream
    }

    fn needs_request_context(&self) -> bool {
        true
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if let Err(action) = capture_append_owner(ctx) {
            return Ok(action);
        }
        let Some((matched, operation)) = Self::matched_operation(ctx)? else {
            // The body hook runs before the request-head classifier during
            // StreamBuffer pre-read, so it may have retained a body for a
            // request that turns out to belong to another protocol. Release
            // that handle before allowing an unrelated request upstream.
            Self::discard_request_state(ctx);
            if match_route(ctx.request.method.as_str(), ctx.request.uri.path()).is_some() {
                // Conversations is a proxy-owned API. A missing classifier
                // match means the dependency was absent, ordered later, or
                // skipped by conditions (including an open failure mode), so
                // forwarding here would silently bypass local handling.
                return Ok(FilterAction::Reject(reject_classifier_unavailable()));
            }
            return Ok(FilterAction::Continue);
        };

        // The classifier publishes registry body metadata. Pair it with the
        // typed operation so corrupt or manually fabricated extension state
        // fails closed instead of selecting the wrong dispatch phase.
        if matched.request_body.is_present() {
            Box::pin(self.begin_body_operation(ctx, matched, operation)).await
        } else {
            // Bodyless local operations never consume a deferred body. They
            // terminate locally, but clearing the state keeps this invariant
            // explicit and avoids retaining it through response handling.
            Self::discard_request_state(ctx);
            Box::pin(self.dispatch_read_operation(ctx, matched, operation)).await
        }
    }

    async fn on_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        // StreamBuffer pre-reading runs before request-header filters. At that
        // point the classifier has not published an operation yet, so retain
        // the completed bytes without trying to infer identity from the URI.
        if !Self::request_filters_ran(ctx) {
            return Ok(Self::defer_body_until_request_filters(ctx, body.as_ref()));
        }

        if let Err(action) = capture_append_owner(ctx) {
            return Ok(action);
        }
        let Some((matched, operation)) = Self::matched_operation(ctx)? else {
            return Ok(FilterAction::Continue);
        };
        if !matched.request_body.is_present() {
            return Ok(FilterAction::Continue);
        }

        let empty: &[u8] = &[];
        let bytes = body.as_ref().map_or(empty, |b| b.as_ref());
        let store = match Self::scoped_store(ctx) {
            Ok(store) => store,
            Err(action) => return Ok(action),
        };
        Box::pin(Self::handle_body_operation(ctx, &store, matched, operation, bytes)).await
    }

    async fn on_bound_upstream_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<BoundUpstreamBodyOutcome, FilterError> {
        let action = self.on_request_body(ctx, body, true).await?;
        bound_body_outcome(action)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "append-back eligibility and owner capture are one response-phase decision"
    )]
    async fn on_response(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        if !should_append_back(ctx) {
            ctx.extensions.insert(ConversationResponseState {
                round: response_round(ctx),
                append_attempted: false,
                append_owner: None,
                streaming: false,
            });
            return Ok(FilterAction::Continue);
        }

        let resp = ctx.response_header.as_ref();
        let is_success = resp.is_none_or(|r| r.status.is_success());
        let content_type = resp
            .and_then(|r| r.headers.get(http::header::CONTENT_TYPE))
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default();
        let is_json = content_type
            .split(';')
            .next()
            .unwrap_or_default()
            .trim()
            .eq_ignore_ascii_case("application/json");
        let is_stream = is_streaming_request(ctx) && is_event_stream_content_type(content_type);

        let armed = is_success && (is_json || is_stream);
        if !armed {
            trace!("conversation append-back skipped (non-2xx or unsupported response content type)");
        }
        if armed {
            let owner = ctx
                .extensions
                .get::<CapturedAppendOwner>()
                .map(|captured| captured.0.clone())
                .ok_or_else(|| FilterError::from("openai_conversations: append-back owner was not captured"))?;
            ctx.extensions.insert(ConversationResponseState {
                round: response_round(ctx),
                append_attempted: false,
                append_owner: Some(owner),
                streaming: is_stream,
            });
            if is_json {
                // An agentic terminal has already been canonicalized by the
                // inner IRR before the outer response header hooks run. For a
                // budgeted conversation turn, append here so a rejection can
                // carry a Responses JSON body before headers are committed.
                if ctx
                    .extensions
                    .get::<ResponsesState>()
                    .is_some_and(|state| state.retained_payload_limit().is_some())
                {
                    let status = ctx
                        .extensions
                        .get::<ResponsesState>()
                        .and_then(|state| state.response_object.get("status"))
                        .and_then(Value::as_str);
                    let completed = crate::openai::responses::buffered_canonical_completed(ctx);
                    let explicitly_unsuccessful = matches!(status, Some("incomplete" | "failed"));
                    if completed {
                        if awaiting_response_store(ctx) {
                            return Ok(FilterAction::Continue);
                        }
                        let mut no_body = None;
                        if !canonical_append_is_empty(ctx) {
                            let wire_bytes = ctx
                                .extensions
                                .get::<ResponsesState>()
                                .and_then(|state| retained_json_bytes(&state.response_object))
                                .ok_or_else(|| FilterError::from("openai_conversations: response size overflow"))?;
                            if !append_extraction_fits(ctx, &no_body, true) {
                                return conversation_budget_failure(ctx, false, &mut no_body);
                            }
                            let append_owner = ctx
                                .extensions
                                .get::<ConversationResponseState>()
                                .and_then(|state| state.append_owner.clone())
                                .ok_or_else(|| FilterError::from("openai_conversations: append owner missing"))?;
                            let items = extract_streaming_append_back_items(ctx, append_owner)
                                .ok_or_else(|| FilterError::from("openai_conversations: append items missing"))?;
                            if !append_record_construction_fits(ctx, &items.all_items) {
                                return conversation_budget_failure(ctx, false, &mut no_body);
                            }
                            let store = resolve_store(ctx, &items.owner).ok_or_else(|| {
                                FilterError::from("openai_conversations: store unavailable for append-back")
                            })?;
                            match persist_items(&store, &items.conversation_id, ctx, items.all_items, wire_bytes).await
                            {
                                Ok(()) => {},
                                Err(AppendError::Budget) => {
                                    return conversation_budget_failure(ctx, false, &mut no_body);
                                },
                                Err(AppendError::Store(error)) => return Err(error),
                            }
                        }
                        if let Some(state) = ctx.extensions.get_mut::<ConversationResponseState>() {
                            state.append_attempted = true;
                        }
                    } else {
                        // A selected branch may skip the agentic loop while
                        // retaining its request-wide budget policy. In that
                        // case the completed upstream body is the only source
                        // of append items. Bound the shared framework buffer
                        // to the remaining headroom before its first chunk.
                        let Some(max_bytes) = buffered_response_headroom(ctx) else {
                            return conversation_budget_failure(ctx, false, &mut None);
                        };
                        if existing_response_buffer_exceeds(ctx, max_bytes) {
                            // Core ratchets to the larger StreamBuffer cap. A
                            // prior filter's buffer cannot be narrowed here,
                            // so reject before it can retain unadmitted bytes.
                            return conversation_budget_failure(ctx, false, &mut None);
                        }
                        ctx.set_response_body_mode(BodyMode::StreamBuffer {
                            max_bytes: Some(max_bytes),
                        });
                        if explicitly_unsuccessful
                            && let Some(state) = ctx.extensions.get_mut::<ConversationResponseState>()
                        {
                            state.append_attempted = true;
                        }
                    }
                } else {
                    ctx.set_response_body_mode(BodyMode::StreamBuffer {
                        max_bytes: Some(MAX_JSON_BODY_BYTES),
                    });
                }
            }
        } else {
            ctx.extensions.insert(ConversationResponseState {
                round: response_round(ctx),
                append_attempted: false,
                append_owner: None,
                streaming: false,
            });
        }

        Ok(FilterAction::Continue)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "terminal append and rollback decisions share one callback"
    )]
    fn on_response_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        let response_state = ctx.extensions.get::<ConversationResponseState>();
        if !conversation_body_selected(ctx)
            || response_state.is_none_or(|state| state.append_owner.is_none() || state.append_attempted)
        {
            // This filter is composed with other response-body consumers, such
            // as `openai_response_store`. Releasing here drains a shared
            // StreamBuffer before those filters see end-of-stream, which can
            // turn a complete chunked response into several unpersistable
            // chunks (#1265). A filter that has no work for this exchange must
            // leave release ownership to the pipeline as a whole.
            return Ok(FilterAction::Continue);
        }
        let streaming = response_state.is_some_and(|state| state.streaming);

        if awaiting_response_store(ctx) {
            return Ok(FilterAction::Continue);
        }

        if streaming {
            // Deferred terminals are delivered in this non-EOS chunk. Local
            // completions leave the flag unset and carry the full terminal in
            // a single IRR chunk before the empty EOS callback. A completed-looking
            // state without either frame must not append after a failed stream.
            if !streaming_terminal_emitted(ctx) && !contains_completed_terminal(body) {
                return Ok(FilterAction::Continue);
            }
        } else if !end_of_stream {
            return Ok(FilterAction::Continue);
        }
        let Some(append_owner) = response_state.and_then(|state| state.append_owner.clone()) else {
            return Ok(FilterAction::Continue);
        };

        if canonical_append_is_empty(ctx) {
            if let Some(state) = ctx.extensions.get_mut::<ConversationResponseState>() {
                state.append_attempted = true;
            }
            return Ok(FilterAction::Continue);
        }

        if !append_extraction_fits(ctx, body, streaming) {
            return conversation_budget_failure(ctx, streaming, body);
        }
        let items = if streaming {
            extract_streaming_append_back_items(ctx, append_owner)
        } else {
            extract_append_back_items(ctx, body, append_owner)
        };
        let Some(items) = items else {
            return Ok(FilterAction::Continue);
        };
        if !append_record_construction_fits(ctx, &items.all_items) {
            return conversation_budget_failure(ctx, streaming, body);
        }

        let conv_id = items.conversation_id;
        // Append before the completed JSON body or streaming terminal frame is
        // released. Under the default `failure_mode: closed`, a persistence
        // failure withholds that success from the client (#837). Pingora may
        // return a clean 500 if headers are unflushed, or reset a committed 2xx.
        // `failure_mode: open` is an explicit operator opt-out of that guarantee:
        // the pipeline logs this error and releases the terminal even though
        // items were lost. Item insertion and cache rebuild are transactional.
        if let Some(state) = ctx.extensions.get_mut::<ConversationResponseState>() {
            state.append_attempted = true;
        }
        match Self::append_items_blocking(
            &items.owner,
            &conv_id,
            ctx,
            items.all_items,
            body.as_ref().map_or(0, Bytes::len),
        ) {
            Ok(()) => {},
            Err(AppendError::Budget) => return conversation_budget_failure(ctx, streaming, body),
            Err(AppendError::Store(error)) => {
                warn!(error = %error, conversation_id = %conv_id, "conversation append-back failed");
                return Err(error);
            },
        }

        Ok(FilterAction::Continue)
    }
}

/// Whether this request should trigger conversation append-back on
/// the response path.
fn should_append_back(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.get_metadata("openai_responses_format.has_conversation") == Some("true")
        && ctx.get_metadata("responses.conversation_id").is_some()
        && ctx.get_metadata("openai_responses_format.background") != Some("true")
}

/// Whether the classified Responses request selected streamed delivery.
fn is_streaming_request(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.get_metadata("openai_responses_format.stream") == Some("true")
}

/// Whether the stream composer placed its canonical terminal in this chunk.
fn streaming_terminal_emitted(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.extensions
        .get::<ResponsesState>()
        .is_some_and(|state| state.logical_stream_terminal_emitted)
}

/// Prove no append records exist only after the loop finalized this turn.
fn canonical_append_is_empty(ctx: &HttpFilterContext<'_>) -> bool {
    crate::openai::responses::buffered_canonical_completed(ctx)
        && ctx.extensions.get::<ResponsesState>().is_some_and(|state| {
            state.input.is_empty()
                && state
                    .response_object
                    .get("output")
                    .and_then(Value::as_array)
                    .is_some_and(Vec::is_empty)
        })
}

/// Bound a conditional noncanonical JSON buffer by unused request allowance.
fn buffered_response_headroom(ctx: &HttpFilterContext<'_>) -> Option<usize> {
    let state = ctx.extensions.get::<ResponsesState>()?;
    let limit = state.retained_payload_limit()?;
    let current = state.retained_payload_bytes_bounded(limit)?;
    limit.checked_sub(current).map(|bytes| bytes.min(MAX_JSON_BODY_BYTES))
}

/// Detect a prior filter buffer that core's ratchet cannot narrow.
fn existing_response_buffer_exceeds(ctx: &HttpFilterContext<'_>, cap: usize) -> bool {
    match &ctx.response_body_mode {
        BodyMode::StreamBuffer { max_bytes } => max_bytes.is_none_or(|existing| existing > cap),
        _ => false,
    }
}

/// Check the canonical local-completion frame delivered as one IRR chunk. The
/// composer writes this ASCII event header in one chunk; no SSE body is accumulated.
fn contains_completed_terminal(body: &Option<Bytes>) -> bool {
    const EVENT_HEADER: &[u8] = b"event: response.completed\n";
    body.as_deref()
        .is_some_and(|chunk| chunk.windows(EVENT_HEADER.len()).any(|window| window == EVENT_HEADER))
}

/// Admit every new JSON owner before parsing buffered output or cloning the
/// canonical streaming output. The body remains live during this operation.
#[expect(
    clippy::too_many_lines,
    reason = "both buffered parsing and canonical clone owners need one admission"
)]
fn append_extraction_fits(ctx: &HttpFilterContext<'_>, body: &Option<Bytes>, streaming: bool) -> bool {
    let Some(state) = ctx.extensions.get::<ResponsesState>() else {
        return true;
    };
    if state.retained_payload_limit().is_none() {
        return true;
    }
    let input_bytes = retained_json_values_bytes(&state.input);
    let staging = if streaming {
        let output_bytes = state
            .response_object
            .get("output")
            .and_then(Value::as_array)
            .map_or(Some(0), |output| retained_json_values_bytes(output));
        input_bytes
            .and_then(|bytes| bytes.checked_add(output_bytes?))
            .and_then(|bytes| {
                if body.is_none() {
                    bytes.checked_add(retained_json_bytes(&state.response_object)?)
                } else {
                    Some(bytes)
                }
            })
    } else {
        let Some(bytes) = body.as_ref() else {
            return true;
        };
        input_bytes.and_then(|input| input.checked_add(buffered_parsed_json_bytes_upper_bound(bytes)?))
    };
    staging
        .and_then(|bytes| bytes.checked_add(body.as_ref().map_or(0, Bytes::len)))
        .is_some_and(|bytes| state.can_retain_payload(bytes))
}

/// Record normalization can add IDs, statuses, and structured message parts.
/// Admit that transient expansion before consuming the extracted item vector.
fn append_record_construction_fits(ctx: &HttpFilterContext<'_>, items: &[Value]) -> bool {
    let Some(state) = ctx.extensions.get::<ResponsesState>() else {
        return true;
    };
    if state.retained_payload_limit().is_none() {
        return true;
    }
    retained_json_values_bytes(items)
        .and_then(|bytes| bytes.checked_mul(4))
        .and_then(|bytes| bytes.checked_add(items.len().checked_mul(512)?))
        .is_some_and(|bytes| state.can_retain_payload(bytes))
}

/// Hold normalized records and their SQL serialization/bind while reserving
/// the remaining allowance for the full owner-scoped cache rebuild.
fn append_rebuild_allowance(
    ctx: &HttpFilterContext<'_>,
    records: &[crate::store::ConversationItemRecord],
    body_bytes: usize,
) -> Result<Option<usize>, AppendError> {
    let Some(state) = ctx.extensions.get::<ResponsesState>() else {
        return Ok(None);
    };
    let Some(limit) = state.retained_payload_limit() else {
        return Ok(None);
    };
    let record_bytes = records.iter().try_fold(0_usize, |used, record| {
        used.checked_add(retained_json_bytes(&record.item_data)?)?
            .checked_add(record.item_id.len())?
            .checked_add(record.conversation_id.len())?
            .checked_add(record.owner.tenant_id().len())?
            .checked_add(record.owner.issuer().len())?
            .checked_add(record.owner.subject().len())
    });
    let staging = record_bytes
        .and_then(|bytes| bytes.checked_mul(3))
        .and_then(|bytes| bytes.checked_add(body_bytes))
        .ok_or(AppendError::Budget)?;
    let current = state.retained_payload_bytes_bounded(limit).ok_or(AppendError::Budget)?;
    let remaining = limit
        .checked_sub(current)
        .and_then(|bytes| bytes.checked_sub(staging))
        .ok_or(AppendError::Budget)?;
    Ok(Some(remaining))
}

/// Suppress success and remove a response persisted earlier in this exchange.
/// A marker is installed only after the Responses store's own durable write.
#[expect(
    clippy::too_many_lines,
    reason = "rollback and terminal wire error must be sequenced together"
)]
fn conversation_budget_failure(
    ctx: &mut HttpFilterContext<'_>,
    streaming: bool,
    body: &mut Option<Bytes>,
) -> Result<FilterAction, FilterError> {
    if let Some(marker) = ctx.extensions.remove::<PersistedResponseForConversation>() {
        let owner = ctx
            .extensions
            .get::<CapturedAppendOwner>()
            .ok_or_else(|| FilterError::from("openai_conversations: append owner missing for rollback"))?;
        let store = ctx
            .extensions
            .get::<ResponseStoreRegistry>()
            .and_then(|registry| registry.get_scoped(crate::openai::responses::DEFAULT_STORE_NAME, &owner.0))
            .ok_or_else(|| FilterError::from("openai_conversations: response store missing for rollback"))?;
        let handle = tokio::runtime::Handle::current();
        tokio::task::block_in_place(|| handle.block_on(store.delete_response(&marker.0)))
            .map_err(|error| -> FilterError { Box::new(error) })?;
    }
    if let Some(state) = ctx.extensions.get_mut::<ConversationResponseState>() {
        state.append_attempted = true;
    }
    ctx.set_metadata("responses.skip_persist", "true");
    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        state.discard_payload_for_budget_error();
    }
    if streaming {
        crate::openai::responses::fs_end_stream_with_error_ctx(ctx, "server_error", APPEND_BUDGET_MESSAGE);
        *body = crate::openai::responses::stream_events::encode_local_error(ctx, "server_error", APPEND_BUDGET_MESSAGE);
        return Ok(FilterAction::Continue);
    }
    Ok(FilterAction::Reject(responses_error_rejection(
        502,
        "server_error",
        APPEND_BUDGET_MESSAGE,
    )))
}

// -----------------------------------------------------------------------------
// Append-Back
// -----------------------------------------------------------------------------

/// Collected items for append-back persistence.
struct AppendBackItems {
    /// Target conversation ID.
    conversation_id: String,
    /// Immutable owner scope for the conversation.
    owner: StateOwner,
    /// Input + output items to persist.
    all_items: Vec<Value>,
}

/// Extract and merge input+output items from the response body for
/// append-back. Returns `None` when there is nothing to persist.
fn extract_append_back_items(
    ctx: &HttpFilterContext<'_>,
    body: &Option<Bytes>,
    owner: StateOwner,
) -> Option<AppendBackItems> {
    let bytes = body.as_ref().filter(|b| !b.is_empty())?;
    let conv_id = ctx.get_metadata("responses.conversation_id")?.to_owned();

    let all_items = merge_input_output_items(ctx, bytes)?;

    Some(AppendBackItems {
        conversation_id: conv_id,
        owner,
        all_items,
    })
}

/// Use the stream composer's canonical terminal resource. The event bytes stay
/// streaming; only the completed output items cross the persistence boundary.
fn extract_streaming_append_back_items(ctx: &HttpFilterContext<'_>, owner: StateOwner) -> Option<AppendBackItems> {
    let state = ctx.extensions.get::<ResponsesState>()?;
    if state.response_object.get("status").and_then(Value::as_str) != Some("completed") {
        return None;
    }
    let conv_id = ctx.get_metadata("responses.conversation_id")?.to_owned();
    // The canonical resource remains available to response-store and terminal
    // handling, so its output must be copied at this durable item boundary.
    let mut all_items = state.input.clone();
    if let Some(output) = state.response_object.get("output").and_then(Value::as_array) {
        all_items.extend(output.iter().cloned());
    }
    if all_items.is_empty() {
        return None;
    }
    Some(AppendBackItems {
        conversation_id: conv_id,
        owner,
        all_items,
    })
}

/// Parse the response body and combine request input items with
/// response output items. Returns `None` when both are empty.
fn merge_input_output_items(ctx: &HttpFilterContext<'_>, bytes: &[u8]) -> Option<Vec<Value>> {
    let mut response_json: Value = match serde_json::from_slice(bytes) {
        Ok(v) => v,
        Err(e) => {
            warn!(error = %e, "conversation append-back: invalid response JSON");
            return None;
        },
    };

    {
        let status = response_json.get("status").and_then(Value::as_str).unwrap_or_default();
        if status != "completed" {
            trace!(status, "conversation append-back skipped (response not completed)");
            return None;
        }
    }

    let output_items = match response_json.get_mut("output").map(Value::take) {
        Some(Value::Array(items)) => items,
        _ => Vec::new(),
    };

    let input_items = ctx
        .extensions
        .get::<ResponsesState>()
        .map(|state| state.input.clone())
        .unwrap_or_default();

    if input_items.is_empty() && output_items.is_empty() {
        return None;
    }

    let mut all_items = input_items;
    all_items.extend(output_items);
    Some(all_items)
}

/// Persist items and refresh the denormalized message cache.
///
/// Records are built with the handle's bound owner, so the owner-scoped write
/// path accepts them; a record under any other owner would be rejected.
async fn persist_items(
    store: &OwnerScopedResponseStore,
    conversation_id: &str,
    ctx: &HttpFilterContext<'_>,
    items: Vec<Value>,
    body_bytes: usize,
) -> Result<(), AppendError> {
    let created_at = handlers::current_timestamp(ctx);

    let records = build_item_records(store.owner(), conversation_id, created_at, 0, items, || {
        handlers::generated_item_id(ctx)
    })
    .map_err(|e| AppendError::Store(Box::new(e)))?;

    if records.is_empty() {
        return Ok(());
    }

    let count = records.len();
    let allowance = append_rebuild_allowance(ctx, &records, body_bytes)?;
    let result = if let Some(max_rebuild_bytes) = allowance {
        store
            .create_items_and_sync_messages_bounded(conversation_id, &records, max_rebuild_bytes)
            .await
    } else {
        store.create_items_and_sync_messages(conversation_id, &records).await
    };
    match result {
        Ok(()) => {},
        Err(StoreError::PayloadTooLarge) => return Err(AppendError::Budget),
        Err(error) => return Err(AppendError::Store(Box::new(error))),
    }

    debug!(conversation_id, count, "conversation items appended from response");

    Ok(())
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Build a 500 rejection when the store is unavailable.
fn reject_store_unavailable() -> Rejection {
    let body = serde_json::json!({
        "error": {
            "message": "Internal server error.",
            "type": "server_error",
        }
    });
    Rejection::status(500)
        .with_header("content-type", "application/json")
        .with_body(serde_json::to_vec(&body).unwrap_or_default())
}

/// Build a 500 rejection when the required operation classifier did not run.
fn reject_classifier_unavailable() -> Rejection {
    let body = serde_json::json!({
        "error": {
            "message": "Internal server error.",
            "type": "server_error",
        }
    });
    Rejection::status(500)
        .with_header("content-type", "application/json")
        .with_body(serde_json::to_vec(&body).unwrap_or_default())
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    #[test]
    fn reject_store_unavailable_returns_500_server_error() {
        let rejection = reject_store_unavailable();
        assert_eq!(rejection.status, 500);
        let body: Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["message"], "Internal server error.");
    }

    #[test]
    fn reject_store_unavailable_sets_json_content_type() {
        let rejection = reject_store_unavailable();
        let ct = rejection
            .headers
            .iter()
            .find(|(k, _)| k == "content-type")
            .map(|(_, v)| v.as_str());
        assert_eq!(ct, Some("application/json"), "should set application/json content-type");
    }

    #[test]
    fn reject_classifier_unavailable_returns_500_server_error() {
        let rejection = reject_classifier_unavailable();
        assert_eq!(rejection.status, 500);
        let body: Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["message"], "Internal server error.");
    }
}
