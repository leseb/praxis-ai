// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Extracts document content from `input_file` parts and converts
//! them to `input_text` for inference backends that do not natively
//! support `input_file` (e.g. vLLM).
//!
//! Walks `message` content arrays and `function_call_output` output
//! arrays, finds `input_file` parts with inline `file_data`, and
//! replaces text-safe documents with `input_text` containing the
//! decoded UTF-8 text.
//!
//! This filter is an explicitly configured backend adapter. It
//! should only be enabled for routes to backends that cannot consume
//! `input_file` parts directly. For `OpenAI`-compatible backends
//! with native document support, leave this filter out of the
//! pipeline.
//!
//! Runs after `openai_file_resolve` (which resolves `file_id` to
//! inline `file_data`) and before `openai_responses_proxy` (which
//! rebuilds the body from state). Parts without inline `file_data`
//! (unresolved `file_id` or `file_url`) are skipped — this filter
//! does not perform network I/O.
//!
//! Text-safe MIME types (`text/*`, `application/json`,
//! `application/xml`) are decoded from base64 and validated as
//! UTF-8. Unsupported MIME types are either left unchanged
//! (`on_unsupported: continue`) or rejected (`on_unsupported:
//! reject`).
//!
//! When [`ResponsesState`] is present (e.g. after `rehydrate`),
//! converted content is synced back into `state.request_body`,
//! `state.messages`, and `state.persisted_messages` so that
//! `responses_proxy` does not overwrite the rewritten body.
//!
//! [`ResponsesState`]: super::state::ResponsesState

pub(crate) mod config;
mod extract;

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests;

use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, BoundUpstreamBodyOutcome, FilterAction, FilterError, HttpFilter, HttpFilterContext,
    Rejection, body::MAX_JSON_BODY_BYTES, parse_filter_config,
};
use tracing::{debug, trace, warn};

use self::{
    config::{DocExtractConfig, validate_config},
    extract::{ExtractError, ExtractionBudget, extract_input_file, parse_data_uri},
};
use super::{
    agentic_loop::{AgenticBudgetPolicy, buffered_parsed_json_bytes_upper_bound},
    body_limits::reject_rewritten_body_too_large,
    bound_body_outcome,
    content_parts::{content_parts, content_parts_mut, infer_mime_from_filename},
    openai_responses_proxy::serialized_outbound_body_len,
    state::{ResponsesState, retained_json_bytes, retained_json_values_bytes},
};
use crate::{classifier::is_responses_create, json_body::serialize_json_body};

/// Converts `input_file` content parts to `input_text` for backends
/// that do not support `input_file` natively (e.g. vLLM, llm-d).
///
/// # YAML
///
/// ```yaml
/// filter: openai_doc_extract
/// allow_pre_security_callout: true
/// ```
///
/// # Full YAML
///
/// ```yaml
/// filter: openai_doc_extract
/// allow_pre_security_callout: true
/// on_unsupported: continue
/// max_rewritten_body_bytes: 67108864
/// max_content_bytes: 10485760
/// max_file_references: 32
/// max_total_text_bytes: 67108864
/// ```
pub struct DocExtractFilter {
    /// Validated filter configuration.
    config: DocExtractConfig,
}

impl DocExtractFilter {
    /// Create a filter from parsed YAML config.
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the YAML config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: DocExtractConfig = parse_filter_config("openai_doc_extract", config)?;
        let validated = validate_config(cfg)?;

        Ok(Box::new(Self { config: validated }))
    }
}

#[async_trait]
impl HttpFilter for DocExtractFilter {
    fn name(&self) -> &'static str {
        "openai_doc_extract"
    }

    fn request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn bound_upstream_request_body_access(&self) -> BodyAccess {
        BodyAccess::ReadWrite
    }

    fn request_body_mode(&self) -> BodyMode {
        // Accept up to the absolute ceiling; the pipeline's body_limits
        // decides the real raw cap. max_rewritten_body_bytes bounds only
        // the body produced after input_file → input_text conversion.
        BodyMode::StreamBuffer {
            max_bytes: Some(MAX_JSON_BODY_BYTES),
        }
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
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

        if !is_responses_create(&ctx.request.method, ctx.request.uri.path()) {
            trace!("skipping non-create request");
            return Ok(FilterAction::Release);
        }

        if ctx.get_metadata("openai_responses_format.format") != Some("openai_responses") {
            trace!("skipping non-responses request");
            return Ok(FilterAction::Release);
        }

        let Some(raw) = body.as_ref() else {
            trace!("no body, releasing");
            return Ok(FilterAction::Release);
        };

        if !document_parse_fits_budget(ctx, raw) {
            return Ok(reject_retained_extraction_budget(ctx));
        }

        let parsed: serde_json::Value = match serde_json::from_slice(raw) {
            Ok(v) => v,
            Err(e) => {
                debug!(error = %e, "body is not valid JSON, releasing");
                return Ok(FilterAction::Release);
            },
        };

        extract_and_rewrite(self, ctx, body, parsed)
    }

    async fn on_bound_upstream_request_body(
        &self,
        ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
    ) -> Result<BoundUpstreamBodyOutcome, FilterError> {
        let action = self.on_request_body(ctx, body, true).await?;
        bound_body_outcome(action)
    }
}

/// Run extraction on the current input and history, then rewrite
/// the body and sync state.
///
/// Takes ownership of the parsed body so the extracted value can be
/// moved into [`ResponsesState`] instead of deep-cloned.
fn extract_and_rewrite(
    filter: &DocExtractFilter,
    ctx: &mut HttpFilterContext<'_>,
    body: &mut Option<Bytes>,
    mut parsed: serde_json::Value,
) -> Result<FilterAction, FilterError> {
    let Some(raw_len) = body.as_ref().map(Bytes::len) else {
        return Ok(reject_retained_extraction_budget(ctx));
    };
    if extraction_budget_active(ctx)
        && !projected_extraction_peak(ctx, &parsed, raw_len).is_some_and(|bytes| extraction_fits_budget(ctx, bytes))
    {
        return Ok(reject_retained_extraction_budget(ctx));
    }
    let mut budget = ExtractionBudget::new(&filter.config);

    let count = match extract_current_input(&mut parsed, &mut budget) {
        Ok(count) => count,
        Err(e) => return Ok(reject_extract_error(&e)),
    };

    if count == 0 {
        return finish_history_only(ctx, &mut budget, raw_len, filter.config.max_rewritten_body_bytes);
    }

    debug!(count, "extracted input_file parts");
    if let Some(rejection) = rewrite_body(body, &parsed, filter.config.max_rewritten_body_bytes, filter.name())? {
        return Ok(rejection);
    }
    if let Err(e) = sync_state_after_rewrite(ctx, parsed, &mut budget) {
        return Ok(reject_extract_error(&e));
    }
    if !extraction_fits_budget(ctx, body.as_ref().map_or(0, Bytes::len)) {
        return Ok(reject_retained_extraction_budget(ctx));
    }
    if let Some(rejection) = reject_oversized_state_body(ctx, filter.config.max_rewritten_body_bytes)? {
        return Ok(rejection);
    }

    Ok(FilterAction::Continue)
}

/// Complete a request whose current input was unchanged but whose rehydrated
/// history may still contain inline documents.
fn finish_history_only(
    ctx: &mut HttpFilterContext<'_>,
    budget: &mut ExtractionBudget,
    raw_len: usize,
    max_rewritten_body_bytes: usize,
) -> Result<FilterAction, FilterError> {
    trace!("no input_file parts to extract");
    if let Err(error) = extract_state_history(ctx, budget) {
        return Ok(reject_extract_error(&error));
    }
    if !extraction_fits_budget(ctx, raw_len) {
        return Ok(reject_retained_extraction_budget(ctx));
    }
    if let Some(rejection) = reject_oversized_state_body(ctx, max_rewritten_body_bytes)? {
        return Ok(rejection);
    }
    Ok(FilterAction::Continue)
}

/// The framework body and a second parsed tree coexist with shared state.
fn document_parse_fits_budget(ctx: &mut HttpFilterContext<'_>, raw: &Bytes) -> bool {
    if let Some(limit) = ctx
        .extensions
        .get::<AgenticBudgetPolicy>()
        .map(|policy| policy.max_retained_bytes())
        && let Some(state) = ctx.extensions.get_mut::<ResponsesState>()
    {
        state.apply_retained_payload_limit(limit);
    }
    if !extraction_budget_active(ctx) {
        return true;
    }
    // This filter may precede the Responses request classifier. Apply its
    // allocation-free structural admission before constructing a JSON Value;
    // compact arrays can own far more nodes than their wire length suggests.
    if super::initial_budget_rejection(ctx, raw).is_some()
        || !super::initial_json_parse_peak_bytes(raw).is_some_and(|bytes| extraction_fits_budget(ctx, bytes))
    {
        return false;
    }
    raw.len()
        .checked_add(buffered_parsed_json_bytes_upper_bound(raw).unwrap_or(usize::MAX))
        .is_some_and(|bytes| extraction_fits_budget(ctx, bytes))
}

/// Shared limit can be published before `ResponsesState` exists on a direct
/// request, or applied to state by an earlier request-body filter.
fn extraction_budget_active(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.extensions
        .get::<ResponsesState>()
        .and_then(ResponsesState::retained_payload_limit)
        .is_some()
        || ctx.extensions.get::<AgenticBudgetPolicy>().is_some()
}

/// Check a filter-local owner against shared state or the pipeline policy.
fn extraction_fits_budget(ctx: &HttpFilterContext<'_>, additional_bytes: usize) -> bool {
    if let Some(state) = ctx.extensions.get::<ResponsesState>()
        && state.retained_payload_limit().is_some()
    {
        return state.can_retain_payload(additional_bytes);
    }
    ctx.extensions
        .get::<AgenticBudgetPolicy>()
        .is_none_or(|policy| additional_bytes <= policy.max_retained_bytes())
}

/// Bound the simultaneous raw body, parsed tree, rewritten body, decoded text,
/// and two independently owned history-tail copies before any file is decoded.
/// Existing state owners are measured by `extraction_fits_budget` separately.
fn projected_extraction_peak(ctx: &HttpFilterContext<'_>, parsed: &serde_json::Value, raw_len: usize) -> Option<usize> {
    let parsed_bytes = retained_json_bytes(parsed)?;
    let input = parsed.get("input").and_then(serde_json::Value::as_array);
    let current = input.map_or_else(
        || Some(ExtractionGrowth::default()),
        |input| measure_extraction_growth(input),
    )?;
    let history = ctx
        .extensions
        .get::<ResponsesState>()
        .map_or_else(|| Some(ExtractionGrowth::default()), history_extraction_growth)?;
    let rewritten_bytes = if current.extractable {
        parsed_bytes.checked_add(current.json_growth)?
    } else {
        0
    };
    let tail_copies = if current.extractable && ctx.extensions.get::<ResponsesState>().is_some() {
        retained_json_values_bytes(input?)?
            .checked_add(current.json_growth)?
            .checked_mul(2)?
    } else {
        0
    };
    raw_len
        .checked_add(parsed_bytes)?
        .checked_add(rewritten_bytes)?
        .checked_add(tail_copies)?
        .checked_add(history.json_growth)?
        .checked_add(current.largest_text.max(history.largest_text).checked_mul(2)?)
}

/// Project extraction only across history prefixes; current tails are replaced.
fn history_extraction_growth(state: &ResponsesState) -> Option<ExtractionGrowth> {
    let input_len = state.input.len();
    let messages_end = state.messages.len().saturating_sub(input_len);
    let persisted_end = state.persisted_messages.len().saturating_sub(input_len);
    measure_extraction_growth(state.messages.get(..messages_end)?)?.combine(measure_extraction_growth(
        state.persisted_messages.get(..persisted_end)?,
    )?)
}

/// Growth beyond the already charged encoded part. JSON may escape one decoded
/// input byte into six wire bytes (for example `\\u0000`).
#[derive(Clone, Copy, Default)]
struct ExtractionGrowth {
    /// Maximum extra compact-JSON bytes after replacing encoded parts.
    json_growth: usize,
    /// Largest one-file decoded text, including its optional source label.
    largest_text: usize,
    /// Whether at least one text-safe inline file would be converted.
    extractable: bool,
}

impl ExtractionGrowth {
    /// Sum independent owners while retaining the largest transient decode.
    fn combine(self, other: Self) -> Option<Self> {
        Some(Self {
            json_growth: self.json_growth.checked_add(other.json_growth)?,
            largest_text: self.largest_text.max(other.largest_text),
            extractable: self.extractable || other.extractable,
        })
    }
}

/// Project compact-JSON growth for text-safe inline files in these items.
fn measure_extraction_growth(items: &[serde_json::Value]) -> Option<ExtractionGrowth> {
    let mut growth = ExtractionGrowth::default();
    for item in items {
        let Some(parts) = content_parts(item) else {
            continue;
        };
        for part in parts {
            let Some(text_bound) = extractable_text_bound(part).ok()? else {
                continue;
            };
            let new_part_bound = b"{\"type\":\"input_text\",\"text\":\"\"}"
                .len()
                .checked_add(text_bound.checked_mul(6)?)?;
            let part_growth = new_part_bound.saturating_sub(retained_json_bytes(part)?);
            growth.json_growth = growth.json_growth.checked_add(part_growth)?;
            growth.largest_text = growth.largest_text.max(text_bound);
            growth.extractable = true;
        }
    }
    Some(growth)
}

/// Base64 decoding never yields more bytes than its encoded input. Include the
/// optional source label that extraction prefixes to the decoded text.
fn extractable_text_bound(part: &serde_json::Value) -> Result<Option<usize>, ()> {
    if part.get("type").and_then(serde_json::Value::as_str) != Some("input_file") {
        return Ok(None);
    }
    let Some(data) = part.get("file_data").and_then(serde_json::Value::as_str) else {
        return Ok(None);
    };
    let filename = part.get("filename").and_then(serde_json::Value::as_str);
    let mime = parse_data_uri(data)
        .map(|uri| uri.mime)
        .or_else(|| infer_mime_from_filename(filename))
        .unwrap_or("application/octet-stream");
    if !config::is_text_safe_mime(mime) {
        return Ok(None);
    }
    let prefix = filename
        .filter(|name| !name.is_empty())
        .map_or(Some(0), |name| name.len().checked_add(11))
        .ok_or(())?;
    Ok(Some(data.len().checked_add(prefix).ok_or(())?))
}

/// Publish one terminal shared-budget error before another dispatch or store.
fn reject_retained_extraction_budget(ctx: &mut HttpFilterContext<'_>) -> FilterAction {
    super::budget_error::reject_request(
        ctx,
        "agentic retained payload exceeded openai_agentic_loop.max_retained_bytes during document extraction",
    )
}

/// Walk the current request input and extract text-safe `input_file` parts.
fn extract_current_input(parsed: &mut serde_json::Value, budget: &mut ExtractionBudget) -> Result<usize, ExtractError> {
    let Some(items) = parsed.get_mut("input").and_then(serde_json::Value::as_array_mut) else {
        return Ok(0);
    };
    extract_items(items, budget)
}

/// Walk items and extract text-safe `input_file` content parts.
fn extract_items(items: &mut [serde_json::Value], budget: &mut ExtractionBudget) -> Result<usize, ExtractError> {
    let mut count = 0;
    for item in items.iter_mut() {
        count += extract_item_parts(item, budget)?;
    }
    Ok(count)
}

/// Extract text-safe `input_file` parts from a single item.
fn extract_item_parts(item: &mut serde_json::Value, budget: &mut ExtractionBudget) -> Result<usize, ExtractError> {
    let Some(parts) = content_parts_mut(item) else {
        return Ok(0);
    };
    let mut count = 0;
    for part in parts.iter_mut() {
        if part.get("type").and_then(serde_json::Value::as_str) != Some("input_file") {
            continue;
        }
        if let Some(text) = extract_input_file(part, budget)? {
            *part = serde_json::json!({"type": "input_text", "text": text});
            count += 1;
        }
    }
    Ok(count)
}

/// Serialize the extracted JSON and replace the buffered request body.
fn rewrite_body(
    body: &mut Option<Bytes>,
    parsed: &serde_json::Value,
    max_rewritten_body_bytes: usize,
    filter_name: &'static str,
) -> Result<Option<FilterAction>, FilterError> {
    let rewritten = serialize_json_body(parsed)
        .map_err(|e| -> FilterError { format!("{filter_name}: failed to serialize body: {e}").into() })?;
    if rewritten.len() > max_rewritten_body_bytes {
        warn!(
            actual = rewritten.len(),
            limit = max_rewritten_body_bytes,
            "rewritten request body exceeds configured limit"
        );
        return Ok(Some(reject_rewritten_body_too_large(
            rewritten.len(),
            max_rewritten_body_bytes,
        )));
    }
    rewritten.commit(body, filter_name, "input");
    Ok(None)
}

/// Sync converted content back into [`ResponsesState`] after a body
/// rewrite.
///
/// Takes `resolved_body` by value and moves it into `request_body`
/// rather than deep-cloning a tree that may carry inlined file data.
fn sync_state_after_rewrite(
    ctx: &mut HttpFilterContext<'_>,
    resolved_body: serde_json::Value,
    budget: &mut ExtractionBudget,
) -> Result<(), ExtractError> {
    let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
        return Ok(());
    };

    state.request_body = resolved_body;
    state.mark_replay_stable_payload_changed();

    let input_len = state.input.len();
    let ResponsesState {
        request_body,
        messages,
        persisted_messages,
        ..
    } = state;

    let Some(resolved_input) = request_body.get("input").and_then(serde_json::Value::as_array) else {
        return Ok(());
    };

    sync_message_history(messages, input_len, Some(resolved_input), budget)?;
    sync_persisted_history(persisted_messages, input_len, Some(resolved_input), budget)
}

/// Test helper that creates an isolated request extraction budget.
#[cfg(test)]
fn sync_state(
    ctx: &mut HttpFilterContext<'_>,
    resolved_body: serde_json::Value,
    config: &DocExtractConfig,
) -> Result<(), ExtractError> {
    let mut budget = ExtractionBudget::new(config);
    sync_state_after_rewrite(ctx, resolved_body, &mut budget)
}

/// Extract `input_file` parts in rehydrated history when the
/// current input had no `input_file` parts to extract.
fn extract_state_history(ctx: &mut HttpFilterContext<'_>, budget: &mut ExtractionBudget) -> Result<(), ExtractError> {
    let Some(state) = ctx.extensions.get_mut::<ResponsesState>() else {
        return Ok(());
    };

    let input_len = state.input.len();

    sync_message_history(&mut state.messages, input_len, None, budget)?;
    sync_persisted_history(&mut state.persisted_messages, input_len, None, budget)
}

/// Sync the persisted-messages mirror with independent count and
/// byte accounting.
fn sync_persisted_history(
    messages: &mut [serde_json::Value],
    input_len: usize,
    resolved_input: Option<&[serde_json::Value]>,
    budget: &mut ExtractionBudget,
) -> Result<(), ExtractError> {
    let saved = budget.begin_independent_accounting();
    let result = sync_message_history(messages, input_len, resolved_input, budget);
    budget.restore_accounting(&saved);
    result
}

/// Replace the current-input tail, then extract text-safe
/// `input_file` parts from the history prefix.
fn sync_message_history(
    messages: &mut [serde_json::Value],
    input_len: usize,
    resolved_input: Option<&[serde_json::Value]>,
    budget: &mut ExtractionBudget,
) -> Result<(), ExtractError> {
    let Some(history_end) = messages.len().checked_sub(input_len) else {
        return Ok(());
    };
    if let Some(resolved_input) = resolved_input {
        replace_tail(messages, history_end, resolved_input);
    }
    extract_history(messages, history_end, budget)
}

/// Copy resolved input items into the current-input tail of a
/// message vector, starting at `history_end`.
fn replace_tail(messages: &mut [serde_json::Value], history_end: usize, resolved_input: &[serde_json::Value]) {
    for (i, item) in resolved_input.iter().enumerate() {
        if let Some(slot) = messages.get_mut(history_end + i) {
            *slot = item.clone();
        }
    }
}

/// Extract text-safe `input_file` parts from history messages (the
/// prefix before the current input).
fn extract_history(
    messages: &mut [serde_json::Value],
    history_end: usize,
    budget: &mut ExtractionBudget,
) -> Result<(), ExtractError> {
    if history_end == 0 {
        return Ok(());
    }
    let Some(history) = messages.get_mut(..history_end) else {
        return Ok(());
    };
    extract_items(history, budget).map(|_count| ())
}

/// Enforce the body limit against the exact request shape that
/// `openai_responses_proxy` will later serialize from state.
fn reject_oversized_state_body(
    ctx: &HttpFilterContext<'_>,
    max_rewritten_body_bytes: usize,
) -> Result<Option<FilterAction>, FilterError> {
    let Some(state) = ctx.extensions.get::<ResponsesState>() else {
        return Ok(None);
    };
    let len = serialized_outbound_body_len(state).map_err(|e| -> FilterError {
        format!("openai_doc_extract: failed to measure rebuilt request body: {e}").into()
    })?;
    Ok((len > max_rewritten_body_bytes).then(|| {
        warn!(
            actual = len,
            limit = max_rewritten_body_bytes,
            "rebuilt state body exceeds configured limit"
        );
        reject_rewritten_body_too_large(len, max_rewritten_body_bytes)
    }))
}

// -- Error responses ------------------------------------------------

/// Map one extraction error to an HTTP rejection.
fn reject_extract_error(err: &ExtractError) -> FilterAction {
    let (status, message) = extract_error_response(err);

    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "doc_extract_error"
        }
    })
    .to_string();

    FilterAction::Reject(
        Rejection::status(status)
            .with_header("content-type", "application/json")
            .with_body(Bytes::from(body)),
    )
}

/// Map an extraction error to an HTTP status code and message.
fn extract_error_response(err: &ExtractError) -> (u16, String) {
    let (status, message) = match err {
        ExtractError::DecodeFailed { detail } => (400, format!("file_data decode failed: {detail}")),
        ExtractError::TooManyReferences { limit } => (413, format!("request exceeds {limit} input_file references")),
        ExtractError::TooLarge { detail, limit } => (413, format!("extracted content exceeds {limit} bytes: {detail}")),
        ExtractError::Unsupported { mime } => (400, format!("unsupported file type: {mime}")),
    };
    warn!(%err, "extraction error");
    (status, message)
}
