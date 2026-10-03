// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Aggregate budget failures shared by request and response filters.

use praxis_filter::{FilterAction, HttpFilterContext, IterationState, Rejection};

use super::{
    ObservedResponsesSse,
    error::responses_error_rejection,
    state::ResponsesState,
    stream_events::{encode_local_error, encode_retained_payload_error},
};

/// Both the Responses loop and IRR must still be on their first round.
fn is_initial_request(ctx: &HttpFilterContext<'_>) -> bool {
    ctx.extensions
        .get::<ResponsesState>()
        .is_none_or(|state| state.iteration == 0)
        && ctx
            .extensions
            .get::<IterationState>()
            .is_none_or(|step| step.iteration() == 0)
}

/// Drop retained payload and reject in the wire format already established
/// by the logical response. An IRR step can run after SSE was committed even
/// when a buffered step has replaced `IterationState.previous_response`.
pub(super) fn reject_retained_payload_budget(ctx: &mut HttpFilterContext<'_>, message: &str) -> FilterAction {
    let initial = is_initial_request(ctx);
    reject_retained_payload_budget_with_status(ctx, message, initial)
}

/// Reject payload growth caused by a provider response as a server error,
/// including when it happens in the first inference round.
pub(super) fn reject_retained_payload_response_budget(ctx: &mut HttpFilterContext<'_>, message: &str) -> FilterAction {
    reject_retained_payload_budget_with_status(ctx, message, false)
}

/// Shared cleanup and committed-stream encoding for both admission phases.
fn reject_retained_payload_budget_with_status(
    ctx: &mut HttpFilterContext<'_>,
    message: &str,
    initial: bool,
) -> FilterAction {
    let committed_stream = ctx.extensions.get::<ObservedResponsesSse>().is_some();
    ctx.set_metadata("responses.skip_persist", "true");
    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        state.discard_payload_for_budget_error();
    }
    #[cfg(feature = "store")]
    super::store::discard_retained_request_payload(ctx);
    if committed_stream {
        let body =
            encode_local_error(ctx, "server_error", message).unwrap_or_else(|| encode_retained_payload_error(ctx));
        return FilterAction::Reject(
            Rejection::status(200)
                .with_header("content-type", "text/event-stream")
                .with_body(body)
                .preserving_keepalive(),
        );
    }
    let (status, code) = if initial {
        (413, "invalid_request_error")
    } else {
        (502, "server_error")
    };
    FilterAction::Reject(responses_error_rejection(status, code, message))
}
