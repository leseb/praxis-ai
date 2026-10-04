// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Wire errors for request-wide retained-payload admission.

use praxis_filter::{FilterAction, HttpFilterContext, IterationState, Rejection};

use super::{
    ObservedResponsesSse,
    error::responses_error_rejection,
    state::ResponsesState,
    stream_events::{encode_local_error, encode_retained_payload_error},
};

/// Reject a request-side admission before dispatching another provider call.
/// The HTTP status is determined by the logical response, not its `stream` bit:
/// the first input is a 413, an uncommitted continuation is a 502, and
/// a committed response requires an in-band error.
pub(crate) fn reject_request(ctx: &mut HttpFilterContext<'_>, message: &str) -> FilterAction {
    FilterAction::Reject(request_rejection(ctx, message))
}

/// The selected upstream body hook needs a `Rejection` directly.
pub(crate) fn request_rejection(ctx: &mut HttpFilterContext<'_>, message: &str) -> Rejection {
    let initial = ctx
        .extensions
        .get::<ResponsesState>()
        .is_none_or(|state| state.iteration == 0)
        && ctx
            .extensions
            .get::<IterationState>()
            .is_none_or(|step| step.iteration() == 0);
    rejection(ctx, message, initial)
}

/// Reject a provider response that exceeds the shared budget.
pub(crate) fn response_rejection(ctx: &mut HttpFilterContext<'_>, message: &str) -> Rejection {
    rejection(ctx, message, false)
}

fn rejection(ctx: &mut HttpFilterContext<'_>, message: &str, initial: bool) -> Rejection {
    let committed = ctx.extensions.get::<ObservedResponsesSse>().is_some()
        || ctx
            .extensions
            .get::<praxis_filter::ClientResponseHeadersCommitted>()
            .is_some();
    ctx.set_metadata("responses.skip_persist", "true");
    if let Some(state) = ctx.extensions.get_mut::<ResponsesState>() {
        state.discard_payload_for_budget_error();
    }
    #[cfg(feature = "store")]
    super::store::discard_retained_request_payload(ctx);

    if committed {
        let body =
            encode_local_error(ctx, "server_error", message).unwrap_or_else(|| encode_retained_payload_error(ctx));
        return Rejection::status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(body)
            .preserving_keepalive();
    }

    let (status, code) = if initial {
        (413, "invalid_request_error")
    } else {
        (502, "server_error")
    };
    responses_error_rejection(status, code, message)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn request_phase_tracks_actual_sse_commitment() {
        let request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
        let mut ctx = crate::test_utils::make_filter_context(&request);
        ctx.extensions
            .insert(ResponsesState::from_request_body(json!({"stream": true})));
        assert_eq!(request_rejection(&mut ctx, "over budget").status, 413);

        let mut continued = ResponsesState::from_request_body(json!({"stream": true}));
        continued.iteration = 1;
        ctx.extensions.insert(continued);
        let precommit = request_rejection(&mut ctx, "over budget");
        assert_eq!(precommit.status, 502, "a stream bit alone cannot claim HTTP 200");

        let mut committed = ResponsesState::from_request_body(json!({"stream": true}));
        committed.iteration = 1;
        ctx.extensions.insert(committed);
        ctx.extensions.insert(ObservedResponsesSse);
        let in_band = request_rejection(&mut ctx, "over budget");
        assert_eq!(in_band.status, 200);
        let body = std::str::from_utf8(in_band.body.as_deref().unwrap()).unwrap();
        assert!(body.contains("event: error"));
        assert!(!body.contains("response.completed"));
        assert!(!body.contains("[DONE]"));

        let mut headers_only = ResponsesState::from_request_body(json!({"stream": true}));
        headers_only.iteration = 1;
        ctx.extensions.insert(headers_only);
        ctx.extensions.remove::<ObservedResponsesSse>();
        ctx.extensions.insert(praxis_filter::ClientResponseHeadersCommitted);
        let first_pull = request_rejection(&mut ctx, "over budget");
        assert_eq!(
            first_pull.status, 200,
            "headers commit the status before the first body pull"
        );
        assert!(
            std::str::from_utf8(first_pull.body.as_deref().unwrap())
                .unwrap()
                .contains("event: error")
        );
    }
}
