// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Streaming restoration of lowered Codex client tools (#1159).
//!
//! `openai_client_tool_compat` lowers rich client tools (`custom`, `namespace`
//! members, local `shell`, client-executed `tool_search`) to private `function`
//! tools on the outbound request. When the model returns a lowered
//! `function_call` in the Responses SSE stream, this module plans how to restore
//! the original typed item live, inside the single `openai_stream_events` logical
//! owner — no second accumulator, no full-body buffering.
//!
//! The plan pass ([`plan_client_tool_restore`]) is fallible and runs between the
//! commit's phase-2a accumulate and phase-2b append; the disposition applier is
//! infallible and runs inside phase 2b. Native passthrough (empty lowering map)
//! skips planning entirely for zero overhead.

use std::collections::HashMap;

use serde_json::Value;

use crate::openai::responses::state::{ClientToolEcho, ClientToolRestore, LoweredClientTool};
use crate::openai::sse::SseParseError;
use crate::openai::sse::responses::ResponsesEvent;

/// Per-item lifecycle progress through a lowered `function_call`'s SSE events.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ClientToolPhase {
    /// `output_item.added` seen; arguments not yet complete.
    Opened,
    /// `function_call_arguments.done` seen; the arguments are final.
    ArgsComplete,
    /// `output_item.done` seen; the item is finalized.
    Done,
}

/// A lowered client-tool item tracked across its streaming lifecycle.
///
/// Created at `output_item.added` and advanced by later argument/finalizer
/// events so the plan pass can enforce lifecycle order (fail closed on a
/// premature `output_item.done`) and, in Tasks 5-7, synthesize the typed
/// restoration for `Custom`/`Shell`/`ToolSearch` kinds.
#[derive(Clone, Debug)]
#[expect(
    dead_code,
    reason = "#1159 Tasks 5-7 read output_index/item_id for EmitCustomInput/EmitCustom* synthesis"
)]
pub(super) struct ClientToolStreamItem {
    /// Stable key matching this item across its lifecycle events (`item:{id}` or
    /// `index:{output_index}`).
    pub key: String,
    /// The private lowered name the backend returned (`agentic_ns__{ns}__{member}`
    /// or a private `custom`/`shell`/`tool_search` name). Never leaked to clients.
    pub private_name: String,
    /// The typed item this lowered `function_call` restores to.
    pub restore: ClientToolRestore,
    /// Lifecycle progress observed so far.
    pub phase: ClientToolPhase,
    /// Absolute output index carried by the item's lifecycle events.
    pub output_index: u64,
    /// The item id (`item.id`), when the backend supplied one.
    pub item_id: Option<String>,
}

/// A completed private `function_call` item captured from accumulated storage at
/// `function_call_arguments.done`, keyed for later typed synthesis (Tasks 5-7).
#[derive(Clone, Debug)]
#[expect(
    dead_code,
    reason = "#1159 Task 5 builds completions in the commit phase-2a capture; consumed by Tasks 5-7 synthesis"
)]
pub(super) struct ClientToolCompletion {
    /// Stable key matching the completion to its tracked stream item.
    pub key: String,
    /// The completed private `function_call` item, cloned from storage.
    pub item: Value,
}

/// How one committed SSE event must be restored before it reaches the client.
///
/// Task 4 constructs only `Passthrough` and `RetypeInPlace`; the remaining
/// dispositions are produced by Tasks 5-7 for `Custom`/`Shell`/`ToolSearch`
/// synthesis and terminal snapshot restoration.
#[derive(Clone, Debug)]
#[expect(
    dead_code,
    reason = "#1159 Tasks 5-7 construct the Suppress/EmitCustom*/EmitTyped*/RestoreSnapshot dispositions"
)]
pub(super) enum ClientToolDisposition {
    /// Forward the event unchanged.
    Passthrough,
    /// Drop the event entirely (the typed replacement is synthesized elsewhere).
    Suppress,
    /// Retype the event's `item` in place: set `type`/`name` and re-add or remove
    /// `namespace`. Used for `Namespace` members (retype-in-place, no synthesis).
    RetypeInPlace {
        /// The restored output-item type (`function_call`).
        item_type: &'static str,
        /// The original member name to restore.
        name: String,
        /// The namespace to re-add, or `None` to remove it.
        namespace: Option<String>,
    },
    /// Emit a synthesized `custom_tool_call` `output_item.added` (Task 6).
    EmitCustomShell {
        /// The synthesized output item.
        item: Value,
    },
    /// Emit a synthesized custom-tool input event (Task 6).
    EmitCustomInput {
        /// Stable key of the tracked stream item.
        key: String,
        /// The item id to stamp on the synthesized event.
        item_id: String,
        /// The absolute output index.
        output_index: u64,
        /// The unwrapped plain-string input.
        input: String,
    },
    /// Emit a synthesized custom-tool `output_item.done` (Task 6).
    EmitCustomItemDone {
        /// The synthesized output item.
        item: Value,
    },
    /// Emit a synthesized typed `output_item.added` (Task 7).
    EmitTypedAdded {
        /// The synthesized output item.
        item: Value,
    },
    /// Emit a synthesized typed `output_item.done` (Task 7).
    EmitTypedDone {
        /// The synthesized output item.
        item: Value,
    },
    /// Restore the terminal response snapshot's output (Task 7).
    RestoreSnapshot {
        /// The restored response object.
        response: Value,
    },
}

/// The planned restoration for one committed chunk: one disposition per event,
/// plus the advanced per-item lifecycle state to store back on the owner.
pub(super) struct PlannedRestore {
    /// One disposition per input event, in order.
    pub dispositions: Vec<ClientToolDisposition>,
    /// The tracked stream items after applying this chunk's lifecycle advances.
    pub next_items: Vec<ClientToolStreamItem>,
}

/// Map each committed-this-chunk event to a restoration disposition, staging the
/// per-item lifecycle advance in `next_items`. Fail closed (`Err`) on any
/// lifecycle-order violation, missing artifact, or lossy restore. Native
/// passthrough (empty `reverse`) returns an empty plan with zero work.
pub(super) fn plan_client_tool_restore(
    reverse: &HashMap<String, LoweredClientTool>,
    echo: Option<&ClientToolEcho>,
    committed: &[ClientToolStreamItem],
    events: &[ResponsesEvent],
    completions: &[ClientToolCompletion],
) -> Result<PlannedRestore, SseParseError> {
    let mut plan = PlannedRestore {
        dispositions: Vec::new(),
        next_items: committed.to_vec(),
    };
    if reverse.is_empty() {
        return Ok(plan);
    }
    let _ = (echo, completions); // consumed by Tasks 5-7
    for event in events {
        let disposition = plan_one_event(reverse, &mut plan.next_items, event)?;
        plan.dispositions.push(disposition);
    }
    Ok(plan)
}

/// Plan the restoration for a single committed event, advancing `next_items`.
fn plan_one_event(
    reverse: &HashMap<String, LoweredClientTool>,
    next_items: &mut Vec<ClientToolStreamItem>,
    event: &ResponsesEvent,
) -> Result<ClientToolDisposition, SseParseError> {
    match event {
        ResponsesEvent::OutputItemAdded(payload) => Ok(plan_output_item_added(reverse, next_items, payload)),
        ResponsesEvent::FunctionCallArgumentsDone(payload) => Ok(plan_arguments_done(next_items, payload)),
        ResponsesEvent::OutputItemDone(payload) => plan_output_item_done(reverse, next_items, payload),
        _ => Ok(ClientToolDisposition::Passthrough),
    }
}

/// Plan an `output_item.added`: open tracking for a lowered `Namespace` member and
/// retype it in place. Non-lowered items and (for now) other lowered kinds pass
/// through unchanged.
fn plan_output_item_added(
    reverse: &HashMap<String, LoweredClientTool>,
    next_items: &mut Vec<ClientToolStreamItem>,
    payload: &Value,
) -> ClientToolDisposition {
    let Some(item) = payload.get("item") else {
        return ClientToolDisposition::Passthrough;
    };
    let Some(name) = item.get("name").and_then(Value::as_str) else {
        return ClientToolDisposition::Passthrough;
    };
    let Some(lowered) = reverse.get(name) else {
        return ClientToolDisposition::Passthrough;
    };

    // Only Namespace members are restored in this task; other lowered kinds pass
    // through untracked until Tasks 5-7 add their synthesis.
    if lowered.restore != ClientToolRestore::Namespace {
        return ClientToolDisposition::Passthrough;
    }

    let Some(key) = client_tool_event_key(payload) else {
        return ClientToolDisposition::Passthrough;
    };
    next_items.push(ClientToolStreamItem {
        key,
        private_name: name.to_owned(),
        restore: lowered.restore,
        phase: ClientToolPhase::Opened,
        output_index: payload.get("output_index").and_then(Value::as_u64).unwrap_or_default(),
        item_id: item.get("id").and_then(Value::as_str).map(ToOwned::to_owned),
    });
    ClientToolDisposition::RetypeInPlace {
        item_type: "function_call",
        name: lowered.original_name.clone(),
        namespace: lowered.namespace.clone(),
    }
}

/// Plan a `function_call_arguments.done`: advance a tracked `Namespace` item to
/// `ArgsComplete`. The arguments frame carries no typed name, so lowered-ness is
/// resolved by matching the tracked items by key.
fn plan_arguments_done(next_items: &mut [ClientToolStreamItem], payload: &Value) -> ClientToolDisposition {
    let Some(key) = client_tool_event_key(payload) else {
        return ClientToolDisposition::Passthrough;
    };
    let Some(tracked) = next_items.iter_mut().find(|tracked| tracked.key == key) else {
        return ClientToolDisposition::Passthrough;
    };
    if tracked.restore == ClientToolRestore::Namespace {
        tracked.phase = ClientToolPhase::ArgsComplete;
    }
    ClientToolDisposition::Passthrough
}

/// Plan an `output_item.done`: finalize a tracked `Namespace` item and retype it
/// in place. Fail closed if the finalizer arrives before `arguments.done` (C4).
fn plan_output_item_done(
    reverse: &HashMap<String, LoweredClientTool>,
    next_items: &mut [ClientToolStreamItem],
    payload: &Value,
) -> Result<ClientToolDisposition, SseParseError> {
    let Some(key) = client_tool_event_key(payload) else {
        return Ok(ClientToolDisposition::Passthrough);
    };
    let Some(tracked) = next_items.iter_mut().find(|tracked| tracked.key == key) else {
        return Ok(ClientToolDisposition::Passthrough);
    };
    if tracked.restore != ClientToolRestore::Namespace {
        return Ok(ClientToolDisposition::Passthrough);
    }

    // C4 lifecycle-order check: an `output_item.done` before `arguments.done`
    // (item still `Opened`) is a malformed lifecycle; fail closed.
    if tracked.phase == ClientToolPhase::Opened {
        // #1159 Task 9: swap to SseParseError::ClientToolRestore { key, reason }
        return Err(SseParseError::MalformedJson {
            event_type: "response.output_item.done".to_owned(),
            err: format!("client-tool restore for '{}': output_item.done before arguments.done", tracked.key),
        });
    }
    tracked.phase = ClientToolPhase::Done;

    // Restore the original member name and namespace from the lowering map keyed
    // by the private name the backend returned; never leak the lowered name.
    let Some(lowered) = reverse.get(tracked.private_name.as_str()) else {
        return Ok(ClientToolDisposition::Passthrough);
    };
    Ok(ClientToolDisposition::RetypeInPlace {
        item_type: "function_call",
        name: lowered.original_name.clone(),
        namespace: lowered.namespace.clone(),
    })
}

/// Stable key matching a lowered client-tool item across its lifecycle events.
///
/// `output_item.added`/`.done` carry the id nested under `item`; the
/// `function_call_arguments.*` events carry a top-level `item_id`. Prefer either
/// id form, falling back to `output_index`, so one key matches the whole
/// lifecycle. Mirrors [`super::accumulator::tool_call_key`]'s
/// `item:{id}`/`index:{n}` shape.
fn client_tool_event_key(payload: &Value) -> Option<String> {
    let nested_id = payload.get("item").and_then(|item| item.get("id")).and_then(Value::as_str);
    let top_id = payload.get("item_id").and_then(Value::as_str);
    if let Some(id) = nested_id.or(top_id) {
        return Some(format!("item:{id}"));
    }
    payload
        .get("output_index")
        .and_then(Value::as_u64)
        .map(|output_index| format!("index:{output_index}"))
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    reason = "test assertions favor direct unwrap/index/panic for clear failures"
)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::openai::responses::state::{ClientToolRestore, LoweredClientTool};

    fn reverse_namespace() -> HashMap<String, LoweredClientTool> {
        let mut m = HashMap::new();
        m.insert("agentic_ns__fs__read".to_owned(), LoweredClientTool {
            original_name: "read".to_owned(),
            namespace: Some("fs".to_owned()),
            restore: ClientToolRestore::Namespace,
        });
        m
    }

    #[test]
    fn native_passthrough_plans_nothing() {
        let reverse = HashMap::new();
        let plan = plan_client_tool_restore(&reverse, None, &[], &[], &[]).unwrap();
        assert!(plan.dispositions.is_empty());
        assert!(plan.next_items.is_empty());
    }

    #[test]
    fn namespace_output_item_added_retypes_in_place() {
        let reverse = reverse_namespace();
        let added = ResponsesEvent::OutputItemAdded(serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "name": "agentic_ns__fs__read",
                     "call_id": "c1", "id": "fc_1"}
        }));
        let plan = plan_client_tool_restore(&reverse, None, &[], std::slice::from_ref(&added), &[]).unwrap();
        assert_eq!(plan.dispositions.len(), 1);
        match &plan.dispositions[0] {
            ClientToolDisposition::RetypeInPlace { item_type, name, namespace } => {
                assert_eq!(*item_type, "function_call");
                assert_eq!(name, "read");
                assert_eq!(namespace.as_deref(), Some("fs"));
            },
            other => panic!("expected RetypeInPlace, got {other:?}"),
        }
        assert_eq!(plan.next_items.len(), 1);
        assert!(matches!(plan.next_items[0].phase, ClientToolPhase::Opened));
    }

    #[test]
    fn namespace_full_lifecycle_advances_and_retypes_done() {
        let reverse = reverse_namespace();
        let added = ResponsesEvent::OutputItemAdded(serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "name": "agentic_ns__fs__read",
                     "call_id": "c1", "id": "fc_1"}
        }));
        let args_done = ResponsesEvent::FunctionCallArgumentsDone(serde_json::json!({
            "type": "response.function_call_arguments.done",
            "output_index": 0,
            "item_id": "fc_1",
            "arguments": "{\"path\":\"/etc\"}"
        }));
        let item_done = ResponsesEvent::OutputItemDone(serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "function_call", "name": "agentic_ns__fs__read",
                     "call_id": "c1", "id": "fc_1", "status": "completed"}
        }));
        let events = [added, args_done, item_done];
        let plan = plan_client_tool_restore(&reverse, None, &[], &events, &[]).unwrap();

        assert_eq!(plan.dispositions.len(), 3);
        assert!(matches!(plan.dispositions[0], ClientToolDisposition::RetypeInPlace { .. }));
        assert!(matches!(plan.dispositions[1], ClientToolDisposition::Passthrough));
        match &plan.dispositions[2] {
            ClientToolDisposition::RetypeInPlace { name, namespace, .. } => {
                assert_eq!(name, "read");
                assert_eq!(namespace.as_deref(), Some("fs"));
            },
            other => panic!("expected RetypeInPlace on done, got {other:?}"),
        }
        assert_eq!(plan.next_items.len(), 1);
        assert!(matches!(plan.next_items[0].phase, ClientToolPhase::Done));
        // The private lowered name is tracked internally but never surfaces in a
        // client-visible disposition.
        assert_eq!(plan.next_items[0].private_name, "agentic_ns__fs__read");
    }

    #[test]
    fn namespace_output_item_done_before_args_done_fails_closed() {
        let reverse = reverse_namespace();
        let added = ResponsesEvent::OutputItemAdded(serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "name": "agentic_ns__fs__read",
                     "call_id": "c1", "id": "fc_1"}
        }));
        let item_done = ResponsesEvent::OutputItemDone(serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"type": "function_call", "name": "agentic_ns__fs__read",
                     "call_id": "c1", "id": "fc_1", "status": "completed"}
        }));
        // output_item.done arriving while the item is still `Opened` (no
        // arguments.done seen) is a C4 lifecycle-order violation: fail closed.
        let events = [added, item_done];
        let result = plan_client_tool_restore(&reverse, None, &[], &events, &[]);
        assert!(result.is_err(), "output_item.done before arguments.done must fail closed");
    }
}
