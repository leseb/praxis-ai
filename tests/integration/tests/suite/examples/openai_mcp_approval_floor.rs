// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the mcp-approval-floor example config.
//!
//! These validate that the `openai_mcp_dispatch` operator approval floor
//! (`minimum_approval: always`) is a valid configuration that builds and leaves
//! non-MCP traffic untouched. The end-to-end behavior — a client
//! `"require_approval": "never"` still producing an `mcp_approval_request` and
//! zero `tools/call` before approval — is exercised against the full agentic
//! loop in `openai_agentic_loop.rs` (`approval_floor_*`).

use std::collections::HashMap;

use praxis_test_utils::{
    free_port, http_send, json_post, load_example_config, parse_body, parse_status, start_backend_with_shutdown,
    start_proxy,
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn openai_mcp_approval_floor_example_passes_through_non_mcp_request() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/mcp-approval-floor.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"Hello, world!"}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "non-MCP request should pass through");
    assert_eq!(
        parse_body(&raw),
        "inference",
        "non-MCP request should reach inference backend"
    );
}

#[test]
fn openai_mcp_approval_floor_example_passes_through_function_tools() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/mcp-approval-floor.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"function","name":"calc"}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 200, "function tools should pass through");
    assert_eq!(
        parse_body(&raw),
        "inference",
        "function tools should reach inference backend"
    );
}

#[test]
fn openai_mcp_approval_floor_example_rejects_ssrf_mcp_url() {
    let backend_guard = start_backend_with_shutdown("inference");
    let proxy_port = free_port();

    let config = load_example_config(
        "openai/responses/mcp-approval-floor.yaml",
        proxy_port,
        HashMap::from([("127.0.0.1:3001", backend_guard.port())]),
    );
    let proxy = start_proxy(&config);

    // The approval floor is orthogonal to the SSRF guard: link-local metadata
    // (169.254.0.0/16) is blocked unconditionally on the MCP callout path, even
    // when private upstreams are permitted.
    let body = r#"{"model":"gpt-4.1","input":"test","tools":[{"type":"mcp","server_label":"evil","server_url":"http://169.254.169.254/mcp","allowed_tools":["x"]}]}"#;
    let raw = http_send(proxy.addr(), &json_post("/v1/responses", body));

    assert_eq!(parse_status(&raw), 502, "SSRF URL should be rejected");
    let response_body = parse_body(&raw);
    assert!(
        response_body.contains("SSRF"),
        "rejection should mention SSRF: {response_body}"
    );
}
