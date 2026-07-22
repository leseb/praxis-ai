// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Request handlers for the `/v1/conversations` endpoints.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet, VecDeque},
};

use percent_encoding::percent_decode_str;
use praxis_filter::{FilterAction, FilterError, HttpFilterContext, Rejection};
use serde_json::{Map, Value};
use tracing::debug;

use super::validate::validate_metadata;
use crate::{
    openai::responses::{
        DEFAULT_TENANT_ID, LOCAL_FILE_SEARCH_PUBLIC_ID_FINGERPRINT_FIELD, TENANT_METADATA_KEY,
        local_file_search_marker_triplet, local_file_search_public_id_fingerprint,
        store::{DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT},
    },
    store::{ConversationItemRecord, ConversationItemStore, ConversationRecord, StoreError},
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Maximum number of items accepted by create endpoints.
const MAX_ITEMS_PER_REQUEST: usize = 20;

/// Maximum optimistic retries for a contended conversation cache refresh.
const MAX_MESSAGE_SYNC_ATTEMPTS: usize = 16;

// -----------------------------------------------------------------------------
// ItemListParams
// -----------------------------------------------------------------------------

/// Cursor pagination parameters for conversation item listing.
struct ItemListParams {
    /// Item ID to page after.
    after_item_id: Option<String>,

    /// Maximum number of items to return.
    limit: u32,

    /// Whether results should be oldest-first.
    ascending: bool,
}

impl Default for ItemListParams {
    fn default() -> Self {
        Self {
            after_item_id: None,
            limit: DEFAULT_PAGE_LIMIT,
            ascending: false,
        }
    }
}

impl ItemListParams {
    /// Return the effective limit clamped to the API bounds.
    fn effective_limit(&self) -> u32 {
        self.limit.clamp(1, MAX_PAGE_LIMIT)
    }
}

// -----------------------------------------------------------------------------
// Conversation Lifecycle
// -----------------------------------------------------------------------------

/// Handle `POST /v1/conversations` — create a new conversation.
#[expect(clippy::too_many_lines, reason = "sequential guard-clause pipeline")]
pub(super) async fn handle_create_conversation(
    ctx: &HttpFilterContext<'_>,
    store: &dyn ConversationItemStore,
    body: &[u8],
) -> Result<FilterAction, FilterError> {
    let tenant_id = ctx.get_metadata(TENANT_METADATA_KEY).unwrap_or(DEFAULT_TENANT_ID);
    let input = match parse_json_object_body(body) {
        Ok(v) => v,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    let metadata = input.get("metadata").cloned().unwrap_or(Value::Null);
    if let Err(msg) = validate_metadata(&metadata) {
        return Ok(FilterAction::Reject(invalid_input_response(&msg)?));
    }
    let metadata = if metadata.is_null() {
        Value::Object(Map::new())
    } else {
        metadata
    };

    let raw_id = ctx.id_generator.generate(ctx.time_source);
    let conversation_id = format!("conv_{raw_id}");
    let created_at = current_timestamp(ctx);
    let item_values = match parse_items_field(input.get("items"), false) {
        Ok(items) => items,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    let item_records = match build_item_records(ctx, tenant_id, &conversation_id, created_at, 1, item_values) {
        Ok(records) => records,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    if let Some(item_id) = duplicate_item_id(&item_records) {
        return Ok(FilterAction::Reject(invalid_input_response(
            &duplicate_item_id_message(item_id),
        )?));
    }
    let messages = Value::Array(item_records.iter().map(|item| item.item_data.clone()).collect());

    let record = ConversationRecord {
        conversation_id: conversation_id.clone(),
        tenant_id: tenant_id.to_owned(),
        created_at,
        metadata: metadata.clone(),
        messages,
    };

    if let Err(e) = store.upsert_conversation(&record).await {
        return Ok(FilterAction::Reject(store_error_response(&e)?));
    }
    if !item_records.is_empty()
        && let Err(e) = store.create_conversation_items(&item_records).await
    {
        return Ok(FilterAction::Reject(store_error_response(&e)?));
    }
    debug!(conversation_id, tenant_id, "conversation created");

    let body = conversation_to_json(&record);
    Ok(FilterAction::Reject(json_response(200, &body)?))
}

/// Handle `GET /v1/conversations/{id}` — retrieve a conversation.
pub(super) async fn handle_get_conversation(
    ctx: &HttpFilterContext<'_>,
    store: &dyn ConversationItemStore,
    conversation_id: &str,
) -> Result<FilterAction, FilterError> {
    let tenant_id = ctx.get_metadata(TENANT_METADATA_KEY).unwrap_or(DEFAULT_TENANT_ID);

    match store.get_conversation(tenant_id, conversation_id).await {
        Ok(Some(record)) => {
            let body = conversation_to_json(&record);
            Ok(FilterAction::Reject(json_response(200, &body)?))
        },
        Ok(None) => {
            debug!(conversation_id, "conversation not found");
            Ok(FilterAction::Reject(not_found_response(&format!(
                "No conversation found with id: '{conversation_id}'."
            ))?))
        },
        Err(e) => Ok(FilterAction::Reject(store_error_response(&e)?)),
    }
}

/// Handle `POST /v1/conversations/{id}` — update a conversation.
#[expect(clippy::too_many_lines, reason = "sequential guard-clause pipeline")]
pub(super) async fn handle_update_conversation(
    ctx: &HttpFilterContext<'_>,
    store: &dyn ConversationItemStore,
    conversation_id: &str,
    body: &[u8],
) -> Result<FilterAction, FilterError> {
    let tenant_id = ctx.get_metadata(TENANT_METADATA_KEY).unwrap_or(DEFAULT_TENANT_ID);
    let input = match parse_json_object_body(body) {
        Ok(v) => v,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    let metadata_update = input.get("metadata").cloned();
    if let Some(metadata) = &metadata_update
        && let Err(msg) = validate_metadata(metadata)
    {
        return Ok(FilterAction::Reject(invalid_input_response(&msg)?));
    }

    let existing = match store.get_conversation(tenant_id, conversation_id).await {
        Ok(record) => record,
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    };
    let Some(existing) = existing else {
        debug!(conversation_id, "conversation not found for update");
        return Ok(FilterAction::Reject(not_found_response(&format!(
            "No conversation found with id: '{conversation_id}'."
        ))?));
    };

    let metadata = metadata_update.map_or_else(
        || existing.metadata.clone(),
        |metadata| {
            if metadata.is_null() {
                Value::Object(Map::new())
            } else {
                metadata
            }
        },
    );

    let record = ConversationRecord {
        conversation_id: conversation_id.to_owned(),
        tenant_id: tenant_id.to_owned(),
        created_at: existing.created_at,
        metadata: metadata.clone(),
        messages: existing.messages,
    };

    if let Err(e) = store.upsert_conversation(&record).await {
        return Ok(FilterAction::Reject(store_error_response(&e)?));
    }
    debug!(conversation_id, tenant_id, "conversation updated");

    let body = conversation_to_json(&record);
    Ok(FilterAction::Reject(json_response(200, &body)?))
}

/// Handle `DELETE /v1/conversations/{id}` — delete a conversation.
///
/// This intentionally deletes only the conversation record. The OpenAI
/// Conversations API specifies that deleting a conversation does not delete
/// its items; item cleanup belongs to item deletion or a separate retention
/// policy, not this endpoint.
pub(super) async fn handle_delete_conversation(
    ctx: &HttpFilterContext<'_>,
    store: &dyn ConversationItemStore,
    conversation_id: &str,
) -> Result<FilterAction, FilterError> {
    let tenant_id = ctx.get_metadata(TENANT_METADATA_KEY).unwrap_or(DEFAULT_TENANT_ID);

    match store.delete_conversation(tenant_id, conversation_id).await {
        Ok(true) => {
            debug!(conversation_id, tenant_id, "conversation deleted");
            let body = serde_json::json!({
                "id": conversation_id,
                "object": "conversation.deleted",
                "deleted": true,
            });
            Ok(FilterAction::Reject(json_response(200, &body)?))
        },
        Ok(false) => {
            debug!(conversation_id, "conversation not found for delete");
            Ok(FilterAction::Reject(not_found_response(&format!(
                "No conversation found with id: '{conversation_id}'."
            ))?))
        },
        Err(e) => Ok(FilterAction::Reject(store_error_response(&e)?)),
    }
}

// -----------------------------------------------------------------------------
// Conversation Items
// -----------------------------------------------------------------------------

/// Handle `POST /v1/conversations/{id}/items` — create items.
#[expect(clippy::too_many_lines, reason = "sequential guard-clause pipeline")]
pub(super) async fn handle_create_items(
    ctx: &HttpFilterContext<'_>,
    store: &dyn ConversationItemStore,
    conversation_id: &str,
    body: &[u8],
) -> Result<FilterAction, FilterError> {
    let tenant_id = ctx.get_metadata(TENANT_METADATA_KEY).unwrap_or(DEFAULT_TENANT_ID);
    let input = match parse_json_object_body(body) {
        Ok(v) => v,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    let existing = match store.get_conversation(tenant_id, conversation_id).await {
        Ok(record) => record,
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    };
    let Some(existing) = existing else {
        debug!(conversation_id, "conversation not found for item create");
        return Ok(FilterAction::Reject(not_found_response(
            &conversation_not_found_message(conversation_id),
        )?));
    };

    let item_values = match parse_items_field(input.get("items"), true) {
        Ok(items) => items,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    let start_position = match store.max_item_position(tenant_id, conversation_id).await {
        Ok(pos) => pos.saturating_add(1),
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    };
    let created_at = current_timestamp(ctx);
    let item_records =
        match build_item_records(ctx, tenant_id, conversation_id, created_at, start_position, item_values) {
            Ok(records) => records,
            Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
        };
    if let Some(item_id) = duplicate_item_id(&item_records) {
        return Ok(FilterAction::Reject(invalid_input_response(
            &duplicate_item_id_message(item_id),
        )?));
    }
    let requested_ids: Vec<&str> = item_records.iter().map(|r| r.item_id.as_str()).collect();
    let already_present = match store
        .get_existing_conversation_item_ids(tenant_id, conversation_id, &requested_ids)
        .await
    {
        Ok(ids) => ids,
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    };
    if let Some(item_id) = already_present.first() {
        return Ok(FilterAction::Reject(invalid_input_response(
            &existing_item_id_message(item_id),
        )?));
    }

    if let Err(e) = store.create_conversation_items(&item_records).await {
        return Ok(FilterAction::Reject(store_error_response(&e)?));
    }
    if let Err(e) = sync_conversation_messages(store, existing).await {
        return Ok(FilterAction::Reject(store_error_response(&e)?));
    }
    debug!(
        conversation_id,
        tenant_id,
        count = item_records.len(),
        "conversation items created"
    );

    let body = conversation_items_to_json(&item_records, false);
    Ok(FilterAction::Reject(json_response(200, &body)?))
}

/// Handle `GET /v1/conversations/{id}/items` — list items.
#[expect(clippy::too_many_lines, reason = "sequential guard-clause pipeline")]
pub(super) async fn handle_list_items(
    ctx: &HttpFilterContext<'_>,
    store: &dyn ConversationItemStore,
    conversation_id: &str,
) -> Result<FilterAction, FilterError> {
    let tenant_id = ctx.get_metadata(TENANT_METADATA_KEY).unwrap_or(DEFAULT_TENANT_ID);
    match store.get_conversation(tenant_id, conversation_id).await {
        Ok(Some(_)) => {},
        Ok(None) => {
            debug!(conversation_id, "conversation not found for item list");
            return Ok(FilterAction::Reject(not_found_response(
                &conversation_not_found_message(conversation_id),
            )?));
        },
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    }

    let params = parse_item_list_params(ctx.request.uri.query());
    let limit = params.effective_limit();
    let rows = match store
        .list_conversation_items(
            tenant_id,
            conversation_id,
            params.after_item_id.as_deref(),
            limit.saturating_add(1),
            params.ascending,
        )
        .await
    {
        Ok(rows) => rows,
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    };
    let take_limit = usize::try_from(limit).unwrap_or(usize::MAX);
    let has_more = rows.len() > take_limit;
    let data: Vec<_> = rows.into_iter().take(take_limit).collect();

    let body = conversation_items_to_json(&data, has_more);
    Ok(FilterAction::Reject(json_response(200, &body)?))
}

/// Handle `GET /v1/conversations/{id}/items/{item_id}` — retrieve one item.
pub(super) async fn handle_get_item(
    ctx: &HttpFilterContext<'_>,
    store: &dyn ConversationItemStore,
    conversation_id: &str,
    item_id: &str,
) -> Result<FilterAction, FilterError> {
    let tenant_id = ctx.get_metadata(TENANT_METADATA_KEY).unwrap_or(DEFAULT_TENANT_ID);
    let item_id = match decode_item_id_path_segment(item_id) {
        Ok(id) => id,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    let item_id = item_id.as_ref();
    match store.get_conversation_item(tenant_id, conversation_id, item_id).await {
        Ok(Some(record)) => Ok(FilterAction::Reject(json_response(200, &record.item_data)?)),
        Ok(None) => {
            debug!(conversation_id, item_id, "conversation item not found");
            Ok(FilterAction::Reject(not_found_response(&item_not_found_message(
                item_id,
            ))?))
        },
        Err(e) => Ok(FilterAction::Reject(store_error_response(&e)?)),
    }
}

/// Handle `DELETE /v1/conversations/{id}/items/{item_id}` — delete one item.
#[expect(clippy::too_many_lines, reason = "sequential guard-clause pipeline")]
#[expect(clippy::cognitive_complexity, reason = "tracing macros inflate complexity")]
pub(super) async fn handle_delete_item(
    ctx: &HttpFilterContext<'_>,
    store: &dyn ConversationItemStore,
    conversation_id: &str,
    item_id: &str,
) -> Result<FilterAction, FilterError> {
    let tenant_id = ctx.get_metadata(TENANT_METADATA_KEY).unwrap_or(DEFAULT_TENANT_ID);
    let item_id = match decode_item_id_path_segment(item_id) {
        Ok(id) => id,
        Err(msg) => return Ok(FilterAction::Reject(invalid_input_response(&msg)?)),
    };
    let item_id = item_id.as_ref();
    let existing = match store.get_conversation(tenant_id, conversation_id).await {
        Ok(Some(record)) => record,
        Ok(None) => {
            debug!(conversation_id, item_id, "conversation not found for item delete");
            return Ok(FilterAction::Reject(not_found_response(
                &conversation_not_found_message(conversation_id),
            )?));
        },
        Err(e) => return Ok(FilterAction::Reject(store_error_response(&e)?)),
    };

    match store
        .delete_conversation_item(tenant_id, conversation_id, item_id)
        .await
    {
        Ok(true) => {
            if let Err(e) = sync_conversation_messages(store, existing).await {
                return Ok(FilterAction::Reject(store_error_response(&e)?));
            }
            debug!(conversation_id, item_id, tenant_id, "conversation item deleted");
            match store.get_conversation(tenant_id, conversation_id).await {
                Ok(Some(record)) => Ok(FilterAction::Reject(json_response(
                    200,
                    &conversation_to_json(&record),
                )?)),
                Ok(None) => Ok(FilterAction::Reject(not_found_response(
                    &conversation_not_found_message(conversation_id),
                )?)),
                Err(e) => Ok(FilterAction::Reject(store_error_response(&e)?)),
            }
        },
        Ok(false) => {
            debug!(conversation_id, item_id, "conversation item not found for delete");
            Ok(FilterAction::Reject(not_found_response(&item_not_found_message(
                item_id,
            ))?))
        },
        Err(e) => Ok(FilterAction::Reject(store_error_response(&e)?)),
    }
}

// -----------------------------------------------------------------------------
// JSON Helpers
// -----------------------------------------------------------------------------

/// Parse request body as JSON.
fn parse_json_body(body: &[u8]) -> Result<Value, String> {
    serde_json::from_slice(body).map_err(|e| format!("invalid JSON body: {e}"))
}

/// Parse request body as a JSON object.
fn parse_json_object_body(body: &[u8]) -> Result<Map<String, Value>, String> {
    match parse_json_body(body)? {
        Value::Object(map) => Ok(map),
        _ => Err("request body must be a JSON object".to_owned()),
    }
}

/// Parse the optional or required `items` body field.
fn parse_items_field(field: Option<&Value>, required: bool) -> Result<Vec<Value>, String> {
    let Some(value) = field else {
        return if required {
            Err("'items' is required".to_owned())
        } else {
            Ok(Vec::new())
        };
    };
    let Value::Array(items) = value else {
        return Err("items must be a JSON array".to_owned());
    };
    if items.len() > MAX_ITEMS_PER_REQUEST {
        return Err(format!("items may contain at most {MAX_ITEMS_PER_REQUEST} entries"));
    }
    Ok(items.clone())
}

/// Return the first duplicate item ID in a create request.
fn duplicate_item_id(items: &[ConversationItemRecord]) -> Option<&str> {
    let mut seen = HashSet::new();
    for item in items {
        if !seen.insert(item.item_id.as_str()) {
            return Some(item.item_id.as_str());
        }
    }
    None
}

/// Build store records for normalized conversation item JSON values.
#[expect(clippy::too_many_arguments, reason = "factoring into struct would add indirection")]
pub(super) fn build_item_records(
    ctx: &HttpFilterContext<'_>,
    tenant_id: &str,
    conversation_id: &str,
    created_at: i64,
    start_position: i64,
    items: Vec<Value>,
) -> Result<Vec<ConversationItemRecord>, String> {
    items
        .into_iter()
        .enumerate()
        .map(|(index, item)| {
            let (item_id, item_data) = normalize_item(ctx, item)?;
            let offset = i64::try_from(index).unwrap_or(i64::MAX);
            Ok(ConversationItemRecord {
                item_id,
                tenant_id: tenant_id.to_owned(),
                conversation_id: conversation_id.to_owned(),
                item_data,
                created_at,
                position: start_position.saturating_add(offset),
            })
        })
        .collect()
}

/// Ensure an item is an object and has a usable ID.
pub(super) fn normalize_item(ctx: &HttpFilterContext<'_>, item: Value) -> Result<(String, Value), String> {
    let Value::Object(mut map) = item else {
        return Err("each item must be a JSON object".to_owned());
    };
    let item_id = match map.get("id") {
        Some(Value::String(id)) if !id.is_empty() => id.clone(),
        Some(Value::String(_)) => return Err("item id must not be empty".to_owned()),
        Some(Value::Null) | None => generated_item_id(ctx),
        Some(_) => return Err("item id must be a string".to_owned()),
    };
    map.remove(LOCAL_FILE_SEARCH_PUBLIC_ID_FINGERPRINT_FIELD);
    map.insert("id".to_owned(), Value::String(item_id.clone()));
    normalize_message_item(&mut map)?;
    Ok((item_id, Value::Object(map)))
}

/// Normalize easy SDK message inputs into conversation message response objects.
fn normalize_message_item(map: &mut Map<String, Value>) -> Result<(), String> {
    if map.get("type").and_then(Value::as_str) != Some("message") {
        return Ok(());
    }

    let role = match map.get("role") {
        Some(Value::String(role)) if !role.is_empty() => role.clone(),
        Some(Value::String(_)) => return Err("message role must not be empty".to_owned()),
        Some(_) => return Err("message role must be a string".to_owned()),
        None => return Err("message role is required".to_owned()),
    };

    let content = map
        .remove("content")
        .ok_or_else(|| "message content is required".to_owned())?;
    map.insert("content".to_owned(), normalize_message_content(&role, content)?);
    map.entry("status".to_owned())
        .or_insert_with(|| Value::String("completed".to_owned()));

    Ok(())
}

/// Convert string message content to the list-form content returned by the API.
fn normalize_message_content(role: &str, content: Value) -> Result<Value, String> {
    match content {
        Value::String(text) => {
            let content_item = if role == "assistant" {
                serde_json::json!({
                    "type": "output_text",
                    "text": text,
                    "annotations": [],
                })
            } else {
                serde_json::json!({
                    "type": "input_text",
                    "text": text,
                })
            };
            Ok(Value::Array(vec![content_item]))
        },
        Value::Array(_) => Ok(content),
        _ => Err("message content must be a string or array".to_owned()),
    }
}

/// Generate a conversation item ID.
pub(super) fn generated_item_id(ctx: &HttpFilterContext<'_>) -> String {
    let raw_id = ctx.id_generator.generate(ctx.time_source);
    format!("item_{raw_id}")
}

/// Decode an item ID path segment the same way clients encode path parameters.
fn decode_item_id_path_segment(item_id: &str) -> Result<Cow<'_, str>, String> {
    percent_decode_str(item_id)
        .decode_utf8()
        .map_err(|e| format!("item id path segment must be valid UTF-8: {e}"))
}

/// Convert a `ConversationRecord` to a JSON response object.
fn conversation_to_json(record: &ConversationRecord) -> Value {
    serde_json::json!({
        "id": record.conversation_id,
        "object": "conversation",
        "created_at": record.created_at,
        "metadata": record.metadata,
    })
}

/// Convert item records to an `OpenAI` list response object.
fn conversation_items_to_json(records: &[ConversationItemRecord], has_more: bool) -> Value {
    let first_id = records.first().map_or("", |record| record.item_id.as_str());
    let last_id = records.last().map_or("", |record| record.item_id.as_str());
    let data: Vec<Value> = records.iter().map(|record| record.item_data.clone()).collect();

    serde_json::json!({
        "object": "list",
        "data": data,
        "has_more": has_more,
        "first_id": first_id,
        "last_id": last_id,
    })
}

/// Parse cursor-based pagination parameters from a query string.
fn parse_item_list_params(query: Option<&str>) -> ItemListParams {
    let Some(qs) = query else {
        return ItemListParams::default();
    };

    let mut params = ItemListParams::default();
    for pair in qs.split('&') {
        let Some((key, value)) = pair.split_once('=') else {
            continue;
        };
        match key {
            "after" => {
                params.after_item_id = Some(decode_query_component(value));
            },
            "limit" => {
                if let Ok(n) = value.parse::<u32>() {
                    params.limit = n;
                }
            },
            "order" => match value {
                "asc" => params.ascending = true,
                "desc" => params.ascending = false,
                _ => {},
            },
            _ => {},
        }
    }
    params
}

/// Decode one application/x-www-form-urlencoded query component.
fn decode_query_component(value: &str) -> String {
    let normalized = value.replace('+', " ");
    percent_decode_str(&normalized).decode_utf8_lossy().into_owned()
}

/// Return the current Unix timestamp as an `i64`.
pub(super) fn current_timestamp(ctx: &HttpFilterContext<'_>) -> i64 {
    i64::try_from(ctx.time_source.now().as_secs()).unwrap_or(i64::MAX)
}

/// Build a JSON response with the given status code.
fn json_response(status: u16, body: &Value) -> Result<Rejection, FilterError> {
    let bytes = serde_json::to_vec(body)
        .map_err(|e| FilterError::from(format!("openai_conversations: serialize failed: {e}")))?;
    Ok(Rejection::status(status)
        .with_header("content-type", "application/json")
        .with_body(bytes))
}

/// Build a 400 JSON response for invalid input.
fn invalid_input_response(message: &str) -> Result<Rejection, FilterError> {
    json_response(
        400,
        &serde_json::json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
            }
        }),
    )
}

/// Build a 404 JSON response.
fn not_found_response(message: &str) -> Result<Rejection, FilterError> {
    json_response(
        404,
        &serde_json::json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
            }
        }),
    )
}

/// Build the standard conversation not-found message.
fn conversation_not_found_message(conversation_id: &str) -> String {
    format!("No conversation found with id: '{conversation_id}'.")
}

/// Build the standard item not-found message.
fn item_not_found_message(item_id: &str) -> String {
    format!("No conversation item found with id: '{item_id}'.")
}

/// Build a duplicate-item client error message.
fn duplicate_item_id_message(item_id: &str) -> String {
    format!("duplicate item id in request: '{item_id}'")
}

/// Build an existing-item client error message.
fn existing_item_id_message(item_id: &str) -> String {
    format!("item id already exists in conversation: '{item_id}'")
}

/// Build a 500 JSON response from a store error.
fn store_error_response(error: &StoreError) -> Result<Rejection, FilterError> {
    let message = match error {
        StoreError::InvalidInput(msg) => {
            return json_response(
                400,
                &serde_json::json!({
                    "error": {
                        "message": msg,
                        "type": "invalid_request_error",
                    }
                }),
            );
        },
        _ => "Internal server error.",
    };
    json_response(
        500,
        &serde_json::json!({
            "error": {
                "message": message,
                "type": "server_error",
            }
        }),
    )
}

/// Refresh the denormalized conversation message cache from item rows.
///
/// Re-reads all items on every mutation rather than patching the JSON
/// array incrementally. Acceptable because conversations hold a small
/// number of items; incremental updates would risk drift.
pub(super) async fn sync_conversation_messages(
    store: &dyn ConversationItemStore,
    record: ConversationRecord,
) -> Result<(), StoreError> {
    sync_conversation_messages_inner(store, record, None).await
}

/// Refresh replay history while adding private markers from the current turn.
pub(super) async fn sync_conversation_messages_with_overlay(
    store: &dyn ConversationItemStore,
    record: ConversationRecord,
    additional_hidden_messages: &Value,
) -> Result<(), StoreError> {
    sync_conversation_messages_inner(store, record, Some(additional_hidden_messages)).await
}

/// Rebuild the private cache from public rows and hidden provenance.
#[expect(
    clippy::too_many_lines,
    reason = "the bounded compare-and-swap retry keeps each synchronization step visible"
)]
async fn sync_conversation_messages_inner(
    store: &dyn ConversationItemStore,
    mut record: ConversationRecord,
    additional_hidden_messages: Option<&Value>,
) -> Result<(), StoreError> {
    for _attempt in 0..MAX_MESSAGE_SYNC_ATTEMPTS {
        let public_messages = collect_conversation_messages(store, &record.tenant_id, &record.conversation_id).await?;
        let messages = Value::Array(preserve_local_file_search_markers(
            public_messages,
            &record.messages,
            additional_hidden_messages,
        ));
        if store
            .compare_and_swap_conversation_messages(
                &record.tenant_id,
                &record.conversation_id,
                &record.messages,
                &messages,
            )
            .await?
        {
            return Ok(());
        }
        record = store
            .get_conversation(&record.tenant_id, &record.conversation_id)
            .await?
            .ok_or_else(|| {
                StoreError::Database(format!(
                    "conversation disappeared during message sync: {}",
                    record.conversation_id
                ))
            })?;
    }
    Err(StoreError::Database(format!(
        "conversation message sync remained contended after {MAX_MESSAGE_SYNC_ATTEMPTS} attempts: {}",
        record.conversation_id
    )))
}

/// Restore private local-search markers without changing public item rows.
fn preserve_local_file_search_markers(
    public_messages: Vec<Value>,
    hidden_messages: &Value,
    additional_hidden_messages: Option<&Value>,
) -> Vec<Value> {
    let mut markers = hidden_messages
        .as_array()
        .map_or_else(HashMap::new, |hidden| local_file_search_marker_queues(hidden));
    if let Some(additional) = additional_hidden_messages.and_then(Value::as_array) {
        append_local_file_search_marker_queues(&mut markers, additional);
    }
    if markers.is_empty() {
        return public_messages;
    }

    let mut replay_messages = Vec::with_capacity(public_messages.len().saturating_add(markers.len().saturating_mul(2)));
    for item in public_messages {
        if let Some(marker) = pop_local_file_search_marker(&mut markers, &item) {
            replay_messages.extend(marker.into_iter().cloned());
        } else {
            replay_messages.push(item);
        }
    }
    replay_messages
}

/// Pop the matching private marker for one public call occurrence.
fn pop_local_file_search_marker<'a>(
    markers: &mut HashMap<String, VecDeque<[&'a Value; 3]>>,
    item: &Value,
) -> Option<[&'a Value; 3]> {
    let id = item
        .get("type")
        .and_then(Value::as_str)
        .filter(|item_type| *item_type == "file_search_call")
        .and_then(|_| item.get("id"))
        .and_then(Value::as_str)?;
    let fingerprint_key = format!("fingerprint:{}", local_file_search_public_id_fingerprint(id));
    markers.get_mut(&fingerprint_key).and_then(VecDeque::pop_front)
}

/// Collect every exact marker in occurrence order without collapsing IDs.
fn local_file_search_marker_queues(hidden_messages: &[Value]) -> HashMap<String, VecDeque<[&Value; 3]>> {
    let mut markers: HashMap<_, VecDeque<_>> = HashMap::new();
    append_local_file_search_marker_queues(&mut markers, hidden_messages);
    markers
}

/// Append exact marker occurrences to an existing association index.
fn append_local_file_search_marker_queues<'a>(
    markers: &mut HashMap<String, VecDeque<[&'a Value; 3]>>,
    hidden_messages: &'a [Value],
) {
    for window in hidden_messages.windows(3) {
        if let Some((id, marker)) = local_file_search_marker(window) {
            markers.entry(id).or_default().push_back(marker);
        }
    }
}

/// Index one exact private compact local-search marker triplet.
fn local_file_search_marker(items: &[Value]) -> Option<(String, [&Value; 3])> {
    let (fingerprint, marker) = local_file_search_marker_triplet(items)?;
    Some((format!("fingerprint:{fingerprint}"), marker))
}

/// Collect all item JSON values for a conversation in ascending order.
async fn collect_conversation_messages(
    store: &dyn ConversationItemStore,
    tenant_id: &str,
    conversation_id: &str,
) -> Result<Vec<Value>, StoreError> {
    let mut after = None;
    let mut messages = Vec::new();
    loop {
        let rows = store
            .list_conversation_items(tenant_id, conversation_id, after.as_deref(), MAX_PAGE_LIMIT, true)
            .await?;
        if rows.is_empty() {
            break;
        }
        after = rows.last().map(|record| record.item_id.clone());
        let row_count = rows.len();
        messages.extend(rows.into_iter().map(|record| record.item_data));
        if row_count < usize::try_from(MAX_PAGE_LIMIT).unwrap_or(usize::MAX) {
            break;
        }
    }
    Ok(messages)
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;
    use crate::openai::responses::LOCAL_FILE_SEARCH_MARKER_ARGUMENTS;

    #[test]
    #[expect(clippy::too_many_lines, reason = "constructs two complete marker occurrences")]
    fn current_marker_overlay_follows_existing_duplicate_id_occurrence() {
        let fingerprint = local_file_search_public_id_fingerprint("fs-duplicate");
        let compact_first = serde_json::json!({
            "type":"file_search_call","id":"fs-duplicate","status":"completed",
            LOCAL_FILE_SEARCH_PUBLIC_ID_FINGERPRINT_FIELD:fingerprint,
            "results":[{"file_id":"file-a","filename":"a.txt","score":0.0,"text":""}]
        });
        let marker_first = serde_json::json!({
            "type":"function_call","call_id":"file_search_0_0123456789abcdef","name":"file_search",
            "arguments":LOCAL_FILE_SEARCH_MARKER_ARGUMENTS,"status":"completed"
        });
        let output_first = serde_json::json!({
            "type":"function_call_output","call_id":"file_search_0_0123456789abcdef","output":""
        });
        let compact_second = serde_json::json!({
            "type":"file_search_call","id":"fs-duplicate","status":"completed",
            LOCAL_FILE_SEARCH_PUBLIC_ID_FINGERPRINT_FIELD:fingerprint,
            "results":[{"file_id":"file-b","filename":"b.txt","score":0.0,"text":""}]
        });
        let marker_second = serde_json::json!({
            "type":"function_call","call_id":"file_search_1_fedcba9876543210","name":"file_search",
            "arguments":LOCAL_FILE_SEARCH_MARKER_ARGUMENTS,"status":"completed"
        });
        let output_second = serde_json::json!({
            "type":"function_call_output","call_id":"file_search_1_fedcba9876543210","output":""
        });
        let hidden = serde_json::json!([compact_first, marker_first, output_first]);
        let current = serde_json::json!([compact_second, marker_second, output_second]);
        let public = vec![
            serde_json::json!({"type":"file_search_call","id":"fs-duplicate","status":"completed"}),
            serde_json::json!({"type":"message","id":"intervening","role":"assistant","content":[]}),
            serde_json::json!({"type":"file_search_call","id":"fs-duplicate","status":"completed"}),
        ];

        let replay = preserve_local_file_search_markers(public, &hidden, Some(&current));

        assert_eq!(replay.len(), 7);
        assert_eq!(
            replay[0].pointer("/results/0/file_id"),
            Some(&Value::String("file-a".to_owned()))
        );
        assert_eq!(
            replay[4].pointer("/results/0/file_id"),
            Some(&Value::String("file-b".to_owned()))
        );
        assert_eq!(replay[3]["id"], "intervening");
        assert_eq!(replay[1]["call_id"], "file_search_0_0123456789abcdef");
        assert_eq!(replay[5]["call_id"], "file_search_1_fedcba9876543210");
    }

    #[test]
    fn private_fingerprint_matches_an_untruncated_public_id() {
        let public_id = format!("fs-{}", "x".repeat(512));
        let fingerprint = local_file_search_public_id_fingerprint(&public_id);
        let truncated_id: String = public_id.chars().take(256).collect();
        let hidden = serde_json::json!([
            {
                "type":"file_search_call","id":truncated_id,
                LOCAL_FILE_SEARCH_PUBLIC_ID_FINGERPRINT_FIELD:fingerprint,
                "status":"completed","results":[]
            },
            {
                "type":"function_call","call_id":"file_search_0_0123456789abcdef","name":"file_search",
                "arguments":LOCAL_FILE_SEARCH_MARKER_ARGUMENTS,"status":"completed"
            },
            {"type":"function_call_output","call_id":"file_search_0_0123456789abcdef","output":""}
        ]);
        let public = vec![serde_json::json!({"type":"file_search_call","id":public_id,"status":"completed"})];

        let replay = preserve_local_file_search_markers(public, &hidden, None);

        assert_eq!(replay.len(), 3);
        assert_eq!(replay[0][LOCAL_FILE_SEARCH_PUBLIC_ID_FINGERPRINT_FIELD], fingerprint);
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "covers both current and legacy public marker lookalikes across two syncs"
    )]
    fn public_marker_lookalikes_remain_stable_across_repeated_syncs() {
        let req = crate::test_utils::make_request(http::Method::POST, "/v1/conversations/conv/items");
        let ctx = crate::test_utils::make_filter_context(&req);
        let public = vec![
            serde_json::json!({
                "type":"file_search_call","id":"fs-public","status":"completed","results":[],
                LOCAL_FILE_SEARCH_PUBLIC_ID_FINGERPRINT_FIELD:local_file_search_public_id_fingerprint("fs-public")
            }),
            serde_json::json!({
                "type":"function_call","id":"fc-public","call_id":"file_search_0_0123456789abcdef",
                "name":"file_search","arguments":LOCAL_FILE_SEARCH_MARKER_ARGUMENTS,"status":"completed"
            }),
            serde_json::json!({
                "type":"function_call_output","id":"fco-public","call_id":"file_search_0_0123456789abcdef",
                "output":""
            }),
            serde_json::json!({
                "type":"file_search_call","id":"fs-legacy","status":"completed","results":[]
            }),
            serde_json::json!({
                "type":"function_call","id":"fc-legacy","call_id":"legacy","name":"file_search",
                "arguments":"{}","status":"completed"
            }),
            serde_json::json!({
                "type":"function_call_output","id":"fco-legacy","call_id":"legacy","output":""
            }),
        ];
        let normalized: Vec<_> = public
            .into_iter()
            .map(|item| normalize_item(&ctx, item).expect("public item must normalize").1)
            .collect();

        let first = preserve_local_file_search_markers(normalized.clone(), &Value::Null, None);
        let second = preserve_local_file_search_markers(normalized.clone(), &Value::Array(first), None);

        assert_eq!(second, normalized);
        assert!(
            second
                .iter()
                .all(|item| item.get(LOCAL_FILE_SEARCH_PUBLIC_ID_FINGERPRINT_FIELD).is_none())
        );
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "constructs two concurrent persisted marker overlays"
    )]
    async fn concurrent_marker_overlays_retry_without_losing_provenance() {
        use crate::store::SqliteResponseStore;

        let store = SqliteResponseStore::new("sqlite::memory:", "responses", "conversations", Some("items"))
            .await
            .expect("store must initialize");
        let conversation = ConversationRecord {
            conversation_id: "conv-race".to_owned(),
            tenant_id: "tenant".to_owned(),
            created_at: 1,
            metadata: serde_json::json!({}),
            messages: serde_json::json!([]),
        };
        ConversationItemStore::upsert_conversation(&store, &conversation)
            .await
            .expect("conversation must persist");
        let public_ids = ["fs-a", "fs-b"];
        let rows: Vec<_> = public_ids
            .iter()
            .enumerate()
            .map(|(position, id)| ConversationItemRecord {
                item_id: (*id).to_owned(),
                tenant_id: "tenant".to_owned(),
                conversation_id: "conv-race".to_owned(),
                item_data: serde_json::json!({"type":"file_search_call","id":id,"status":"completed"}),
                created_at: 1,
                position: i64::try_from(position).expect("position must fit"),
            })
            .collect();
        store
            .create_conversation_items(&rows)
            .await
            .expect("public rows must persist");
        let first_snapshot = ConversationItemStore::get_conversation(&store, "tenant", "conv-race")
            .await
            .expect("read must succeed")
            .expect("conversation must exist");
        let second_snapshot = ConversationItemStore::get_conversation(&store, "tenant", "conv-race")
            .await
            .expect("read must succeed")
            .expect("conversation must exist");
        let overlay = |id: &str| {
            let call_id = if id == "fs-a" {
                "file_search_0_0123456789abcdef"
            } else {
                "file_search_1_fedcba9876543210"
            };
            serde_json::json!([
                {
                    "type":"file_search_call","id":id,
                    LOCAL_FILE_SEARCH_PUBLIC_ID_FINGERPRINT_FIELD:local_file_search_public_id_fingerprint(id),
                    "status":"completed","results":[]
                },
                {
                    "type":"function_call","call_id":call_id,"name":"file_search",
                    "arguments":LOCAL_FILE_SEARCH_MARKER_ARGUMENTS,"status":"completed"
                },
                {"type":"function_call_output","call_id":call_id,"output":""}
            ])
        };
        let overlay_a = overlay("fs-a");
        let overlay_b = overlay("fs-b");

        let (first, second) = tokio::join!(
            sync_conversation_messages_with_overlay(&store, first_snapshot, &overlay_a),
            sync_conversation_messages_with_overlay(&store, second_snapshot, &overlay_b)
        );
        first.expect("first overlay must sync");
        second.expect("second overlay must retry and sync");
        let final_record = ConversationItemStore::get_conversation(&store, "tenant", "conv-race")
            .await
            .expect("read must succeed")
            .expect("conversation must exist");
        let messages = final_record.messages.as_array().expect("messages must be an array");
        assert_eq!(messages.len(), 6);
        assert_eq!(
            messages.iter().filter(|item| item["type"] == "function_call").count(),
            2
        );
    }

    // -------------------------------------------------------------------------
    // store_error_response
    // -------------------------------------------------------------------------

    #[test]
    fn store_error_invalid_input_returns_400() {
        let error = StoreError::InvalidInput("bad cursor".to_owned());
        let rejection = store_error_response(&error).unwrap();
        assert_eq!(rejection.status, 400);
        let body: Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["message"], "bad cursor");
    }

    #[test]
    fn store_error_database_returns_500() {
        let error = StoreError::Database("connection lost".to_owned());
        let rejection = store_error_response(&error).unwrap();
        assert_eq!(rejection.status, 500);
        let body: Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["message"], "Internal server error.");
    }

    // -------------------------------------------------------------------------
    // parse_item_list_params
    // -------------------------------------------------------------------------

    #[test]
    fn parse_params_skips_pair_without_separator() {
        let params = parse_item_list_params(Some("noseparator&limit=5"));
        assert_eq!(params.limit, 5);
        assert!(params.after_item_id.is_none());
    }

    #[test]
    fn parse_params_unknown_order_stays_default() {
        let params = parse_item_list_params(Some("order=random"));
        assert!(!params.ascending, "unknown order should keep default descending");
    }

    #[test]
    fn parse_params_non_numeric_limit_uses_default() {
        let params = parse_item_list_params(Some("limit=abc"));
        assert_eq!(params.limit, DEFAULT_PAGE_LIMIT);
    }

    // -------------------------------------------------------------------------
    // decode_query_component / decode_item_id_path_segment
    // -------------------------------------------------------------------------

    #[test]
    fn decode_query_component_invalid_utf8_uses_lossy() {
        let result = decode_query_component("%FF%FE");
        assert!(
            result.contains('\u{FFFD}'),
            "invalid UTF-8 should produce replacement characters"
        );
    }

    #[test]
    fn decode_item_id_path_segment_invalid_utf8_returns_error() {
        let result = decode_item_id_path_segment("%FF%FE");
        assert!(result.is_err(), "invalid UTF-8 should return error");
        assert!(
            result.unwrap_err().contains("valid UTF-8"),
            "error should mention UTF-8 requirement"
        );
    }

    // -------------------------------------------------------------------------
    // ItemListParams::effective_limit
    // -------------------------------------------------------------------------

    #[test]
    fn effective_limit_clamps_zero_to_one() {
        let params = ItemListParams {
            limit: 0,
            ..ItemListParams::default()
        };
        assert_eq!(params.effective_limit(), 1);
    }

    #[test]
    fn effective_limit_clamps_above_max() {
        let params = ItemListParams {
            limit: MAX_PAGE_LIMIT + 50,
            ..ItemListParams::default()
        };
        assert_eq!(params.effective_limit(), MAX_PAGE_LIMIT);
    }

    #[test]
    fn effective_limit_returns_value_within_range() {
        let params = ItemListParams {
            limit: 50,
            ..ItemListParams::default()
        };
        assert_eq!(params.effective_limit(), 50);
    }

    // -------------------------------------------------------------------------
    // store_error_response — catch-all variants
    // -------------------------------------------------------------------------

    #[test]
    fn store_error_serialization_returns_500() {
        let error = StoreError::Serialization("corrupt data".to_owned());
        let rejection = store_error_response(&error).unwrap();
        assert_eq!(rejection.status, 500);
        let body: Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["message"], "Internal server error.");
    }

    #[test]
    fn store_error_unavailable_returns_500() {
        let error = StoreError::Unavailable("not connected".to_owned());
        let rejection = store_error_response(&error).unwrap();
        assert_eq!(rejection.status, 500);
        let body: Value = serde_json::from_slice(rejection.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["message"], "Internal server error.");
    }

    // -------------------------------------------------------------------------
    // parse_item_list_params — additional edges
    // -------------------------------------------------------------------------

    #[test]
    fn parse_params_none_query_returns_defaults() {
        let params = parse_item_list_params(None);
        assert_eq!(params.limit, DEFAULT_PAGE_LIMIT);
        assert!(!params.ascending);
        assert!(params.after_item_id.is_none());
    }

    #[test]
    fn parse_params_valid_after_parameter() {
        let params = parse_item_list_params(Some("after=item_abc123&limit=10"));
        assert_eq!(params.after_item_id.as_deref(), Some("item_abc123"));
        assert_eq!(params.limit, 10);
    }

    #[test]
    fn parse_params_asc_order() {
        let params = parse_item_list_params(Some("order=asc"));
        assert!(params.ascending, "order=asc should set ascending");
    }

    #[test]
    fn parse_params_desc_order() {
        let params = parse_item_list_params(Some("order=desc"));
        assert!(!params.ascending, "order=desc should set descending");
    }

    #[test]
    fn parse_params_negative_limit_uses_default() {
        let params = parse_item_list_params(Some("limit=-5"));
        assert_eq!(
            params.limit, DEFAULT_PAGE_LIMIT,
            "negative limit should not parse as u32"
        );
    }

    #[test]
    fn parse_params_percent_encoded_after() {
        let params = parse_item_list_params(Some("after=item%20with+space"));
        assert_eq!(
            params.after_item_id.as_deref(),
            Some("item with space"),
            "percent-encoded and plus-encoded values should decode"
        );
    }

    // -------------------------------------------------------------------------
    // decode_item_id_path_segment — additional cases
    // -------------------------------------------------------------------------

    #[test]
    fn decode_item_id_plain_ascii_passes_through() {
        let result = decode_item_id_path_segment("item_abc123").unwrap();
        assert_eq!(result.as_ref(), "item_abc123");
    }

    #[test]
    fn decode_item_id_percent_encoded_ascii() {
        let result = decode_item_id_path_segment("item%5Fabc").unwrap();
        assert_eq!(result.as_ref(), "item_abc", "percent-encoded underscore should decode");
    }
}
