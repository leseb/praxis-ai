// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Generated `OpenAPI` description for local Conversations endpoints.

#![allow(dead_code, reason = "schema-only marker functions and types")]
#![expect(
    clippy::large_stack_frames,
    reason = "utoipa macro-generated schema builders allocate large temporary values"
)]

use std::collections::BTreeMap;

use serde_json::{Value, json};
use utoipa::{OpenApi, ToSchema};

/// Generate the local Conversations implementation `OpenAPI` document as
/// pretty JSON.
///
/// # Errors
///
/// Returns an error if the generated `OpenAPI` document cannot be serialized
/// as JSON.
pub fn implementation_openapi_json() -> Result<String, serde_json::Error> {
    let mut spec = serde_json::to_value(ConversationsOpenApi::openapi())?;
    normalize_discriminator_property(&mut spec, "ConversationResource", "conversation");
    normalize_discriminator_property(&mut spec, "DeletedConversationResource", "conversation.deleted");
    normalize_discriminator_property(&mut spec, "ConversationItemList", "list");
    serde_json::to_string_pretty(&spec)
}

/// Normalize generated one-variant enum component refs into OpenAI's inline
/// string enum + default shape.
fn normalize_discriminator_property(spec: &mut Value, schema: &str, value: &str) {
    let Some(property) = spec
        .pointer_mut(&format!("/components/schemas/{schema}/properties/object"))
        .and_then(Value::as_object_mut)
    else {
        return;
    };

    property.clear();
    property.insert("type".to_owned(), json!("string"));
    property.insert("enum".to_owned(), json!([value]));
    property.insert("default".to_owned(), json!(value));
}

/// Local Conversations implementation `OpenAPI` document.
#[derive(OpenApi)]
#[openapi(
    info(title = "Praxis AI OpenAI Conversations implementation", version = "0.1.0"),
    paths(
        create_conversation,
        get_conversation,
        update_conversation,
        delete_conversation,
        create_conversation_items,
        list_conversation_items,
        get_conversation_item,
        delete_conversation_item,
    ),
    components(schemas(
        CreateConversationRequest,
        UpdateConversationRequest,
        CreateConversationItemsRequest,
        ConversationResource,
        DeletedConversationResource,
        ConversationItemList,
        ConversationItem,
        ConversationObject,
        DeletedConversationObject,
        ListObject,
        ItemOrder,
    )),
    tags((name = "Conversations"))
)]
struct ConversationsOpenApi;

/// Create a conversation.
#[utoipa::path(
    post,
    path = "/conversations",
    tag = "Conversations",
    request_body(content = CreateConversationRequest, content_type = "application/json"),
    responses((status = 200, description = "OK", body = ConversationResource))
)]
fn create_conversation() {}

/// Get a conversation.
#[utoipa::path(
    get,
    path = "/conversations/{conversation_id}",
    tag = "Conversations",
    params(("conversation_id" = String, Path, description = "The ID of the conversation to retrieve.")),
    responses((status = 200, description = "OK", body = ConversationResource))
)]
fn get_conversation() {}

/// Update a conversation.
#[utoipa::path(
    post,
    path = "/conversations/{conversation_id}",
    tag = "Conversations",
    params(("conversation_id" = String, Path, description = "The ID of the conversation to update.")),
    request_body(content = UpdateConversationRequest, content_type = "application/json"),
    responses((status = 200, description = "OK", body = ConversationResource))
)]
fn update_conversation() {}

/// Delete a conversation.
#[utoipa::path(
    delete,
    path = "/conversations/{conversation_id}",
    tag = "Conversations",
    params(("conversation_id" = String, Path, description = "The ID of the conversation to delete.")),
    responses((status = 200, description = "OK", body = DeletedConversationResource))
)]
fn delete_conversation() {}

/// Create conversation items.
#[utoipa::path(
    post,
    path = "/conversations/{conversation_id}/items",
    tag = "Conversations",
    params(("conversation_id" = String, Path, description = "The ID of the conversation to add the items to.")),
    request_body(content = CreateConversationItemsRequest, content_type = "application/json"),
    responses((status = 200, description = "OK", body = ConversationItemList))
)]
fn create_conversation_items() {}

/// List conversation items.
#[utoipa::path(
    get,
    path = "/conversations/{conversation_id}/items",
    tag = "Conversations",
    params(
        ("conversation_id" = String, Path, description = "The ID of the conversation to list items for."),
        ("limit" = Option<u32>, Query, description = "Maximum number of items to return."),
        ("order" = Option<ItemOrder>, Query, description = "Sort order for returned items."),
        ("after" = Option<String>, Query, description = "Item ID to list after.")
    ),
    responses((status = 200, description = "OK", body = ConversationItemList))
)]
fn list_conversation_items() {}

/// Get one conversation item.
#[utoipa::path(
    get,
    path = "/conversations/{conversation_id}/items/{item_id}",
    tag = "Conversations",
    params(
        ("conversation_id" = String, Path, description = "The ID of the conversation that contains the item."),
        ("item_id" = String, Path, description = "The ID of the item to retrieve.")
    ),
    responses((status = 200, description = "OK", body = ConversationItem))
)]
fn get_conversation_item() {}

/// Delete one conversation item.
#[utoipa::path(
    delete,
    path = "/conversations/{conversation_id}/items/{item_id}",
    tag = "Conversations",
    params(
        ("conversation_id" = String, Path, description = "The ID of the conversation that contains the item."),
        ("item_id" = String, Path, description = "The ID of the item to delete.")
    ),
    responses((status = 200, description = "OK", body = ConversationResource))
)]
fn delete_conversation_item() {}

/// Request body accepted by `POST /conversations`.
#[derive(ToSchema)]
struct CreateConversationRequest {
    /// Optional metadata map accepted by the local implementation.
    metadata: Option<Metadata>,

    /// Optional initial items to add to the conversation.
    #[schema(max_items = 20)]
    items: Option<Vec<ConversationItem>>,
}

/// Request body accepted by `POST /conversations/{conversation_id}`.
#[derive(ToSchema)]
struct UpdateConversationRequest {
    /// Optional replacement metadata.
    metadata: Option<Metadata>,
}

/// Request body accepted by `POST /conversations/{conversation_id}/items`.
#[derive(ToSchema)]
struct CreateConversationItemsRequest {
    /// Items to create.
    #[schema(max_items = 20)]
    items: Vec<ConversationItem>,
}

/// Metadata object accepted by the local implementation.
type Metadata = BTreeMap<String, String>;

/// Local conversation response object.
#[derive(ToSchema)]
struct ConversationResource {
    /// Conversation ID.
    id: String,

    /// Object discriminator.
    #[schema(default = "conversation")]
    object: ConversationObject,

    /// Creation timestamp.
    #[schema(format = Int64)]
    created_at: i64,

    /// Conversation metadata.
    metadata: Metadata,
}

/// Delete conversation response object.
#[derive(ToSchema)]
struct DeletedConversationResource {
    /// Conversation ID.
    id: String,

    /// Object discriminator.
    #[schema(default = "conversation.deleted")]
    object: DeletedConversationObject,

    /// Whether the object was deleted.
    deleted: bool,
}

/// Conversation item list response object.
#[derive(ToSchema)]
struct ConversationItemList {
    /// Object discriminator.
    #[schema(default = "list")]
    object: ListObject,

    /// Conversation items.
    data: Vec<ConversationItem>,

    /// Whether more items are available.
    has_more: bool,

    /// First item ID in this page.
    first_id: String,

    /// Last item ID in this page.
    last_id: String,
}

/// Opaque conversation item object stored by the local implementation.
#[derive(ToSchema)]
struct ConversationItem;

/// Conversation object discriminator.
#[derive(ToSchema)]
#[schema(rename_all = "snake_case")]
enum ConversationObject {
    /// Conversation resource.
    Conversation,
}

/// Deleted conversation object discriminator.
#[derive(ToSchema)]
enum DeletedConversationObject {
    /// Deleted conversation resource.
    #[schema(rename = "conversation.deleted")]
    ConversationDeleted,
}

/// List object discriminator.
#[derive(ToSchema)]
#[schema(rename_all = "snake_case")]
enum ListObject {
    /// List resource.
    List,
}

/// Supported item list ordering.
#[derive(ToSchema)]
#[schema(rename_all = "snake_case")]
enum ItemOrder {
    /// Oldest item first.
    Asc,

    /// Newest item first.
    Desc,
}
