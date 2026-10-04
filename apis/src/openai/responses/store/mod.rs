// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `OpenAI` Responses API store utilities.
//!
//! Helpers that operate on the generic [`ResponseStore`] but are
//! specific to the `OpenAI` Responses API (e.g., input item
//! pagination for the `/v1/responses/{id}/input_items` endpoint).
//!
//! [`ResponseStore`]: crate::store::ResponseStore

mod config;
mod filter;

#[cfg(feature = "openai-conversations")]
pub(crate) use self::filter::PersistedResponseForConversation;
pub use self::filter::ResponseStoreFilter;
pub(crate) use self::filter::{
    discard_retained_request_payload, mark_retained_request_payload_charged, request_persistence_armed,
    retained_request_payload_bytes,
};
#[cfg(feature = "openai-conversations")]
pub(crate) use self::filter::{mark_store_response_header_skipped, store_response_header_skipped};

#[cfg(test)]
#[cfg(all(feature = "store-postgres", feature = "store-sqlite"))]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cognitive_complexity,
    reason = "tests"
)]
mod tests;
