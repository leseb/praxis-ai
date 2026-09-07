// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! SSE frame reassembly, adapted onto the shared Praxis SSE codec.
//!
//! This module keeps the OpenAI-facing [`SseFrame`] / [`SseFrameParser`] /
//! [`SseParseError`] surface stable while delegating the byte-level record
//! framing to [`praxis_filter::sse::SseDecoder`]. Provider JSON typing, the
//! `[DONE]` sentinel, the event-count budget, timeouts, and lifecycle rules
//! stay in the OpenAI consumers (see `responses::parser` and the
//! `responses::stream_events` filter) — this layer only turns a byte stream
//! into data-bearing [`SseFrame`] values.

use std::{fmt, time::Duration};

use bytes::Bytes;
use praxis_filter::sse::{SseDecodeError, SseDecoder, SseLimits, SseRecord};

/// A completed SSE frame: one event boundary's worth of data.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SseFrame {
    /// Value from the last `event:` field, if present (lossily UTF-8 decoded).
    pub event_type: Option<String>,
    /// Joined `data:` field values, separated by `\n`.
    pub data: Vec<u8>,
}

/// Incremental SSE frame parser.
///
/// Feeds each body chunk to a request-scoped [`SseDecoder`] and yields the
/// data-bearing records it completed as [`SseFrame`] values on each blank-line
/// event boundary. Comment-only, `event`-only, and `id`/`retry`-only blocks are
/// framed by the decoder but carry no `data`, so they never surface as frames —
/// matching the pre-codec parser, which only dispatched blocks with at least one
/// `data:` field.
pub(crate) struct SseFrameParser {
    /// Shared-codec decoder holding the cross-chunk line/record state.
    decoder: SseDecoder,
}

impl SseFrameParser {
    /// Create a new parser with the given buffer byte limit.
    ///
    /// The single OpenAI `max_buffer_bytes` budget is applied to *both* shared-codec
    /// framing bounds — `max_line_bytes` (one partial line held across chunks) and
    /// `max_record_bytes` (the committed fields of one in-progress record) — and
    /// either overflow surfaces as [`SseParseError::BufferOverflow`].
    ///
    /// For the blocks the Responses API emits — a single `data:` line each — the
    /// record *is* that one line, so both the overflow trip point and the peak
    /// retained bytes (`max_buffer_bytes`) match the pre-codec parser. The two
    /// accountings do differ for a *multi-line* record, which the Responses API
    /// never produces: the old parser summed its carried line and joined data into
    /// one budget, whereas the codec bounds the carried line and the committed
    /// record separately, so such a record can retain up to two budgets transiently
    /// and is accepted slightly past the old combined trip point. This only ever
    /// loosens the bound — no stream the old parser accepted is now rejected.
    ///
    /// The codec's default `max_fields_per_record` is left in place: it never trips
    /// on Responses events and bounds an otherwise unbounded field vector (e.g. many
    /// empty `data:` lines), so it strengthens rather than weakens backpressure.
    pub fn new(max_buffer_bytes: usize) -> Self {
        Self {
            decoder: SseDecoder::with_limits(SseLimits {
                max_line_bytes: max_buffer_bytes,
                max_record_bytes: max_buffer_bytes,
                ..SseLimits::default()
            }),
        }
    }

    /// Feed a chunk of bytes, returning any complete SSE frames.
    pub fn parse_chunk(&mut self, chunk: &Bytes) -> Result<Vec<SseFrame>, SseParseError> {
        self.parse_chunk_with_counted_event_limit(chunk, 0, usize::MAX, |_| true)
    }

    /// Feed a chunk and stop before emitting more frames than the event budget allows.
    pub fn parse_chunk_with_event_limit(
        &mut self,
        chunk: &Bytes,
        current_events: usize,
        max_events: usize,
    ) -> Result<Vec<SseFrame>, SseParseError> {
        self.parse_chunk_with_counted_event_limit(chunk, current_events, max_events, |_| true)
    }

    /// Feed a chunk and count only selected frames against the event budget.
    ///
    /// Frames complete by this chunk are built in order; a frame for which
    /// `count_frame` returns `true` is checked against the budget *before* it is
    /// emitted, so exceeding the budget returns [`SseParseError::EventLimitExceeded`]
    /// and discards the whole chunk's frames — the pre-codec behavior. A framing
    /// limit violation likewise discards the chunk's frames and surfaces as
    /// [`SseParseError::BufferOverflow`].
    pub fn parse_chunk_with_counted_event_limit(
        &mut self,
        chunk: &Bytes,
        current_events: usize,
        max_events: usize,
        mut count_frame: impl FnMut(&SseFrame) -> bool,
    ) -> Result<Vec<SseFrame>, SseParseError> {
        // Inspect only: `push` reads the chunk (a cheap `Bytes` refcount bump for
        // Pingora), never the forwarded body, which the filter leaves untouched.
        let batch = self.decoder.push(chunk);

        let mut frames = Vec::new();
        let mut counted_in_chunk = 0;

        for record in &batch.records {
            // The decoder frames every blank-line-terminated block; only blocks
            // with a `data:` field are dispatchable events (WHATWG), matching the
            // old parser's `has_data` gate.
            if !record.is_event() {
                continue;
            }

            let frame = frame_from_record(record);
            if count_frame(&frame) {
                check_event_limit(current_events, counted_in_chunk, max_events)?;
                counted_in_chunk = counted_in_chunk.saturating_add(1);
            }
            frames.push(frame);
        }

        // A framing-limit violation poisons the decoder; report it (and drop this
        // chunk's frames) only after the event budget, preserving the byte-order
        // precedence of the pre-codec parser. An unterminated final block is
        // never salvaged: `finish` is intentionally not called, so a trailing
        // partial record stays buffered and is discarded (WHATWG), which the
        // consumers observe as a missing terminal event.
        if let Some(err) = batch.error {
            return Err(map_decode_error(err));
        }

        Ok(frames)
    }
}

/// Build an [`SseFrame`] from a data-bearing decoder record.
///
/// The owned copy mirrors the pre-codec `SseFrame`: it is an internal inspection
/// view, distinct from the response body, which the filter forwards byte-exact.
fn frame_from_record(record: &SseRecord) -> SseFrame {
    SseFrame {
        event_type: record.event().map(|bytes| String::from_utf8_lossy(bytes).into_owned()),
        data: record.data().to_vec(),
    }
}

/// Check whether another counted frame would exceed the event budget.
fn check_event_limit(current_events: usize, counted_in_chunk: usize, max_events: usize) -> Result<(), SseParseError> {
    let count = current_events.saturating_add(counted_in_chunk).saturating_add(1);
    if count > max_events {
        return Err(SseParseError::EventLimitExceeded {
            count,
            limit: max_events,
        });
    }

    Ok(())
}

/// Map a shared-codec framing error onto the OpenAI-facing [`SseParseError`].
///
/// Every retained-memory bound the codec enforces (line, record, or field count)
/// is a framing overflow from this layer's perspective, so all three collapse to
/// [`SseParseError::BufferOverflow`], preserving the pre-codec error surface.
/// `Finished` is unreachable: this adapter only ever calls `push`, never
/// `finish`, so the decoder never enters the finished state.
fn map_decode_error(err: SseDecodeError) -> SseParseError {
    let (buffered_bytes, limit) = match err {
        SseDecodeError::LineTooLong { size, limit } | SseDecodeError::RecordTooLarge { size, limit } => (size, limit),
        SseDecodeError::TooManyFields { count, limit } => (count, limit),
        SseDecodeError::Finished => (0, 0),
    };
    SseParseError::BufferOverflow { buffered_bytes, limit }
}

/// Errors from SSE parsing, shared across frame and event layers.
#[derive(Debug)]
pub(crate) enum SseParseError {
    /// Buffered bytes exceeded the configured limit.
    BufferOverflow {
        /// The number of bytes currently buffered.
        buffered_bytes: usize,
        /// The maximum allowed buffered bytes.
        limit: usize,
    },

    /// A `data:` payload was not valid JSON.
    MalformedJson {
        /// The SSE event type that had invalid JSON.
        event_type: String,
        /// The JSON parsing error description.
        err: String,
    },

    /// A required event type field was missing or not a string.
    MissingEventType {
        /// Missing field name.
        field: &'static str,
        /// Event type observed in the other event location.
        event_type: String,
    },

    /// The SSE `event:` field did not match the JSON payload `type`.
    EventTypeMismatch {
        /// Event type from the SSE `event:` field.
        sse_event_type: String,
        /// Event type from the JSON payload `type` field.
        data_event_type: String,
    },

    /// The number of parsed events exceeded the configured limit.
    EventLimitExceeded {
        /// The actual event count.
        count: usize,
        /// The maximum allowed events.
        limit: usize,
    },

    /// The stream exceeded the configured timeout.
    Timeout {
        /// The elapsed time since stream start.
        elapsed: Duration,
        /// The maximum allowed time.
        limit: Duration,
    },

    /// Stream ended without a terminal event.
    MissingTerminalEvent,

    /// A non-error event arrived after stream termination.
    EventAfterTerminal {
        /// Event type observed after termination.
        event_type: String,
    },
}

impl fmt::Display for SseParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferOverflow { buffered_bytes, limit } => write!(
                f,
                "SSE buffer overflow: {buffered_bytes} bytes exceeds {limit} byte limit"
            ),
            Self::MalformedJson { event_type, err } => {
                write!(f, "malformed JSON in SSE event '{event_type}': {err}")
            },
            Self::MissingEventType { field, event_type } => {
                write!(f, "missing string SSE event type field '{field}' near '{event_type}'")
            },
            Self::EventTypeMismatch {
                sse_event_type,
                data_event_type,
            } => write!(
                f,
                "SSE event type '{sse_event_type}' does not match JSON payload type '{data_event_type}'"
            ),
            Self::EventLimitExceeded { count, limit } => {
                write!(f, "SSE event limit exceeded: {count} events exceeds {limit} limit")
            },
            Self::Timeout { elapsed, limit } => {
                write!(f, "SSE stream timeout: {elapsed:?} exceeds {limit:?} limit")
            },
            Self::MissingTerminalEvent => write!(f, "SSE stream ended without terminal event"),
            Self::EventAfterTerminal { event_type } => {
                write!(f, "SSE event '{event_type}' arrived after terminal event")
            },
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
#[expect(clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    const MAX_BUF: usize = 65_536;

    /// Wrap a byte slice as the `Bytes` the parser (and Pingora) hand around.
    fn chunk(bytes: &'static [u8]) -> Bytes {
        Bytes::from_static(bytes)
    }

    // These are adapter tests: they cover this layer's contract with the shared
    // codec — mapping a data-bearing `SseRecord` to an `SseFrame`, skipping
    // non-event records, translating framing-limit errors, enforcing the
    // consumer event budget, and never salvaging an unterminated tail. Byte-level
    // framing (CR/CRLF/LF handling, BOM stripping, cross-chunk line reassembly,
    // optional-space and colon-less field parsing) belongs to the codec and is
    // exercised by its own conformance tests, not duplicated here.

    // -------------------------------------------------------------------------
    // Record -> SseFrame mapping
    // -------------------------------------------------------------------------

    #[test]
    fn data_record_maps_to_frame() {
        let mut parser = SseFrameParser::new(MAX_BUF);
        let frames = parser.parse_chunk(&chunk(b"data: hello\n\n")).unwrap();
        assert_eq!(frames.len(), 1, "single complete frame should dispatch");
        assert_eq!(frames[0].data, b"hello", "frame data should match");
        assert_eq!(frames[0].event_type, None, "frame should not have event type");
    }

    #[test]
    fn event_field_maps_to_frame_event_type() {
        let mut parser = SseFrameParser::new(MAX_BUF);
        let frames = parser
            .parse_chunk(&chunk(b"event: response.created\ndata: {\"id\":\"r1\"}\n\n"))
            .unwrap();
        assert_eq!(frames.len(), 1, "single event frame should dispatch");
        assert_eq!(
            frames[0].event_type.as_deref(),
            Some("response.created"),
            "event type should be captured"
        );
        assert_eq!(frames[0].data, b"{\"id\":\"r1\"}", "event data should match");
    }

    #[test]
    fn every_record_in_chunk_maps_to_a_frame() {
        let mut parser = SseFrameParser::new(MAX_BUF);
        let frames = parser.parse_chunk(&chunk(b"data: first\n\ndata: second\n\n")).unwrap();
        assert_eq!(frames.len(), 2, "both records in the chunk should map to frames");
        assert_eq!(frames[0].data, b"first", "first frame data should match");
        assert_eq!(frames[1].data, b"second", "second frame data should match");
    }

    #[test]
    fn multiline_data_is_joined_into_frame() {
        let mut parser = SseFrameParser::new(MAX_BUF);
        let frames = parser
            .parse_chunk(&chunk(b"data: line1\ndata: line2\ndata: line3\n\n"))
            .unwrap();
        assert_eq!(frames.len(), 1, "multiline data should dispatch one frame");
        assert_eq!(
            frames[0].data, b"line1\nline2\nline3",
            "the record's joined data should surface in the frame"
        );
    }

    #[test]
    fn decoder_state_persists_across_chunks() {
        // The adapter owns the request-scoped decoder, so a record split across
        // two `parse_chunk` calls must still complete.
        let mut parser = SseFrameParser::new(MAX_BUF);
        let frames1 = parser.parse_chunk(&chunk(b"data: hel")).unwrap();
        assert!(frames1.is_empty(), "partial frame should not dispatch");
        let frames2 = parser.parse_chunk(&chunk(b"lo\n\n")).unwrap();
        assert_eq!(frames2.len(), 1, "completed split frame should dispatch");
        assert_eq!(frames2[0].data, b"hello", "joined frame data should match");
    }

    #[test]
    fn lossy_event_type_from_invalid_utf8() {
        let mut parser = SseFrameParser::new(MAX_BUF);
        let frames = parser.parse_chunk(&chunk(b"event: \xFF\xFF\ndata: x\n\n")).unwrap();
        assert_eq!(
            frames.len(),
            1,
            "invalid-UTF-8 event type should still dispatch its data frame"
        );
        assert_eq!(
            frames[0].event_type.as_deref(),
            Some("\u{FFFD}\u{FFFD}"),
            "invalid UTF-8 in the event type should be replaced lossily"
        );
        assert_eq!(
            frames[0].data, b"x",
            "data should be preserved alongside the lossy event type"
        );
    }

    // -------------------------------------------------------------------------
    // Non-event records are skipped (is_event filtering)
    // -------------------------------------------------------------------------

    #[test]
    fn comment_only_record_is_not_a_frame() {
        let mut parser = SseFrameParser::new(MAX_BUF);
        let frames = parser.parse_chunk(&chunk(b": keepalive\n\ndata: hello\n\n")).unwrap();
        assert_eq!(frames.len(), 1, "a comment-only block is not a dispatchable event");
        assert_eq!(frames[0].data, b"hello", "only the data-bearing frame dispatches");
    }

    #[test]
    fn event_only_record_is_not_a_frame() {
        let mut parser = SseFrameParser::new(MAX_BUF);
        let frames = parser.parse_chunk(&chunk(b"event: ping\n\ndata: hello\n\n")).unwrap();
        assert_eq!(
            frames.len(),
            1,
            "an event-only block carries no data and does not dispatch"
        );
        assert_eq!(frames[0].data, b"hello", "only the data-bearing frame dispatches");
        assert_eq!(
            frames[0].event_type, None,
            "the discarded event: does not leak into the next frame"
        );
    }

    #[test]
    fn unterminated_final_block_is_discarded() {
        let mut parser = SseFrameParser::new(MAX_BUF);
        let frames = parser.parse_chunk(&chunk(b"data: no terminator")).unwrap();
        assert!(
            frames.is_empty(),
            "an unterminated final block is never dispatched (finish is not called; WHATWG discard)"
        );
    }

    // -------------------------------------------------------------------------
    // Framing-limit -> BufferOverflow mapping
    // -------------------------------------------------------------------------

    #[test]
    fn line_overflow_maps_to_buffer_overflow() {
        let mut parser = SseFrameParser::new(10);
        let result = parser.parse_chunk(&chunk(b"data: this line is way too long for the limit\n\n"));
        assert!(
            matches!(result, Err(SseParseError::BufferOverflow { .. })),
            "an over-long line should map to a buffer overflow error"
        );
    }

    #[test]
    fn record_overflow_maps_to_buffer_overflow() {
        let mut parser = SseFrameParser::new(20);
        let result = parser.parse_chunk(&chunk(b"event: 1234567890123\ndata: 12345678\n\n"));
        assert!(
            matches!(result, Err(SseParseError::BufferOverflow { .. })),
            "retained record bytes over the limit should map to a buffer overflow error"
        );
    }

    #[test]
    fn overflow_after_a_frame_discards_the_chunk_frames() {
        let mut parser = SseFrameParser::new(15);
        let result = parser.parse_chunk(&chunk(b"data: ok\n\ndata: this-is-way-too-long\n\n"));
        assert!(
            matches!(result, Err(SseParseError::BufferOverflow { .. })),
            "an overflow after a completed frame should still return an error, discarding the chunk's frames"
        );
    }

    // -------------------------------------------------------------------------
    // Event budget (a consumer limit layered above framing)
    // -------------------------------------------------------------------------

    #[test]
    fn event_limit_allows_frames_within_budget() {
        let mut parser = SseFrameParser::new(MAX_BUF);
        let frames = parser
            .parse_chunk_with_event_limit(&chunk(b"data: one\n\ndata: two\n\n"), 0, 5)
            .unwrap();
        assert_eq!(frames.len(), 2, "both frames should be emitted within budget");
    }

    #[test]
    fn event_limit_exact_boundary_succeeds() {
        let mut parser = SseFrameParser::new(MAX_BUF);
        let frames = parser
            .parse_chunk_with_event_limit(&chunk(b"data: one\n\ndata: two\n\n"), 0, 2)
            .unwrap();
        assert_eq!(frames.len(), 2, "exactly at limit should succeed");
    }

    #[test]
    fn event_limit_exceeded_returns_error_before_extra_frame() {
        let mut parser = SseFrameParser::new(MAX_BUF);
        let result = parser.parse_chunk_with_event_limit(&chunk(b"data: one\n\ndata: two\n\ndata: three\n\n"), 0, 2);
        assert!(
            matches!(result, Err(SseParseError::EventLimitExceeded { count: 3, limit: 2 })),
            "third frame should exceed limit of 2"
        );
    }

    #[test]
    fn event_limit_accounts_for_current_events() {
        let mut parser = SseFrameParser::new(MAX_BUF);
        let result = parser.parse_chunk_with_event_limit(&chunk(b"data: one\n\n"), 5, 5);
        assert!(
            matches!(result, Err(SseParseError::EventLimitExceeded { count: 6, limit: 5 })),
            "current_events at limit should reject the next frame"
        );
    }

    #[test]
    fn event_limit_no_frames_always_succeeds() {
        let mut parser = SseFrameParser::new(MAX_BUF);
        let frames = parser
            .parse_chunk_with_event_limit(&chunk(b"data: partial"), 100, 100)
            .unwrap();
        assert!(
            frames.is_empty(),
            "no complete frames should succeed regardless of current count"
        );
    }

    #[test]
    fn counted_event_limit_skips_uncounted_frames() {
        let mut parser = SseFrameParser::new(MAX_BUF);
        // The `[DONE]` sentinel is not counted against the budget but is still
        // emitted, mirroring the OpenAI consumers' `count_frame` predicate.
        let frames = parser
            .parse_chunk_with_counted_event_limit(&chunk(b"data: one\n\ndata: [DONE]\n\n"), 0, 1, |frame| {
                frame.data != b"[DONE]"
            })
            .unwrap();
        assert_eq!(
            frames.len(),
            2,
            "the uncounted sentinel frame is still emitted within budget"
        );
        assert_eq!(frames[1].data, b"[DONE]", "sentinel frame preserved");
    }

    // -------------------------------------------------------------------------
    // SseParseError Display
    // -------------------------------------------------------------------------

    #[test]
    fn display_malformed_json() {
        let err = SseParseError::MalformedJson {
            event_type: "response.created".to_owned(),
            err: "expected value at line 1 column 1".to_owned(),
        };
        let msg = err.to_string();
        assert!(msg.contains("response.created"), "should mention the event type");
        assert!(msg.contains("expected value"), "should include the JSON error");
    }

    #[test]
    fn display_missing_event_type() {
        let err = SseParseError::MissingEventType {
            field: "data.type",
            event_type: "response.created".to_owned(),
        };
        let msg = err.to_string();
        assert!(msg.contains("data.type"), "should mention the missing field");
        assert!(
            msg.contains("response.created"),
            "should mention the observed event type"
        );
    }

    #[test]
    fn display_event_type_mismatch() {
        let err = SseParseError::EventTypeMismatch {
            sse_event_type: "response.completed".to_owned(),
            data_event_type: "response.output_text.delta".to_owned(),
        };
        let msg = err.to_string();
        assert!(msg.contains("response.completed"), "should mention the SSE event type");
        assert!(
            msg.contains("response.output_text.delta"),
            "should mention the payload event type"
        );
    }

    #[test]
    fn display_event_limit_exceeded() {
        let err = SseParseError::EventLimitExceeded { count: 101, limit: 100 };
        let msg = err.to_string();
        assert!(msg.contains("101"), "should mention the count");
        assert!(msg.contains("100"), "should mention the limit");
    }

    #[test]
    fn display_timeout() {
        let err = SseParseError::Timeout {
            elapsed: Duration::from_secs(35),
            limit: Duration::from_secs(30),
        };
        let msg = err.to_string();
        assert!(msg.contains("35"), "should mention elapsed time");
        assert!(msg.contains("30"), "should mention the limit");
    }

    #[test]
    fn display_missing_terminal_event() {
        let err = SseParseError::MissingTerminalEvent;
        let msg = err.to_string();
        assert!(msg.contains("terminal"), "should mention missing terminal event");
    }

    #[test]
    fn display_buffer_overflow() {
        let err = SseParseError::BufferOverflow {
            buffered_bytes: 70_000,
            limit: 65_536,
        };
        let msg = err.to_string();
        assert!(msg.contains("70000"), "should mention buffered bytes");
        assert!(msg.contains("65536"), "should mention the limit");
    }

    #[test]
    fn display_event_after_terminal() {
        let err = SseParseError::EventAfterTerminal {
            event_type: "response.output_text.delta".to_owned(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("response.output_text.delta"),
            "should mention the event type"
        );
        assert!(msg.contains("terminal"), "should mention terminal context");
    }
}
