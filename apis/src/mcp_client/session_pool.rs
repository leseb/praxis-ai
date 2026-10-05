// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Per-execution pool of initialized MCP sessions (#1019).
//!
//! The pool is stored in request extensions so consecutive agentic rounds can
//! reuse an initialized rmcp service. Keys are opaque and include both the
//! dispatch-filter instance and the validated target identity: two dispatchers
//! in one execution can therefore never exchange sessions whose outbound
//! pipeline, timeout, or forwarded-header policy differs.
//!
//! A session's response limit remains immutable. The pool is keyed only by
//! identity, but checkout accepts the required limit and rejects mismatched
//! sessions for explicit background closure. This preserves the transport's
//! fixed response bound without retaining one bucket for every per-round limit.
//!
//! Idle sessions are deliberately scarce: one per identity and sixteen per
//! execution. An idle timer cancels the rmcp worker after one minute, stopping
//! its standalone GET SSE reconnect loop even when model inference is still in
//! progress. Checkout also rejects expired or already-closed services before a
//! tool request is sent, making a fresh connection safe under the at-most-once
//! policy.

use std::{
    collections::HashMap,
    num::NonZeroUsize,
    sync::{
        Arc, Mutex, MutexGuard, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use futures::future::join_all;
use rmcp::{RoleClient, service::RunningService};
use tokio::sync::oneshot;

use super::subrequest_transport::{TransportSignal, TransportSignalState};
use crate::openai::responses::state::retained_json_bytes;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Maximum idle sessions retained per dispatcher + target identity.
pub(crate) const MAX_IDLE_PER_KEY: usize = 1;

/// Maximum idle sessions retained across one logical Responses execution.
pub(crate) const MAX_TOTAL_IDLE: usize = 16;

/// Maximum time a parked session may keep its standalone SSE stream alive.
pub(crate) const MAX_IDLE_AGE: Duration = Duration::from_secs(60);

/// Maximum graceful-shutdown wait during final request teardown.
const MAX_CLOSE_WAIT: Duration = Duration::from_secs(5);

/// Monotonic source for process-local dispatcher namespaces.
static NEXT_POOL_NAMESPACE: AtomicU64 = AtomicU64::new(1);

// -----------------------------------------------------------------------------
// Pool Identity
// -----------------------------------------------------------------------------

/// Unique namespace assigned to one `openai_mcp_dispatch` filter instance.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct McpPoolNamespace(u64);

impl McpPoolNamespace {
    /// Allocate a namespace that cannot collide with another live dispatcher.
    pub(crate) fn new() -> Self {
        let id = NEXT_POOL_NAMESPACE.fetch_add(1, Ordering::Relaxed);
        assert!(id != 0, "MCP pool namespace counter exhausted");
        Self(id)
    }
}

/// Opaque identity for one dispatcher's validated MCP target.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct McpPoolKey {
    /// Dispatcher instance whose immutable transport policy owns the session.
    namespace: McpPoolNamespace,
    /// Credential- and target-bound identity computed by dispatch admission.
    target_fingerprint: String,
}

impl McpPoolKey {
    /// Build a reusable key, returning `None` for the fail-closed empty target
    /// fingerprint sentinel.
    pub(crate) fn new(namespace: McpPoolNamespace, target_fingerprint: String) -> Option<Self> {
        (!target_fingerprint.is_empty()).then_some(Self {
            namespace,
            target_fingerprint,
        })
    }
}

// -----------------------------------------------------------------------------
// PooledSession
// -----------------------------------------------------------------------------

/// Synchronized ownership claim for one parked session's idle cancellation.
struct IdleTimer {
    /// Wakes the timer task early after checkout or explicit closure.
    disarm: oneshot::Sender<()>,
    /// Exactly one side may claim the parked state: checkout or the timer.
    parked: Arc<AtomicBool>,
}

/// A live, initialized MCP session ready for another exclusive `tools/call`.
pub(crate) struct PooledSession {
    /// Initialized rmcp client worker.
    service: RunningService<RoleClient, ()>,
    /// Replaceable per-call transport classification slot.
    signal_state: Arc<TransportSignalState>,
    /// Immutable result limit baked into this session's transport.
    payload_limit: usize,
    /// Immutable initialize ceiling baked into this session's transport.
    initialize_limit: usize,
    /// Bound for the standalone GET parser while rmcp keeps polling this
    /// session after a successful tool call.
    stream_retained_reserve: Option<NonZeroUsize>,
    /// Instant when the session most recently entered the idle pool.
    last_used: Instant,
    /// Guard that disarms the idle cancellation task when taken or closed.
    idle_timer: Option<IdleTimer>,
}

impl PooledSession {
    /// Bind a freshly initialized service to its transport state and immutable
    /// response limit.
    pub(crate) fn new(
        service: RunningService<RoleClient, ()>,
        signal_state: Arc<TransportSignalState>,
        payload_limit: usize,
        initialize_limit: usize,
    ) -> Self {
        Self {
            service,
            signal_state,
            payload_limit,
            initialize_limit,
            stream_retained_reserve: super::subrequest_transport::tool_stream_retained_reserve(
                payload_limit,
                initialize_limit,
            )
            .and_then(NonZeroUsize::new),
            last_used: Instant::now(),
            idle_timer: None,
        }
    }

    /// The running rmcp client service used for issuing a tool request.
    pub(crate) fn service(&self) -> &RunningService<RoleClient, ()> {
        &self.service
    }

    /// Peer metadata, a possible GET parser, one serialized control reply,
    /// and the future buffered DELETE remain charged from parking through
    /// background cleanup. Closing must not need new headroom after an idle
    /// timeout or terminal drain.
    fn retained_payload_bytes(&self) -> Option<usize> {
        let info = self.service.peer_info()?;
        retained_json_bytes(info.as_ref())?
            .checked_add(self.stream_retained_reserve?.get())?
            .checked_add(super::subrequest_transport::tool_delete_retained_reserve(
                self.initialize_limit,
            )?)?
            .checked_add(super::subrequest_transport::tool_control_retained_reserve(
                self.initialize_limit,
            )?)
    }

    #[cfg(test)]
    /// Let pool tests simulate idle GET traffic after the session is parked.
    pub(crate) fn signal_state_for_test(&self) -> Arc<TransportSignalState> {
        Arc::clone(&self.signal_state)
    }

    /// Start a new call-error generation, isolating this request from any
    /// signal recorded by an earlier exchange on the same session.
    pub(crate) fn begin_call(&self) -> Arc<OnceLock<TransportSignal>> {
        self.signal_state.begin_exchange()
    }

    /// Clear this call's signal generation after rmcp has completed or failed it.
    pub(crate) fn finish_call(&self, signal: &Arc<OnceLock<TransportSignal>>) {
        self.signal_state.finish_exchange(signal);
    }

    /// Return whether the rmcp worker terminated while the session was parked.
    fn is_closed(&self) -> bool {
        self.service.is_closed()
    }

    /// Return whether this session has exceeded the bounded idle lifetime.
    fn is_expired_at(&self, now: Instant) -> bool {
        now.saturating_duration_since(self.last_used) >= MAX_IDLE_AGE
    }

    /// Arm cancellation of the rmcp worker if the session stays parked.
    fn park(&mut self) {
        self.last_used = Instant::now();
        let (disarm, timer_guard) = oneshot::channel();
        let parked = Arc::new(AtomicBool::new(true));
        let timer_parked = Arc::clone(&parked);
        let service_cancellation = self.service.cancellation_token();
        tokio::spawn(async move {
            tokio::select! {
                () = tokio::time::sleep(MAX_IDLE_AGE) => {
                    if timer_parked
                        .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        service_cancellation.cancel();
                    }
                },
                _ = timer_guard => {},
            }
        });
        self.idle_timer = Some(IdleTimer { disarm, parked });
    }

    /// Claim the parked session before its timer does.
    ///
    /// A `false` result means the timer already won and cancellation is pending,
    /// even if [`RunningService::is_closed`] has not observed it yet.
    fn unpark(&mut self) -> bool {
        if let Some(timer) = self.idle_timer.take() {
            let claimed = timer
                .parked
                .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
                .is_ok();
            let _timer_was_closed = timer.disarm.send(()).is_err();
            claimed
        } else {
            true
        }
    }

    /// Close the service and await rmcp's best-effort DELETE up to `timeout`.
    async fn close_with_timeout(mut self, timeout: Duration) {
        let _claimed_before_idle_timeout = self.unpark();
        drop(self.service.close_with_timeout(timeout).await);
    }

    /// Close the service within the normal teardown bound.
    pub(crate) async fn close(self) {
        self.close_with_timeout(MAX_CLOSE_WAIT).await;
    }

    /// A detached pool close keeps its charge until rmcp's cleanup task exits.
    /// `close_with_timeout` can return while rmcp still owns peer information.
    async fn close_fully(mut self) {
        let _claimed_before_idle_timeout = self.unpark();
        drop(self.service.close().await);
    }

    /// Close without waiting past an active tool call's absolute deadline.
    pub(crate) async fn close_before(self, deadline: tokio::time::Instant) {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        self.close_with_timeout(remaining).await;
    }

    #[cfg(test)]
    pub(crate) fn mark_expired(&mut self) {
        self.last_used = Instant::now() - MAX_IDLE_AGE;
    }

    #[cfg(test)]
    /// Simulate the timer atomically claiming cancellation before checkout.
    fn claim_idle_timeout_for_test(&self) {
        assert!(self.idle_timer.is_some(), "parked test session must have an idle timer");
        if let Some(timer) = self.idle_timer.as_ref() {
            assert!(
                timer
                    .parked
                    .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok(),
                "test idle timer must win the parked-session claim"
            );
        }
    }
}

// -----------------------------------------------------------------------------
// McpSessionPool
// -----------------------------------------------------------------------------

/// Sessions accepted and rejected by one checkout operation.
pub(crate) struct PoolCheckout {
    /// A compatible live session, when one was available.
    pub(crate) session: Option<PooledSession>,
    /// Stale, closed, or limit-mismatched sessions requiring explicit closure.
    pub(crate) rejected: Vec<PooledSession>,
}

/// Request-scoped reusable MCP sessions.
#[derive(Clone)]
pub(crate) struct McpSessionPool {
    /// Shared idle-session map and cancellation-path cleanup guard.
    inner: Arc<PoolInner>,
}

/// Shared pool state whose final drop is the cancellation-path cleanup guard.
struct PoolInner {
    /// Idle sessions grouped by opaque dispatcher + target identity.
    sessions: Mutex<HashMap<McpPoolKey, Vec<PooledSession>>>,
    /// Weak charges for sessions moved to bounded background closure. The
    /// close task owns the strong charge until its rmcp service is dropped.
    closing: Mutex<Vec<Weak<ClosingCharge>>>,
}

/// Snapshot of independently owned metadata retained by a closing session.
/// `None` keeps aggregate admission fail-closed if a size cannot be measured.
struct ClosingCharge {
    /// Snapshot of the session's bounded live payload, or unknown on overflow.
    bytes: Option<usize>,
}

impl Drop for PoolInner {
    fn drop(&mut self) {
        let map = self
            .sessions
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let sessions: Vec<_> = std::mem::take(map).into_values().flatten().collect();
        if sessions.is_empty() {
            return;
        }

        // Normal response completion calls `drain()` and awaits shutdown. If a
        // streamed response is cancelled before its EOS body hook, final pool
        // ownership is dropped instead; keep the same explicit bounded-close
        // behavior when a runtime is still available. Without one, dropping the
        // sessions still cancels their rmcp workers through RunningService.
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            drop(runtime.spawn(close_sessions(sessions)));
        }
    }
}

impl McpSessionPool {
    /// Create an empty pool.
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(PoolInner {
                sessions: Mutex::new(HashMap::new()),
                closing: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Payload retained by parked rmcp peer information, pool identities, and
    /// a possible incomplete standalone GET SSE event.
    /// rmcp keeps initialize `instructions` and `_meta` in `peer_info`; read the
    /// live value so a transparent reinitialization cannot leave a stale charge.
    pub(crate) fn retained_payload_parts(&self) -> Option<(usize, usize)> {
        let parked = self.lock().iter().try_fold(0_usize, |used, (key, sessions)| {
            let used = used.checked_add(key.target_fingerprint.len())?;
            sessions.iter().try_fold(used, |used, session| {
                used.checked_add(session.retained_payload_bytes()?)
            })
        })?;
        let mut closing_entries = self
            .inner
            .closing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        closing_entries.retain(|charge| charge.strong_count() > 0);
        let closing = closing_entries.iter().try_fold(0_usize, |used, weak| {
            weak.upgrade()
                .map_or(Some(used), |charge| used.checked_add(charge.bytes?))
        })?;
        drop(closing_entries);
        Some((parked, closing))
    }

    /// Payload retained by sessions waiting for background cleanup.
    pub(crate) fn retained_closing_payload_bytes(&self) -> Option<usize> {
        let mut closing = self
            .inner
            .closing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        closing.retain(|charge| charge.strong_count() > 0);
        closing.iter().try_fold(0_usize, |used, weak| {
            weak.upgrade()
                .map_or(Some(used), |charge| used.checked_add(charge.bytes?))
        })
    }

    /// Payload retained by parked and closing sessions.
    #[cfg(test)]
    pub(crate) fn retained_payload_bytes(&self) -> Option<usize> {
        let (parked, closing) = self.retained_payload_parts()?;
        parked.checked_add(closing)
    }

    /// Take one compatible session and return all unusable entries separately
    /// so the caller can close them outside the synchronous mutex boundary.
    pub(crate) fn checkout(&self, key: &McpPoolKey, payload_limit: usize) -> PoolCheckout {
        self.checkout_with_initialize_limit(key, payload_limit, super::MAX_CONTROL_RESPONSE_BYTES, false)
    }

    /// Checkout additionally binds the immutable handshake limit, preventing a
    /// session initialized without a budget from being reused under a smaller one.
    pub(crate) fn checkout_with_initialize_limit(
        &self,
        key: &McpPoolKey,
        payload_limit: usize,
        initialize_limit: usize,
        budgeted: bool,
    ) -> PoolCheckout {
        let mut map = self.lock();
        let Some(stack) = map.get_mut(key) else {
            return PoolCheckout {
                session: None,
                rejected: Vec::new(),
            };
        };

        let now = Instant::now();
        let mut session = None;
        let mut rejected = Vec::new();
        for mut candidate in std::mem::take(stack) {
            let idle_timer_disarmed = candidate.unpark();
            if session.is_none()
                && idle_timer_disarmed
                && candidate.payload_limit == payload_limit
                && candidate.initialize_limit == initialize_limit
                && candidate.signal_state.has_get_stream_budget() == budgeted
                && !candidate.signal_state.get_stream_exhausted()
                && !candidate.is_closed()
                && !candidate.is_expired_at(now)
            {
                session = Some(candidate);
            } else {
                rejected.push(candidate);
            }
        }
        map.remove(key);
        drop(map);
        PoolCheckout { session, rejected }
    }

    /// Return a healthy session. Rejected and superseded sessions are returned
    /// for explicit asynchronous closure by the caller.
    pub(crate) fn checkin(&self, key: McpPoolKey, mut session: PooledSession) -> Vec<PooledSession> {
        let mut map = self.lock();

        let mut rejected = Vec::new();
        if let Some(existing) = map.get_mut(&key) {
            let mut retained = Vec::with_capacity(existing.len());
            for prior in std::mem::take(existing) {
                if prior.payload_limit == session.payload_limit
                    && prior.initialize_limit == session.initialize_limit
                    && prior.signal_state.has_get_stream_budget() == session.signal_state.has_get_stream_budget()
                {
                    retained.push(prior);
                } else {
                    rejected.push(prior);
                }
            }
            *existing = retained;
        }

        let total: usize = map.values().map(Vec::len).sum();
        let stack = map.entry(key).or_default();
        if stack.len() >= MAX_IDLE_PER_KEY || total >= MAX_TOTAL_IDLE {
            rejected.push(session);
        } else {
            session.park();
            stack.push(session);
        }
        drop(map);
        rejected
    }

    /// Remove every idle session from the pool and close them concurrently.
    pub(crate) async fn drain(&self) {
        let sessions = self.take_all();
        close_sessions(sessions).await;
    }

    /// Remove every idle session and schedule bounded graceful closure.
    pub(crate) fn drain_in_background(&self) {
        self.close_sessions_in_background(self.take_all());
    }

    /// Keep rejected sessions charged until their bounded DELETE completes.
    pub(crate) fn close_sessions_in_background(&self, sessions: Vec<PooledSession>) {
        if sessions.is_empty() {
            return;
        }
        let tracked: Vec<_> = sessions
            .into_iter()
            .map(|session| {
                let charge = Arc::new(ClosingCharge {
                    bytes: session.retained_payload_bytes(),
                });
                (session, charge)
            })
            .collect();
        {
            let mut closing = self
                .inner
                .closing
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            closing.retain(|entry| entry.strong_count() > 0);
            closing.extend(tracked.iter().map(|(_, charge)| Arc::downgrade(charge)));
        }
        drop(tokio::spawn(async move {
            join_all(tracked.into_iter().map(|(session, charge)| async move {
                session.close_fully().await;
                drop(charge);
            }))
            .await;
        }));
    }

    #[cfg(test)]
    pub(crate) fn expire_all_for_test(&self) {
        for session in self.lock().values_mut().flatten() {
            session.mark_expired();
        }
    }

    #[cfg(test)]
    /// Simulate every parked timer winning its synchronization race.
    pub(crate) fn claim_all_idle_timeouts_for_test(&self) {
        for session in self.lock().values_mut().flatten() {
            session.claim_idle_timeout_for_test();
        }
    }

    /// Remove every session without awaiting while holding the pool mutex.
    fn take_all(&self) -> Vec<PooledSession> {
        let mut map = self.lock();
        std::mem::take(&mut *map).into_values().flatten().collect()
    }

    /// Recover the map even if an earlier test or caller panicked while holding
    /// the mutex; cleanup must not silently abandon live sessions.
    fn lock(&self) -> MutexGuard<'_, HashMap<McpPoolKey, Vec<PooledSession>>> {
        self.inner
            .sessions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

// -----------------------------------------------------------------------------
// Cleanup
// -----------------------------------------------------------------------------

/// Explicitly close rejected sessions concurrently.
pub(crate) async fn close_sessions(sessions: Vec<PooledSession>) {
    join_all(sessions.into_iter().map(PooledSession::close)).await;
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::future::Future;

    use rmcp::{
        service::{RxJsonRpcMessage, TxJsonRpcMessage},
        transport::Transport,
    };

    use super::*;

    /// A close that outlives the outer five-second wait in rmcp's timed API.
    struct GatedClose {
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }

    impl Transport<RoleClient> for GatedClose {
        type Error = std::io::Error;

        fn send(
            &mut self,
            _: TxJsonRpcMessage<RoleClient>,
        ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'static {
            std::future::ready(Ok(()))
        }

        async fn receive(&mut self) -> Option<RxJsonRpcMessage<RoleClient>> {
            std::future::pending().await
        }

        async fn close(&mut self) -> Result<(), Self::Error> {
            self.started.notify_one();
            self.release.notified().await;
            Ok(())
        }
    }

    #[tokio::test]
    #[expect(
        clippy::too_many_lines,
        reason = "test keeps rmcp cleanup gated past the timed-close boundary"
    )]
    async fn closing_charge_waits_for_rmcp_task_past_outer_close_timeout() {
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let service = rmcp::service::serve_directly::<RoleClient, _, _, _, _>(
            (),
            GatedClose {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
            },
            Some(rmcp::model::ServerConfig::default().into()),
        );
        let session = PooledSession::new(service, Arc::new(TransportSignalState::new(None)), 1_024, 1_024);
        let pool = McpSessionPool::new();
        pool.close_sessions_in_background(vec![session]);
        assert!(
            tokio::time::timeout(Duration::from_secs(1), started.notified())
                .await
                .is_ok(),
            "rmcp transport close must begin"
        );

        // rmcp's close_with_timeout would detach its JoinHandle after five
        // seconds even though this transport still owns the initialized peer.
        tokio::time::sleep(MAX_CLOSE_WAIT + Duration::from_millis(250)).await;
        assert!(pool.retained_payload_bytes().is_some_and(|bytes| bytes > 0));
        release.notify_one();
        assert!(
            tokio::time::timeout(Duration::from_secs(1), async {
                while pool.retained_payload_bytes() != Some(0) {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .is_ok(),
            "charge must clear when the rmcp cleanup task exits"
        );
    }

    #[test]
    fn empty_fingerprint_has_no_pool_key() {
        assert!(
            McpPoolKey::new(McpPoolNamespace::new(), String::new()).is_none(),
            "ambiguous targets must not construct reusable keys"
        );
    }

    #[test]
    #[expect(clippy::panic, reason = "test setup invariant")]
    fn dispatcher_namespaces_isolate_identical_targets() {
        let target = "same-target".to_owned();
        let Some(first) = McpPoolKey::new(McpPoolNamespace::new(), target.clone()) else {
            panic!("non-empty target must construct a key");
        };
        let Some(second) = McpPoolKey::new(McpPoolNamespace::new(), target) else {
            panic!("non-empty target must construct a key");
        };
        assert_ne!(first, second, "separate dispatch filters must never share sessions");
    }

    #[test]
    fn pool_clone_shares_backing_map() {
        let pool = McpSessionPool::new();
        let handle = pool.clone();
        assert!(Arc::ptr_eq(&pool.inner, &handle.inner), "clone must share the map");
    }
}
