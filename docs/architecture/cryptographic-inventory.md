<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Praxis Contributors -->

# Cryptographic operations inventory

Operation-level inventory of the cryptography reachable from the Praxis AI
proxy, tracked for [#1220][i1220]. Unlike a dependency list, this connects each
reachable operation to its **caller**, **algorithm**, the **implementation
provider that actually executes** the primitive (not just the top-level crate),
its **build/feature conditions**, and its **production-vs-test** status.

- **Status:** initial inventory (spike). Maintained by the reviewable check
  `cargo xtask check-crypto-inventory` (see [Maintenance](#maintenance)).
- **Profile of record:** the default production binary
  `praxis-ai-proxy` with `--no-default-features --features store-postgres`
  (the workspace default). Other profiles are called out per row.
- **Method:** every claim is verified against the actual resolved graph and
  against the source at the cited `file:line`. "In the production **binary**"
  means the runtime graph `cargo tree --edges normal` (excludes both dev- and
  build-dependencies — matching the repo's own guard at `Makefile:94`); build
  host tooling is checked separately with `--edges no-dev`. Reproduction
  commands are in the [appendix](#appendix-reproduction).
- **Intended validated module:** `aws-lc-rs` in its FIPS-140-3 build mode. Any
  operation is only labelled `validated` when the executing path is proven to
  run that module in that mode. Today **no path qualifies** (see the two
  headline findings below).

Related investigations — referenced here, **not** duplicated:
[#814][i814] (SigV4 FIPS signing) and [#1217][i1217] (PostgreSQL cryptographic
boundary).

## Headline findings

1. **The process-wide rustls provider is `ring`, not `aws-lc-rs`.** `rustls` is
   compiled with *both* the `aws-lc-rs` and `ring` features, but the Pingora
   fork installs `ring` as the process default
   (`CryptoProvider::install_default(rustls::crypto::ring::default_provider())`)
   at the very first step of server start-up, before any TLS object is built.
   Every rustls consumer that resolves via `CryptoProvider::get_default()` —
   the data-plane listener, the upstream connector, all `reqwest` outbound
   callouts, and the Redis/Valkey session client — therefore runs **ring**. The
   `aws-lc-rs` code is linked but only actually executes for **JWT verification**
   (a praxis-core identity plugin that calls `aws-lc-rs` directly, bypassing the
   rustls default). Consequence: the `prefer-post-quantum` `X25519MLKEM768`
   group is aws-lc-rs-only and is therefore **inert** at runtime.

2. **FIPS build mode is not enabled anywhere.** `aws-lc-rs@1.18.1` is built with
   `alloc, aws-lc-sys, default, prebuilt-nasm, ring-io, ring-sig-verify` — the
   `fips` feature is **absent**. So even the JWT path (the one place aws-lc-rs
   runs) is not a validated build, and even after fixing finding 1 the module
   would still need its `fips` feature to be compliant.

3. **The PostgreSQL store TLS stack diverges from the data plane.** The default
   `store-postgres` feature selects `sqlx/tls-native-tls`, routing store TLS
   through the platform `native-tls` backend (OpenSSL on Linux,
   Security.framework on macOS, schannel on Windows) — a completely different
   module from the rustls/ring data plane. PostgreSQL authentication crypto
   (SCRAM-SHA-256, legacy MD5) runs on RustCrypto via `sqlx-postgres`. Tracked
   by [#1217][i1217].

4. **Test-only cryptography is provably contained.** `rcgen` (self-signed test
   cert generation), `yasna`, and `sha1` (WebSocket accept-key) are absent from
   both the runtime and the build graph. `sha3` (Cedar grammar codegen) is a
   **build-host-only** build-dependency: absent from `--edges normal` (the
   runtime binary), present only under `--edges no-dev`. None reaches the
   shipped artifact. See
   [Test-only cryptography](#test-only-cryptography-and-containment).

## Build-profile model

| Feature | Default? | Crypto it activates |
| --- | --- | --- |
| `store-postgres` | **yes** | `sqlx-postgres` (SCRAM/MD5 auth) + `native-tls` (OpenSSL/Security.framework/schannel) store TLS |
| `store-sqlite` | no | none (bundled plaintext SQLite; no SQLCipher/TLS/auth) |
| `azure-ad-filter` | no (experimental) | outbound OAuth token fetch over reqwest TLS (no local signing) |
| `gcp-adc-filter` | no (experimental) | outbound ADC/metadata token fetch over reqwest TLS (no local signing) |
| `token-rate-limit-filter` | no (experimental) | `redis` client + optional `rediss://` rustls TLS; SHA-256 keyspace derivation |
| `basic-auth-filter` | no (experimental) | praxis-core caller-credential verification (A12): unsalted SHA-256 password digest + constant-time compare (upstream) |

`aws-sigv4` and `jsonwebtoken` are **not** feature-gated — their crypto is
compiled into every default binary regardless of whether an operator configures
the `aws_sigv4_sign` filter or a JWT identity policy.

## Provider selection (the load-bearing detail)

```
server/src/server.rs:65  run_server() → create_subrequest_client(&config)   ← FIRST statement
  → server/src/subrequest.rs:46  SubRequestConnector::with_options
  → praxis-proxy-core  subrequest/internals.rs:120  Connector::new (pingora http)
  → quixotic-plecostomus-core  connectors/mod.rs:205  tls::Connector::new
  → quixotic-plecostomus-rustls/src/lib.rs:44-45
        CryptoProvider::install_default(rustls::crypto::ring::default_provider())
```

This runs **before** `build_full_registry` (server.rs:66), pipeline resolution
(server.rs:102), and protocol/listener registration (server.rs:106).
`install_default` is first-writer-wins, so `ring` wins the process default.
Downstream:

- `reqwest@0.13.5` (`client.rs:721`) selects
  `CryptoProvider::get_default()…unwrap_or_else(default_rustls_crypto_provider)`
  — it uses the installed default (**ring**) and only falls back to its
  compiled-in `aws-lc-rs` when *nothing* is installed. It never reaches the
  fallback here.
- `praxis-tls` and `redis`/`tokio-rustls` likewise resolve via `get_default()`
  → **ring**.
- `jsonwebtoken@11.1.0` (feature `aws_lc_rs`) calls `aws-lc-rs` **directly**,
  independent of the rustls default → this is the only runtime aws-lc-rs
  execution.

The AI crates install **no** process-default provider of their own (grep for
`install_default`/`CryptoProvider` across `server/src apis/src filters/src` is
empty). The root-cause remediation (R1) is to install
`aws-lc-rs` as the default before `create_subrequest_client`.

## Production-reachable cryptographic operations

Provider = the crate whose code actually executes the primitive. Reachability
values: `prod-default` (in the default binary), `prod-optional` (only under a
named non-default feature), `unreachable-compliance` (linked but dead under the
compliance profile). Disposition (the set the check enforces on every
`production` entry): `validated`, `non-validated`, `needs-remediation`,
`upstream-owned` (crypto lives in praxis-core/third-party), `n-a` (non-security
integrity/encoding).

### Transport TLS (data plane + outbound callouts)

All rows execute on **ring@0.17.14** at runtime (finding 1).

| # | Operation | Caller | Algorithm | Provider | Reach | Disp | Remediation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | Process rustls provider install | `quixotic-plecostomus-rustls/src/lib.rs:44` via `server/src/server.rs:65` | `install_default(ring)` | ring | prod-default | needs-remediation | **R1** |
| T2 | Upstream data-plane handshake (proxy→LLM) | pingora `HttpProxy` client_upstream connector | TLS 1.2/1.3, ECDHE-X25519/P-256/P-384, AES-GCM/ChaCha20-Poly1305 | ring (via rustls) | prod-default | non-validated | R1 |
| T3 | AI subrequest callout handshake (web_search, file_search/api_client, guardrails/NeMo, compact) | `praxis_core::subrequest::SubRequestClient` (e.g. `apis/src/web_search/provider.rs:398`, `apis/src/openai/api_client/mod.rs:593`, `filters/src/guardrails/providers/nemo.rs:96`) | TLS 1.2/1.3 client + X.509 verify | ring | prod-default | non-validated | R1 |
| T4 | Downstream listener handshake (client→proxy) | `praxis-tls setup.rs:191-214 build_server_config_base` | TLS 1.2/1.3 server | ring | prod-default | needs-remediation | R1 |
| T5 | Outbound reqwest TLS: MCP callouts + `file_resolve` file_url | `apis/src/mcp_client/mod.rs:315/407/719`, `apis/src/openai/responses/file_resolve/resolve_url.rs:360/572` | TLS 1.2/1.3 client | ring (reqwest `get_default()`) | prod-default | needs-remediation | R1 |
| T6 | Outbound reqwest TLS: Azure AD / GCP token fetch | `apis/src/callout_target.rs:232` ← `filters/src/azure/azure_ad.rs:277`, `filters/src/gcp/token.rs:106` | TLS 1.2/1.3 client | ring | prod-optional (`azure-ad-filter`/`gcp-adc-filter`) | non-validated | R1 |
| T7 | Redis/Valkey session-store TLS (`rediss://`) | `praxis-policy-session-valkey` → `deadpool-redis` → `redis@1.7.0` | TLS 1.2/1.3 client | ring | prod-default (compiled); active only with `rediss://` | non-validated | R1 (upstream) |
| T8 | Valkey TLS for token-rate-limit | `filters/src/token_rate_limit/backend.rs:21` | TLS 1.2/1.3 client | ring | prod-optional (`token-rate-limit-filter`) | non-validated | R1 |
| T9 | TLS handshake CSPRNG (randoms, nonces, ECDHE scalars, session keys) | internal to active provider | `ring::rand::SystemRandom` (OS-seeded) | ring | prod-default | non-validated | R1 |
| T10 | Post-quantum hybrid KX `X25519MLKEM768` | rustls provider construction (`prefer-post-quantum`) | ML-KEM-768 + X25519 | aws-lc-rs only | **unreachable-compliance** (inert under ring) | needs-remediation | R1 makes it live |

### Certificate / trust-anchor handling (transport)

| # | Operation | Caller | Algorithm | Provider | Reach | Disp | Remediation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | Upstream + reqwest server-cert chain & hostname verification | pingora `connectors/tls/rustls/mod.rs` connect(); `reqwest client.rs:759/798` | X.509 path validation + ECDSA/RSA sig verify | **pingora upstream:** `rustls-webpki@0.103.15` (sig algs from ring). **reqwest default path:** `rustls-platform-verifier@0.7.0` — Linux runs rustls `WebPkiServerVerifier` (ring); **macOS uses Security.framework and Windows uses CryptoAPI** for chain-building + hostname validation, routing only the TLS-handshake signature check through the rustls provider (see note) | prod-default | non-validated | **R1 (partial — see note)**; also audit operator `verify_cert=false`/empty-SNI `SkipAll` paths |
| C2 | Downstream mTLS client-cert verification (data-plane listener) | `praxis-tls setup.rs:205-213` → `client_auth::build_client_verifier` | X.509 client-cert path + sig verify (+ optional CRL) | `rustls-webpki` (ring) | prod-optional (listener `client_auth`) | non-validated | R1 (distinct from PG mTLS #1217) |
| C3 | Listener cert + private-key PEM load & consistency check | `praxis-tls setup/loader.rs:44-60` | PEM→DER decode; PKCS1/PKCS8/SEC1 key parse; `keys_match` | `rustls-pemfile@2.2.0` + `rustls-pki-types@1.15.1`; `zeroize` for key buffers | prod-default | n-a | none (sound; signing backend follows provider) |
| C4 | Upstream cert SHA-256 fingerprint (telemetry/identity) | `quixotic-plecostomus-rustls/src/lib.rs:188` `hash_certificate()` | SHA-256 (unkeyed, non-secret) | ring `ring::digest::SHA256` | prod-default | n-a | none |
| C5 | OS native trust-store load | pingora `load_platform_certs_incl_env_into_store`; reqwest path | Load system CA X.509 (no primitive) | `rustls-native-certs@0.7.3` + `0.8.4` | prod-default | n-a | none |
| C6 | X.509 peer/chain structural parse (mTLS/CA extraction) | pingora / praxis-core | ASN.1 X.509 v3 parse | `x509-parser@0.18.1` → `der-parser@10` → `asn1-rs@0.7` | prod-default | upstream-owned | upstream |
| C7 | JWT key ASN.1/PEM parse | praxis-core identity-jwt via `jsonwebtoken` | PKCS1/PKCS8/SEC1 DER + PEM decode | `simple_asn1@0.6.4` + `pem@3.0.6` | prod-default | upstream-owned | upstream (see A1) |

> **C1 is not fully migrated by R1.** `rustls-platform-verifier@0.7.0` is
> platform-split. On **Linux** it delegates to rustls' own
> `WebPkiServerVerifier` (chain-building, hostname match, and signature check all
> run through the process `CryptoProvider` — `ring` today, `aws-lc-rs` after R1),
> so R1 migrates it completely — as it does the pingora upstream path (`rustls-webpki`)
> on every platform. On **macOS** (`apple::Verifier`, Security.framework) and
> **Windows** (`windows::Verifier`, CryptoAPI) the reqwest default path performs
> chain-building and hostname validation in the **OS** verifier; only
> `verify_tls12_signature`/`verify_tls13_signature` (the handshake `CertificateVerify`
> signature) is routed back through the rustls provider. So on those two targets
> R1 migrates only the handshake signature check — the trust decision stays with
> the platform module and is out of scope for the validated-provider goal.

### Authentication / signing

| # | Operation | Caller | Algorithm | Provider | Reach | Disp | Remediation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | JWT signature **verify** (identity authn; decode-only) | `praxis-policy-plugin-identity-jwt` (no AI call site): builds only `DecodingKey` + `Validation` (`config.rs` `from_rsa_pem`/`from_ec_pem`/`from_ed_pem`/`from_jwk`/`from_secret`) — no `EncodingKey`/`encode`, so production **verifies, never signs** | HS256/RS256/PS256/ES256/EdDSA (accepted-verify set upstream) | **aws-lc-rs@1.18.1 (no `fips`)** via `jsonwebtoken@11.1.0`; signing/verifying traits via `signature@2.2.0`; JWT-key ASN.1/PEM via `simple_asn1`/`pem` (C7) | prod-default | non-validated (no FIPS build) | **R2** (upstream: pin aws-lc-rs FIPS) |
| A2 | OAuth delegated-token **cache-key** derivation (keyed) | `praxis-policy-plugin-delegator-oauth` `src/cache/key.rs` `derive()` (no AI call site) — HMAC over a length-prefixed identity+payload encoding under a per-process secret; **not** OAuth state/nonce/PKCE (this is a token-delegation/exchange plugin, no authorization-code flow) | HMAC-SHA256 | `hmac@0.13.0` + `sha2@0.11.0` (RustCrypto); per-process secret from `getrandom` (R-4) | prod-default (compiled); executes only when a delegator `cache:` block is configured | upstream-owned | **R7** (upstream `praxis-proxy/policy`) |
| A3 | AWS SigV4 signing-key derivation + StringToSign | `filters/src/aws/sigv4.rs:151` `sign_headers`; registered `register.rs:89` (unconditional) | HMAC-SHA256 4-stage key derivation | `hmac@0.13.0` + `sha2@0.11.0` (RustCrypto) via `aws-sigv4@1.5.3` | prod-default (compiled); executes only when `aws_sigv4_sign` configured | non-validated | **#814** |
| A4 | SigV4 payload digest (`x-amz-content-sha256`) | `filters/src/aws/sigv4.rs:136` | SHA-256 over buffered body | `sha2@0.11.0` via `aws-sigv4` | prod-default | non-validated | **#814** |
| A5 | PG SCRAM-SHA-256 SASL auth | `sqlx-postgres sasl.rs` ← `apis/src/store/postgres.rs:102` | HMAC-SHA-256 + SHA-256 + salted-password `Hi` iteration | `hmac@0.13.0` + `sha2@0.11.0` (RustCrypto) | prod-default | non-validated | **#1217** |
| A6 | PG SCRAM SASLprep | `sqlx-postgres sasl.rs` | RFC 4013 Unicode normalization | `stringprep@0.1.5` | prod-default | n-a | #1217 |
| A7 | PG SCRAM client-nonce RNG | `sqlx-postgres sasl.rs:175` `gen_nonce()` | ChaCha CSPRNG (OS-seeded) | `rand@0.10.2` → `getrandom@0.4.3` | prod-default | non-validated | #1217 |
| A8 | PG legacy MD5 password auth | `sqlx-postgres password.rs:62` | `MD5(md5(pw+user)+salt)` challenge (broken) | `md-5@0.11.0` (RustCrypto) | prod-default | non-validated | **#1217** (disallow under compliance) |
| A9 | PG cleartext password auth | `sqlx-postgres password.rs` | none (relies on store TLS) | n/a; secret wrapped by `secrecy@0.10.3` | prod-default | n-a | #1217 |
| A10 | PG store TLS handshake + cert verify (`VerifyFull` default) | `apis/src/store/postgres.rs:102` `connect_with` | TLS 1.2/1.3 + X.509 chain/hostname | **native-tls@0.2.18** → OpenSSL@0.10 (Linux) / Security.framework@3 (macOS) / schannel (Windows) | prod-default | non-validated | **#1217** / **R4** |
| A11 | PG client-cert auth (mTLS, pg_hba `cert`) | `apis/src/store/postgres.rs:102` (URL `sslcert`/`sslkey`) | X.509 client cert + TLS CertificateVerify sig | native-tls → platform backend | prod-optional (operator config) | non-validated | **#1217** |
| A12 | Basic Auth caller-credential **verify** | praxis-core builtin `.../basic_auth/filter.rs:237` (`Sha256::digest`) + `:226` (`ct_eq`) | unsalted SHA-256 password digest + constant-time compare | `sha2@0.11.0` + `subtle@2.6.1` (RustCrypto) | prod-optional (`basic-auth-filter`; **not** in the profile of record) | upstream-owned | upstream (weak unsalted scheme; **R7**-adjacent) |

### Application hashing (our code) — security relevance called out

| # | Operation | Caller | Algorithm | Provider | Reach | Disp | Remediation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H1 | **MCP approval target-identity fingerprint** (binds an `mcp_approval_response` to the exact resolved target incl. the `authorization` token) | `apis/src/openai/responses/mcp_dispatch/approval.rs:261-294` | SHA-256, unkeyed, length-framed, hex | `sha2@0.11.0` (RustCrypto) | prod-default | **non-validated** (only genuine security control among our hashes) | **R3** |
| H2 | Routing candidate `stable_id` fallback (session-affinity id) | `filters/src/routing/descriptor.rs:242-261` | SHA-256, unkeyed | `sha2@0.11.0` | prod-default | n-a (non-secret affinity key) | none |
| H3 | Routing overlay change-detection hash (skip re-parse when file byte-identical) | `filters/src/routing/overlay.rs:433/1072` | SHA-256 over file bytes | `sha2@0.11.0` | prod-default | n-a | none |
| H4 | Routing overlay semantic content-digest self-check (RFC-8785 canonical) | `filters/src/routing/overlay.rs:692-715` | SHA-256 over canonical JSON | `sha2@0.11.0` | prod-default | n-a (self-consistency check) | none |
| H5 | token_rate_limit subject → bucket pseudonymization | `filters/src/token_rate_limit/mod.rs:1342` | SHA-256 + base64url | `sha2@0.11.0` | prod-optional (`token-rate-limit-filter`) | n-a | none |
| H6 | token_rate_limit Valkey key-name derivation | `filters/src/token_rate_limit/backend.rs:796/1138` | SHA-256, hex | `sha2@0.11.0` | prod-optional | n-a | none |

### Upstream policy-engine hashing (praxis-core policy path)

Reached via the praxis-core policy engine, not AI code, but in the profile-of-record graph and security-relevant, so inventoried here.

| # | Operation | Caller | Algorithm | Provider | Reach | Disp | Remediation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | **Session-identity binding** (derives the session key from `sha256(subject : caller_workload : this_workload)[:16]` so an attacker-chosen raw header cannot forge a session) | `praxis-policy-apl-runtime@0.2.0/src/session_resolver.rs:82/84` (identity tier) | SHA-256, unkeyed, first 64 bits, hex | `sha2@0.11.0` (RustCrypto) | prod-default (executes when the policy engine binds session identity) | upstream-owned | **R7** (upstream `praxis-proxy/policy`) |
| P2 | Valkey session-store key-name derivation | `praxis-policy-session-valkey@0.2.0/src/store.rs:78/80` | SHA-256 over `session_id`, hex | `sha2@0.11.0` (RustCrypto) | prod-default (compiled); active only with a Valkey session store configured | n-a (non-secret key name) | none |

> AI code instantiates no keyed-MAC, signature, or constant-time-equality
> primitive directly: `grep` for `Hmac`/`Mac`/`new_from_slice`/`subtle` across
> `apis/filters/server` returns nothing. What the enclosing **process** runs is
> broader:
>
> - **Keyed HMACs:** SigV4 request signing (A3/A4, our `sigv4` filter via
>   `aws-sigv4`), the upstream OAuth delegated-token cache key (A2), PG
>   SCRAM-SHA-256 client/server proofs (A5, via `sqlx-postgres`), and the TLS
>   handshake key schedule itself (the active provider's HMAC/PRF — `ring` today).
> - **KDFs:** TLS 1.3 runs HKDF-Extract/Expand **inside `ring`** (rustls'
>   `RingHkdf` over `ring::hkdf`), and TLS 1.2 uses ring's PRF — these execute on
>   every handshake. The standalone RustCrypto `hkdf@0.13.0` crate is a *different*
>   code path: pulled in only by `sqlx-postgres` (`PgAdvisoryLock`) and **not
>   invoked** by the AI store (see the integrity table).
> - **Constant-time comparison:** `subtle@2.6.1` executes transitively inside
>   rustls, and — only when the non-default `basic-auth-filter` feature is enabled
>   — in the Basic Auth builtin's credential compare (A12). The AI proxy itself
>   injects credentials on the outbound side and does not verify caller
>   credentials unless that upstream feature is turned on.

### Randomness

| # | Operation | Caller | Algorithm | Provider | Reach | Disp | Remediation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R-1 | Load-balancing candidate selection (`Random`/`WeightedRandom`) | `filters/src/routing/picker.rs:69/74` | ChaCha12 thread RNG | `rand@0.10.2` → `chacha20@0.10.2` → `getrandom@0.4.3` | prod-default | n-a (unpredictability not required) | none |
| R-2 | Correlation/idempotency ids (`resp_*`, `conv_*`, compact, metering) | `apis/src/openai/responses/validate/mod.rs:123` &al. | praxis-core `IdGenerator`: 48-bit µs clock + 32-bit per-instance seed | praxis-core (`id.rs`), one-time `rand@0.10.2` seed | prod-default | upstream-owned (**not** a security boundary today) | none unless promoted to a secret |
| R-3 | MCP session id (`mcp-session-id` header) | `filters/src/agentic/mcp/broker/mod.rs:824` | same `IdGenerator` → **predictable** | praxis-core | prod-default | upstream-owned | note: switch to CSPRNG if it ever becomes session-binding |
| R-4 | OAuth delegator **cache-key secret** (per-process, 32 bytes) | `praxis-policy-plugin-delegator-oauth` `src/cache/key.rs` `KeySecret::random()` (seeds the A2 HMAC; there is no state/nonce/PKCE in this token-delegation plugin) | `getrandom@0.4.3` `fill()` → OS CSPRNG | OS CSPRNG via `getrandom` | prod-default (compiled); drawn only when a delegator `cache:` block is configured | upstream-owned | **R7** (upstream `praxis-proxy/policy`) |

## Non-cryptographic / integrity-only (out of validated-module scope)

Recorded so they are not mistaken for untraced crypto. None is a security
primitive.

| Operation | Algorithm | Provider | Reach | Notes |
| --- | --- | --- | --- | --- |
| Pingora HTTP cache key hashing | **BLAKE2b-128** | `blake2@0.11.0` (via `quixotic-plecostomus-cache`) | prod-default | A real crypto hash, but used only as a non-secret cache key (replaces md5). Upstream-owned. |
| Streaming item-dedup digest | SipHash-1-3 (`DefaultHasher`) | Rust std | prod-default | `apis/.../stream_events/mod.rs:1168`; single-process dedup |
| Opaque `mcp_list_tools` id | SipHash-1-3 (`DefaultHasher`) | Rust std | prod-default | `apis/.../openai_mcp_tool_resolve/mod.rs:2360` |
| Anthropic fallback `request_id` | `SystemTime` nanos, hex | Rust std (no RNG) | prod-default | `apis/src/anthropic/messages_format/mod.rs:186` |
| Concurrent hashmap (flurry) | aHash | `ahash@0.8.12` | prod-default | Pingora fork; DoS-resistant hashmap |
| Deflate/gzip integrity | CRC-32 | `crc32fast@1.5.2` | prod-default | Pingora fork |
| BPE tokenizer vocab decode | base64 + CRC-32 | `tiktoken-rs@0.12.0` | prod-default | `apis/.../compact/mod.rs:197` |
| HTTP/2 header hashing | FNV | `fnv@1.0.7` (via `h2`) | prod-default | non-crypto |
| `ahash` compile-time seed | Keccak-f permutation | `tiny-keccak@2.0.2` (via `const-random-macro`) | **build-host only** | Keccak-named, but reachable only beneath `const-random-macro` (a proc-macro), so it runs on the build host to seed a hashmap and is **not linked into the runtime binary** — classified `build_only`, not runtime content (see [containment](#test-only-cryptography-and-containment)) |
| RustCrypto digest support | — (OID constants, block buffering, type-level ints) | `const-oid`, `digest`, `block-buffer`, `crypto-common`, `generic-array`, `hybrid-array`, `cpufeatures`, `cmov`, `typenum` | prod-default | primitive-support layer under `sha2`/`hmac`/`hkdf`/`md-5`/`blake2`; no standalone operation |
| Perfect-hash lookups (chrono-tz, cedar) | SipHash (phf) | `siphasher@1.0.3` | prod-default | via praxis-core OPA/policy path |
| sqlx migration checksum / protocol CRC | SHA-384/256 / CRC | `sha2@0.10.9` / `crc@3` (via `sqlx-core`) | prod-default | **compiled but not invoked** — AI store emits DDL directly, never runs sqlx migrations |
| `PgAdvisoryLock` key derivation | HKDF-SHA256 | `hkdf@0.13.0` (via `sqlx-postgres`) | prod-default | **compiled but not invoked** by the AI store |
| base64/base64url transport encode | base64 | `base64@0.23.1`/`0.22.1`/`base64-simd` | prod-default | encoding, not crypto (`apis/src/state_owner.rs:756`) |
| Metrics reservoir sampling | xoshiro/ChaCha | `rand@0.9.5` + `rand_xoshiro` (metrics-util) | prod-default | non-security statistical sampling |
| Temp-file naming / retry jitter | wyrand | `fastrand@2.5.0` | prod-default | non-crypto (native-tls temp files, redis backoff) |
| UUID v4 (policy/cache internals) | 122-bit random | `uuid@1.26.1` → `rand@0.10.2` | prod-default | upstream policy/cache; not an AI-issued token |
| Rego bignum arithmetic | — | `num-bigint` (regorus) | prod-default | not crypto |

## Test-only cryptography and containment

Acceptance criterion: *test-only cryptography is demonstrably absent from
production artifacts.* **Verified.** In the runtime graph
`cargo tree -p praxis-ai-proxy --edges normal --no-default-features --features
store-postgres -i <crate>` returns *"did not match any packages"* for every row
below:

| Crate | Operation | Where it lives | In runtime binary? | In build graph? |
| --- | --- | --- | --- | --- |
| `rcgen@0.14.10` (→ ring, `yasna`) | Self-signed test CA/server/client cert + key generation | `tests/utils` (`tests/utils/src/net/tls.rs`), consumed only by `xtask`/`tests/*` dev-deps | no | no |
| `yasna@0.6.0` | DER writer for `rcgen` | via `rcgen` (dev) | no | no |
| `sha1@0.11.0` | WebSocket `Sec-WebSocket-Accept` digest | `tungstenite` ← `tokio-tungstenite` ← `praxis-test-utils` | no | no |
| `sha3@0.10.9` + `keccak` | Cedar grammar parser-table hashing | `lalrpop` **build-dependency** of `cedar-policy-core` | **no** | yes (build host only) |

`rcgen`/`yasna`/`sha1` are `[dev-dependencies]`-only — absent from both graphs.
`sha3` is a build-dependency: it runs on the build host to generate Cedar's
parser tables and is **not linked into the runtime binary** (`--edges normal`
excludes it; only `--edges no-dev` shows it).

One further build-host case is subtler: `tiny-keccak@2.0.2` (a Keccak-f
implementation) **does** appear under `--edges normal`, but only beneath
`const-random-macro` — a proc-macro. A proc-macro is a build-host code generator,
so `tiny-keccak` executes at compile time (to seed `ahash`) and is **not linked
into the shipped binary** either. `cargo tree -i tiny-keccak` therefore shows it,
but every path to the root passes through a proc-macro node. It is classified
`build_only`, not runtime content. The maintenance check models this precisely:
it computes *runtime linkage* by walking the tree from the root and refusing to
cross proc-macro nodes, so a crate reachable only beneath one is treated as
build-host tooling — and, conversely, a `production`/`allow` entry that turns out
to be reachable only that way is flagged (`runtime-linkage misfile`). A CI guard
against all of these regressions is provided by the maintenance check (below).

## Remediation register

Existing issues (referenced, not duplicated): **[#814][i814]** owns SigV4 (A3,
A4); **[#1217][i1217]** owns PostgreSQL TLS + auth (A5–A11).

Proposed follow-ups. #1220's acceptance criterion is that remediation is split
into **linked** issues. The plan of record: file **R1–R3** as new issues in this
repo; **fold R4 into #1217** rather than duplicating it; treat **R5** as optional
hardening and **defer R6** until its signing surface exists; and file **R7** as a
linked upstream `praxis-proxy/policy` issue grouping that repo's non-validated
crypto questions — the OAuth cache-key HMAC/secret (A2/R-4) and the
session-identity-binding SHA-256 (P1) — so each has a tracked owner (a bare
"upstream" label does not satisfy the linked-remediation criterion). This register
is updated with each issue number as it is filed (the maintainer files them — this
table is not self-linking):

| ID | Title | Covers | Type |
| --- | --- | --- | --- |
| **R1** | Install `aws-lc-rs` as the process-wide rustls `CryptoProvider` default at start-up (before `create_subrequest_client`) | T1–T10, C2; C1 fully on the pingora path + Linux reqwest, but only the handshake signature check on macOS/Windows reqwest (Security.framework/CryptoAPI keep the trust decision — see the C1 note) | new |
| **R2** | Build/verify `aws-lc-rs` in FIPS mode for the compliance profile (enable `fips` feature; pin & assert) | A1 + all rustls TLS once R1 lands | new |
| **R3** | Route the MCP-approval `target_fingerprint` SHA-256 (H1) through the selected validated provider under the compliance profile | H1 | new |
| **R4** | Evaluate moving PostgreSQL store TLS off `native-tls` onto rustls+`aws-lc-rs` so store TLS shares the validated module | A10, A11 | **fold into #1217** (no standalone issue) |
| **R5** | Wrap `azure_ad` `client_secret` (plain `String` at `azure_ad.rs:192/234`) in `SecretString`/`Zeroizing` for parity with `credential_inject`/`web_search` | hardening | new (optional; deferred) |
| **R6** | When `azure_ad` `private_key_jwt` / `gcp_adc` service-account RS256 signing land, require a validated provider and re-inventory | latent signing surfaces | tracking (deferred until the signing surface exists) |
| **R7** | Track the non-validated `praxis-proxy/policy` crypto as a validated-provider question in the owning plugins: the OAuth delegated-token cache-key HMAC (A2) + its `getrandom` per-process secret (R-4), and the session-identity-binding SHA-256 (P1) | A2, R-4, P1 | new (upstream `praxis-proxy/policy`) |

Upstream-owned (praxis-core / third-party) — flag, cannot fix here: A1 (JWT
verify), A12 (Basic Auth digest/compare, non-default feature), T7 (Valkey session
TLS), P2 (Valkey session key name, non-secret), BLAKE2b cache hashing, the
`IdGenerator` predictability of R-2/R-3. The `praxis-proxy/policy` crypto that is
a genuine validated-provider question — the OAuth cache-key HMAC + its `getrandom`
secret (A2/R-4) and the session-identity SHA-256 (P1) — is, instead of a bare
flag, tracked by the linked upstream issue **R7** above.

## Maintenance

The machine-readable companion `cryptographic-inventory.yaml` lists every
crypto/crypto-adjacent crate expected in the profile-of-record graph, with its
`disposition`, and pins the load-bearing providers under `providers:`.
`cargo xtask check-crypto-inventory` (run by `make lint` and CI) resolves the
runtime graph (`cargo tree --edges normal --no-default-features --features
store-postgres`) for the profile of record on **all three tier-1 targets**
(`x86_64-unknown-linux-gnu`, `aarch64-apple-darwin`, `x86_64-pc-windows-msvc`) —
so the result is **host-independent** and CI validates the macOS/Windows
declarations even though it runs on Linux — and fails if:

- a crate is classified in more than one list — `production`/`allow`/`proc_macro`/
  `test_only`/`build_only` — or a pinned provider spills outside `production` — a
  **duplicate classification**, which would let one bucket's guard mask another's;
  or
- a crypto-named crate (matched by the manifest's `watch_tokens`) appears in a
  target's resolved tree but is declared for that platform in none of
  `production`, `allow`, `proc_macro`, or `build_only` (a `platform:`-tagged entry
  covers only its own target) — an **undeclared** crypto path, which cannot be
  silently accepted; or
- a `production` entry is missing from the graph of a target it applies to
  (a `platform:`-tagged entry on its own target, or an untagged entry on any of
  the three) — **stale**: the operation was removed or the manifest drifted; or
- a `production` entry carries no `disposition`, or one outside the documented
  set (`validated`/`non-validated`/`needs-remediation`/`n-a`/`upstream-owned`) —
  an **unclassified** entry, so a missing or misspelled classification cannot
  pass silently; or
- a pinned `providers:` entry drifts — wrong version, a missing
  `require_features`, or a present `forbid_features` (e.g. `aws-lc-rs` gaining
  `fips`, `rustls` losing `ring`, `jsonwebtoken` losing `aws_lc_rs`) — so the
  provider-selection findings cannot rot while the crate name stays put; unlisted
  extra features are permitted by design, since feature sets grow across patch
  releases and an exact-set pin would churn on benign additions; or
- a `proc_macro` entry is not actually a `(proc-macro)` in the resolved graph (a
  runtime crate hiding in the build-host bucket), or a `production`/`allow` entry
  *is* a proc-macro (build-host code generator misfiled as runtime content); or
- a `production`/`allow` entry resolves into a target's tree but only *beneath* a
  proc-macro subtree, so it is not linked into the runtime binary — a
  **runtime-linkage misfile** that overstates the shipped contents and must move
  to `build_only` (e.g. `tiny-keccak` under `const-random-macro`); or
- a crate listed under `test_only` or `build_only` is actually **linked into** any
  target's runtime binary — reachable from the root without crossing a proc-macro
  node, not merely present in `--edges normal` output (**containment regression**,
  e.g. `rcgen`, `sha3`).

The check compares crate names, versions, features, proc-macro status, and
runtime linkage (root-to-crate reachability skipping proc-macro nodes);
callers, algorithms, and operation metadata in the tables above are reviewed by
hand — update the manifest **and** this document together when the check fails.
The manifest is the stable inventory; this prose explains it.

### Known limitations (deliberate scope)

The check is a reviewable tripwire, not a totalizing gate; two boundaries are
stated here so its guarantees are not overread:

- **Undeclared detection is name-based.** A crate is required to be declared only
  when its name matches a `watch_tokens` entry. A genuinely novel crypto crate
  whose name tokenizes to no known token (e.g. `libcrux-ml-kem` → `libcrux`,
  `ml`, `kem`) would not trip this guard; such a crate is instead caught when it
  first lands in `Cargo.lock` (reviewed on every dependency change, alongside
  `cargo deny`/`cargo audit`), at which point `watch_tokens` is widened. The
  token list is kept broad (curves, AEAD modes, PQC families, alternative
  libraries) precisely to shrink this gap.
- **Test/build-only crates are checked for non-leakage, not presence.** The
  containment check fails if a `test_only`/`build_only` crate appears in the
  runtime binary graph (the security-critical direction). It does not resolve the
  dev/build closures to assert those crates are *still present* there, so a fully
  removed entry becomes harmless dead documentation rather than a failure —
  removing it is a manifest-hygiene edit, not a security event.
- **Proc-macros — and their subtrees — are tracked, not counted as runtime
  content.** `cargo tree --edges normal` lists proc-macro crates (e.g.
  `zeroize_derive`, `asn1-rs-derive`) because they are normal-dependency edges,
  but they execute on the build host as code generators and are not linked into
  the shipped binary. They live in the manifest's `proc_macro` list. Crucially,
  the crates a proc-macro *itself* depends on (e.g. `tiny-keccak` under
  `const-random-macro`) are build-host tooling too, even though they carry no
  `(proc-macro)` marker: the check computes runtime linkage by walking from the
  root and refusing to cross proc-macro nodes, so anything reachable only that way
  is `build_only`, not runtime content. The check enforces that a runtime list
  never contains a proc-macro nor a crate reachable only beneath one, and that a
  `proc_macro` entry is genuinely one.

## Appendix: reproduction

```console
# Runtime binary graph (profile of record) — what the check resolves:
cargo tree -p praxis-ai-proxy --edges normal --no-default-features --features store-postgres -i <crate>

# Feature flags actually enabled on a crate (e.g. prove aws-lc-rs has no `fips`):
cargo tree -p praxis-ai-proxy --edges normal --no-default-features --features store-postgres -f "{p} {f}" | grep 'aws-lc-rs v'

# Host-independent, per-target graph (what the check resolves — same for all hosts):
for T in x86_64-unknown-linux-gnu aarch64-apple-darwin x86_64-pc-windows-msvc; do \
  cargo tree -p praxis-ai-proxy --edges normal --no-default-features --features store-postgres --target "$T" -f "{p} {f}"; done

# Build-host tooling (build-deps INCLUDED): shows sha3/keccak that --edges normal hides:
cargo tree -p praxis-ai-proxy --edges no-dev --no-default-features --features store-postgres -i sha3

# Prove a crate is dev/test-only (present in the full graph, absent from --edges no-dev):
cargo tree -p praxis-ai-proxy --no-default-features --features store-postgres -i rcgen

# SQLite profile:
cargo tree -p praxis-ai-apis --edges normal --no-default-features --features store-sqlite -i <crate>

# Optional-feature reachability:
cargo tree -p praxis-ai-proxy --edges normal --no-default-features --features "store-postgres token-rate-limit-filter" -i redis
```

[i1220]: https://github.com/praxis-proxy/ai/issues/1220
[i814]: https://github.com/praxis-proxy/ai/issues/814
[i1217]: https://github.com/praxis-proxy/ai/issues/1217
