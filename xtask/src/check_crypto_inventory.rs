// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `cargo xtask check-crypto-inventory` — verify the cryptographic inventory
//! manifest against the actual runtime dependency graphs of the shipped
//! profiles, and generate the prose companion from it.
//!
//! The manifest (`docs/architecture/cryptographic-inventory.yaml`) is the
//! authoritative inventory; the prose companion
//! (`docs/architecture/cryptographic-inventory.md`) is generated from it by this
//! task (`--fix`). The AI proxy ships two build profiles and both are tracked:
//!
//! * `full` — the published container image (`PRAXIS_AI_FEATURES ?= full`);
//! * `fips` — the reduced FIPS runtime, built `--no-default-features` with the feature set shared from
//!   [`crate::fips::FIPS_FEATURES`].
//!
//! For each profile the check resolves
//! `cargo tree -p praxis-ai-proxy --edges normal --no-default-features
//! --features <profile> --target <triple>` on all three tier-1 targets, so the
//! result is host-independent and CI validates the macOS/Windows declarations
//! even though it runs on Linux. The edge and runtime-linkage graphs are keyed on
//! package **identity** (name + version), so two co-resolved versions of one
//! crate are never merged. The check fails when the manifest and the resolved
//! graphs disagree, so:
//!
//! * a crypto-family-named crate (matched by the manifest's `watch_tokens`, or named exactly on the FIPS denylist
//!   [`crate::fips::graph::DENIED`]) cannot enter a target's runtime graph undeclared — on *every* profile, so a
//!   denylisted cipher (e.g. `chacha20poly1305`, `ctr`, `cbc`) whose concatenated name the token split cannot reduce is
//!   still caught in the published `full` image, not only in `fips`. The guard is a name tripwire, not a totalizing
//!   gate: a genuinely novel crypto crate whose name matches neither is caught instead at Cargo.lock / `cargo deny`
//!   review, where new dependencies land (widen `watch_tokens` when one appears);
//! * a `production` entry cannot go unclassified (missing/unknown `disposition`) or unassigned to a profile;
//! * a `production` entry cannot appear in a profile it does not declare (a profile leak), nor be declared for a
//!   profile/target where it is absent (a stale declaration);
//! * a pinned provider cannot drift its version, gain a forbidden feature, or lose a required one while the crate name
//!   stays put (e.g. `aws-lc-rs` gaining `fips`, `rustls` regaining `ring`, or `openssl` gaining `vendored`); unlisted
//!   features are permitted by design, since feature sets grow across patch releases and an exact-set pin would churn
//!   on benign additions;
//! * a proc-macro crate (build-host code generator, present under `--edges normal` but not linked into the shipped
//!   binary — e.g. `zeroize_derive`) cannot be misfiled as runtime-binary content, and a runtime crate cannot hide in
//!   the build-host `proc_macro` bucket;
//! * a crate reachable only *beneath* a proc-macro subtree (a code generator's own dependency — e.g. `tiny-keccak`
//!   under `const-random-macro`) executes on the build host too, so it cannot be filed as runtime content: runtime
//!   linkage is computed by walking the tree from the root and refusing to cross into proc-macro nodes, not by mere
//!   presence in `--edges normal` output;
//! * test/build-only cryptography cannot silently link into the runtime binary;
//! * a crate on Red Hat's FIPS denylist (shared from [`crate::fips::graph::DENIED`]) cannot enter any `fips` graph, nor
//!   be laundered into the runtime-linked `allow` ("reviewed non-primitive") bucket on any profile — a denied name is a
//!   real primitive by definition, so it must be tracked in `production`, not silenced as an exception;
//! * a crate cannot carry two classifications at once (a duplicate across `production`/`allow`/`proc_macro`/
//!   `test_only`/`build_only` would let one bucket's guard mask another's).

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt::Write as _,
    path::{Path, PathBuf},
    process::Command,
};

use clap::Parser;
use serde::Deserialize;

// -----------------------------------------------------------------------------
// CLI Arguments
// -----------------------------------------------------------------------------

/// CLI arguments for `cargo xtask check-crypto-inventory`.
#[derive(Parser)]
pub(crate) struct Args {
    /// Regenerate the prose companion from the manifest instead of only checking
    /// that it is in sync.
    #[arg(long)]
    fix: bool,
}

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Relative path from the workspace root to the manifest.
const MANIFEST_REL: &str = "docs/architecture/cryptographic-inventory.yaml";

/// Relative path to the generated prose companion.
const DOC_REL: &str = "docs/architecture/cryptographic-inventory.md";

/// Basename of the manifest, used for the same-directory link in the prose.
const MANIFEST_BASENAME: &str = "cryptographic-inventory.yaml";

/// Tier-1 targets resolved so the check is host-independent: the graphs come
/// from the lockfile, not the host, so CI on Linux still validates the
/// macOS/Windows declarations. `(platform label, target triple)`.
const TARGETS: [(&str, &str); 3] = [
    ("linux", "x86_64-unknown-linux-gnu"),
    ("macos", "aarch64-apple-darwin"),
    ("windows", "x86_64-pc-windows-msvc"),
];

/// The shipped build profiles the inventory tracks, in display order.
const PROFILES: [Profile; 2] = [
    Profile {
        label: "full",
        features: "full",
        description: "Published container image (`standard` + `openai-all` + `store-postgres`).",
    },
    Profile {
        label: "fips",
        features: crate::fips::FIPS_FEATURES,
        description: "Reduced FIPS runtime, built `--no-default-features`.",
    },
];

/// Valid profile labels (the `label` of every [`PROFILES`] entry). Kept as a
/// standalone array so error messages can print the expected set; a unit test
/// asserts it stays in step with [`PROFILES`].
const PROFILE_LABELS: [&str; 2] = ["full", "fips"];

/// Dispositions every `production` entry must declare (kept in sync with the
/// generated prose legend, which is rendered from this list).
const VALID_DISPOSITIONS: [&str; 5] = [
    "validated",
    "non-validated",
    "needs-remediation",
    "upstream-owned",
    "n-a",
];

/// A shipped build profile: a label, its `--features` argument (always built
/// `--no-default-features`), and a one-line description rendered into the prose.
struct Profile {
    /// Manifest/label token (`full` / `fips`).
    label: &'static str,
    /// The `--features` argument value.
    features: &'static str,
    /// One-line description for the generated companion document.
    description: &'static str,
}

/// Human-readable meaning of each disposition, rendered into the prose legend.
fn disposition_doc(disposition: &str) -> &'static str {
    match disposition {
        "validated" => "Executes the intended FIPS-validated module with FIPS mode proven in effect.",
        "non-validated" => "A real cryptographic primitive whose validated execution is not proven in this build.",
        "needs-remediation" => "A primitive that should change (e.g. move onto the OpenSSL-backed provider).",
        "upstream-owned" => "A primitive owned and executed entirely within an upstream dependency.",
        "n-a" => "Not a security-function primitive (identifier hashing, key zeroization, or a support layer).",
        _ => "",
    }
}

// -----------------------------------------------------------------------------
// Manifest
// -----------------------------------------------------------------------------

/// The parsed cryptographic inventory manifest.
///
/// `deny_unknown_fields` makes a mistyped key a hard parse error rather than a
/// silently dropped section (e.g. `provders:` losing the entire drift guard).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    /// Name tokens that mark a crate as crypto-relevant.
    watch_tokens: Vec<String>,
    /// Crypto/crypto-support crates expected in the runtime binary.
    production: Vec<Entry>,
    /// Watch-matched crates that are not tracked primitives (reviewed).
    allow: Vec<Entry>,
    /// Crypto that must never appear in the runtime graph (dev-only).
    test_only: Vec<Entry>,
    /// Crypto that runs only on the build host (build-dependencies).
    build_only: Vec<Entry>,
    /// Crypto-adjacent proc-macro crates: present under `--edges normal` but
    /// executing on the build host as code generators, not linked into the
    /// shipped binary. Tracked separately so they are not counted as runtime
    /// contents (e.g. `zeroize_derive`, `asn1-rs-derive`).
    #[serde(default)]
    proc_macro: Vec<Entry>,
    /// Load-bearing providers whose version and features are pinned.
    #[serde(default)]
    providers: Vec<Provider>,
    /// Production-reachable cryptographic operations — the operation-level
    /// companion to `production` (#1220). Empty is permitted (the profile/graph
    /// guards still run); when populated, [`check_operations`] enforces unique
    /// ids, complete fields, profile consistency against the referenced crates,
    /// remediation links for non-compliant dispositions, and coverage of every
    /// real primitive.
    #[serde(default)]
    operations: Vec<Operation>,
    /// Production primitives deliberately exempt from operation coverage, each
    /// with a reason.
    #[serde(default)]
    operation_exempt: Vec<OperationExempt>,
}

/// One crate entry in the manifest.
///
/// `deny_unknown_fields` rejects mistyped keys (e.g. `platfrom:`) instead of
/// silently defaulting them, which would quietly change the entry's scope.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    /// The crate name as it appears in `cargo tree`.
    #[serde(rename = "crate")]
    crate_name: String,
    /// The host OS this entry is present on (`linux`/`macos`/`windows`).
    ///
    /// When set, the checks only apply on that target; an unset platform means
    /// the crate is expected on every target.
    #[serde(default)]
    platform: Option<String>,
    /// The build profiles this crate is present in (`full`/`fips`). Required and
    /// validated for `production` entries; must be empty on the other buckets,
    /// which are profile-agnostic.
    #[serde(default)]
    profiles: Vec<String>,
    /// The classification (`validated`/`non-validated`/…). Required and
    /// validated for `production` entries; advisory elsewhere.
    #[serde(default)]
    disposition: Option<String>,
    /// Human-readable rationale. Validated non-empty for `production` entries so
    /// no tracked crypto path is left undocumented.
    #[serde(default)]
    note: String,
}

/// A provider crate whose version and feature set are pinned so the
/// provider-selection findings cannot rot while the crate name stays put.
///
/// `deny_unknown_fields` is load-bearing here: a mistyped `forbid_feature:`
/// would otherwise be dropped, leaving an empty constraint that silently accepts
/// a forbidden feature (e.g. `fips`) and defeats the drift guard.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Provider {
    /// The provider crate name.
    #[serde(rename = "crate")]
    crate_name: String,
    /// The exact version expected in the lockfile.
    version: String,
    /// The build profiles this provider is pinned in (`full`/`fips`).
    #[serde(default)]
    profiles: Vec<String>,
    /// Features that must all be enabled.
    #[serde(default)]
    require_features: Vec<String>,
    /// Features that must all be absent.
    #[serde(default)]
    forbid_features: Vec<String>,
    /// Human-readable rationale for the pin. Validated non-empty.
    #[serde(default)]
    note: String,
}

/// One production-reachable cryptographic operation: a caller invoking a
/// primitive through a specific implementation provider.
///
/// This is the operation-level companion to the crate-centric [`production`]
/// inventory. The two answer different questions and neither subsumes the other:
/// `production` answers "what cryptographic code can ship" (drift-guarded against
/// the resolved dependency graph); `operations` answers "where is cryptography
/// used, for what algorithm, under whose ownership, and what remediation is
/// required" (#1220's acceptance criterion). An operation's `crates` tie it back
/// to the `production` entries that implement it, so [`check_operations`] can
/// prove the two stay consistent (every referenced crate exists and is resolved
/// in the operation's profiles, and every real primitive is covered by an
/// operation).
///
/// `deny_unknown_fields` makes a mistyped key a hard parse error rather than a
/// silently dropped field.
///
/// [`production`]: Manifest::production
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Operation {
    /// Stable operation identifier (e.g. `A3`, `H1`, `T1`). Must be unique.
    id: String,
    /// Short human-readable operation name.
    name: String,
    /// The component/repository responsible for the operation and its remediation
    /// (e.g. `praxis-ai (this repo)`, `praxis-core tls`, `praxis-proxy/policy
    /// (upstream)`). Distinct from `caller` (where it is invoked) and `provider`
    /// (what executes the primitive): an in-repo caller can invoke an
    /// upstream-owned primitive.
    owner: String,
    /// Who invokes the operation, ideally with a `file:line` anchor (in this repo
    /// or an upstream dependency).
    caller: String,
    /// The algorithm(s) the operation executes.
    algorithm: String,
    /// Human-readable label for the implementation provider (e.g.
    /// `OpenSSL EVP via apis::hash`, `aws-lc-rs via jsonwebtoken`).
    provider: String,
    /// The `production` crate names that implement or support this operation.
    /// Every entry must be a declared `production` crate, and each is validated to
    /// be resolved in every profile the operation declares.
    crates: Vec<String>,
    /// The build profiles the operation executes in (`full`/`fips`).
    profiles: Vec<String>,
    /// The operation's disposition (same vocabulary as `production`:
    /// [`VALID_DISPOSITIONS`]).
    disposition: String,
    /// Remediation reference. Required (non-empty and pointing at an issue,
    /// docs path, or upstream repo — see [`looks_like_reference`]) when the
    /// disposition is non-compliant (`non-validated` or `needs-remediation`).
    #[serde(default)]
    remediation: String,
    /// Human-readable rationale/context. Validated non-empty.
    #[serde(default)]
    note: String,
}

/// A `production` primitive deliberately NOT tied to an [`Operation`], with a
/// stated reason.
///
/// The coverage arm of [`check_operations`] requires every non-`n-a` production
/// primitive to be referenced by an operation; this bucket is the explicit escape
/// hatch for a real primitive that legitimately carries no standalone operation
/// of its own (e.g. a support crate surfaced only transitively). Keeping the
/// exemption explicit — rather than loosening the coverage check — means the gap
/// is reviewed and documented instead of silent.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OperationExempt {
    /// The production crate name being exempted from operation coverage.
    #[serde(rename = "crate")]
    crate_name: String,
    /// Why this primitive carries no standalone operation. Validated non-empty.
    reason: String,
}

// -----------------------------------------------------------------------------
// Resolved graph
// -----------------------------------------------------------------------------

/// A resolved package identity: crate name plus exact version.
///
/// Keying the edge and linkage graphs on `(name, version)` keeps two co-resolved
/// versions of one crate as distinct nodes, so the runtime-linkage walk never
/// merges a build-host-only copy with a runtime-linked one of the same name.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct PkgId {
    /// The crate name.
    name: String,
    /// The resolved version (without the leading `v`).
    version: String,
}

/// Version and enabled features of one crate in a target's runtime graph.
#[derive(Debug)]
struct CrateFacts {
    /// The resolved crate version (without the leading `v`).
    version: String,
    /// The set of features cargo reports enabled for the crate.
    features: BTreeSet<String>,
    /// Whether cargo marks this crate `(proc-macro)` — a build-host code
    /// generator that is not linked into the runtime binary.
    is_proc_macro: bool,
}

/// One resolved `(profile, target)` runtime graph: crate name -> every resolved
/// version.
///
/// The `facts` value is a `Vec` because Cargo can co-resolve several versions of
/// the same crate; keeping them all is what lets the provider-drift check see a
/// second, upgraded copy of a pinned provider instead of silently accepting the
/// first occurrence.
struct TargetGraph {
    /// Profile label this graph was resolved for (`full`/`fips`).
    profile: &'static str,
    /// Platform label (`linux`/`macos`/`windows`).
    platform: String,
    /// Resolved crate facts keyed by crate name (one entry per resolved version).
    facts: BTreeMap<String, Vec<CrateFacts>>,
    /// Package identities actually linked into the runtime binary: reachable from
    /// the root crate over `--edges normal` edges *without crossing a proc-macro
    /// node*. Proc-macros and any crate reachable only beneath one (a code
    /// generator's own dependency, e.g. `tiny-keccak` under `const-random-macro`)
    /// run on the build host and are excluded even though `cargo tree` lists them.
    runtime_linked: BTreeSet<PkgId>,
}

impl TargetGraph {
    /// Whether the crate appears anywhere in this graph's resolved tree
    /// (including build-host-only subtrees). Presence, not runtime linkage — use
    /// [`Self::is_runtime_linked`] to ask what actually ships in the binary.
    fn contains(&self, name: &str) -> bool {
        self.facts.contains_key(name)
    }

    /// Whether any resolved copy of the crate is a build-host proc-macro.
    fn is_proc_macro(&self, name: &str) -> bool {
        self.facts.get(name).is_some_and(|v| v.iter().any(|f| f.is_proc_macro))
    }

    /// Whether any resolved copy of the crate is linked into the runtime binary —
    /// reachable from the root over normal edges without passing through a
    /// proc-macro node. The underlying set is instance-precise (keyed on
    /// [`PkgId`]); this name-level query answers "does this crate ship".
    fn is_runtime_linked(&self, name: &str) -> bool {
        self.runtime_linked.iter().any(|id| id.name == name)
    }
}

// -----------------------------------------------------------------------------
// Entry Point
// -----------------------------------------------------------------------------

/// Verify the manifest against the resolved runtime graphs, then check (or, with
/// `--fix`, regenerate) the prose companion.
#[expect(
    clippy::too_many_lines,
    reason = "linear read → resolve → evaluate → generate/compare flow"
)]
pub(crate) fn run(args: &Args) {
    let root = workspace_root();
    let manifest_path = root.join(MANIFEST_REL);
    let doc_path = root.join(DOC_REL);

    let raw = std::fs::read_to_string(&manifest_path).unwrap_or_else(|err| {
        eprintln!("failed to read {}: {err}", manifest_path.display());
        std::process::exit(1);
    });
    let manifest: Manifest = serde_yaml::from_str(&raw).unwrap_or_else(|err| {
        eprintln!("failed to parse {MANIFEST_REL}: {err}");
        std::process::exit(1);
    });

    let graphs = resolve_graphs(&root);

    let violations = evaluate(&manifest, &graphs);
    if !violations.is_empty() {
        eprintln!("crypto inventory check failed ({} violation(s)):", violations.len());
        for violation in &violations {
            eprintln!("  - {violation}");
        }
        eprintln!("\nReview {DOC_REL} and update {MANIFEST_REL} to match the change.");
        std::process::exit(1);
    }

    // The manifest is authoritative and internally consistent; the prose is
    // generated from it, never edited by hand.
    let content = render_doc(&manifest);
    if args.fix {
        std::fs::write(&doc_path, &content).unwrap_or_else(|err| {
            eprintln!("failed to write {}: {err}", doc_path.display());
            std::process::exit(1);
        });
        println!("wrote {DOC_REL}");
    } else {
        let current = std::fs::read_to_string(&doc_path).unwrap_or_default();
        if current == content {
            print_summary(&manifest, &graphs);
        } else {
            eprintln!("{DOC_REL} is stale");
            eprintln!("\nrun: cargo xtask check-crypto-inventory --fix");
            std::process::exit(1);
        }
    }
}

/// Resolve every `(profile, target)` combination into a [`TargetGraph`].
fn resolve_graphs(root: &Path) -> Vec<TargetGraph> {
    let mut graphs = Vec::with_capacity(PROFILES.len() * TARGETS.len());
    for profile in &PROFILES {
        for (platform, triple) in TARGETS {
            graphs.push(resolve_target(root, profile, platform, triple));
        }
    }
    graphs
}

/// Print the success summary: manifest counts plus the resolved profile/target
/// set.
fn print_summary(manifest: &Manifest, graphs: &[TargetGraph]) {
    let union: BTreeSet<&str> = graphs.iter().flat_map(|g| g.facts.keys().map(String::as_str)).collect();
    let profiles: Vec<&str> = PROFILES.iter().map(|p| p.label).collect();
    let targets: Vec<&str> = TARGETS.iter().map(|(p, _)| *p).collect();
    println!(
        "crypto inventory in sync ({} production, {} allow, {} providers, {} proc-macro, {} test-only, {} build-only; profiles: {}; targets: {}; {} graphs; {} crates across graphs)",
        manifest.production.len(),
        manifest.allow.len(),
        manifest.providers.len(),
        manifest.proc_macro.len(),
        manifest.test_only.len(),
        manifest.build_only.len(),
        profiles.join("+"),
        targets.join("/"),
        graphs.len(),
        union.len(),
    );
}

// -----------------------------------------------------------------------------
// Evaluation
// -----------------------------------------------------------------------------

/// Compare the manifest against the resolved per-`(profile, target)` `graphs`,
/// returning a human-readable violation for each mismatch. Pure: all IO happens
/// in [`run`].
fn evaluate(manifest: &Manifest, graphs: &[TargetGraph]) -> Vec<String> {
    let mut out = Vec::new();
    check_duplicates(manifest, &mut out);
    check_platforms(manifest, &mut out);
    check_profiles(manifest, &mut out);
    check_dispositions(&manifest.production, &mut out);
    check_notes(manifest, &mut out);
    check_undeclared(manifest, graphs, &mut out);
    check_stale_production(&manifest.production, graphs, &mut out);
    check_profile_leak(&manifest.production, graphs, &mut out);
    check_stale_allow(&manifest.allow, graphs, &mut out);
    check_proc_macro(manifest, graphs, &mut out);
    check_runtime_linkage(manifest, graphs, &mut out);
    check_containment(manifest, graphs, &mut out);
    check_providers(&manifest.providers, graphs, &mut out);
    check_denied_absent(graphs, &mut out);
    check_denied_not_allowed(manifest, &mut out);
    check_operations(manifest, &mut out);
    out
}

/// The `/`-joined list of known platform labels, for error messages.
fn platform_labels() -> String {
    TARGETS.iter().map(|(p, _)| *p).collect::<Vec<_>>().join("/")
}

/// Whether a `platform:`-tagged entry applies to `target`: an untagged entry
/// applies to every target, a tagged one only to its own.
fn platform_matches(platform: &Option<String>, target: &str) -> bool {
    platform.as_deref().is_none_or(|p| p == target)
}

/// Whether a `production` entry is declared for `profile`.
fn entry_in_profile(entry: &Entry, profile: &str) -> bool {
    entry.profiles.iter().any(|p| p == profile)
}

/// Uniqueness check: every crate must carry exactly one classification. A crate
/// declared in two lists lets one bucket's guard mask another's (e.g. a crate in
/// both `build_only` and `production` is exempt from the containment check while
/// still counted as runtime content). Providers are excluded from the mutual-
/// exclusion union because they intentionally overlay `production`; they are
/// instead checked for self-duplication and for spilling into any non-`production`
/// list.
fn check_duplicates(manifest: &Manifest, out: &mut Vec<String>) {
    let mut buckets: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (label, entries) in labelled_buckets(manifest) {
        for entry in entries {
            buckets.entry(entry.crate_name.as_str()).or_default().push(label);
        }
    }
    for (name, labels) in &buckets {
        if labels.len() > 1 {
            out.push(format!(
                "duplicate manifest entry: `{name}` is classified in {} lists ({}) but must have exactly one classification",
                labels.len(),
                labels.join(", ")
            ));
        }
    }
    check_provider_overlap(&manifest.providers, &buckets, out);
}

/// The five mutually-exclusive classification buckets, in a fixed order.
fn labelled_buckets(manifest: &Manifest) -> [(&'static str, &[Entry]); 5] {
    [
        ("production", &manifest.production),
        ("allow", &manifest.allow),
        ("proc_macro", &manifest.proc_macro),
        ("test_only", &manifest.test_only),
        ("build_only", &manifest.build_only),
    ]
}

/// Duplicate sub-check for the pinned providers: a provider may be listed only
/// once and may overlap only the `production` classification (never `allow`,
/// `proc_macro`, `test_only`, or `build_only`).
fn check_provider_overlap(providers: &[Provider], buckets: &BTreeMap<&str, Vec<&str>>, out: &mut Vec<String>) {
    let mut seen_providers: BTreeSet<&str> = BTreeSet::new();
    for provider in providers {
        let name = provider.crate_name.as_str();
        if !seen_providers.insert(name) {
            out.push(format!("duplicate pinned provider: `{name}` is listed more than once"));
        }
        if let Some(labels) = buckets
            .get(name)
            .filter(|labels| labels.iter().any(|l| *l != "production"))
        {
            out.push(format!(
                "pinned provider `{name}` also appears in a non-`production` list ({}) — a provider may only overlap `production`",
                labels.join(", ")
            ));
        }
    }
}

/// Platform check: any `platform:`-tagged entry (in any bucket) must name a known
/// target, so a typo like `platfrom: linux` (or an unsupported OS) is caught
/// rather than silently applying to nothing.
fn check_platforms(manifest: &Manifest, out: &mut Vec<String>) {
    let valid: BTreeSet<&str> = TARGETS.iter().map(|(p, _)| *p).collect();
    let known = platform_labels();
    for (label, entries) in labelled_buckets(manifest) {
        for entry in entries {
            if let Some(platform) = entry.platform.as_deref()
                && !valid.contains(platform)
            {
                out.push(format!(
                    "`{label}` entry `{}` has unknown platform `{platform}` (expected one of {known})",
                    entry.crate_name
                ));
            }
        }
    }
}

/// Profile check: every `production` entry and pinned provider must declare a
/// non-empty subset of the known profiles, and the profile-agnostic buckets
/// (`allow`/`test_only`/`build_only`/`proc_macro`) must not declare any.
#[expect(clippy::too_many_lines, reason = "one linear pass over each bucket's profile rules")]
fn check_profiles(manifest: &Manifest, out: &mut Vec<String>) {
    let valid: BTreeSet<&str> = PROFILES.iter().map(|p| p.label).collect();
    for entry in &manifest.production {
        if entry.profiles.is_empty() {
            out.push(format!(
                "`production` entry `{}` declares no `profiles` (expected a non-empty subset of {PROFILE_LABELS:?})",
                entry.crate_name
            ));
        }
        for profile in &entry.profiles {
            if !valid.contains(profile.as_str()) {
                out.push(format!(
                    "`production` entry `{}` has unknown profile `{profile}` (expected one of {PROFILE_LABELS:?})",
                    entry.crate_name
                ));
            }
        }
    }
    for provider in &manifest.providers {
        if provider.profiles.is_empty() {
            out.push(format!(
                "pinned provider `{}` declares no `profiles` (expected a non-empty subset of {PROFILE_LABELS:?})",
                provider.crate_name
            ));
        }
        for profile in &provider.profiles {
            if !valid.contains(profile.as_str()) {
                out.push(format!(
                    "pinned provider `{}` has unknown profile `{profile}` (expected one of {PROFILE_LABELS:?})",
                    provider.crate_name
                ));
            }
        }
    }
    let non_production: [(&str, &[Entry]); 4] = [
        ("allow", &manifest.allow),
        ("test_only", &manifest.test_only),
        ("build_only", &manifest.build_only),
        ("proc_macro", &manifest.proc_macro),
    ];
    for (label, entries) in non_production {
        for entry in entries {
            if !entry.profiles.is_empty() {
                out.push(format!(
                    "`{label}` entry `{}` declares `profiles`, but only `production` entries are profile-scoped",
                    entry.crate_name
                ));
            }
        }
    }
}

/// Classification check: every `production` entry must carry a known
/// `disposition`, so a missing or misspelled classification cannot pass.
fn check_dispositions(production: &[Entry], out: &mut Vec<String>) {
    let valid: BTreeSet<&str> = VALID_DISPOSITIONS.iter().copied().collect();
    for entry in production {
        match entry.disposition.as_deref() {
            None => out.push(format!(
                "`production` entry `{}` has no `disposition` (expected one of {VALID_DISPOSITIONS:?})",
                entry.crate_name
            )),
            Some(d) if !valid.contains(d) => out.push(format!(
                "`production` entry `{}` has unknown disposition `{d}` (expected one of {VALID_DISPOSITIONS:?})",
                entry.crate_name
            )),
            Some(_) => {},
        }
    }
}

/// Documentation check: every load-bearing entry (production primitives, tracked
/// proc-macros, and pinned providers) must carry a non-empty `note`, so no
/// tracked crypto path is left undocumented. Reading `note` here also anchors the
/// field that `deny_unknown_fields` requires be declared to accept the manifest's
/// `note:`.
fn check_notes(manifest: &Manifest, out: &mut Vec<String>) {
    for entry in manifest.production.iter().chain(&manifest.proc_macro) {
        if entry.note.trim().is_empty() {
            out.push(format!(
                "`{}` has an empty `note` (document its rationale)",
                entry.crate_name
            ));
        }
    }
    for provider in &manifest.providers {
        if provider.note.trim().is_empty() {
            out.push(format!("pinned provider `{}` has an empty `note`", provider.crate_name));
        }
    }
}

/// Undeclared-crypto check: a crypto-relevant crate present in a graph that is not
/// declared applicable to THAT platform in either `production`, `allow`,
/// `proc_macro`, or `build_only`. A crate is crypto-relevant when it matches a
/// `watch_tokens` entry OR is named exactly on the FIPS denylist
/// [`crate::fips::graph::DENIED`]. The denylist arm closes a hole in the token
/// tripwire: the split-on-`-`/`_`-then-strip-trailing-digits tokenizer cannot
/// reduce a separator-less concatenated cipher name (e.g. `chacha20poly1305`) to
/// its `chacha`/`poly1305` tokens, and mode crates like `ctr`/`cbc` have no token
/// at all, so without an exact-name check such a crate could link into the
/// published `full` image undetected ([`check_denied_absent`] only guards `fips`).
///
/// Each graph is checked against the entries that apply to it (by name and
/// platform), so a `platform:`-tagged entry (e.g. Linux-only `openssl`) does not
/// silently suppress the same crate appearing unexpectedly on another target.
/// Profile scoping is enforced separately by [`check_profile_leak`]; declaration
/// here is by name so a known crypto crate (e.g. `sha2`/`hmac`/`md-5`, legitimate
/// in `full`) is never reported as undeclared merely because it also appears in a
/// second profile.
fn check_undeclared(manifest: &Manifest, graphs: &[TargetGraph], out: &mut Vec<String>) {
    let watch: BTreeSet<String> = manifest.watch_tokens.iter().map(|t| t.to_ascii_lowercase()).collect();
    let denied: BTreeSet<&str> = crate::fips::graph::DENIED.iter().copied().collect();
    for g in graphs {
        for name in g.facts.keys() {
            let crypto_relevant = is_watch_match(name, &watch) || denied.contains(name.as_str());
            if crypto_relevant
                && !declared_for(&manifest.production, name, &g.platform)
                && !declared_for(&manifest.allow, name, &g.platform)
                && !declared_for(&manifest.proc_macro, name, &g.platform)
                && !declared_for(&manifest.build_only, name, &g.platform)
            {
                out.push(format!(
                    "undeclared crypto crate in the {}/{} resolved tree: `{name}` is crypto-relevant (matches a watch token or the FIPS denylist) but is declared in none of `production`, `allow`, `proc_macro`, or `build_only` for that platform",
                    g.profile, g.platform
                ));
            }
        }
    }
}

/// Whether `entries` declares `name` as applicable to `platform`: an untagged
/// entry applies to every target, a `platform:`-tagged one only to its target.
fn declared_for(entries: &[Entry], name: &str, platform: &str) -> bool {
    entries
        .iter()
        .any(|e| e.crate_name == name && platform_matches(&e.platform, platform))
}

/// Stale-declaration check: a `production` entry that is declared for a
/// `(profile, target)` it applies to but is absent from that resolved graph.
fn check_stale_production(entries: &[Entry], graphs: &[TargetGraph], out: &mut Vec<String>) {
    for entry in entries {
        for g in graphs {
            if entry_in_profile(entry, g.profile)
                && platform_matches(&entry.platform, &g.platform)
                && !g.contains(&entry.crate_name)
            {
                out.push(format!(
                    "stale `production` entry: `{}` is declared for {}/{} but absent from that runtime graph",
                    entry.crate_name, g.profile, g.platform
                ));
            }
        }
    }
}

/// Profile-leak check: a `production` entry present in a graph whose profile it
/// does not declare. The dual of [`check_stale_production`]: stale = declared but
/// absent; leak = present but not declared. Both use presence (`contains`), so a
/// missing or an unexpected profile assignment is caught symmetrically.
fn check_profile_leak(entries: &[Entry], graphs: &[TargetGraph], out: &mut Vec<String>) {
    for entry in entries {
        for g in graphs {
            if platform_matches(&entry.platform, &g.platform)
                && g.contains(&entry.crate_name)
                && !entry_in_profile(entry, g.profile)
            {
                out.push(format!(
                    "profile leak: `{}` is present in the {}/{} runtime graph but the manifest declares it only for [{}]",
                    entry.crate_name,
                    g.profile,
                    g.platform,
                    entry.profiles.join(", ")
                ));
            }
        }
    }
}

/// Stale-exception check: an `allow` entry that no longer resolves on any
/// applicable graph (it stops suppressing a real crate and becomes dead
/// documentation). `allow` is profile-agnostic, so presence on any profile's
/// applicable target keeps it live.
fn check_stale_allow(entries: &[Entry], graphs: &[TargetGraph], out: &mut Vec<String>) {
    for entry in entries {
        let present = graphs
            .iter()
            .filter(|g| platform_matches(&entry.platform, &g.platform))
            .any(|g| g.contains(&entry.crate_name));
        if !present {
            match entry.platform.as_deref() {
                Some(p) => out.push(format!(
                    "stale `allow` entry: `{}` (platform: {p}) is declared but absent from every {p} runtime graph",
                    entry.crate_name
                )),
                None => out.push(format!(
                    "stale `allow` entry: `{}` is declared but absent from every runtime graph",
                    entry.crate_name
                )),
            }
        }
    }
}

/// Proc-macro classification check. Proc-macros are build-host code generators:
/// `cargo tree --edges normal` lists them, but they are not linked into the
/// shipped binary, so they must not be counted as runtime-binary content. Both
/// directions are enforced:
///
/// * every `proc_macro` entry must be present AND actually marked `(proc-macro)` in some graph — otherwise a normal
///   (linked) crate is hiding in the build-host bucket, dodging runtime scrutiny;
/// * no `production` or `allow` entry may be a proc-macro — that would overstate the runtime-binary contents (the
///   inaccuracy this check exists to prevent).
fn check_proc_macro(manifest: &Manifest, graphs: &[TargetGraph], out: &mut Vec<String>) {
    for entry in &manifest.proc_macro {
        if !graphs.iter().any(|g| g.contains(&entry.crate_name)) {
            out.push(format!(
                "stale `proc_macro` entry: `{}` is absent from every runtime graph",
                entry.crate_name
            ));
        } else if !graphs.iter().any(|g| g.is_proc_macro(&entry.crate_name)) {
            out.push(format!(
                "misclassified `proc_macro` entry: `{}` is a normal linked crate, not a proc-macro — move it to `production`/`allow`",
                entry.crate_name
            ));
        }
    }
    for entry in manifest.production.iter().chain(&manifest.allow) {
        if graphs.iter().any(|g| g.is_proc_macro(&entry.crate_name)) {
            out.push(format!(
                "`{}` is a build-host proc-macro (not linked into the runtime binary) but is declared in a runtime list — move it to `proc_macro`",
                entry.crate_name
            ));
        }
    }
}

/// Runtime-linkage check: a `production`/`allow` entry that resolves into a
/// graph but is NOT linked into the runtime binary there — reachable only beneath
/// a proc-macro subtree (a build-host code generator's own dependency, e.g.
/// `tiny-keccak` under `const-random-macro`) — overstates the runtime contents
/// and must move to `build_only`. Proc-macro entries are handled by
/// [`check_proc_macro`] and skipped here, so this fires only on ordinary crates
/// buried under a code generator.
fn check_runtime_linkage(manifest: &Manifest, graphs: &[TargetGraph], out: &mut Vec<String>) {
    for entry in manifest.production.iter().chain(&manifest.allow) {
        for g in graphs {
            if g.contains(&entry.crate_name)
                && !g.is_runtime_linked(&entry.crate_name)
                && !g.is_proc_macro(&entry.crate_name)
            {
                out.push(format!(
                    "`{}` resolves in the {}/{} tree but is reachable only via a proc-macro subtree (build-host only, not linked into the runtime binary) — move it to `build_only`",
                    entry.crate_name, g.profile, g.platform
                ));
            }
        }
    }
}

/// Containment check: dev-only or build-only crypto actually linked into any
/// runtime binary. Keyed on runtime linkage, not mere presence, so a `build_only`
/// crate that only appears beneath a proc-macro subtree (its legitimate
/// build-host home) is allowed while a genuine leak into the shipped binary is
/// still caught.
fn check_containment(manifest: &Manifest, graphs: &[TargetGraph], out: &mut Vec<String>) {
    for entry in manifest.test_only.iter().chain(&manifest.build_only) {
        for g in graphs {
            if g.is_runtime_linked(&entry.crate_name) {
                out.push(format!(
                    "containment regression: `{}` is linked into the {}/{} runtime binary",
                    entry.crate_name, g.profile, g.platform
                ));
            }
        }
    }
}

/// Provider-drift check across all pinned providers.
fn check_providers(providers: &[Provider], graphs: &[TargetGraph], out: &mut Vec<String>) {
    for provider in providers {
        check_provider(provider, graphs, out);
    }
}

/// Verify one pinned provider's version and feature set on every profile it is
/// declared in, and that it appears at all in each — the load-bearing
/// provider-selection facts cannot rot while the crate name stays put.
fn check_provider(provider: &Provider, graphs: &[TargetGraph], out: &mut Vec<String>) {
    for profile in &provider.profiles {
        let profile_graphs: Vec<&TargetGraph> = graphs.iter().filter(|g| g.profile == profile.as_str()).collect();
        if profile_graphs.is_empty() {
            continue;
        }
        let mut seen = false;
        for g in profile_graphs {
            let Some(instances) = g.facts.get(&provider.crate_name) else {
                continue;
            };
            seen = true;
            // Every resolved copy must match the pin. An off-version copy is drift
            // on its own; features are only meaningful on the pinned-version copy.
            for facts in instances {
                if facts.version == provider.version {
                    check_provider_features(provider, facts, g.profile, &g.platform, out);
                } else {
                    out.push(format!(
                        "provider drift: `{}` is `{}` in the {}/{} graph but the manifest pins `{}`",
                        provider.crate_name, facts.version, g.profile, g.platform, provider.version
                    ));
                }
            }
        }
        if !seen {
            out.push(format!(
                "pinned provider `{}` is absent from the {profile} profile runtime graph",
                provider.crate_name
            ));
        }
    }
}

/// Verify a provider's required features are present and forbidden ones absent
/// in one graph's resolved feature set. Unlisted (extra) features are permitted
/// by design: feature sets legitimately grow across patch releases, so exact-set
/// pinning would churn on benign additions. Only the security-relevant invariants
/// are expressed — as `require_features` (must stay) and `forbid_features` (must
/// not appear, e.g. `fips`). A provider with no constraints (e.g. `ring`) pins
/// only its version, because none of its features affect provider selection.
fn check_provider_features(
    provider: &Provider,
    facts: &CrateFacts,
    profile: &str,
    platform: &str,
    out: &mut Vec<String>,
) {
    for want in &provider.require_features {
        if !facts.features.contains(want) {
            out.push(format!(
                "provider drift: `{}` is missing required feature `{want}` in the {profile}/{platform} graph",
                provider.crate_name
            ));
        }
    }
    for deny in &provider.forbid_features {
        if facts.features.contains(deny) {
            out.push(format!(
                "provider drift: `{}` has forbidden feature `{deny}` enabled in the {profile}/{platform} graph",
                provider.crate_name
            ));
        }
    }
}

/// FIPS denylist check: no crate on Red Hat's denylist (shared with the FIPS
/// report through [`crate::fips::graph::DENIED`]) may enter a `fips` runtime
/// graph. This is a defense-in-depth backstop over the profile assignments: even
/// if a `production` entry were mis-tagged with the `fips` profile, a genuinely
/// denied crate in the FIPS build is caught here.
fn check_denied_absent(graphs: &[TargetGraph], out: &mut Vec<String>) {
    for g in graphs {
        if g.profile != "fips" {
            continue;
        }
        for &denied in crate::fips::graph::DENIED {
            if g.contains(denied) {
                out.push(format!(
                    "FIPS denylist crate `{denied}` is present in the fips/{} runtime graph (must never enter the FIPS build)",
                    g.platform
                ));
            }
        }
    }
}

/// Denylist-classification check: a crate on Red Hat's FIPS denylist
/// ([`crate::fips::graph::DENIED`]) is by definition a real cryptographic
/// primitive, so it can never be a "reviewed non-primitive". It must live in
/// `production` (carrying a disposition, note, and profile assignment), never in
/// `allow`.
///
/// Without this guard the `allow` bucket is an escape hatch for the *`full`*
/// profile: [`check_undeclared`]'s denylist arm is silenced the moment a crate
/// is declared in any bucket (including `allow`), and [`check_denied_absent`]
/// only inspects the `fips` graph — so a denied primitive that ships in `full`
/// (e.g. `sha2`/`hmac`/`md-5`, or a newly-pulled `aes-gcm`) could be laundered
/// into `allow` and pass unnoticed. This is a manifest-level guard (no graph
/// needed): a denied name in `allow` is wrong regardless of which profile
/// resolves it. `test_only`/`build_only` are not covered here because they
/// legitimately hold denied dev/build-host crypto (`sha1`/`sha3`) and
/// [`check_containment`] already proves they never link into the runtime binary;
/// `proc_macro` is likewise backstopped by [`check_proc_macro`].
fn check_denied_not_allowed(manifest: &Manifest, out: &mut Vec<String>) {
    let denied: BTreeSet<&str> = crate::fips::graph::DENIED.iter().copied().collect();
    for entry in &manifest.allow {
        if denied.contains(entry.crate_name.as_str()) {
            out.push(format!(
                "FIPS denylist crate `{}` is declared in `allow` — a denylisted primitive is never a reviewed non-primitive; move it to `production` with a disposition",
                entry.crate_name
            ));
        }
    }
}

/// The dispositions that are non-compliant in a FIPS inventory: a real primitive
/// whose validated execution is not proven (`non-validated`), or one that should
/// change (`needs-remediation`). Operations carrying either MUST cite a
/// remediation reference. `validated`/`n-a`/`upstream-owned` are not required to
/// (compliant, non-primitive, or owned and tracked elsewhere).
const NON_COMPLIANT_DISPOSITIONS: [&str; 2] = ["non-validated", "needs-remediation"];

/// Whether `remediation` cites something actionable: an issue (`#<digit>`), a URL
/// (`http`), a docs path (`.md`), or an upstream repo (`praxis-proxy/`). Used to
/// reject a bare non-empty string (e.g. `"TODO"`) as a remediation "link".
fn looks_like_reference(remediation: &str) -> bool {
    remediation.contains("http")
        || remediation.contains(".md")
        || remediation.contains("praxis-proxy/")
        || remediation
            .as_bytes()
            .windows(2)
            .any(|w| matches!(w, [b'#', d] if d.is_ascii_digit()))
}

/// Operations check: validate the operation-level inventory and its consistency
/// with the crate-centric `production` list (#1220).
///
/// Enforces, for the declared `operations`:
/// 1. **unique ids and complete fields** — every operation has a unique `id` and non-empty
///    `name`/`caller`/`algorithm`/`provider`/`note`, a non-empty `crates` list, a non-empty `profiles` list over
///    `{full, fips}`, and a `disposition` in [`VALID_DISPOSITIONS`];
/// 2. **profile consistency** — every crate an operation references is a declared `production` entry, and is resolved
///    in every profile the operation declares (an operation cannot claim to run in a profile one of its implementing
///    crates is absent from);
/// 3. **remediation links** — an operation whose disposition is non-compliant ([`NON_COMPLIANT_DISPOSITIONS`]) cites an
///    actionable remediation reference ([`looks_like_reference`]);
/// 4. **coverage** — every non-`n-a` `production` primitive is referenced by at least one operation's `crates` list or
///    is listed in `operation_exempt` with a reason (so a real primitive can never enter the runtime with no documented
///    operation). Coverage is enforced only when at least one operation is declared: an empty `operations` set has
///    nothing to be complete against, so the manifest-shape guards above still apply while a fixture or opted-out
///    manifest is not forced to enumerate operations.
///
/// `operation_exempt` entries are validated too: each must name a real
/// `production` crate and carry a non-empty reason.
#[expect(
    clippy::too_many_lines,
    reason = "one linear validation pass per operation invariant"
)]
fn check_operations(manifest: &Manifest, out: &mut Vec<String>) {
    let valid_disp: BTreeSet<&str> = VALID_DISPOSITIONS.iter().copied().collect();
    let valid_profiles: BTreeSet<&str> = PROFILE_LABELS.iter().copied().collect();

    // production crate name -> the profiles it is resolved in (unioned across any
    // platform-split entries of the same name).
    let mut prod_profiles: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for entry in &manifest.production {
        let set = prod_profiles.entry(entry.crate_name.as_str()).or_default();
        for p in &entry.profiles {
            set.insert(p.as_str());
        }
    }

    // Per-operation field, profile, and remediation validation.
    let mut seen_ids: BTreeSet<&str> = BTreeSet::new();
    for op in &manifest.operations {
        let id = op.id.trim();
        let label = if id.is_empty() { "<empty id>" } else { id };
        if id.is_empty() {
            out.push("operation has an empty `id`".to_owned());
        } else if !seen_ids.insert(id) {
            out.push(format!("operation id `{id}` is declared more than once"));
        }
        if op.name.trim().is_empty() {
            out.push(format!("operation `{label}` has an empty `name`"));
        }
        if op.owner.trim().is_empty() {
            out.push(format!("operation `{label}` has an empty `owner`"));
        }
        if op.caller.trim().is_empty() {
            out.push(format!("operation `{label}` has an empty `caller`"));
        }
        if op.algorithm.trim().is_empty() {
            out.push(format!("operation `{label}` has an empty `algorithm`"));
        }
        if op.provider.trim().is_empty() {
            out.push(format!("operation `{label}` has an empty `provider`"));
        }
        if op.note.trim().is_empty() {
            out.push(format!("operation `{label}` has an empty `note`"));
        }
        if !valid_disp.contains(op.disposition.as_str()) {
            out.push(format!(
                "operation `{label}` has missing/invalid disposition `{}` (expected one of {VALID_DISPOSITIONS:?})",
                op.disposition
            ));
        }
        if op.profiles.is_empty() {
            out.push(format!("operation `{label}` declares no `profiles`"));
        }
        for p in &op.profiles {
            if !valid_profiles.contains(p.as_str()) {
                out.push(format!(
                    "operation `{label}` declares invalid profile `{p}` (expected one of {PROFILE_LABELS:?})"
                ));
            }
        }
        // A crate-backed operation must name its crates. An `n-a` operation may
        // reference none: it documents non-cryptographic hashing implemented on
        // the standard library (reviewed and dispositioned out of scope), which
        // has no entry in the crate-centric manifest.
        if op.crates.is_empty() && op.disposition != "n-a" {
            out.push(format!("operation `{label}` references no `crates`"));
        }
        for crate_name in &op.crates {
            match prod_profiles.get(crate_name.as_str()) {
                None => out.push(format!(
                    "operation `{label}` references crate `{crate_name}` which is not a `production` entry"
                )),
                Some(crate_profiles) => {
                    for p in &op.profiles {
                        if valid_profiles.contains(p.as_str()) && !crate_profiles.contains(p.as_str()) {
                            out.push(format!(
                                "operation `{label}` declares profile `{p}` but its crate `{crate_name}` is not resolved in that profile"
                            ));
                        }
                    }
                },
            }
        }
        if NON_COMPLIANT_DISPOSITIONS.contains(&op.disposition.as_str()) && !looks_like_reference(&op.remediation) {
            out.push(format!(
                "operation `{label}` has disposition `{}` but no remediation reference (cite an issue #NNN, a docs path, or an upstream repo)",
                op.disposition
            ));
        }
    }

    check_operation_coverage(manifest, &prod_profiles, out);
}

/// `operation_exempt` validation plus coverage (extracted from
/// [`check_operations`] to keep each pass small): each exemption must name a real
/// `production` crate and carry a non-empty reason, and every non-`n-a`
/// `production` primitive must be referenced by an operation or explicitly
/// exempted. Coverage is skipped when no operations are declared.
#[expect(clippy::too_many_lines, reason = "one linear pass over exemptions then coverage")]
fn check_operation_coverage(
    manifest: &Manifest,
    prod_profiles: &BTreeMap<&str, BTreeSet<&str>>,
    out: &mut Vec<String>,
) {
    // `operation_exempt` validation: real crate + non-empty reason.
    let mut exempt: BTreeSet<&str> = BTreeSet::new();
    for ex in &manifest.operation_exempt {
        if !prod_profiles.contains_key(ex.crate_name.as_str()) {
            out.push(format!(
                "operation_exempt crate `{}` is not a `production` entry",
                ex.crate_name
            ));
        }
        if ex.reason.trim().is_empty() {
            out.push(format!(
                "operation_exempt crate `{}` has an empty `reason`",
                ex.crate_name
            ));
        }
        exempt.insert(ex.crate_name.as_str());
    }

    // Coverage: every non-`n-a` production primitive must be referenced by an
    // operation or explicitly exempted. Skipped when no operations are declared.
    if manifest.operations.is_empty() {
        return;
    }
    let referenced: BTreeSet<&str> = manifest
        .operations
        .iter()
        .flat_map(|op| op.crates.iter().map(String::as_str))
        .collect();
    for entry in &manifest.production {
        if entry.disposition.as_deref() == Some("n-a") {
            continue;
        }
        let name = entry.crate_name.as_str();
        if !referenced.contains(name) && !exempt.contains(name) {
            out.push(format!(
                "production primitive `{name}` (disposition `{}`) is referenced by no operation and is not in `operation_exempt`",
                entry.disposition.as_deref().unwrap_or("")
            ));
        }
    }
}

/// Whether `name` matches any watch token, using the same tokenization the
/// manifest documents: split on `-`/`_`, then also compare each part with its
/// trailing digits stripped (so `sha2` matches `sha`, `md-5` matches `md`).
fn is_watch_match(name: &str, watch: &BTreeSet<String>) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.split(['-', '_']).any(|part| {
        if watch.contains(part) {
            return true;
        }
        let base = part.trim_end_matches(|c: char| c.is_ascii_digit());
        !base.is_empty() && watch.contains(base)
    })
}

// -----------------------------------------------------------------------------
// Prose generation
// -----------------------------------------------------------------------------

/// Render the full generated companion document from the manifest. Pure and
/// deterministic: iterates the manifest in declaration order, so the same
/// manifest always renders byte-for-byte the same document.
#[expect(clippy::too_many_lines, reason = "one writeln! per generated document section")]
fn render_doc(manifest: &Manifest) -> String {
    let mut out = String::new();

    writeln!(out, "<!-- SPDX-License-Identifier: Apache-2.0 -->").unwrap();
    writeln!(out, "<!-- Copyright (c) 2026 Praxis Contributors -->").unwrap();
    writeln!(
        out,
        "<!-- Generated by `cargo xtask check-crypto-inventory --fix`. Do not edit"
    )
    .unwrap();
    writeln!(out, "     by hand — edit {MANIFEST_BASENAME} and regenerate. -->").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "# Cryptographic operations inventory").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "This page is generated from the machine-readable manifest").unwrap();
    writeln!(
        out,
        "[`{MANIFEST_BASENAME}`]({MANIFEST_BASENAME}), which is the authoritative source of"
    )
    .unwrap();
    writeln!(
        out,
        "record for the Praxis AI proxy's production cryptography ([#1220][i1220])."
    )
    .unwrap();
    writeln!(
        out,
        "`cargo xtask check-crypto-inventory` resolves the runtime dependency graph of"
    )
    .unwrap();
    writeln!(
        out,
        "each shipped profile on all three tier-1 targets and fails if the manifest and"
    )
    .unwrap();
    writeln!(
        out,
        "the resolved graphs disagree; run it with `--fix` to regenerate this page."
    )
    .unwrap();
    writeln!(out).unwrap();

    render_profiles(&mut out);
    render_dispositions(&mut out);
    render_production(manifest, &mut out);
    render_providers(manifest, &mut out);
    render_operations(manifest, &mut out);
    render_named_bucket(&mut out, "Reviewed non-primitives (`allow`)", &manifest.allow);
    render_named_bucket(&mut out, "Test-only cryptography (`test_only`)", &manifest.test_only);
    render_named_bucket(&mut out, "Build-host cryptography (`build_only`)", &manifest.build_only);
    render_named_bucket(&mut out, "Build-host proc-macros (`proc_macro`)", &manifest.proc_macro);
    render_watch_tokens(manifest, &mut out);
    render_maintenance(&mut out);

    writeln!(out).unwrap();
    writeln!(out, "[i1220]: https://github.com/praxis-proxy/ai/issues/1220").unwrap();

    out
}

/// Render the build-profiles table.
fn render_profiles(out: &mut String) {
    writeln!(out, "## Build profiles").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "The inventory tracks two shipped profiles; every `production` crate and pinned"
    )
    .unwrap();
    writeln!(
        out,
        "provider is assigned to the profiles it is resolved in. Both are built"
    )
    .unwrap();
    writeln!(out, "`--no-default-features`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "| Profile | Cargo features | Description |").unwrap();
    writeln!(out, "| --- | --- | --- |").unwrap();
    for profile in &PROFILES {
        writeln!(
            out,
            "| `{}` | `{}` | {} |",
            profile.label,
            profile.features,
            cell(profile.description)
        )
        .unwrap();
    }
    writeln!(out).unwrap();
}

/// Render the disposition legend.
fn render_dispositions(out: &mut String) {
    writeln!(out, "## Dispositions").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "Every `production` primitive carries one disposition:").unwrap();
    writeln!(out).unwrap();
    for disposition in VALID_DISPOSITIONS {
        writeln!(out, "- `{disposition}` — {}", disposition_doc(disposition)).unwrap();
    }
    writeln!(out).unwrap();
}

/// Render the production primitives table.
fn render_production(manifest: &Manifest, out: &mut String) {
    writeln!(out, "## Production primitives").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Crypto and crypto-support crates resolved into the runtime binary."
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(out, "| Crate | Profiles | Platform | Disposition | Notes |").unwrap();
    writeln!(out, "| --- | --- | --- | --- | --- |").unwrap();
    for entry in &manifest.production {
        writeln!(
            out,
            "| `{}` | {} | {} | {} | {} |",
            entry.crate_name,
            entry.profiles.join(", "),
            entry.platform.as_deref().unwrap_or("all"),
            entry.disposition.as_deref().unwrap_or(""),
            cell(&entry.note),
        )
        .unwrap();
    }
    writeln!(out).unwrap();
}

/// Render the pinned-providers table (no version pins in prose; the manifest is
/// the authoritative source for versions).
fn render_providers(manifest: &Manifest, out: &mut String) {
    writeln!(out, "## Pinned providers").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Providers whose selection is load-bearing: the manifest pins the version and"
    )
    .unwrap();
    writeln!(out, "the security-relevant features so a drift cannot pass unnoticed.").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "| Crate | Profiles | Required features | Forbidden features | Notes |"
    )
    .unwrap();
    writeln!(out, "| --- | --- | --- | --- | --- |").unwrap();
    for provider in &manifest.providers {
        writeln!(
            out,
            "| `{}` | {} | {} | {} | {} |",
            provider.crate_name,
            provider.profiles.join(", "),
            features_cell(&provider.require_features),
            features_cell(&provider.forbid_features),
            cell(&provider.note),
        )
        .unwrap();
    }
    writeln!(out).unwrap();
}

/// Render the operations inventory: one row per production-reachable
/// cryptographic operation, tying each caller to its algorithm, provider,
/// implementing crates, profiles, disposition, and remediation. Rendered only
/// when the manifest declares operations.
#[expect(clippy::too_many_lines, reason = "one writeln! per operation table column set")]
fn render_operations(manifest: &Manifest, out: &mut String) {
    if manifest.operations.is_empty() {
        return;
    }
    writeln!(out, "## Cryptographic operations").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Every production-reachable cryptographic operation: where it is used, for"
    )
    .unwrap();
    writeln!(
        out,
        "what algorithm, under whose ownership, by which crates, and what remediation"
    )
    .unwrap();
    writeln!(
        out,
        "it requires. The crate-centric table above answers \"what cryptographic code"
    )
    .unwrap();
    writeln!(
        out,
        "can ship\"; this one answers \"where is cryptography actually used\". Each"
    )
    .unwrap();
    writeln!(
        out,
        "operation's crates are cross-checked against the `production` inventory."
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "| ID | Operation | Owner | Caller | Algorithm | Provider | Crates | Profiles | Disposition | Remediation | Notes |"
    )
    .unwrap();
    writeln!(
        out,
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    .unwrap();
    for op in &manifest.operations {
        // `n-a` std operations reference no crypto crate; show an em dash there.
        let crates = if op.crates.is_empty() {
            "—".to_owned()
        } else {
            op.crates
                .iter()
                .map(|c| format!("`{c}`"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            cell(&op.id),
            cell(&op.name),
            cell(&op.owner),
            cell(&op.caller),
            cell(&op.algorithm),
            cell(&op.provider),
            cell(&crates),
            op.profiles.join(", "),
            cell(&op.disposition),
            cell(&op.remediation),
            cell(&op.note),
        )
        .unwrap();
    }
    writeln!(out).unwrap();

    if !manifest.operation_exempt.is_empty() {
        writeln!(
            out,
            "Production primitives with no standalone operation (support layers), exempt"
        )
        .unwrap();
        writeln!(out, "from operation coverage with a reason:").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "| Crate | Reason |").unwrap();
        writeln!(out, "| --- | --- |").unwrap();
        for ex in &manifest.operation_exempt {
            writeln!(out, "| `{}` | {} |", ex.crate_name, cell(&ex.reason)).unwrap();
        }
        writeln!(out).unwrap();
    }
}

/// Render a simple `Crate | Notes` bucket table under `title`.
fn render_named_bucket(out: &mut String, title: &str, entries: &[Entry]) {
    writeln!(out, "## {title}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "| Crate | Notes |").unwrap();
    writeln!(out, "| --- | --- |").unwrap();
    for entry in entries {
        writeln!(out, "| `{}` | {} |", entry.crate_name, cell(&entry.note)).unwrap();
    }
    writeln!(out).unwrap();
}

/// Render the watch-tokens note.
fn render_watch_tokens(manifest: &Manifest, out: &mut String) {
    writeln!(out, "## Watch tokens").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "The check treats a crate as crypto-relevant when its name matches one of these"
    )
    .unwrap();
    writeln!(out, "tokens (split on `-`/`_`, trailing digits stripped):").unwrap();
    writeln!(out).unwrap();
    let tokens: Vec<String> = manifest.watch_tokens.iter().map(|t| format!("`{t}`")).collect();
    writeln!(out, "{}", tokens.join(", ")).unwrap();
    writeln!(out).unwrap();
}

/// Render the maintenance note.
fn render_maintenance(out: &mut String) {
    writeln!(out, "## Maintenance").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "Edit [`{MANIFEST_BASENAME}`]({MANIFEST_BASENAME}) and run:").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "```console").unwrap();
    writeln!(out, "cargo xtask check-crypto-inventory --fix").unwrap();
    writeln!(out, "```").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "The same check without `--fix` runs in `make lint` and fails on any drift"
    )
    .unwrap();
    writeln!(
        out,
        "between the manifest, this page, and the resolved dependency graphs."
    )
    .unwrap();
}

/// Escape one cell of a Markdown table: collapse newlines to spaces and escape
/// pipes so a `note` cannot break the table layout.
fn cell(text: &str) -> String {
    text.replace('\n', " ").replace('|', "\\|").trim().to_owned()
}

/// Render a feature list into one table cell, or an em dash when empty.
fn features_cell(features: &[String]) -> String {
    if features.is_empty() {
        "—".to_owned()
    } else {
        features.iter().map(|f| format!("`{f}`")).collect::<Vec<_>>().join(", ")
    }
}

// -----------------------------------------------------------------------------
// Cargo Graph
// -----------------------------------------------------------------------------

/// Resolve the runtime (`--edges normal`) dependency graph of one profile for one
/// target triple and return its crate facts.
#[expect(
    clippy::too_many_lines,
    reason = "single cargo-tree invocation with a long argument vector"
)]
fn resolve_target(root: &Path, profile: &Profile, platform: &str, triple: &str) -> TargetGraph {
    let output = Command::new("cargo")
        .current_dir(root)
        .args([
            "tree",
            "-p",
            "praxis-ai-proxy",
            "--edges",
            "normal",
            "--no-default-features",
            "--features",
            profile.features,
            "--prefix",
            "depth",
            "--format",
            "{p}|{f}",
            "--target",
            triple,
        ])
        .output()
        .expect("failed to run cargo tree");

    if !output.status.success() {
        eprintln!(
            "cargo tree (profile {}, --target {triple}) failed:\n{}",
            profile.label,
            String::from_utf8_lossy(&output.stderr)
        );
        std::process::exit(1);
    }

    let text = String::from_utf8(output.stdout).expect("cargo tree output is not UTF-8");
    TargetGraph {
        profile: profile.label,
        platform: platform.to_owned(),
        runtime_linked: compute_runtime_linked(&text),
        facts: parse_facts(&text),
    }
}

/// Split a `--prefix depth` line into its integer depth and the remaining text.
///
/// `cargo tree --prefix depth` glues the depth digits directly onto the crate
/// name (`0praxis-ai-proxy v0.3.0 …`, `6tiny-keccak v2.0.2 …`). Lines with no
/// leading digit (the synthetic test fixtures, which omit the prefix) parse as
/// depth 0 with the whole line returned — no crate name begins with a digit, so
/// this is unambiguous.
fn split_depth(line: &str) -> (usize, &str) {
    let end = line
        .char_indices()
        .find(|(_, c)| !c.is_ascii_digit())
        .map_or(line.len(), |(i, _)| i);
    let (digits, rest) = line.split_at(end);
    (digits.parse::<usize>().unwrap_or(0), rest)
}

/// Parse the `name`/`version` and proc-macro marker out of the `{p}` (left) half
/// of a formatted line. `{p}` prints `name vX.Y.Z [(source)] [(proc-macro)]`.
fn parse_pkg(left: &str) -> Option<(PkgId, bool)> {
    let is_proc_macro = left.contains("(proc-macro)");
    let mut fields = left.split_whitespace();
    let name = fields.next()?.to_owned();
    let version = fields.next().unwrap_or("").trim_start_matches('v').to_owned();
    Some((PkgId { name, version }, is_proc_macro))
}

/// Parse `cargo tree --prefix depth --format "{p}|{f}"` output into crate facts.
///
/// Each line is `<depth>name vX.Y.Z [(source)] [(proc-macro)]|feat1,feat2 [(*)]`;
/// the leading digits are the tree depth (stripped here — linkage lives in
/// [`compute_runtime_linked`]), the trailing `(*)` marks a subtree cargo already
/// printed, and a `(proc-macro)` marker flags a build-host code generator that is
/// not linked into the runtime binary. Every distinct resolved version of a crate
/// is retained (the `(*)` repeats of the same version are de-duplicated), so a
/// crate co-resolved at two versions keeps both.
fn parse_facts(text: &str) -> BTreeMap<String, Vec<CrateFacts>> {
    let mut facts: BTreeMap<String, Vec<CrateFacts>> = BTreeMap::new();
    for raw in text.lines() {
        let trimmed = raw.trim().trim_end_matches("(*)").trim_end();
        if trimmed.is_empty() {
            continue;
        }
        let (_, line) = split_depth(trimmed);
        let (left, right) = line.split_once('|').unwrap_or((line, ""));
        let Some((id, is_proc_macro)) = parse_pkg(left) else {
            continue;
        };
        let features: BTreeSet<String> = right
            .split(',')
            .map(str::trim)
            .filter(|f| !f.is_empty())
            .map(str::to_owned)
            .collect();
        let versions = facts.entry(id.name).or_default();
        if !versions.iter().any(|f| f.version == id.version) {
            versions.push(CrateFacts {
                version: id.version,
                features,
                is_proc_macro,
            });
        }
    }
    facts
}

/// The identity-keyed edge set of a `cargo tree --prefix depth` graph, plus the
/// set of proc-macro nodes and the root package.
struct TreeEdges {
    /// For each package identity, the set of package identities it depends on.
    edges: BTreeMap<PkgId, BTreeSet<PkgId>>,
    /// Package identities marked `(proc-macro)` (build-host code generators).
    proc_macros: BTreeSet<PkgId>,
    /// The root package (`praxis-ai-proxy`), if any line was at depth 0.
    root: Option<PkgId>,
}

/// Reconstruct the dependency edges from a `cargo tree --prefix depth` walk.
///
/// `cargo tree --prefix depth` prints a depth-first walk with an integer depth on
/// every line, so the parent of a line at depth `d` is the most recent line at
/// depth `d - 1`. That reconstructs a global, identity-keyed edge set (a `(*)`
/// repeat still contributes its parent edge; its children were already printed at
/// the first, fully expanded occurrence). Keying on [`PkgId`] rather than the bare
/// name keeps two co-resolved versions of a crate as distinct nodes.
fn parse_tree_edges(text: &str) -> TreeEdges {
    let mut edges: BTreeMap<PkgId, BTreeSet<PkgId>> = BTreeMap::new();
    let mut proc_macros: BTreeSet<PkgId> = BTreeSet::new();
    let mut root: Option<PkgId> = None;
    let mut stack: Vec<PkgId> = Vec::new();
    for raw in text.lines() {
        let Some((depth, id, is_proc_macro)) = parse_tree_line(raw) else {
            continue;
        };
        if is_proc_macro {
            proc_macros.insert(id.clone());
        }
        if depth == 0 {
            root.get_or_insert_with(|| id.clone());
        } else if let Some(parent) = stack.get(depth - 1) {
            edges.entry(parent.clone()).or_default().insert(id.clone());
        }
        stack.truncate(depth);
        stack.push(id);
    }
    TreeEdges {
        edges,
        proc_macros,
        root,
    }
}

/// Parse one `cargo tree --prefix depth` line into `(depth, package identity,
/// is-proc-macro)`, or `None` for a blank line. The trailing `(*)` (a subtree
/// cargo already printed) is stripped before parsing; it still contributes a
/// parent edge but its children were printed at the first occurrence.
fn parse_tree_line(raw: &str) -> Option<(usize, PkgId, bool)> {
    let trimmed = raw.trim().trim_end_matches("(*)").trim_end();
    if trimmed.is_empty() {
        return None;
    }
    let (depth, line) = split_depth(trimmed);
    let left = line.split_once('|').map_or(line, |(l, _)| l);
    let (id, is_proc_macro) = parse_pkg(left)?;
    Some((depth, id, is_proc_macro))
}

/// Compute the set of package identities linked into the runtime binary:
/// reachable from the root package over `--edges normal` edges without passing
/// through a proc-macro node.
///
/// A breadth-first walk from the root collects every reachable package while
/// refusing to enter proc-macro nodes — so a proc-macro (a build-host code
/// generator) and anything reachable *only* beneath one (e.g. `tiny-keccak` under
/// `const-random-macro`) is excluded, while a crate that is *also* reachable by a
/// normal path stays linked. Keying on [`PkgId`] means a build-host-only copy of a
/// crate is never conflated with a runtime-linked copy of the same name.
fn compute_runtime_linked(text: &str) -> BTreeSet<PkgId> {
    let TreeEdges {
        edges,
        proc_macros,
        root,
    } = parse_tree_edges(text);
    let mut linked: BTreeSet<PkgId> = BTreeSet::new();
    let Some(root) = root else { return linked };
    if proc_macros.contains(&root) {
        return linked;
    }
    let mut queue: VecDeque<PkgId> = VecDeque::new();
    linked.insert(root.clone());
    queue.push_back(root);
    while let Some(node) = queue.pop_front() {
        let Some(children) = edges.get(&node) else { continue };
        for child in children {
            if proc_macros.contains(child) || linked.contains(child) {
                continue;
            }
            linked.insert(child.clone());
            queue.push_back(child.clone());
        }
    }
    linked
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

/// Find the workspace root by locating the top-level `Cargo.toml`.
fn workspace_root() -> PathBuf {
    let output = Command::new("cargo")
        .args(["locate-project", "--workspace", "--message-format=plain"])
        .output()
        .expect("failed to run cargo locate-project");
    let path = String::from_utf8(output.stdout).expect("non-utf8 path");
    PathBuf::from(path.trim())
        .parent()
        .expect("Cargo.toml has no parent")
        .to_owned()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    /// A synthetic package identity for tests (version is irrelevant to the
    /// name-level queries most tests exercise).
    fn pkg(name: &str) -> PkgId {
        PkgId {
            name: name.to_owned(),
            version: "0.0.0".to_owned(),
        }
    }

    /// The crate names of a runtime-linked set, for name-level assertions.
    fn linked_names(linked: &BTreeSet<PkgId>) -> BTreeSet<String> {
        linked.iter().map(|id| id.name.clone()).collect()
    }

    /// Assert no crate appears in more than one classification list. Providers
    /// intentionally overlap `production`, so they are excluded.
    fn assert_no_duplicate_entries(manifest: &Manifest) {
        let mut seen = BTreeSet::new();
        for entry in manifest
            .production
            .iter()
            .chain(&manifest.allow)
            .chain(&manifest.test_only)
            .chain(&manifest.build_only)
            .chain(&manifest.proc_macro)
        {
            assert!(
                seen.insert(entry.crate_name.clone()),
                "duplicate manifest entry: {}",
                entry.crate_name
            );
        }
    }

    #[expect(clippy::too_many_lines, reason = "inline YAML fixture")]
    fn sample_manifest() -> Manifest {
        // No `providers:` block here — provider drift has its own focused tests
        // so the name-based tests can build graphs from plain crate names.
        let yaml = "
watch_tokens: [sha, md, hmac, rustls, ring, openssl, aws, rand]
production:
  - crate: sha2
    profiles: [full]
    disposition: non-validated
    note: test
  - crate: rustls
    profiles: [full, fips]
    disposition: non-validated
    note: test
  - crate: openssl
    profiles: [full, fips]
    disposition: non-validated
    platform: linux
    note: test
  - crate: security-framework
    profiles: [full, fips]
    disposition: non-validated
    platform: macos
    note: test
allow:
  - crate: aws-smithy-types
  - crate: openssl-macros
test_only:
  - crate: rcgen
  - crate: sha1
build_only:
  - crate: sha3
";
        serde_yaml::from_str(yaml).expect("sample manifest parses")
    }

    /// Build a `full`-profile target graph from bare crate names (version/features
    /// irrelevant). Every name is treated as a normal, runtime-linked crate;
    /// proc-macro / build-host-only crates are layered on via [`insert_proc_macro`]
    /// / [`insert_build_host_only`].
    fn names_tg(platform: &str, names: &[&str]) -> TargetGraph {
        TargetGraph {
            profile: "full",
            platform: platform.to_owned(),
            facts: names
                .iter()
                .map(|n| {
                    (
                        (*n).to_owned(),
                        vec![CrateFacts {
                            version: "0.0.0".to_owned(),
                            features: BTreeSet::new(),
                            is_proc_macro: false,
                        }],
                    )
                })
                .collect(),
            runtime_linked: names.iter().map(|n| pkg(n)).collect(),
        }
    }

    /// Insert `name` into `g` marked as a build-host proc-macro (present in the
    /// tree but never linked into the runtime binary).
    fn insert_proc_macro(g: &mut TargetGraph, name: &str) {
        g.facts.insert(
            name.to_owned(),
            vec![CrateFacts {
                version: "0.0.0".to_owned(),
                features: BTreeSet::new(),
                is_proc_macro: true,
            }],
        );
        g.runtime_linked.remove(&pkg(name));
    }

    /// Insert `name` into `g` as an ordinary crate that is present in the tree but
    /// NOT runtime-linked — simulating a crate reachable only beneath a proc-macro
    /// subtree (e.g. `tiny-keccak` under `const-random-macro`).
    fn insert_build_host_only(g: &mut TargetGraph, name: &str) {
        g.facts.insert(
            name.to_owned(),
            vec![CrateFacts {
                version: "0.0.0".to_owned(),
                features: BTreeSet::new(),
                is_proc_macro: false,
            }],
        );
        g.runtime_linked.remove(&pkg(name));
    }

    /// Build a `full`-profile target graph with `normal` crates plus `proc_macros`
    /// marked as build-host proc-macros.
    fn tg_with(platform: &str, normal: &[&str], proc_macros: &[&str]) -> TargetGraph {
        let mut g = names_tg(platform, normal);
        for name in proc_macros {
            insert_proc_macro(&mut g, name);
        }
        g
    }

    /// The three-target `full`-profile graph set for a clean sample tree.
    fn clean_graphs() -> Vec<TargetGraph> {
        vec![
            names_tg(
                "linux",
                &[
                    "sha2",
                    "rustls",
                    "openssl",
                    "openssl-macros",
                    "aws-smithy-types",
                    "serde",
                ],
            ),
            names_tg("macos", &["sha2", "rustls", "security-framework", "serde"]),
            names_tg("windows", &["sha2", "rustls", "serde"]),
        ]
    }

    /// A `fips`-profile graph from bare crate names.
    fn fips_tg(platform: &str, names: &[&str]) -> TargetGraph {
        let mut g = names_tg(platform, names);
        g.profile = "fips";
        g
    }

    #[test]
    fn profile_labels_match_profiles() {
        let labels: Vec<&str> = PROFILES.iter().map(|p| p.label).collect();
        assert_eq!(labels, PROFILE_LABELS, "PROFILE_LABELS must mirror PROFILES");
    }

    #[test]
    fn tokenizer_matches_digit_suffixes_and_parts() {
        let watch: BTreeSet<String> = ["sha", "md", "rustls", "ring", "aws"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        assert!(is_watch_match("sha2", &watch), "sha2 -> sha");
        assert!(is_watch_match("md-5", &watch), "md-5 -> md");
        assert!(is_watch_match("tokio-rustls", &watch), "part rustls");
        assert!(is_watch_match("ring", &watch), "exact ring");
        assert!(is_watch_match("aws-smithy-types", &watch), "part aws");
        assert!(!is_watch_match("serde_json", &watch), "no crypto token");
        assert!(
            !is_watch_match("string_cache", &watch),
            "ring is not a token of string_cache"
        );
    }

    #[test]
    fn clean_graphs_have_no_violations() {
        let violations = evaluate(&sample_manifest(), &clean_graphs());
        assert!(violations.is_empty(), "expected no violations, got: {violations:?}");
    }

    #[test]
    fn undeclared_crypto_crate_is_flagged_from_any_target() {
        // `hmac` present only on Linux is still caught: every target graph is
        // resolved from the lockfile and checked, so the host does not matter.
        let mut graphs = clean_graphs();
        graphs[0].facts.insert(
            "hmac".to_owned(),
            vec![CrateFacts {
                version: "0.0.0".to_owned(),
                features: BTreeSet::new(),
                is_proc_macro: false,
            }],
        );
        let violations = evaluate(&sample_manifest(), &graphs);
        assert_eq!(violations.len(), 1, "hmac is undeclared: {violations:?}");
        let msg = violations.first().expect("exactly one violation");
        assert!(msg.contains("hmac"), "{msg}");
        assert!(msg.contains("undeclared"), "{msg}");
    }

    #[test]
    fn allow_list_suppresses_undeclared() {
        // openssl-macros matches `openssl` but is on the allow list -> quiet.
        let violations = evaluate(&sample_manifest(), &clean_graphs());
        assert!(
            !violations.iter().any(|v| v.contains("openssl-macros")),
            "allow-listed crate must not be flagged: {violations:?}"
        );
    }

    #[test]
    fn denylisted_cipher_without_watch_token_is_flagged_in_full_profile() {
        // Regression: the token split (on `-`/`_`, then strip trailing digits)
        // cannot reduce the separator-less name `chacha20poly1305` to its
        // `chacha`/`poly1305` tokens, and `ctr`/`cbc` have no token at all, so the
        // watch tripwire alone misses them. Because they are on
        // `fips::graph::DENIED`, the exact-name arm of `check_undeclared` still
        // catches them in the published `full` image, where `check_denied_absent`
        // (fips-only) does not run.
        let watch: BTreeSet<String> = sample_manifest()
            .watch_tokens
            .iter()
            .map(|t| t.to_ascii_lowercase())
            .collect();
        assert!(
            !is_watch_match("chacha20poly1305", &watch),
            "precondition: the watch tripwire alone must miss chacha20poly1305"
        );
        assert!(!is_watch_match("ctr", &watch), "precondition: ctr has no watch token");

        let mut graphs = clean_graphs();
        for name in ["chacha20poly1305", "ctr", "cbc"] {
            graphs[0].facts.insert(
                name.to_owned(),
                vec![CrateFacts {
                    version: "0.0.0".to_owned(),
                    features: BTreeSet::new(),
                    is_proc_macro: false,
                }],
            );
            graphs[0].runtime_linked.insert(pkg(name));
        }
        let violations = evaluate(&sample_manifest(), &graphs);
        for name in ["chacha20poly1305", "ctr", "cbc"] {
            assert!(
                violations.iter().any(|v| v.contains(name) && v.contains("undeclared")),
                "denylisted cipher `{name}` must be flagged undeclared in the full profile: {violations:?}"
            );
        }
    }

    #[test]
    fn fused_separator_less_crypto_names_match_as_whole_tokens() {
        // Regression: the token split (on `-`/`_`, then strip trailing digits)
        // cannot reduce these separator-less names to their family tokens
        // (`secp256k1`/`salsa`/`poly1305`/`schnorr`), so the manifest lists them as
        // whole crate names in `watch_tokens`. Assert both halves: the whole-name
        // arm matches, and the family-token split genuinely does not (the reason the
        // whole names are needed).
        let fused = ["libsecp256k1", "xsalsa20poly1305", "schnorrkel"];
        let whole: BTreeSet<String> = fused.iter().map(|t| (*t).to_owned()).collect();
        for name in fused {
            assert!(
                is_watch_match(name, &whole),
                "fused name `{name}` must match as a whole watch token"
            );
        }
        let family: BTreeSet<String> = ["secp256k1", "salsa", "poly1305", "schnorr"]
            .iter()
            .map(|t| (*t).to_owned())
            .collect();
        for name in fused {
            assert!(
                !is_watch_match(name, &family),
                "precondition: the token split cannot reduce `{name}` to a family token"
            );
        }
    }

    #[test]
    fn denylisted_crate_in_allow_bucket_is_flagged() {
        // Regression: the `allow` bucket must not be an escape hatch for the
        // `full` profile. A crate on `fips::graph::DENIED` (a real primitive) that
        // is filed under `allow` silences `check_undeclared`'s denylist arm
        // (declared-in-a-bucket), and `check_denied_absent` only inspects `fips`
        // graphs — so without `check_denied_not_allowed` a denied cipher shipping
        // in `full` (e.g. `aes-gcm`) would pass. Both crates are present in the
        // graph so `check_stale_allow` stays quiet and the only signal is the new
        // denylist-classification guard.
        let yaml = "
watch_tokens: [aws]
production: []
allow:
  - crate: aes-gcm
  - crate: aws-smithy-types
test_only: []
build_only: []
";
        let manifest: Manifest = serde_yaml::from_str(yaml).expect("manifest parses");
        let graphs = vec![names_tg("linux", &["aes-gcm", "aws-smithy-types"])];
        let violations = evaluate(&manifest, &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("aes-gcm") && v.contains("allow") && v.contains("denylist")),
            "a denylisted crate in `allow` must be flagged: {violations:?}"
        );
        assert!(
            !violations.iter().any(|v| v.contains("aws-smithy-types")),
            "a genuine non-primitive in `allow` must not be flagged: {violations:?}"
        );
    }

    /// Shared production block for the operations tests: two non-`n-a` primitives
    /// (`openssl` in both profiles, `jsonwebtoken` full-only) and one `n-a` support
    /// crate (`zeroize`). Each test appends its own `operations:` (and optional
    /// `operation_exempt:`) block.
    const OPS_PROD: &str = "
watch_tokens: [sha, openssl, jwt, aws, ring, zeroize]
production:
  - crate: openssl
    profiles: [full, fips]
    disposition: non-validated
    note: t
  - crate: jsonwebtoken
    profiles: [full]
    disposition: non-validated
    note: t
  - crate: zeroize
    profiles: [full, fips]
    disposition: n-a
    note: t
allow: []
test_only: []
build_only: []
";

    /// Run only the operations guard against a manifest built from [`OPS_PROD`]
    /// plus the caller-supplied `operations`/`operation_exempt` tail.
    fn op_violations(tail: &str) -> Vec<String> {
        let yaml = format!("{OPS_PROD}{tail}");
        let manifest: Manifest = serde_yaml::from_str(&yaml).expect("manifest parses");
        let mut out = Vec::new();
        check_operations(&manifest, &mut out);
        out
    }

    #[test]
    fn operations_clean_manifest_passes() {
        // openssl -> T1, jsonwebtoken -> A1; zeroize is `n-a` and needs no
        // operation. Non-compliant dispositions carry remediation references.
        let violations = op_violations(
            "operations:
  - id: T1
    name: TLS on OpenSSL
    owner: praxis-core tls
    caller: server/src/server.rs:65
    algorithm: TLS 1.2/1.3
    provider: OpenSSL libcrypto
    crates: [openssl]
    profiles: [full, fips]
    disposition: non-validated
    remediation: docs/fips.md
    note: t
  - id: A1
    name: JWT verify
    owner: praxis-proxy/policy (upstream)
    caller: identity-jwt config.rs
    algorithm: RS256/ES256 verify
    provider: aws-lc-rs via jsonwebtoken
    crates: [jsonwebtoken]
    profiles: [full]
    disposition: non-validated
    remediation: \"#1220\"
    note: t
",
        );
        assert!(violations.is_empty(), "clean operations manifest: {violations:?}");
    }

    #[test]
    fn operation_duplicate_id_is_flagged() {
        let violations = op_violations(
            "operations:
  - id: T1
    name: a
    owner: o
    caller: c
    algorithm: alg
    provider: p
    crates: [openssl]
    profiles: [full]
    disposition: n-a
    note: t
  - id: T1
    name: b
    owner: o
    caller: c
    algorithm: alg
    provider: p
    crates: [jsonwebtoken]
    profiles: [full]
    disposition: n-a
    note: t
",
        );
        assert!(
            violations
                .iter()
                .any(|v| v.contains("T1") && v.contains("more than once")),
            "duplicate id must be flagged: {violations:?}"
        );
    }

    #[test]
    fn operation_empty_required_field_is_flagged() {
        // Empty owner (a required field distinct from caller/provider).
        let violations = op_violations(
            "operations:
  - id: T1
    name: a
    owner: \"\"
    caller: c
    algorithm: alg
    provider: p
    crates: [openssl]
    profiles: [full]
    disposition: n-a
    note: t
",
        );
        assert!(
            violations.iter().any(|v| v.contains("T1") && v.contains("owner")),
            "empty owner must be flagged: {violations:?}"
        );
    }

    #[test]
    fn operation_references_unknown_crate_is_flagged() {
        let violations = op_violations(
            "operations:
  - id: T1
    name: a
    owner: o
    caller: c
    algorithm: alg
    provider: p
    crates: [ring]
    profiles: [full]
    disposition: n-a
    note: t
",
        );
        assert!(
            violations
                .iter()
                .any(|v| v.contains("ring") && v.contains("not a `production` entry")),
            "unknown crate must be flagged: {violations:?}"
        );
    }

    #[test]
    fn operation_profile_absent_from_crate_is_flagged() {
        // jsonwebtoken is full-only; an op claiming it runs in `fips` is a leak.
        let violations = op_violations(
            "operations:
  - id: A1
    name: JWT
    owner: o
    caller: c
    algorithm: alg
    provider: p
    crates: [jsonwebtoken]
    profiles: [full, fips]
    disposition: non-validated
    remediation: \"#1220\"
    note: t
",
        );
        assert!(
            violations
                .iter()
                .any(|v| v.contains("A1") && v.contains("fips") && v.contains("jsonwebtoken")),
            "profile the crate is absent from must be flagged: {violations:?}"
        );
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "inline YAML fixture")]
    fn operation_non_compliant_without_remediation_is_flagged() {
        // non-validated but no remediation reference at all.
        let missing = op_violations(
            "operations:
  - id: T1
    name: a
    owner: o
    caller: c
    algorithm: alg
    provider: p
    crates: [openssl]
    profiles: [full]
    disposition: non-validated
    note: t
",
        );
        assert!(
            missing.iter().any(|v| v.contains("T1") && v.contains("remediation")),
            "missing remediation must be flagged: {missing:?}"
        );
        // A bare non-reference string ("TODO") does not count as a link.
        let bare = op_violations(
            "operations:
  - id: T1
    name: a
    owner: o
    caller: c
    algorithm: alg
    provider: p
    crates: [openssl]
    profiles: [full]
    disposition: non-validated
    remediation: TODO
    note: t
",
        );
        assert!(
            bare.iter().any(|v| v.contains("T1") && v.contains("remediation")),
            "bare remediation string must be rejected: {bare:?}"
        );
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "inline YAML fixture")]
    fn uncovered_primitive_is_flagged_and_exemption_satisfies_it() {
        // Only openssl is covered; jsonwebtoken (non-`n-a`) is referenced by no op.
        let uncovered = op_violations(
            "operations:
  - id: T1
    name: a
    owner: o
    caller: c
    algorithm: alg
    provider: p
    crates: [openssl]
    profiles: [full]
    disposition: n-a
    note: t
",
        );
        assert!(
            uncovered
                .iter()
                .any(|v| v.contains("jsonwebtoken") && v.contains("referenced by no operation")),
            "uncovered primitive must be flagged: {uncovered:?}"
        );
        // Exempting it (with a reason) clears the coverage gap.
        let exempted = op_violations(
            "operations:
  - id: T1
    name: a
    owner: o
    caller: c
    algorithm: alg
    provider: p
    crates: [openssl]
    profiles: [full]
    disposition: n-a
    note: t
operation_exempt:
  - crate: jsonwebtoken
    reason: support layer under aws-lc-rs
",
        );
        assert!(
            !exempted.iter().any(|v| v.contains("jsonwebtoken")),
            "an explicit exemption must satisfy coverage: {exempted:?}"
        );
    }

    #[test]
    fn operation_exempt_unknown_crate_is_flagged() {
        let violations = op_violations(
            "operations:
  - id: T1
    name: a
    owner: o
    caller: c
    algorithm: alg
    provider: p
    crates: [openssl, jsonwebtoken]
    profiles: [full]
    disposition: n-a
    note: t
operation_exempt:
  - crate: nonexistent
    reason: r
",
        );
        assert!(
            violations
                .iter()
                .any(|v| v.contains("nonexistent") && v.contains("not a `production` entry")),
            "exempting a non-production crate must be flagged: {violations:?}"
        );
    }

    #[test]
    fn empty_operations_skips_coverage() {
        // No operations declared: the manifest-shape guards still run, but coverage
        // is not enforced (nothing to be complete against), so the non-`n-a`
        // primitives are not reported as uncovered.
        let violations = op_violations("operations: []\n");
        assert!(violations.is_empty(), "empty operations must be quiet: {violations:?}");
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "inline YAML fixture")]
    fn na_operation_may_reference_no_crates_but_others_must() {
        // An `n-a` std/non-crypto operation legitimately references no crypto
        // crate (e.g. a SipHash `DefaultHasher` change-detection digest).
        let na = op_violations(
            "operations:
  - id: S1
    name: SipHash affinity digest
    owner: praxis-ai apis
    caller: stream_events/mod.rs:1582
    algorithm: SipHash-1-3
    provider: Rust std DefaultHasher
    crates: []
    profiles: [full]
    disposition: n-a
    note: t
",
        );
        assert!(
            !na.iter()
                .any(|v| v.contains("S1") && v.contains("references no `crates`")),
            "an n-a op may reference no crates: {na:?}"
        );
        // A crate-backed disposition with no crates is still a defect.
        let bad = op_violations(
            "operations:
  - id: S1
    name: x
    owner: o
    caller: c
    algorithm: alg
    provider: p
    crates: []
    profiles: [full]
    disposition: non-validated
    remediation: \"#1220\"
    note: t
",
        );
        assert!(
            bad.iter()
                .any(|v| v.contains("S1") && v.contains("references no `crates`")),
            "a non-`n-a` op with no crates must be flagged: {bad:?}"
        );
    }

    #[test]
    fn looks_like_reference_accepts_links_and_rejects_prose() {
        assert!(looks_like_reference("#1217"));
        assert!(looks_like_reference("see #1217 for details"));
        assert!(looks_like_reference("https://example.com/x"));
        assert!(looks_like_reference("docs/fips.md"));
        assert!(looks_like_reference("upstream praxis-proxy/policy"));
        assert!(!looks_like_reference(""));
        assert!(!looks_like_reference("TODO"));
        assert!(!looks_like_reference("fix later"));
        // A `#` with no following digit is not an issue reference.
        assert!(!looks_like_reference("section #"));
    }

    #[test]
    fn untagged_production_entry_missing_everywhere_is_flagged() {
        // rustls (platform any, profiles full) removed from all three graphs.
        let graphs = vec![
            names_tg("linux", &["sha2", "openssl", "openssl-macros", "aws-smithy-types"]),
            names_tg("macos", &["sha2", "security-framework"]),
            names_tg("windows", &["sha2"]),
        ];
        let violations = evaluate(&sample_manifest(), &graphs);
        let stale: Vec<&String> = violations
            .iter()
            .filter(|v| v.contains("stale") && v.contains("rustls"))
            .collect();
        assert_eq!(stale.len(), 3, "one stale per target: {violations:?}");
    }

    #[test]
    fn off_target_platform_entry_is_not_flagged_stale() {
        // openssl (linux) is absent from macOS/Windows and must NOT be flagged.
        let violations = evaluate(&sample_manifest(), &clean_graphs());
        assert!(
            !violations.iter().any(|v| v.contains("openssl") && v.contains("stale")),
            "{violations:?}"
        );
    }

    #[test]
    fn platform_entry_missing_on_its_own_target_is_flagged() {
        // security-framework (macos) is absent from the macOS graph -> stale.
        let mut graphs = clean_graphs();
        graphs[1] = names_tg("macos", &["sha2", "rustls", "serde"]);
        let violations = evaluate(&sample_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("stale") && v.contains("security-framework")),
            "{violations:?}"
        );
    }

    #[test]
    fn platform_tagged_entry_does_not_suppress_other_platforms() {
        // `openssl` is declared only for linux; an unexpected copy on the windows
        // graph must still be flagged undeclared (platform-scoped suppression).
        let mut graphs = clean_graphs();
        graphs[2] = names_tg("windows", &["sha2", "rustls", "serde", "openssl"]);
        let violations = evaluate(&sample_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("undeclared") && v.contains("openssl") && v.contains("windows")),
            "{violations:?}"
        );
    }

    #[test]
    fn unknown_platform_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.production.push(Entry {
            crate_name: "made-up".to_owned(),
            platform: Some("solaris".to_owned()),
            profiles: vec!["full".to_owned()],
            disposition: Some("n-a".to_owned()),
            note: "test".to_owned(),
        });
        let violations = evaluate(&manifest, &clean_graphs());
        assert!(
            violations
                .iter()
                .any(|v| v.contains("unknown platform") && v.contains("solaris")),
            "{violations:?}"
        );
    }

    #[test]
    fn missing_profiles_on_production_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.production.push(Entry {
            crate_name: "newprim".to_owned(),
            platform: None,
            profiles: Vec::new(),
            disposition: Some("n-a".to_owned()),
            note: "test".to_owned(),
        });
        let violations = evaluate(&manifest, &clean_graphs());
        assert!(
            violations
                .iter()
                .any(|v| v.contains("newprim") && v.contains("profiles")),
            "{violations:?}"
        );
    }

    #[test]
    fn unknown_profile_on_production_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.production.push(Entry {
            crate_name: "newprim".to_owned(),
            platform: None,
            profiles: vec!["turbo".to_owned()],
            disposition: Some("n-a".to_owned()),
            note: "test".to_owned(),
        });
        let violations = evaluate(&manifest, &clean_graphs());
        assert!(
            violations
                .iter()
                .any(|v| v.contains("unknown profile") && v.contains("turbo")),
            "{violations:?}"
        );
    }

    #[test]
    fn profiles_on_non_production_bucket_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.allow.push(Entry {
            crate_name: "openssl-probe".to_owned(),
            platform: None,
            profiles: vec!["full".to_owned()],
            disposition: None,
            note: String::new(),
        });
        let violations = evaluate(&manifest, &clean_graphs());
        assert!(
            violations
                .iter()
                .any(|v| v.contains("openssl-probe") && v.contains("profile")),
            "{violations:?}"
        );
    }

    #[test]
    fn full_only_entry_present_in_fips_graph_is_a_profile_leak() {
        // sha2 is declared full-only; appearing in a fips graph is a leak.
        let mut graphs = clean_graphs();
        graphs.push(fips_tg("linux", &["rustls", "openssl", "sha2"]));
        let violations = evaluate(&sample_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("profile leak") && v.contains("sha2")),
            "{violations:?}"
        );
    }

    #[test]
    fn full_only_entry_absent_from_fips_graph_is_not_stale() {
        // sha2 (full-only) legitimately absent from a fips graph must be quiet.
        let mut graphs = clean_graphs();
        graphs.push(fips_tg("linux", &["rustls", "openssl"]));
        let violations = evaluate(&sample_manifest(), &graphs);
        assert!(
            !violations.iter().any(|v| v.contains("sha2")),
            "full-only crate absent from fips must be quiet: {violations:?}"
        );
    }

    #[test]
    fn fips_denylist_crate_in_fips_graph_is_flagged() {
        // aws-lc-rs is on the FIPS denylist; its presence in a fips graph fails.
        let mut graphs = clean_graphs();
        graphs.push(fips_tg("linux", &["rustls", "openssl", "aws-lc-rs"]));
        let violations = evaluate(&sample_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("FIPS denylist") && v.contains("aws-lc-rs")),
            "{violations:?}"
        );
    }

    #[test]
    fn fips_denylist_crate_in_full_graph_is_not_flagged_by_denylist() {
        // sha2 in a full graph is fine — the denylist only guards fips graphs.
        let violations = evaluate(&sample_manifest(), &clean_graphs());
        assert!(
            !violations.iter().any(|v| v.contains("FIPS denylist")),
            "{violations:?}"
        );
    }

    #[test]
    fn missing_disposition_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.production.push(Entry {
            crate_name: "sha2".to_owned(),
            platform: None,
            profiles: vec!["full".to_owned()],
            disposition: None,
            note: "test".to_owned(),
        });
        let violations = evaluate(&manifest, &clean_graphs());
        assert!(
            violations.iter().any(|v| v.contains("no `disposition`")),
            "{violations:?}"
        );
    }

    #[test]
    fn unknown_disposition_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.production.push(Entry {
            crate_name: "sha2".to_owned(),
            platform: None,
            profiles: vec!["full".to_owned()],
            disposition: Some("bogus".to_owned()),
            note: "test".to_owned(),
        });
        let violations = evaluate(&manifest, &clean_graphs());
        assert!(
            violations
                .iter()
                .any(|v| v.contains("unknown disposition") && v.contains("bogus")),
            "{violations:?}"
        );
    }

    #[test]
    fn empty_production_note_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.production.push(Entry {
            crate_name: "sha2".to_owned(),
            platform: None,
            profiles: vec!["full".to_owned()],
            disposition: Some("n-a".to_owned()),
            note: "  ".to_owned(),
        });
        let violations = evaluate(&manifest, &clean_graphs());
        assert!(violations.iter().any(|v| v.contains("empty `note`")), "{violations:?}");
    }

    #[test]
    fn misspelled_provider_constraint_key_is_rejected() {
        // `forbid_feature` (typo) must be a hard parse error, not a silently
        // dropped constraint that would defeat the drift guard.
        let yaml = "
watch_tokens: [aws]
production:
  - crate: aws-lc-rs
    profiles: [full]
    disposition: non-validated
    note: test
allow: []
test_only: []
build_only: []
providers:
  - crate: aws-lc-rs
    version: \"1.18.1\"
    profiles: [full]
    forbid_feature: [fips]
    note: test
";
        let parsed: Result<Manifest, _> = serde_yaml::from_str(yaml);
        assert!(parsed.is_err(), "unknown provider field must be rejected");
    }

    #[test]
    fn test_only_crate_in_runtime_is_containment_regression() {
        // rcgen linked into the runtime binary (not merely present) is a leak.
        let mut graphs = clean_graphs();
        graphs[0].facts.insert(
            "rcgen".to_owned(),
            vec![CrateFacts {
                version: "0.0.0".to_owned(),
                features: BTreeSet::new(),
                is_proc_macro: false,
            }],
        );
        graphs[0].runtime_linked.insert(pkg("rcgen"));
        let violations = evaluate(&sample_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("containment regression") && v.contains("rcgen")),
            "{violations:?}"
        );
    }

    #[test]
    fn build_only_crate_in_runtime_is_containment_regression() {
        // sha3 linked into the runtime binary (not merely present) is a leak.
        let mut graphs = clean_graphs();
        graphs[2].facts.insert(
            "sha3".to_owned(),
            vec![CrateFacts {
                version: "0.0.0".to_owned(),
                features: BTreeSet::new(),
                is_proc_macro: false,
            }],
        );
        graphs[2].runtime_linked.insert(pkg("sha3"));
        let violations = evaluate(&sample_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("containment regression") && v.contains("sha3")),
            "{violations:?}"
        );
    }

    #[test]
    fn build_only_crate_present_but_not_runtime_linked_is_quiet() {
        // A `build_only` crate that appears in the tree only beneath a proc-macro
        // subtree (present but not runtime-linked) is its legitimate build-host
        // home — no containment regression. Modelled on `tiny-keccak`.
        let mut manifest = sample_manifest();
        manifest.build_only.push(Entry {
            crate_name: "tiny-keccak".to_owned(),
            platform: None,
            profiles: Vec::new(),
            disposition: None,
            note: "build-host only".to_owned(),
        });
        let mut graphs = clean_graphs();
        insert_build_host_only(&mut graphs[0], "tiny-keccak");
        let violations = evaluate(&manifest, &graphs);
        assert!(
            !violations.iter().any(|v| v.contains("tiny-keccak")),
            "build-host-only crate must be quiet: {violations:?}"
        );
    }

    #[test]
    fn runtime_list_entry_not_linked_is_flagged() {
        // A crate declared in `allow` (a runtime list) but reachable only beneath
        // a proc-macro subtree overstates the runtime contents -> flagged with a
        // pointer to `build_only`. This is the tiny-keccak-in-allow regression.
        let mut manifest = sample_manifest();
        manifest.allow.push(Entry {
            crate_name: "tiny-keccak".to_owned(),
            platform: None,
            profiles: Vec::new(),
            disposition: None,
            note: String::new(),
        });
        let mut graphs = clean_graphs();
        insert_build_host_only(&mut graphs[0], "tiny-keccak");
        let violations = evaluate(&manifest, &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("tiny-keccak") && v.contains("build_only")),
            "{violations:?}"
        );
    }

    #[test]
    fn duplicate_entry_across_lists_is_flagged() {
        // A crate classified in two lists lets one bucket's guard mask another's.
        let mut manifest = sample_manifest();
        manifest.build_only.push(Entry {
            crate_name: "sha2".to_owned(), // already a `production` entry
            platform: None,
            profiles: Vec::new(),
            disposition: None,
            note: "dup".to_owned(),
        });
        let violations = evaluate(&manifest, &clean_graphs());
        assert!(
            violations
                .iter()
                .any(|v| v.contains("duplicate manifest entry") && v.contains("sha2")),
            "{violations:?}"
        );
    }

    #[test]
    fn compute_runtime_linked_excludes_proc_macro_subtree() {
        // `under-pm` is reachable only beneath a proc-macro node -> build-host
        // only. `shared` sits under a normal path -> runtime-linked.
        let text = "0root v1.0.0|\n\
                    1normal-a v1.0.0|\n\
                    2shared v1.0.0|\n\
                    1pm-crate v1.0.0 (proc-macro)|\n\
                    2under-pm v1.0.0|\n";
        let linked = linked_names(&compute_runtime_linked(text));
        assert!(linked.contains("root"), "{linked:?}");
        assert!(linked.contains("normal-a"), "{linked:?}");
        assert!(linked.contains("shared"), "{linked:?}");
        assert!(!linked.contains("pm-crate"), "proc-macro excluded: {linked:?}");
        assert!(
            !linked.contains("under-pm"),
            "crate only under a proc-macro excluded: {linked:?}"
        );
    }

    #[test]
    fn compute_runtime_linked_keeps_crate_with_a_normal_path() {
        // `dual` is reachable via a proc-macro AND via a normal crate; the normal
        // path keeps it linked.
        let text = "0root v1.0.0|\n\
                    1pm-crate v1.0.0 (proc-macro)|\n\
                    2dual v1.0.0|\n\
                    1normal-b v1.0.0|\n\
                    2dual v1.0.0|\n";
        let linked = linked_names(&compute_runtime_linked(text));
        assert!(linked.contains("normal-b"), "{linked:?}");
        assert!(
            linked.contains("dual"),
            "reachable via a normal path stays linked: {linked:?}"
        );
        assert!(!linked.contains("pm-crate"), "{linked:?}");
    }

    #[test]
    fn compute_runtime_linked_distinguishes_coresolved_versions() {
        // Two versions of `dep`: v1 under a normal path (linked), v2 only beneath a
        // proc-macro (build-host only). Identity keying keeps them apart.
        let text = "0root v1.0.0|\n\
                    1normal-a v1.0.0|\n\
                    2dep v1.0.0|\n\
                    1pm-crate v1.0.0 (proc-macro)|\n\
                    2dep v2.0.0|\n";
        let linked = compute_runtime_linked(text);
        assert!(
            linked.contains(&PkgId {
                name: "dep".to_owned(),
                version: "1.0.0".to_owned()
            }),
            "v1 under a normal path is linked: {linked:?}"
        );
        assert!(
            !linked.contains(&PkgId {
                name: "dep".to_owned(),
                version: "2.0.0".to_owned()
            }),
            "v2 only beneath a proc-macro is not linked: {linked:?}"
        );
    }

    fn proc_macro_manifest() -> Manifest {
        // `zeroize_derive` matches the `zeroize` token and is declared as a
        // build-host proc-macro; `sha2`/`rustls` are ordinary runtime primitives.
        let yaml = "
watch_tokens: [sha, rustls, zeroize]
production:
  - crate: sha2
    profiles: [full]
    disposition: non-validated
    note: test
  - crate: rustls
    profiles: [full]
    disposition: non-validated
    note: test
allow: []
test_only: []
build_only: []
proc_macro:
  - crate: zeroize_derive
    note: test
";
        serde_yaml::from_str(yaml).expect("proc_macro manifest parses")
    }

    #[test]
    fn parse_facts_detects_proc_macro_marker() {
        let text = "zeroize_derive v1.5.0 (proc-macro)|\n\
                    async-trait v0.1.92 (proc-macro)|feat (*)\n\
                    sha2 v0.10.0|default\n";
        let facts = parse_facts(text);
        assert!(
            facts
                .get("zeroize_derive")
                .and_then(|v| v.first())
                .expect("present")
                .is_proc_macro,
            "marker detected"
        );
        assert!(
            facts
                .get("async-trait")
                .and_then(|v| v.first())
                .expect("present")
                .is_proc_macro,
            "marker detected before trailing (*)"
        );
        assert!(
            !facts
                .get("sha2")
                .and_then(|v| v.first())
                .expect("present")
                .is_proc_macro,
            "normal crate not flagged"
        );
    }

    #[test]
    fn well_classified_proc_macro_has_no_violations() {
        // zeroize_derive is a build-host proc-macro declared in `proc_macro`: it
        // must be suppressed for undeclared, never flagged stale/misclassified,
        // and never counted as runtime content.
        let graphs = vec![
            tg_with("linux", &["sha2", "rustls"], &["zeroize_derive"]),
            tg_with("macos", &["sha2", "rustls"], &["zeroize_derive"]),
            tg_with("windows", &["sha2", "rustls"], &["zeroize_derive"]),
        ];
        let violations = evaluate(&proc_macro_manifest(), &graphs);
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn proc_macro_in_runtime_list_is_flagged() {
        // A production entry reported as a proc-macro in the graph overstates the
        // runtime-binary contents and must be flagged.
        let mut graphs = clean_graphs();
        insert_proc_macro(&mut graphs[0], "sha2");
        let violations = evaluate(&sample_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("sha2") && v.contains("runtime list")),
            "{violations:?}"
        );
    }

    #[test]
    fn misclassified_proc_macro_entry_is_flagged() {
        // zeroize_derive declared as a proc-macro but resolving as a normal
        // (linked) crate means a runtime crate is hiding in the build-host bucket.
        let graphs = vec![
            tg_with("linux", &["sha2", "rustls", "zeroize_derive"], &[]),
            tg_with("macos", &["sha2", "rustls", "zeroize_derive"], &[]),
            tg_with("windows", &["sha2", "rustls", "zeroize_derive"], &[]),
        ];
        let violations = evaluate(&proc_macro_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("misclassified") && v.contains("zeroize_derive")),
            "{violations:?}"
        );
    }

    #[test]
    fn stale_proc_macro_entry_is_flagged() {
        // zeroize_derive declared but absent from every graph -> dead entry.
        let graphs = vec![
            tg_with("linux", &["sha2", "rustls"], &[]),
            tg_with("macos", &["sha2", "rustls"], &[]),
            tg_with("windows", &["sha2", "rustls"], &[]),
        ];
        let violations = evaluate(&proc_macro_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("stale `proc_macro`") && v.contains("zeroize_derive")),
            "{violations:?}"
        );
    }

    fn provider_manifest() -> Manifest {
        // aws-lc-rs is both a production primitive and a pinned provider (as in
        // the real manifest), so the undeclared check stays quiet and only the
        // provider-drift path is exercised.
        let yaml = "
watch_tokens: [aws]
production:
  - crate: aws-lc-rs
    profiles: [full]
    disposition: non-validated
    note: test
allow: []
test_only: []
build_only: []
providers:
  - crate: aws-lc-rs
    version: \"1.18.1\"
    profiles: [full]
    require_features: [aws-lc-sys]
    forbid_features: [fips]
    note: test
";
        serde_yaml::from_str(yaml).expect("provider manifest parses")
    }

    /// Three `full`-profile graphs where `aws-lc-rs` resolves at the given
    /// `(version, &[features])` instances on every target.
    fn provider_graphs(instances: &[(&str, &[&str])]) -> Vec<TargetGraph> {
        let build = || {
            instances
                .iter()
                .map(|(v, f)| CrateFacts {
                    version: (*v).to_owned(),
                    features: f.iter().map(|s| (*s).to_owned()).collect(),
                    is_proc_macro: false,
                })
                .collect::<Vec<_>>()
        };
        TARGETS
            .iter()
            .map(|(platform, _)| {
                let facts = [("aws-lc-rs".to_owned(), build())].into_iter().collect();
                TargetGraph {
                    profile: "full",
                    platform: (*platform).to_owned(),
                    facts,
                    runtime_linked: [pkg("aws-lc-rs")].into_iter().collect(),
                }
            })
            .collect()
    }

    #[test]
    fn provider_clean_has_no_violations() {
        let graphs = provider_graphs(&[("1.18.1", &["aws-lc-sys", "alloc", "default"])]);
        let violations = evaluate(&provider_manifest(), &graphs);
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn provider_version_drift_is_flagged() {
        let graphs = provider_graphs(&[("1.19.0", &["aws-lc-sys"])]);
        let violations = evaluate(&provider_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("provider drift") && v.contains("1.19.0")),
            "{violations:?}"
        );
    }

    #[test]
    fn provider_forbidden_feature_is_flagged() {
        let graphs = provider_graphs(&[("1.18.1", &["aws-lc-sys", "fips"])]);
        let violations = evaluate(&provider_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("forbidden feature") && v.contains("fips")),
            "{violations:?}"
        );
    }

    #[test]
    fn provider_missing_required_feature_is_flagged() {
        let graphs = provider_graphs(&[("1.18.1", &["alloc"])]);
        let violations = evaluate(&provider_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("missing required feature") && v.contains("aws-lc-sys")),
            "{violations:?}"
        );
    }

    #[test]
    fn provider_absent_everywhere_is_flagged() {
        let graphs = vec![
            names_tg("linux", &["serde"]),
            names_tg("macos", &["serde"]),
            names_tg("windows", &["serde"]),
        ];
        let violations = evaluate(&provider_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("pinned provider") && v.contains("aws-lc-rs")),
            "{violations:?}"
        );
    }

    #[test]
    fn parse_facts_extracts_name_version_and_features() {
        let text = "praxis-ai-proxy v0.3.0 (/path/to/server)|store-postgres\n\
                    ring v0.17.14|alloc,default,dev_urandom_fallback\n\
                    aws-lc-rs v1.18.1|alloc,aws-lc-sys,default (*)\n\
                    serde v1.0.0|\n\n";
        let facts = parse_facts(text);
        assert_eq!(
            facts.len(),
            4,
            "blank lines ignored: {:?}",
            facts.keys().collect::<Vec<_>>()
        );
        let ring = facts.get("ring").and_then(|v| v.first()).expect("ring present");
        assert_eq!(ring.version, "0.17.14");
        assert!(ring.features.contains("dev_urandom_fallback"));
        let aws = facts
            .get("aws-lc-rs")
            .and_then(|v| v.first())
            .expect("aws-lc-rs present");
        assert_eq!(aws.version, "1.18.1", "(*) marker stripped");
        assert!(aws.features.contains("aws-lc-sys"));
        assert!(!aws.features.contains("fips"));
        let serde = facts.get("serde").and_then(|v| v.first()).expect("serde present");
        assert!(serde.features.is_empty(), "no features");
    }

    #[test]
    fn parse_facts_keeps_multiple_resolved_versions() {
        // A crate co-resolved at two versions must retain both (provider-drift
        // relies on seeing the second, upgraded copy).
        let text = "aws-lc-rs v1.18.1|aws-lc-sys\n\
                    aws-lc-rs v2.0.0|aws-lc-sys,fips\n\
                    aws-lc-rs v1.18.1|aws-lc-sys (*)\n";
        let facts = parse_facts(text);
        let versions = facts.get("aws-lc-rs").expect("aws-lc-rs present");
        assert_eq!(
            versions.len(),
            2,
            "both versions kept, (*) repeat de-duplicated: {versions:?}"
        );
        assert!(versions.iter().any(|f| f.version == "1.18.1"));
        assert!(
            versions
                .iter()
                .any(|f| f.version == "2.0.0" && f.features.contains("fips"))
        );
    }

    #[test]
    fn provider_second_coresolved_version_is_flagged() {
        // The pinned 1.18.1 copy is fine, but a co-resolved 2.0.0 copy must not
        // be silently accepted — this is the P2 false-negative guard.
        let graphs = provider_graphs(&[("1.18.1", &["aws-lc-sys"]), ("2.0.0", &["aws-lc-sys"])]);
        let violations = evaluate(&provider_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("provider drift") && v.contains("2.0.0")),
            "{violations:?}"
        );
    }

    #[test]
    fn render_doc_is_deterministic() {
        let manifest = sample_manifest();
        assert_eq!(render_doc(&manifest), render_doc(&manifest));
    }

    #[test]
    fn render_doc_lists_content_and_generated_header() {
        let doc = render_doc(&provider_manifest());
        assert!(doc.contains("Generated by"), "{doc}");
        assert!(doc.contains("aws-lc-rs"), "{doc}");
        assert!(doc.contains("## Build profiles"), "{doc}");
        assert!(doc.contains("## Pinned providers"), "{doc}");
        // No dependency version pins in the prose.
        assert!(!doc.contains("1.18.1"), "prose must not pin versions: {doc}");
    }

    #[test]
    fn render_doc_escapes_pipes_in_notes() {
        let mut manifest = sample_manifest();
        manifest.production.push(Entry {
            crate_name: "pipey".to_owned(),
            platform: None,
            profiles: vec!["full".to_owned()],
            disposition: Some("n-a".to_owned()),
            note: "a | b".to_owned(),
        });
        let doc = render_doc(&manifest);
        assert!(doc.contains("a \\| b"), "pipe escaped in cell: {doc}");
    }

    #[test]
    fn generated_doc_matches_committed() {
        let root = workspace_root();
        let raw = std::fs::read_to_string(root.join(MANIFEST_REL)).expect("manifest readable");
        let manifest: Manifest = serde_yaml::from_str(&raw).expect("real manifest parses");
        let generated = render_doc(&manifest);
        let committed = std::fs::read_to_string(root.join(DOC_REL)).unwrap_or_default();
        assert_eq!(
            generated, committed,
            "companion document is stale — run `cargo xtask check-crypto-inventory --fix`"
        );
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "one assertion block per manifest invariant")]
    fn real_manifest_parses_and_is_internally_consistent() {
        let root = workspace_root();
        let raw = std::fs::read_to_string(root.join(MANIFEST_REL)).expect("manifest readable");
        let manifest: Manifest = serde_yaml::from_str(&raw).expect("real manifest parses");
        assert!(!manifest.watch_tokens.is_empty());
        assert!(!manifest.production.is_empty());
        assert!(!manifest.providers.is_empty(), "providers pinned");

        // No crate may appear in more than one classification list (providers
        // intentionally overlap `production`, so they are excluded here).
        assert_no_duplicate_entries(&manifest);

        // Every production entry carries a valid disposition.
        let valid: BTreeSet<&str> = VALID_DISPOSITIONS.iter().copied().collect();
        for entry in &manifest.production {
            let disp = entry.disposition.as_deref();
            assert!(
                disp.is_some_and(|d| valid.contains(d)),
                "{} has missing/invalid disposition {disp:?}",
                entry.crate_name
            );
        }

        // Every production entry and pinned provider declares a valid, non-empty
        // profiles set.
        let valid_profiles: BTreeSet<&str> = PROFILE_LABELS.iter().copied().collect();
        for entry in &manifest.production {
            assert!(!entry.profiles.is_empty(), "{} has no profiles", entry.crate_name);
            for profile in &entry.profiles {
                assert!(
                    valid_profiles.contains(profile.as_str()),
                    "{} has invalid profile {profile}",
                    entry.crate_name
                );
            }
        }
        for provider in &manifest.providers {
            assert!(
                !provider.profiles.is_empty(),
                "provider {} has no profiles",
                provider.crate_name
            );
            for profile in &provider.profiles {
                assert!(
                    valid_profiles.contains(profile.as_str()),
                    "provider {} has invalid profile {profile}",
                    provider.crate_name
                );
            }
        }

        // Every pinned provider is also tracked as a production primitive.
        let production: BTreeSet<&str> = manifest.production.iter().map(|e| e.crate_name.as_str()).collect();
        for provider in &manifest.providers {
            assert!(
                production.contains(provider.crate_name.as_str()),
                "pinned provider {} must also be a production entry",
                provider.crate_name
            );
        }

        // The canonical SHA token guards against an empty/malformed token list.
        let watch: BTreeSet<String> = manifest.watch_tokens.iter().map(|t| t.to_ascii_lowercase()).collect();
        assert!(watch.contains("sha"), "expected canonical token present");

        // The operation-level inventory (#1220) is populated and internally
        // consistent: unique ids, complete fields, profiles that match the
        // referenced crates, remediation for non-compliant dispositions, and
        // coverage of every real primitive. `check_operations` needs no graph, so
        // it runs here directly.
        assert!(!manifest.operations.is_empty(), "operations inventory populated");
        let mut op_out = Vec::new();
        check_operations(&manifest, &mut op_out);
        assert!(op_out.is_empty(), "operations inventory inconsistent: {op_out:?}");
    }
}
