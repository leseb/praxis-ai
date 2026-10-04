#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Praxis Contributors

set -eu

manifest=Cargo.toml
begin='# BEGIN LOCAL PRAXIS OVERRIDE'
end='# END LOCAL PRAXIS OVERRIDE'

case "${1:-}" in
    patch)
        if [ ! -d ../praxis ]; then
            echo 'ERROR: ../praxis not found — clone Praxis core as a sibling directory first' >&2
            exit 1
        fi
        if grep -q "^$begin" "$manifest"; then
            echo "Already patched — run 'make unpatch-praxis' first" >&2
            exit 1
        fi

        pinned=0
        for crate in praxis-proxy-core praxis-proxy-filter praxis-proxy-protocol praxis-proxy-tls praxis-proxy; do
            if grep -q "^$crate = { path = " "$manifest"; then
                echo "Already patched: $crate has a path override" >&2
                exit 1
            fi
            if grep -q "^$crate = { git = " "$manifest"; then
                pinned=$((pinned + 1))
            fi
        done
        if [ "$pinned" -ne 0 ] && [ "$pinned" -ne 5 ]; then
            echo 'ERROR: refusing to replace a partial Praxis git patch table' >&2
            exit 1
        fi

        temporary=$(mktemp "${manifest}.XXXXXX")
        trap 'rm -f "$temporary"' EXIT HUP INT TERM
        if grep -q '^\[patch\.crates-io\]$' "$manifest"; then
            awk '/^praxis-proxy(-core|-filter|-protocol|-tls)? = \{ git = / { print "# " $0; next } { print }' "$manifest" > "$temporary"
            printf '%s existing-table\n' "$begin" >> "$temporary"
        else
            cat "$manifest" > "$temporary"
            printf '%s created-table\n[patch.crates-io]\n' "$begin" >> "$temporary"
        fi
        printf '%s\n' \
            'praxis-proxy-core = { path = "../praxis/core" }' \
            'praxis-proxy-filter = { path = "../praxis/filter" }' \
            'praxis-proxy-protocol = { path = "../praxis/protocol" }' \
            'praxis-proxy-tls = { path = "../praxis/tls" }' \
            'praxis-proxy = { path = "../praxis/server" }' \
            "$end" >> "$temporary"
        cat "$temporary" > "$manifest"
        echo 'Patched Cargo.toml to use ../praxis path dependencies'
        ;;
    unpatch)
        if ! grep -q "^$begin" "$manifest"; then
            echo 'Nothing to unpatch'
            exit 0
        fi
        if ! grep -q "^$end\$" "$manifest"; then
            echo 'ERROR: local Praxis override is missing its end marker' >&2
            exit 1
        fi

        temporary=$(mktemp "${manifest}.XXXXXX")
        trap 'rm -f "$temporary"' EXIT HUP INT TERM
        awk '
            /^# BEGIN LOCAL PRAXIS OVERRIDE / { local = 1; next }
            local && /^# END LOCAL PRAXIS OVERRIDE$/ { local = 0; next }
            local { next }
            /^# praxis-proxy(-core|-filter|-protocol|-tls)? = \{ git = / { sub(/^# /, "") }
            { print }
        ' "$manifest" > "$temporary"
        cat "$temporary" > "$manifest"
        echo 'Restored the committed Praxis dependency pins'
        ;;
    *)
        echo 'usage: sh scripts/patch-praxis.sh patch|unpatch' >&2
        exit 2
        ;;
esac
