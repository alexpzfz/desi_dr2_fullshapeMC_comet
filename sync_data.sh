#!/bin/bash
# Sync the cached data (data/) and the chains (outputs/chains/) with NERSC over rsync.
#
# Run this on the other machine, not on NERSC. NERSC is the source of truth
# for the data; chains are run on both machines and synced both ways.
#
#   ./sync_data.sh pull [data|chains|all] [--delete] [rsync args...]
#   ./sync_data.sh push chains [rsync args...]
#
# pull defaults to data. pull only adds/updates files; --delete (data only)
# also removes local files that are gone from NERSC. Chains are never deleted
# on either side, and a file is not overwritten if the copy being replaced is
# newer. Nautilus snapshots (*_snap.hdf5, *_nautilus.hdf5) are not synced.
# Extra arguments go to rsync, e.g. --dry-run.
#
# Environment:
#   DESI_MC_NERSC_HOST  ssh host for NERSC (default: dtn01.nersc.gov; set the
#                       user and key in ~/.ssh/config)
#   DESI_MC_NERSC_REPO  this repo's path on NERSC
#                       (default: /global/homes/a/alexpzfz/desi_dr2_fullshapeMC_comet)
#   DESI_MC_DATA_DIR    local data dir (default: <repo>/data, as in env.py)
#
# NERSC needs a fresh sshproxy key (valid 24 h) before syncing; see README,
# "Syncing data and chains".

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${DESI_MC_NERSC_HOST:-dtn01.nersc.gov}"
NERSC_REPO="${DESI_MC_NERSC_REPO:-/global/homes/a/alexpzfz/desi_dr2_fullshapeMC_comet}"
DATA_DIR="${DESI_MC_DATA_DIR:-$REPO_ROOT/data}"
CHAINS_DIR="$REPO_ROOT/outputs/chains"

usage() { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

[ $# -ge 1 ] || usage 1
mode="$1"; shift
case "$mode" in -h|--help) usage ;; esac

if [ -n "${NERSC_HOST:-}" ]; then
    echo "Error: run this on the other machine; NERSC is the remote end." >&2
    exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
    echo "Error: rsync not found." >&2
    exit 1
fi

# The trailing slashes make rsync follow the data/ and outputs/chains/
# symlinks on NERSC. Incomplete transfers wait in .rsync-partial/ so a
# half-copied chain never shows up under its real name. Relative symlinks
# inside the tree are kept as links; absolute ones (which would dangle on
# the other side) are replaced by a copy of their target.
opts=(-ah --partial-dir=.rsync-partial --copy-unsafe-links)
[ -t 1 ] && opts+=(--info=progress2)
chain_opts=(--update --exclude='*_snap.hdf5' --exclude='*_nautilus.hdf5' --exclude='.ipynb_checkpoints/')

pull_data() {
    mkdir -p "$DATA_DIR"
    echo "Pulling $HOST:$NERSC_REPO/data -> $DATA_DIR"
    rsync "${opts[@]}" "$@" "$HOST:$NERSC_REPO/data/" "$DATA_DIR/"
}

pull_chains() {
    mkdir -p "$CHAINS_DIR"
    echo "Pulling $HOST:$NERSC_REPO/outputs/chains -> $CHAINS_DIR"
    rsync "${opts[@]}" "${chain_opts[@]}" "$@" "$HOST:$NERSC_REPO/outputs/chains/" "$CHAINS_DIR/"
}

case "$mode" in
    pull)
        what=data
        case "${1:-}" in data|chains|all) what="$1"; shift ;; esac
        delete=()
        if [ "${1:-}" = "--delete" ]; then
            if [ "$what" != data ]; then
                echo "Error: --delete only applies to data; chains are never deleted." >&2
                exit 1
            fi
            delete=(--delete); shift
        fi
        case "$what" in
            data)   pull_data ${delete[@]+"${delete[@]}"} "$@" ;;
            chains) pull_chains "$@" ;;
            all)    pull_data "$@"; pull_chains "$@" ;;
        esac
        ;;
    push)
        if [ "${1:-}" != chains ]; then
            echo "Error: only chains can be pushed; the data is built on NERSC." >&2
            exit 1
        fi
        shift
        if [ ! -d "$CHAINS_DIR" ]; then
            echo "Error: $CHAINS_DIR does not exist; nothing to push." >&2
            exit 1
        fi
        echo "Pushing $CHAINS_DIR -> $HOST:$NERSC_REPO/outputs/chains"
        rsync "${opts[@]}" "${chain_opts[@]}" "$@" "$CHAINS_DIR/" "$HOST:$NERSC_REPO/outputs/chains/"
        ;;
    *)
        echo "Error: unknown mode '$mode'." >&2
        usage 1
        ;;
esac
