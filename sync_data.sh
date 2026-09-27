#!/bin/bash
# Sync the cached data (data/) through a cloud remote (e.g. Nextcloud) with rclone.
#
# NERSC is the source of truth: push from NERSC after rebuilding the cache,
# pull everywhere else.
#
#   ./sync_data.sh push [rclone args...]            mirror data/ to the remote (NERSC only)
#   ./sync_data.sh pull [--delete] [rclone args...]  update data/ from the remote
#
# pull only adds/updates files; --delete also removes local files that are no
# longer on the remote. Extra arguments go to rclone, e.g. --dry-run.
#
# Environment:
#   DESI_MC_RCLONE_REMOTE  rclone remote path (default: nextcloud:desi_mc_data)
#   DESI_MC_DATA_DIR       local data dir (default: <repo>/data, as in env.py)
#
# One-time setup: install rclone and create the remote with `rclone config`
# (see README, "Syncing the cached data").

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="${DESI_MC_RCLONE_REMOTE:-nextcloud:desi_mc_data}"
DATA_DIR="${DESI_MC_DATA_DIR:-$REPO_ROOT/data}"

usage() { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

[ $# -ge 1 ] || usage 1
mode="$1"; shift
case "$mode" in -h|--help) usage ;; esac

if ! command -v rclone >/dev/null 2>&1; then
    echo "Error: rclone not found. Install it (https://rclone.org/install/) and run 'rclone config'." >&2
    exit 1
fi

progress=()
[ -t 1 ] && progress=(--progress)

case "$mode" in
    push)
        if [ -z "${NERSC_HOST:-}" ]; then
            echo "Error: push is meant to run on NERSC, where the cache is built." >&2
            echo "Pushing from elsewhere could overwrite newer data on the remote." >&2
            exit 1
        fi
        # rclone sync mirrors deletions, so an empty/missing source would wipe the remote.
        for sub in cutsky cubic; do
            if [ -z "$(ls -A "$DATA_DIR/$sub" 2>/dev/null)" ]; then
                echo "Error: $DATA_DIR/$sub is missing or empty; refusing to push." >&2
                exit 1
            fi
        done
        echo "Pushing $DATA_DIR -> $REMOTE"
        rclone sync "$DATA_DIR" "$REMOTE" ${progress[@]+"${progress[@]}"} "$@"
        ;;
    pull)
        cmd=copy
        if [ "${1:-}" = "--delete" ]; then
            cmd=sync; shift
        fi
        mkdir -p "$DATA_DIR"
        echo "Pulling $REMOTE -> $DATA_DIR ($cmd)"
        rclone "$cmd" "$REMOTE" "$DATA_DIR" ${progress[@]+"${progress[@]}"} "$@"
        ;;
    *)
        echo "Error: unknown mode '$mode'." >&2
        usage 1
        ;;
esac
