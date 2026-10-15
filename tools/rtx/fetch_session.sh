#!/usr/bin/env bash
# Copy one finished RTX session (without raw/) from the WSL research checkout into this repo. Read-only on the remote.
#
# Usage (Mac): tools/rtx/fetch_session.sh <results subdir, e.g. core_b | advanced | capacity> <session id>
# Refuses to overwrite an existing local session directory.
set -euo pipefail
SUB=${1:?results subdir}; SID=${2:?session id}
case "$SUB$SID" in *..*|*/|/*) echo "refusing: bad path"; exit 2;; esac
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DEST="$REPO/results/$SUB"
[ -e "$DEST/$SID" ] && { echo "refusing: $DEST/$SID already exists"; exit 3; }
mkdir -p "$DEST"
ssh -o BatchMode=yes -o ConnectTimeout=30 gpu-server \
  "wsl.exe -d Ubuntu-22.04 -- bash -lc \"cd ~/llm-serving-lab/results/$SUB && test -f $SID/session.json && tar -c --exclude=raw $SID\"" \
  2>/dev/null | tar -x -C "$DEST"
test -f "$DEST/$SID/session.json" || { echo "fetch failed (no session.json)"; exit 4; }
ls "$DEST/$SID"; du -sh "$DEST/$SID"
