#!/usr/bin/env bash
# Copy one committed session (without raw/) from this repo into the WSL research checkout, e.g. the Mac E3 pilot that
# the RTX formal plan checks (041 v3 §3.7). Writes only results/<subdir>/<id> on WSL; never touches the GPU or a lock.
#
# Usage (Mac): tools/rtx/push_session.sh <results subdir, e.g. stage_e> <session id>
# Refuses a session that is not committed here, has uncommitted changes, or already exists on WSL.
set -euo pipefail
SUB=${1:?results subdir}; SID=${2:?session id}
safe() { [[ "$1" =~ ^[A-Za-z0-9._-]+$ && "$1" != *..* ]]; }
safe "$SUB" && safe "$SID" || { echo "refusing: bad path"; exit 2; }
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SRC="$REPO/results/$SUB/$SID"
[ -f "$SRC/session.json" ] || { echo "refusing: no $SRC/session.json"; exit 2; }
git -C "$REPO" ls-files --error-unmatch "results/$SUB/$SID/session.json" >/dev/null 2>&1 \
  || { echo "refusing: results/$SUB/$SID is not committed"; exit 2; }
[ -z "$(git -C "$REPO" status --porcelain -- "results/$SUB/$SID")" ] \
  || { echo "refusing: results/$SUB/$SID has uncommitted changes"; exit 2; }
remote() {
  ssh -o BatchMode=yes -o ConnectTimeout=30 gpu-server "wsl.exe -d Ubuntu-22.04 -- bash -lc \"$1\""
}
remote "test ! -e ~/llm-serving-lab/results/$SUB/$SID" || { echo "refusing: results/$SUB/$SID exists on WSL"; exit 3; }
COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -c -C "$REPO/results/$SUB" --exclude=raw "$SID" \
  | remote "mkdir -p ~/llm-serving-lab/results/$SUB && tar -x -k -C ~/llm-serving-lab/results/$SUB"
remote "test -f ~/llm-serving-lab/results/$SUB/$SID/session.json" || { echo "push failed (no session.json)"; exit 4; }
echo "pushed results/$SUB/$SID to WSL"
