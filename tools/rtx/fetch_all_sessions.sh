#!/usr/bin/env bash
# Before an analysis of WSL-run sessions (040 補記 14, 041 §0): list every session directory under the WSL research
# checkout's results/<subdir>, fetch the ones this repo does not have yet (tools/rtx/fetch_session.sh, without raw/),
# and write results/<subdir>/wsl-inventory.txt (the UTC time of the listing and every directory found) to commit with
# them, so the analysis demonstrably saw every attempt. Read-only on the remote; never touches the GPU or a lock.
#
# Usage (Mac): tools/rtx/fetch_all_sessions.sh <results subdir, e.g. stage_q | stage_e>
set -euo pipefail
SUB=${1:?results subdir}
[[ "$SUB" =~ ^[A-Za-z0-9._-]+$ && "$SUB" != *..* ]] || { echo "refusing: bad path"; exit 2; }
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LISTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
NAMES=$(ssh -o BatchMode=yes -o ConnectTimeout=30 gpu-server \
  "wsl.exe -d Ubuntu-22.04 -- bash -lc \"cd ~/llm-serving-lab/results/$SUB 2>/dev/null && find . -mindepth 2 -maxdepth 2 -name session.json -o -name plan.json | cut -d/ -f2 | sort -u\"" \
  | tr -d '\r') || { echo "listing failed; nothing fetched"; exit 4; }
mkdir -p "$REPO/results/$SUB"
for SID in $NAMES; do
  [[ "$SID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "refusing: odd directory name $SID"; exit 2; }
  if [ -e "$REPO/results/$SUB/$SID" ]; then
    echo "have $SID"
  else
    "$REPO/tools/rtx/fetch_session.sh" "$SUB" "$SID"
  fi
done
{ echo "# listed_at_utc $LISTED (tools/rtx/fetch_all_sessions.sh; every session directory on WSL)"; printf '%s\n' $NAMES; } \
  > "$REPO/results/$SUB/wsl-inventory.txt"
echo "wrote results/$SUB/wsl-inventory.txt ($(printf '%s\n' $NAMES | grep -c . || true) directories)"
