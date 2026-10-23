#!/usr/bin/env bash
# Copy E1-P's seed check record (decision 041 v3 §1.7) from the WSL research checkout into this repo. Read-only on the
# remote; refuses to overwrite a local record; checks the copy's sha256 against the remote file's. Run it only while
# tools/rtx/gpu_status.sh prints FREE (it reads WSL's disk).
#
# Usage (Mac): tools/rtx/fetch_seed_check.sh
set -euo pipefail
REL=results/stage_e/e1p-seed-check/seed_check.json
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DEST="$REPO/$REL"
[ -e "$DEST" ] && { echo "refusing: $DEST already exists"; exit 3; }
REMOTE=$(ssh -o BatchMode=yes -o ConnectTimeout=30 gpu-server \
  "wsl.exe -d Ubuntu-22.04 -- bash -lc \"cd ~/llm-serving-lab && sha256sum $REL\"" 2>/dev/null | tr -d '\r' | cut -d' ' -f1)
[ ${#REMOTE} -eq 64 ] || { echo "no seed check on WSL ($REL)"; exit 4; }
mkdir -p "$(dirname "$DEST")"
TMP="$DEST.tmp"
ssh -o BatchMode=yes -o ConnectTimeout=30 gpu-server \
  "wsl.exe -d Ubuntu-22.04 -- bash -lc \"cd ~/llm-serving-lab && cat $REL\"" 2>/dev/null > "$TMP"
LOCAL=$(shasum -a 256 "$TMP" | cut -d' ' -f1)
[ "$LOCAL" = "$REMOTE" ] || { rm -f "$TMP"; echo "sha256 mismatch: local $LOCAL, remote $REMOTE"; exit 5; }
mv "$TMP" "$DEST"
echo "$REL $LOCAL"
