#!/usr/bin/env bash
# Copy an explicit list of raw/ files of one RTX session from the WSL research checkout into this repo's git-ignored
# results/capacity/c0-step0/raw/<session id>/ (decision 028 step 0). Read-only on the remote (ls, tar -c); never
# touches the GPU lock.
#
# Usage (Mac): tools/rtx/fetch_raw.sh <results subdir, e.g. core_b | advanced> <session id> <raw file>...
#   e.g. tools/rtx/fetch_raw.sh core_b formal-20261009-coreb-uniform \
#            $(.venv/bin/python benchmarks/c0_step0.py --list-files core)
# 1. every name must be in the session's committed raw_manifest.json (exit 2) and must not exist locally (exit 3);
# 2. a read-only `ls` confirms every file exists in ~/llm-serving-lab/results/<subdir>/<sid>/raw/ (exit 4, nothing
#    copied);
# 3. the files are copied (tar -c on the remote, extracted without overwriting) and each sha256 is checked against
#    raw_manifest.json; a missing or mismatching copy fails (exit 5; a mismatch is kept as <file>.sha256-mismatch).
set -euo pipefail
SUB=${1:?results subdir}; SID=${2:?session id}; shift 2
safe() { [[ "$1" =~ ^[A-Za-z0-9._-]+$ && "$1" != *..* ]]; }
safe "$SUB" && safe "$SID" || { echo "refusing: bad path"; exit 2; }
[ $# -gt 0 ] || { echo "refusing: no raw file listed"; exit 2; }
for f in "$@"; do safe "$f" || { echo "refusing: bad file name $f"; exit 2; }; done
REPO=$(cd "$(dirname "$0")/../.." && pwd)
MANIFEST="$REPO/results/$SUB/$SID/raw_manifest.json"
DEST="$REPO/results/capacity/c0-step0/raw/$SID"
[ -f "$MANIFEST" ] || { echo "refusing: no committed $MANIFEST"; exit 2; }
# run_core_a.write_raw_manifest writes json.dumps(indent=1): the sha256 line follows its "raw/<file>": { line
want_sha() { grep -F -A1 "\"raw/$1\": {" "$MANIFEST" | grep -oE '[0-9a-f]{64}' | head -n 1 || true; }
for f in "$@"; do
  [ -n "$(want_sha "$f")" ] || { echo "refusing: raw/$f is not in $MANIFEST"; exit 2; }
  [ -e "$DEST/$f" ] && { echo "refusing: $DEST/$f already exists"; exit 3; }
done
remote() {
  ssh -o BatchMode=yes -o ConnectTimeout=30 gpu-server \
    "wsl.exe -d Ubuntu-22.04 -- bash -lc \"cd ~/llm-serving-lab/results/$SUB/$SID/raw && $1 -- ${*:2}\""
}
remote ls "$@" >/dev/null || { echo "raw archive incomplete on WSL (ls failed); nothing fetched"; exit 4; }
mkdir -p "$DEST"
remote "tar -c" "$@" 2>/dev/null | tar -x -k -C "$DEST" || echo "the copy pipeline failed; verifying what arrived"
bad=0
for f in "$@"; do
  if [ ! -f "$DEST/$f" ]; then
    echo "missing after fetch: $f"; bad=1
  elif [ "$(shasum -a 256 "$DEST/$f" | cut -d' ' -f1)" != "$(want_sha "$f")" ]; then
    mv "$DEST/$f" "$DEST/$f.sha256-mismatch"; echo "sha256 mismatch: $f (kept as $f.sha256-mismatch)"; bad=1
  fi
done
[ "$bad" = 0 ] || exit 5
echo "fetched $# file(s) into $DEST; sha256 verified against $MANIFEST"
