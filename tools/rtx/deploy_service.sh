#!/usr/bin/env bash
# Deploy the committed HEAD as a new capserve release on the RTX WSL (decision 029 "WSL 的部署目錄"); product only.
#
# Usage (Mac, clean tracked tree): tools/rtx/deploy_service.sh
# 1. scripts/deploy_bundle.py builds the bundle (git archive HEAD minus results/, plus DEPLOYED_COMMIT and
#    DEPLOY_MANIFEST.json); it refuses uncommitted tracked changes. Its git comes from CommandLineTools (NEXT.md:
#    /usr/bin/git is broken on this Mac).
# 2. Under `flock -n /tmp/ai-portfolio-gpu.lock` (so it never overlaps a formal research session; it never waits for,
#    steals or removes the lock) the bundle is extracted into ~/llm-serving-service/releases/<sha> on WSL. It refuses
#    when that release directory exists (a release is never overwritten) or ~/llm-serving-service or its releases/
#    is a symlink (either could land the extraction inside the research checkout).
#    A partial extraction cannot start: `capserve.deploy up` verifies every file against DEPLOY_MANIFEST.json.
# 3. A read-only check that the release's DEPLOYED_COMMIT is <sha>.
# It never writes the research checkout ~/llm-serving-lab (its files, DEPLOYED_COMMIT or venv), installs nothing and
# starts nothing. Exit: 0 deployed, 1 not deployed on WSL (lock held, release exists, ...), 2 no bundle, 3 mismatch.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LOCK=/tmp/ai-portfolio-gpu.lock
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if ! out=$(PATH=/Library/Developer/CommandLineTools/usr/bin:$PATH \
             "$REPO/.venv/bin/python" "$REPO/scripts/deploy_bundle.py" --out "$TMP/svc.tar"); then
  echo "deploy_bundle refused; nothing deployed"
  exit 2
fi
SHA=$(printf '%s\n' "$out" | sed -n 's/.*"commit": *"\([0-9a-f]\{40\}\)".*/\1/p' | head -n 1)
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "no commit sha in the deploy_bundle output: $out"; exit 2; }
REL="llm-serving-service/releases/$SHA"

remote() {
  ssh -o BatchMode=yes -o ConnectTimeout=30 gpu-server "wsl.exe -d Ubuntu-22.04 -- bash -lc \"$1\""
}

if ! remote "test ! -L ~/llm-serving-service && test ! -L ~/llm-serving-service/releases && flock -n $LOCK sh -c 'test ! -e ~/$REL && mkdir -p ~/$REL && tar -x -C ~/$REL'" \
     < "$TMP/svc.tar"; then
  echo "not deployed: the GPU lock is held, ~/$REL already exists, ~/llm-serving-service or its releases/ is a"
  echo "symlink, or the extraction failed (a partial release is refused by capserve.deploy up's manifest check)"
  exit 1
fi
got=$(remote "cat ~/$REL/DEPLOYED_COMMIT" < /dev/null | tr -d '\r\n')
[ "$got" = "$SHA" ] || { echo "~/$REL/DEPLOYED_COMMIT is '$got', expected $SHA"; exit 3; }
echo "deployed $SHA to ~/$REL"
echo "next, under the GPU lock and between research sessions (capserve/itest.py):"
echo "  cd ~/$REL && flock -n $LOCK env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=service:benchmarks \\"
echo "    <research venv>/bin/python -m capserve.itest IT-0 IT-1 IT-2 IT-3 IT-4 IT-5 IT-6 --allow-provisional"
