#!/usr/bin/env bash
# Start capserve as a resident service on the RTX WSL (decision 038 §7): the production config, its pinned verified
# profile, `capserve.deploy up --resident` (no max_hold_s; restarts after a failure, gives up after max_restarts).
#
# ONLY with the user's consent and never while research needs the GPU (user decision, 037): a resident service holds
# the shared GPU lock, and about 15 GB of VRAM, until `capserve.deploy stop`. It never waits for, steals or removes the
# lock: when the lock is held it exits 1 at once and starts nothing.
#
# Usage (WSL, release already deployed by tools/rtx/deploy_service.sh):
#   ~/llm-serving-service/releases/<sha>/tools/rtx/capserve_resident.sh <sha>
# Stop it (WSL):
#   cd ~/llm-serving-service/releases/<sha> && PYTHONPATH=service:benchmarks ~/llm-serving-lab/.venv/bin/python \
#       -m capserve.deploy stop --config service/deploy/rtx-verified.toml
# Start at Windows logon: registering a Task Scheduler task that runs
#   wsl.exe -d Ubuntu-22.04 -- bash -lc "~/llm-serving-service/releases/<sha>/tools/rtx/capserve_resident.sh <sha>"
# is a Windows system setting: the user does it, never this project.
set -euo pipefail
SHA=${1:?usage: capserve_resident.sh <release sha>}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "not a release sha: $SHA"; exit 2; }
REL="$HOME/llm-serving-service/releases/$SHA"
cd "$REL"
exec flock -n /tmp/ai-portfolio-gpu.lock env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=service:benchmarks \
    "$HOME/llm-serving-lab/.venv/bin/python" -m capserve.deploy up --resident \
    --config service/deploy/rtx-verified.toml
