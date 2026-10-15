#!/usr/bin/env bash
# Read-only status of the shared RTX GPU lock and GPU compute processes (MACHINE_POLICY "GPU Hard Lock").
#
# Usage (Mac): tools/rtx/gpu_status.sh
# Prints one final line: FREE, LOCKED (someone holds /tmp/ai-portfolio-gpu.lock), or BUSY (a GPU compute process
# exists without the lock). Never takes, deletes or waits for the lock; never touches another process.
set -uo pipefail
ssh -o BatchMode=yes -o ConnectTimeout=20 gpu-server 'wsl.exe -d Ubuntu-22.04 -- bash -s' <<'REMOTE' 2>&1 | tr -d '\r'
L=/tmp/ai-portfolio-gpu.lock
held=no
if [ -e "$L" ]; then
  ino=$(stat -c %i "$L")
  awk -v ino="$ino" '{ split($6, a, ":"); if (a[3] == ino) found = 1 } END { exit !found }' /proc/locks && held=yes
fi
apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
used=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null)
echo "gpu: $used"
[ -n "$apps" ] && echo "compute apps: $apps"
if [ "$held" = yes ]; then echo LOCKED
elif [ -n "$apps" ]; then echo BUSY
else echo FREE
fi
REMOTE
