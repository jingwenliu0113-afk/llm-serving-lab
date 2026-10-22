#!/usr/bin/env python3
"""041 v3 disclosure 8: Stage C formal-20261017-c1 (another session), M2K and U2K at 2K. Per (workload, C): the mean
total token throughput of the valid runs and the mean short-class P95 TTFT of their derived per-request files
(the derived cls field, with core_b_analysis.percentile), and the M2K KV peak at C 4. Descriptive only; run from the repo root."""

import json
import sys
from pathlib import Path

sys.path.insert(0, "benchmarks")
import core_b_analysis as ba  # noqa: E402

SESSION = Path("results/capacity/formal-20261017-c1")
rows = [json.loads(x) for x in (SESSION / "rows.jsonl").read_text().splitlines() if x.strip()]
out = {}
for r in rows:
    if r.get("workload") not in ("M2K", "U2K") or not r.get("valid") or r.get("phase") == "sentinel":
        continue
    reqs = [json.loads(x) for x in (SESSION / r["derived"]).read_text().splitlines() if x.strip()]
    p95 = ba.percentile([q["ttft_ms"] for q in reqs if q["cls"] == "short"], 95)   # as short_class_p95_ttft_ms
    out.setdefault((r["workload"], r["C"]), []).append(
        (r["summary"].get("total_token_throughput"), p95, (r.get("metrics") or {}).get("kv_cache_usage_peak")))
for (w, c), v in sorted(out.items()):
    thr = sum(x[0] for x in v) / len(v)
    p95s = [x[1] for x in v if x[1] is not None]
    print(f"{w} C {c}: n {len(v)}, total tok/s {thr:.0f}, short P95 TTFT {sum(p95s) / len(p95s):.0f} ms" if p95s else
          f"{w} C {c}: n {len(v)}, total tok/s {thr:.0f}", f"kv_peak max {max(x[2] or 0 for x in v):.3f}")
