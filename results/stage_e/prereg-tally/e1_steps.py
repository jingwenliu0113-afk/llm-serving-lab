import json, glob, os, sys, bisect, statistics as st
ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
sys.path.insert(0, ROOT + "/benchmarks")
import c0_analysis as ca
res = {"slip": [], "normal": []}
for f in sorted(glob.glob(ROOT + "/results/advanced/formal-20261011-adv/derived/*-8192-uniform-r*-requests.jsonl")):
    tag = os.path.basename(f).replace("-requests.jsonl", "")
    cond = tag.split("-")[0]
    if cond not in ("B0", "G", "K3"): continue
    rows = sorted((json.loads(l) for l in open(f)), key=lambda r: r["idx"])
    tt = [r["client_ttft_ms"] for r in rows]
    eps = ca.classify_bench(tt, C=8, L=8192)["episodes"]
    sl = {e["k"] for e in eps}
    steps = [json.loads(l) for l in open(ROOT + f"/results/capacity/c0-step0/raw/formal-20261011-adv/{tag}.steps.jsonl")]
    ts = [s["t"] for s in steps]
    for g in range(1, 24):
        r0 = rows[8*g]
        a = r0["arrival"]; s = a + (r0["server_ttft_ms"] - r0["P_ms"]) / 1000
        i = bisect.bisect_right(ts, a)  # first step stamped after arrival
        prev = ts[i-1]
        # steps until the first step with ctx_tok >= 1000 after arrival
        seq = []
        j = i
        while j < len(steps) and len(seq) < 8:
            seq.append((round((ts[j]-a)*1000,1), steps[j]["ctx_req"], steps[j]["ctx_tok"], steps[j]["gen_req"]))
            if steps[j]["ctx_tok"] >= 1000: break
            j += 1
        n_dec = sum(1 for x in seq if x[2] < 1000)
        kind = "slip" if g in sl else ("prec" if g+1 in sl else "normal")
        rec = (tag, g, round((a-prev)*1000,1), round((ts[i-2] if i>=2 else 0) and (a-ts[i-2])*1000,1), n_dec, round(r0["F_ms"],1), round(r0["Q_ms"],1), seq)
        if kind == "slip": res["slip"].append(rec)
        elif kind == "normal": res["normal"].append(rec)
        if kind != "normal" or g in (5,):
            print(kind, rec)
from collections import Counter
print("n_dec slip", Counter(r[4] for r in res["slip"]), "normal", Counter(r[4] for r in res["normal"]))
print("phase a-t_prev slip", sorted(r[2] for r in res["slip"]))
ph = sorted(r[2] for r in res["normal"]); print("normal phase min/med/max", ph[0], st.median(ph), ph[-1], len(ph))
ph2 = sorted(r[3] for r in res["normal"]); print("normal a-t_{prev-1} min/med/max", ph2[0], st.median(ph2), ph2[-1]); print("slip a - t_prev-1", sorted(r[3] for r in res["slip"]))
print("F slip", sorted(r[5] for r in res["slip"]), "F normal range", min(r[5] for r in res["normal"]), max(r[5] for r in res["normal"]))
