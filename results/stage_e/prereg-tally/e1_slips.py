import json, glob, os, sys, statistics as st
ROOT = "/Users/liujingwen/Desktop/AI-Portfolio/llm-serving-lab"
sys.path.insert(0, ROOT + "/benchmarks")
import c0_analysis as ca

C = 8
def med(x): return st.median(x)

def groups_dev(lat, eps, label):
    """d_j(g) = lat[8g+j]-lat[8g]; dev = d - per-position median over groups 1..last-1 excluding slip/precursor groups."""
    n = len(lat); G = n // C
    d = {g: [lat[8*g+j]-lat[8*g] for j in range(1, 8)] for g in range(G)}
    slipk = {e["k"] for e in eps}
    prec = {k-1 for k in slipk}
    normal = [g for g in range(1, G-1) if g not in slipk and g not in prec]
    base = [med([d[g][j] for g in normal]) for j in range(7)]
    dev = {g: [d[g][j]-base[j] for j in range(7)] for g in range(G)}
    nm = max(max(abs(v) for v in dev[g]) for g in normal)
    out = []
    for k in sorted(slipk):
        out.append((k, [round(min(dev[k-1]),1), round(max(dev[k-1]),1)], [round(min(dev[k-2]),1), round(max(dev[k-2]),1)] if k-2>=0 else None,
                    [round(min(dev[k]),1), round(max(dev[k]),1)]))
    return nm, out, len(normal)

# C0
print("== C0 explore-c0-20261016")
allst = []
nm_all = []
for f in sorted(glob.glob(ROOT + "/results/capacity/explore-c0-20261016/derived/b*-n*-requests.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    rows.sort(key=lambda r: r["idx"])
    tt = [r["ttft_ms"] for r in rows]; lat = [r["latency_ms"] for r in rows]
    r = ca.classify_bench(tt, C=8, L=8192)
    eps = r["episodes"]
    nm, out, nn = groups_dev(lat, eps, f)
    nm_all.append(nm)
    for e in eps: allst.append(("C0", os.path.basename(f), e["start_idx"], e["length"]))
    if eps: print(os.path.basename(f), [(e["start_idx"], e["length"]) for e in eps], "normal max |dev|", round(nm,1), out)
print("C0 normal max |dev| over benches", round(max(nm_all),1))

# Core C8 raw
print("== Core C8 raw")
for r_ in (1,2,3):
    d = json.load(open(ROOT + f"/results/capacity/c0-step0/raw/formal-20261009-coreb-uniform/bench-uniform-8192-r{r_}-c8.json"))
    tt = [x*1000 for x in d["ttfts"]]; lat = [x*1000 for x in d["latencies"]]; stt = d["start_times"]
    r = ca.classify_bench(tt, C=8, L=8192)
    eps = r["episodes"]
    nm, out, nn = groups_dev(lat, eps, r_)
    comp = [stt[i] + lat[i]/1000 for i in range(len(lat))]
    # send lag: start[i+8] - comp[i]  (steady state, closed loop) -> actual releaser = earliest unconsumed completion
    lags = []
    import heapq
    order = sorted(range(len(stt)), key=lambda i: stt[i])
    for i in order[8:]:
        pass
    print("r", r_, [(e["start_idx"], e["length"]) for e in eps], "normal max |dev|", round(nm,1), out)
    for e in eps: allst.append(("Core", r_, e["start_idx"], e["length"]))
    gaps = []
    for g in range(1, len(stt)//8):
        gaps.append((g, round((stt[8*g+1]-stt[8*g])*1000,1)))
    sl = {e["k"] for e in eps}
    print("   slip gaps", [x for x in gaps if x[0] in sl], "normal range", min(x[1] for x in gaps if x[0] not in sl), max(x[1] for x in gaps if x[0] not in sl))

# Advanced
print("== Advanced 8K uniform B0/G/K3")
for f in sorted(glob.glob(ROOT + "/results/advanced/formal-20261011-adv/derived/*-8192-uniform-r*-requests.jsonl")):
    cond = os.path.basename(f).split("-")[0]
    if cond not in ("B0", "G", "K3"): continue
    rows = [json.loads(l) for l in open(f)]; rows.sort(key=lambda r: r["idx"])
    tt = [r["client_ttft_ms"] for r in rows]
    r = ca.classify_bench(tt, C=8, L=8192)
    eps = r["episodes"]
    arr = [r_["arrival"] for r_ in rows]
    sl = {e["k"] for e in eps}
    gaps = [(g, round((arr[8*g+1]-arr[8*g])*1000,1)) for g in range(1, len(arr)//8)]
    for e in eps: allst.append(("Adv", os.path.basename(f), e["start_idx"], e["length"]))
    print(os.path.basename(f), [(e["start_idx"], e["length"]) for e in eps], "slip gaps", [x for x in gaps if x[0] in sl],
          "normal", (min(x[1] for x in gaps if x[0] not in sl), max(x[1] for x in gaps if x[0] not in sl)))

from collections import Counter
print(len(allst), Counter(x[2] for x in allst))
print(Counter((x[0]) for x in allst))
