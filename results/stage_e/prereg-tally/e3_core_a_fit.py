import json, math, sys
ROOT = "/Users/liujingwen/Desktop/AI-Portfolio/llm-serving-lab"
sys.path.insert(0, ROOT + "/benchmarks")
import core_a_analysis as an
files = {'4B-Q4_K_M': 2708804544, '4B-Q6_K': 3464055744, '4B-Q8_0': 4482403264,
         '9B-Q4_K_M': 5629109216, '9B-Q6_K': 7359259616, '9B-Q8_0': 9527501792}

def ols(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs)
    b = sum((x-mx)*(y-my) for x, y in zip(xs, ys))/sxx; a = my-b*mx
    res = [y-(a+b*x) for x, y in zip(xs, ys)]
    s = math.sqrt(sum(r*r for r in res)/(n-2)) if n > 2 else float('nan')
    return a, b, s, res, sxx, mx

for p in ['results/core_a/macos-metal/formal-20261009-a2/analysis.json', 'results/core_a/wsl2-cuda/formal-20261009-rtx-a1/analysis.json']:
    A = json.load(open(ROOT + '/' + p)); sub = A['subtracted_embd_bytes']
    for L in (2048,):
        d = {}
        for k in files:
            c = A['cells'][f'{k}-{L}']['decode']
            d[k] = (an.decode_bytes(files[k], sub[k], L)/1e9, 1000/c['m'], 1000*c['se']/c['m']**2)
        keys = sorted(d)
        a, b, s, res, sxx, mx = ols([d[k][0] for k in keys], [d[k][1] for k in keys])
        seb = s/math.sqrt(sxx); sea = s*math.sqrt(1/6 + mx**2/sxx)
        print(p.split('/')[2], L, 'global a=%.3f (SE %.3f) b=%.3f (SE %.3f) s=%.3f' % (a, sea, b, seb, s))
        print('  cell SE(t) ms:', {k: round(d[k][2], 3) for k in keys})
        print('  residuals ms:', {k: round(r, 3) for k, r in zip(keys, res)})
        # within-size fits
        for sz in ('4B', '9B'):
            ks = [k for k in keys if k.startswith(sz)]
            aa, bb, ss, rr, _, _ = ols([d[k][0] for k in ks], [d[k][1] for k in ks])
            print('  within', sz, 'a=%.3f b=%.3f s=%.3f' % (aa, bb, ss))
        # cross-size per format
        for q in ('Q4_K_M', 'Q6_K', 'Q8_0'):
            x4, y4 = d['4B-'+q][:2]; x9, y9 = d['9B-'+q][:2]
            bq = (y9-y4)/(x9-x4); aq = y4-bq*x4
            print('  cross-size', q, 'b=%.3f a=%.3f' % (bq, aq))
        # ANCOVA: size intercepts + common slope
        xs4 = [d[k][0] for k in keys if k.startswith('4B')]; ys4 = [d[k][1] for k in keys if k.startswith('4B')]
        xs9 = [d[k][0] for k in keys if k.startswith('9B')]; ys9 = [d[k][1] for k in keys if k.startswith('9B')]
        m4x = sum(xs4)/3; m4y = sum(ys4)/3; m9x = sum(xs9)/3; m9y = sum(ys9)/3
        sxx_w = sum((x-m4x)**2 for x in xs4) + sum((x-m9x)**2 for x in xs9)
        sxy_w = sum((x-m4x)*(y-m4y) for x, y in zip(xs4, ys4)) + sum((x-m9x)*(y-m9y) for x, y in zip(xs9, ys9))
        bw = sxy_w/sxx_w; a4 = m4y-bw*m4x; a9 = m9y-bw*m9x
        rw = [y-(a4+bw*x) for x, y in zip(xs4, ys4)] + [y-(a9+bw*x) for x, y in zip(xs9, ys9)]
        sw = math.sqrt(sum(r*r for r in rw)/3)
        P4, P9 = 8424393664/2, 17920697312/2
        a0 = (P9*a4-P4*a9)/(P9-P4); c = (a9-a4)/(P9-P4)*1e9
        print('  ANCOVA common b=%.3f a4=%.3f a9=%.3f s=%.3f -> fixed a0=%.3f, per-Gparam c=%.3f ms' % (bw, a4, a9, sw, a0, c))
        # predicted ratio for the H-A2 pair under global fit vs bytes ratio
        x1, y1 = d['4B-Q4_K_M'][:2]; x2, y2 = d['9B-Q8_0'][:2]
        print('  H-A2 pair obs ratio %.3f bytes ratio %.3f affine-pred ratio %.3f' % (y2/y1, x2/x1, (a+b*x2)/(a+b*x1)))
        # estimated new configs
        for name, fb, eb in (('9B-Q4_0', 5313356768, 572129280), ('4B-Q4_0', 2560000000, 0), ('4B-Q5_K_M', 3074986944, 0)):
            x0 = an.decode_bytes(fb, eb, L)/1e9
            print('  est', name, 'x0=%.3f yhat=%.3f' % (x0, a+b*x0))
        # same-size k-quant control diffs
        r = dict(zip(keys, res))
        print('  control |r(Q6_K)-r(Q4_K_M)| 4B %.3f 9B %.3f' % (abs(r['4B-Q6_K']-r['4B-Q4_K_M']), abs(r['9B-Q6_K']-r['9B-Q4_K_M'])))
        print('  Q8_0 - Q4_K_M residual diffs 4B %.3f 9B %.3f' % (r['4B-Q8_0']-r['4B-Q4_K_M'], r['9B-Q8_0']-r['9B-Q4_K_M']))

print("=== leave-pair-out phi")
for p in ['results/core_a/macos-metal/formal-20261009-a2/analysis.json', 'results/core_a/wsl2-cuda/formal-20261009-rtx-a1/analysis.json']:
    A = json.load(open(ROOT + '/' + p)); sub = A['subtracted_embd_bytes']; L = 2048
    d = {k: (an.decode_bytes(files[k], sub[k], L)/1e9, 1000/A['cells'][f'{k}-{L}']['decode']['m']) for k in files}
    pair = ('4B-Q4_K_M', '9B-Q8_0')
    rest = [k for k in d if k not in pair]
    a, b, s, res, sxx, mx = ols([d[k][0] for k in rest], [d[k][1] for k in rest])
    x1, y1 = d[pair[0]]; x2, y2 = d[pair[1]]
    Rb = x2/x1; Ro = y2/y1; Ra = (a+b*x2)/(a+b*x1)
    print(p.split('/')[2], 'LPO a=%.3f b=%.3f' % (a, b), 'Rbytes %.3f Robs %.3f Raff %.3f phi %.3f' % (Rb, Ro, Ra, (Rb-Ra)/(Rb-Ro)))
    # all 9 cross-size pairs: deviation share explained by in-sample fit
    a6, b6, s6, res6, sxx6, mx6 = ols([d[k][0] for k in sorted(d)], [d[k][1] for k in sorted(d)])
    for k4 in [k for k in d if k.startswith('4B')]:
        for k9 in [k for k in d if k.startswith('9B')]:
            x1, y1 = d[k4]; x2, y2 = d[k9]
            Rb = x2/x1; Ro = y2/y1; Ra = (a6+b6*x2)/(a6+b6*x1)
            print('   ', k4, k9, 'Rb %.3f Ro %.3f Ra %.3f dev %+.1f%% phi %.2f' % (Rb, Ro, Ra, 100*(Ro/Rb-1), (Rb-Ra)/(Rb-Ro) if Rb != Ro else float('nan')))

print("=== cross-size intercepts with SE")
for p in ['results/core_a/macos-metal/formal-20261009-a2/analysis.json', 'results/core_a/wsl2-cuda/formal-20261009-rtx-a1/analysis.json']:
    A = json.load(open(ROOT + '/' + p)); sub = A['subtracted_embd_bytes']; L = 2048
    d = {}
    for k in files:
        c = A['cells'][f'{k}-{L}']['decode']
        d[k] = (an.decode_bytes(files[k], sub[k], L)/1e9, 1000/c['m'], 1000*c['se']/c['m']**2)
    for q in ('Q8_0', 'Q6_K', 'Q4_K_M'):
        x4, t4, s4 = d['4B-'+q]; x9, t9, s9 = d['9B-'+q]
        af = (x9*t4 - x4*t9)/(x9-x4); se = math.sqrt((x9*s4)**2 + (x4*s9)**2)/(x9-x4)
        print(p.split('/')[2], q, 'x4=%.3f x9=%.3f a_f=%.3f SE=%.3f a/SE=%.1f' % (x4, x9, af, se, af/se))
