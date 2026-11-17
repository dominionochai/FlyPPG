import numpy as np, json, flyppg3 as G, flyppg as F
CF = {'shared lighting (reference)': dict(il=1., ae=1., fl=1., tr=0.), 'face-only lighting (screen glow)': dict(il=1., ae=1., fl=1., tr=0., sh=0.),
      'partial coupling (50%)': dict(il=1., ae=1., fl=1., tr=0., sh=.5), 'background-only distractor': dict(tr=0., bgd=1.),
      'bg clutter + shared lighting': dict(il=1., ae=1., fl=1., tr=0., cl=1.), 'clean (no lighting issue)': dict(tr=0.)}
ARMS = {'raw (green)': (lambda v: v, G.rppg), 'POS chrominance only': (lambda v: v, G.rppg_pos), 'surround-norm (green)': (G.bg_norm, G.rppg),
        'surround-norm + POS': (G.bg_norm, G.rppg_pos), 'gated surround + POS': (G.surround_gated, G.rppg_pos), 'gated surround (green)': (G.surround_gated, G.rppg)}
n = 8; out = {}
for c, pr in CF.items():
    res = {a: [] for a in ARMS}
    for k in range(n):
        v, hr, _ = G.make_video3(3000 + k, **pr)
        for a, (pre, ex) in ARMS.items(): res[a] += [est - hr for est, _ in F.window_est(ex(pre(v)), hr)]
    for a, e in res.items(): e = np.array(e); out[f'{c}|{a}'] = dict(MAE=float(np.abs(e).mean()), success=float((np.abs(e) <= 5).mean() * 100), n=len(e))
    print(c, flush=True)
json.dump(out, open('results_failure.json', 'w'), indent=1)
print(f"{'':36}" + ''.join(f'{a[:15]:>16}' for a in ARMS))
for c in CF: print(f'{c:36}' + ''.join(f"{out[f'{c}|{a}']['MAE']:9.2f}/{out[f'{c}|{a}']['success']:3.0f}%  " for a in ARMS))
