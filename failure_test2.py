import numpy as np, json, flyppg3 as G, flyppg as F
CF = {'coloured shared lighting': dict(il=1., ae=1., fl=1., tr=0., cs=1.), 'coloured face-only lighting': dict(il=1., ae=1., fl=1., tr=0., sh=0., cs=1.),
      'coloured + bg clutter': dict(il=1., ae=1., fl=1., tr=0., cs=1., cl=1.)}
ARMS = {'raw (green)': (lambda v: v, G.rppg), 'POS only': (lambda v: v, G.rppg_pos), 'surround scalar (green)': (G.bg_norm, G.rppg),
        'surround per-channel (green)': (G.bg_norm_pc, G.rppg), 'surround per-channel + POS': (G.bg_norm_pc, G.rppg_pos)}
out = {}
for c, pr in CF.items():
    res = {a: [] for a in ARMS}
    for k in range(8):
        v, hr, _ = G.make_video3(3000 + k, **pr)
        for a, (pre, ex) in ARMS.items(): res[a] += [est - hr for est, _ in F.window_est(ex(pre(v)), hr)]
    for a, e in res.items(): e = np.array(e); out[f'{c}|{a}'] = dict(MAE=float(np.abs(e).mean()), success=float((np.abs(e) <= 5).mean() * 100), n=len(e))
json.dump(out, open('results_failure2.json', 'w'), indent=1)
print(f"{'':32}" + ''.join(f'{a[:16]:>18}' for a in ARMS))
for c in CF: print(f'{c:32}' + ''.join(f"{out[f'{c}|{a}']['MAE']:11.2f}/{out[f'{c}|{a}']['success']:3.0f}%  " for a in ARMS))
