import numpy as np, json, flyppg3 as G, flyppg as F
from scipy import ndimage as ndi
def noise_sigma(v):   # per-clip temporal noise estimate (MAD of second temporal difference on the face ROI)
    x = v[:, 24:72, 24:72, 1]; d2 = x[2:] - 2 * x[1:-1] + x[:-2]; return 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6)
def lamina_pool(v):   # light-adaptive pooling: the dimmer/noisier, the wider the spatial summation and the longer the temporal integration
    sn = noise_sigma(v); ss = float(np.clip((sn - .004) / .006 * 1.5, 0, 3.5)); tw = 1 + int(round(2 * np.clip(sn / .02, 0, 1)))
    o = ndi.gaussian_filter(v, (0, ss, ss, 0)) if ss > .1 else v
    return ndi.uniform_filter1d(o, tw, axis=0) if tw > 1 else o
CF = {'JPEG q30': dict(tr=0., jpegq=30), 'JPEG q15': dict(tr=0., jpegq=15), 'dim (noise .02)': dict(tr=0., noise=.02), 'very dim (noise .04)': dict(tr=0., noise=.04),
      'JPEG q30 + dim': dict(tr=0., jpegq=30, noise=.02), 'JPEG q15 + very dim': dict(tr=0., jpegq=15, noise=.04)}
ARMS = {'raw': lambda v: v, 'conventional': F.pre_conventional, 'lamina adaptive pooling': lamina_pool, 'temporal mean(5) only': lambda v: ndi.uniform_filter1d(v, 5, axis=0)}
EX = {'POS': G.rppg_pos, 'green': G.rppg}
out = {}
for c, pr in CF.items():
    res = {(a, e): [] for a in ARMS for e in EX}
    for k in range(6):
        v, hr, _ = G.make_video3(4000 + k, **pr)
        for a, fn in ARMS.items():
            p = fn(v)
            for e, ex in EX.items(): res[(a, e)] += [est - hr for est, _ in F.window_est(ex(p), hr)]
    for (a, e), er in res.items(): er = np.array(er); out[f'{c}|{a}|{e}'] = dict(MAE=float(np.abs(er).mean()), success=float((np.abs(er) <= 5).mean() * 100))
    print(c, flush=True)
json.dump(out, open('results_lowlight.json', 'w'), indent=1)
for e in EX:
    print('\nextractor', e, ' MAE bpm / success%'); print(f"{'':24}" + ''.join(f'{a[:22]:>26}' for a in ARMS))
    for c in CF: print(f'{c:24}' + ''.join(f"{out[f'{c}|{a}|{e}']['MAE']:17.2f}/{out[f'{c}|{a}|{e}']['success']:3.0f}%  " for a in ARMS))
