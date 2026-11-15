import numpy as np, json, sys, os, pickle, time, tracemalloc
from scipy import ndimage as ndi
import flyppg as F, readout as R, readout4 as R4, eval2 as E
K, LAM = 6, 1e-5
d = np.load('tune_feats_grid.npz'); Xt, Yt = d['X'], d['Y']
def taps(X, k=K): return np.concatenate([np.roll(X, j, 0) for j in range(-k, k + 1)], 1)
A = np.vstack([np.c_[taps(x), np.ones(len(x))] for x in Xt]); B = np.vstack(Yt)
W2 = np.linalg.solve(A.T @ A + LAM * np.trace(A.T @ A) / A.shape[1] * np.eye(A.shape[1]), A.T @ B)
np.save('readout_W2.npy', W2)
_c = {}
def fly_motion2(v):
    k = id(v)
    if k not in _c: _c.clear(); _c[k] = np.c_[taps(R4.feats_grid(v)), np.ones(len(v))] @ W2
    return _c[k]
def stab_fly2(v):
    pos = np.cumsum(fly_motion2(v), 0); pos = pos - ndi.gaussian_filter1d(pos, 30, axis=0, mode='nearest')
    return E.apply_shift(v, pos)
PRE = {'raw': lambda v: v, 'conventional': F.pre_conventional,
       'classical stab+step': lambda v: F.step_comp(E.stab_classical(v)),
       'fly step only': F.step_comp,
       'FlyPPG v2 (step+T4/T5 grid)': lambda v: F.step_comp(stab_fly2(v))}
if __name__ == '__main__':
    n = int(sys.argv[1]); res = pickle.load(open('partial3.pkl', 'rb')) if os.path.exists('partial3.pkl') else {}
    for c in E.CONDS:
        if any(k[0] == c for k in res): continue
        vids = [E.gen(1000 + k, c) for k in range(n)]
        for a in PRE: res[(c, a)] = dict(err=[], snr=[])
        for v, hr in vids:
            for a, fn in PRE.items():
                p = fn(v)
                for est, snr in F.window_est(F.rppg_proxy(p), hr): res[(c, a)]['err'].append(est - hr); res[(c, a)]['snr'].append(snr)
        pickle.dump(res, open('partial3.pkl', 'wb')); print(c, 'done', flush=True)
    out = {}
    for (c, a), r in res.items():
        e, s = np.array(r['err']), np.array(r['snr']); g = s > 0
        out[f'{c}|{a}'] = dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())), SNR=float(s.mean()), success=float((np.abs(e) <= 5).mean() * 100),
                               reported=float(g.mean() * 100), MAE_reported=float(np.abs(e[g]).mean()) if g.any() else float('nan'))
    v, _ = E.gen(7000, 'motion+illum'); lat = {}
    for a, fn in PRE.items():
        _c.clear(); E._cache.clear(); t0 = time.perf_counter(); fn(v); lat[a] = (time.perf_counter() - t0) / len(v) * 1000
        _c.clear(); E._cache.clear(); tracemalloc.start(); fn(v); lat[a + '|MB'] = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
    out['_latency'] = lat; json.dump(out, open('results_v2.json', 'w'), indent=1); print('saved', flush=True)
