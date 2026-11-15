import numpy as np, time, json, sys, tracemalloc, cv2, pickle
from scipy import ndimage as ndi
import flyppg as F, readout as R
K, LAM = 8, 1e-5
d = np.load('tune_feats.npz'); Xt, Yt = d['X'], d['Y']
def taps(X, k=K): return np.concatenate([np.roll(X, j, 0) for j in range(-k, k + 1)], 1)
A = np.vstack([np.c_[taps(x), np.ones(len(x))] for x in Xt]); B = np.vstack(Yt)
W = np.linalg.solve(A.T @ A + LAM * np.trace(A.T @ A) / A.shape[1] * np.eye(A.shape[1]), A.T @ B)

_cache = {}
def fly_motion(v):
    k = id(v)
    if k not in _cache:
        P = np.c_[taps(R.feats(v)), np.ones(len(v))] @ W
        _cache.clear(); _cache[k] = P
    return _cache[k]
def apply_shift(v, pos):
    out = np.empty_like(v)
    for t in range(len(v)): out[t] = ndi.shift(v[t], (-pos[t, 0], -pos[t, 1], 0), order=1, mode='nearest')
    return out
def stab_fly(v):
    pos = np.cumsum(fly_motion(v), 0); pos = pos - ndi.gaussian_filter1d(pos, 30, axis=0, mode='nearest')
    return apply_shift(v, pos)

def classical_pos(v):
    L = R.lum_ds(v); ref = L[:30].mean(0); win = np.outer(np.hanning(48), np.hanning(48))
    Fr = np.fft.fft2((ref - ref.mean()) * win); pos = np.zeros((len(v), 2))
    for t in range(len(v)):
        Ft = np.fft.fft2((L[t] - L[t].mean()) * win); c = Fr * np.conj(Ft); c /= np.abs(c) + 1e-9
        r = np.real(np.fft.ifft2(c)); y, x = np.unravel_index(r.argmax(), r.shape)
        def sub(a, m, n): return 0.5 * (a[(m - 1) % n] - a[(m + 1) % n]) / (a[(m - 1) % n] - 2 * a[m] + a[(m + 1) % n] + 1e-12)
        dy = y + sub(r[:, x], y, 48); dx = x + sub(r[y, :], x, 48)
        dy = dy - 48 if dy > 24 else dy; dx = dx - 48 if dx > 24 else dx
        pos[t] = [-2 * dy, -2 * dx]          # content displacement vs ref, full-res px (sign verified in selftest)
    return pos
def stab_classical(v): return apply_shift(v, classical_pos(v))

def selftest():
    v, _ = F.make_video(1, 1, 0); p = classical_pos(v); true = F.make_video.pos - F.make_video.pos[:30].mean(0)
    c = [np.corrcoef(p[:, k], true[:, k])[0, 1] for k in range(2)]; assert min(c) > 0.9, c; return c

PRE = {
 'raw': lambda v: v,
 'conventional': F.pre_conventional,
 'classical stab+conv': lambda v: F.pre_conventional(stab_classical(v)),
 'classical stab+step': lambda v: F.step_comp(stab_classical(v)),
 'fly step only': F.step_comp,
 'fly T4/T5 stab only': stab_fly,
 'FlyPPG v1 (step+T4/T5)': lambda v: F.step_comp(stab_fly(v)),
}
def jpeg(v, q):
    out = np.empty_like(v)
    for t in range(len(v)):
        im = np.clip(v[t] * 255, 0, 255).astype(np.uint8); ok, b = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, q])
        out[t] = cv2.imdecode(b, 1).astype(np.float32) / 255
    return out
CONDS = {  # name: (motion, illum, noise, jpegq)
 'static': (0, 0, .006, None), 'motion': (1, 0, .006, None), 'illum': (0, 1, .006, None),
 'motion+illum': (1, 1, .006, None), 'compressed(q25)+motion': (.6, 0, .006, 25), 'low-light+motion': (.6, 0, .02, None)}
def gen(seed, c):
    mo, il, nz, q = CONDS[c]; v, hr = F.make_video(seed, mo, il, nz)
    return (jpeg(v, q) if q else v), hr
if __name__ == '__main__':
    print('selftest classical corr', np.round(selftest(), 3), flush=True)
    import os
    n = int(sys.argv[1]); res = pickle.load(open('partial.pkl','rb')) if os.path.exists('partial.pkl') else {}
    for c in CONDS:
        if any(k[0]==c for k in res): continue
        vids = [gen(1000 + k, c) for k in range(n)]
        for a, fn in PRE.items(): res[(c, a)] = dict(err=[], snr=[])
        for v, hr in vids:
            for a, fn in PRE.items():
                p = fn(v)
                for est, snr in F.window_est(F.rppg_proxy(p), hr): res[(c, a)]['err'].append(est - hr); res[(c, a)]['snr'].append(snr)
        pickle.dump(res, open('partial.pkl','wb')); print(c, 'done', flush=True)
    out = {}
    for (c, a), r in res.items():
        e, s = np.array(r['err']), np.array(r['snr']); g = s > 0
        out[f'{c}|{a}'] = dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())), SNR=float(s.mean()), success=float((np.abs(e) <= 5).mean() * 100),
                               reported=float(g.mean() * 100), MAE_reported=float(np.abs(e[g]).mean()) if g.any() else float('nan'))
    # latency / memory (no cache), 2 videos
    v, _ = gen(7000, 'motion+illum'); lat = {}
    for a, fn in PRE.items():
        _cache.clear(); t0 = time.perf_counter(); fn(v); lat[a] = (time.perf_counter() - t0) / len(v) * 1000
        _cache.clear(); tracemalloc.start(); fn(v); mem = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop(); lat[a + '|MB'] = mem
    out['_latency'] = lat; json.dump(out, open('results_v1.json', 'w'), indent=1); print('saved')
