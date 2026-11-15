import numpy as np, json, flyppg as F, readout as R, eval2 as E
from scipy import signal
v, hr = F.make_video(1000, 1, 1); pos = F.make_video.pos
X = R.feats(v); P = E.fly_motion(v); Y = np.diff(pos, axis=0, prepend=pos[:1])
n = lambda a, q=98: np.clip(a / (np.percentile(np.abs(a), q) + 1e-9), 0, 1)
Xn = np.stack([n(np.maximum(X[:, k], 0)) for k in range(8)], 1)
Ld = R.lum_ds(v); lam = n(np.abs(np.diff(Ld, axis=0, prepend=Ld[:1])).mean((1, 2)))
g = np.log(v.mean((1, 2, 3))); dg = np.abs(np.diff(g, prepend=g[0])); S = (dg > max(8 * 1.4826 * np.median(np.abs(dg - np.median(dg))), .02)).astype(float)
S = np.clip(np.convolve(S, np.hanning(9) / np.hanning(9).sum() * 3, 'same'), 0, 1)
V = n(np.hypot(*P.T)); Ve = P[:, 1] / (np.percentile(np.abs(Y[:, 1]), 98) + 1e-9); Vt = Y[:, 1] / (np.percentile(np.abs(Y[:, 1]), 98) + 1e-9)
b, a = signal.butter(3, (.7, 3.), 'bandpass', fs=30)
def pulse(x): y = signal.filtfilt(b, a, signal.detrend(x)); return np.clip(y / (3 * y.std()) * 2, -1, 1)
raw = pulse(F.rppg_proxy(v)); fly = pulse(F.rppg_proxy(F.step_comp(E.stab_fly(v))))
r = lambda a: np.round(a, 2).tolist()
D = dict(T=len(v), X=r(Xn), L=r(lam), S=r(S), V=r(V), Vt=r(np.clip(Vt, -1, 1)), Ve=r(np.clip(Ve, -1, 1)), raw=r(raw), fly=r(fly))
for tpl,out in [('brain_tpl.html','brain.html'),('brain3d_tpl.html','brain3d.html')]: open(out, 'w').write(open(tpl).read().replace('/*DATA*/', json.dumps(D, separators=(',', ':'))))
print('ok', len(open('brain.html').read()) // 1024, 'KB; step frames', int(S.sum() > 0), 'corr est/true', round(float(np.corrcoef(Ve, Vt)[0, 1]), 2))
