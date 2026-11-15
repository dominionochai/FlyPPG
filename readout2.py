import numpy as np, pickle, time, flyppg as F, readout as R
from scipy import ndimage as ndi
X, Y = [], []
for i in range(8):
    v, hr = F.make_video(5000 + i, 1, i % 2); pos = F.make_video.pos
    X.append(R.feats(v)); Y.append(np.diff(pos, axis=0, prepend=pos[:1]))
np.savez('tune_feats.npz', X=np.array(X), Y=np.array(Y))
def taps(X, K=4):
    return np.concatenate([np.roll(X, k, 0) for k in range(-K, K + 1)], 1)
def fit(idx, K, lam_s):
    A = np.vstack([np.c_[taps(X[i], K), np.ones(len(X[i]))] for i in idx]); B = np.vstack([Y[i] for i in idx])
    lam = lam_s * np.trace(A.T @ A) / A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ B)
hp = lambda p: p - ndi.gaussian_filter1d(p, 30, axis=0)
for K in [0, 2, 4, 8]:
    for lam_s in [1e-3, 1e-2, 1e-1]:
        Wt = fit(range(6), K, lam_s); res = []
        for i in (6, 7):
            P = np.c_[taps(X[i], K), np.ones(len(X[i]))] @ Wt
            res.append([np.corrcoef(hp(np.cumsum(P, 0))[:, k], hp(np.cumsum(Y[i], 0))[:, k])[0, 1] for k in range(2)])
        print(K, lam_s, np.round(np.mean(res, 0), 3), flush=True)
