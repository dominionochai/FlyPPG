import numpy as np, pickle, time, flyppg as F, fly_conn as C
from scipy import ndimage as ndi
def lum_ds(v):  # 2x mean-pooled luminance
    L = v.mean(3); return L.reshape(len(L), 48, 2, 48, 2).mean((2, 4)).astype(np.float32)
def feats(v):
    o = C.bank(lum_ds(v)); X = np.stack([o[n].mean((1, 2)) for n in C.SUB], 1)
    return X - np.median(X, 0)
if __name__ == '__main__':
    t0 = time.time(); X, Y = [], []
    for i in range(6):
        v, hr = F.make_video(5000 + i, 1, i % 2); pos = F.make_video.pos
        vel = np.diff(pos, axis=0, prepend=pos[:1])         # (row, col) full-res px/frame
        X.append(feats(v)); Y.append(vel); print(i, round(time.time() - t0), flush=True)
    Xtr, Ytr, Xte, Yte = np.vstack(X[:4]), np.vstack(Y[:4]), np.vstack(X[4:]), np.vstack(Y[4:])
    A = np.c_[Xtr, np.ones(len(Xtr))]; lam = 1e-3 * np.trace(A.T @ A) / A.shape[1]
    Wt = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Ytr)
    P = np.c_[Xte, np.ones(len(Xte))] @ Wt
    r2 = 1 - ((Yte - P) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)
    print('held-out velocity R2 (row,col):', r2.round(3))
    # position-level (in-band) check
    pt = np.cumsum(P, 0); pg = np.cumsum(Yte, 0)
    hp = lambda p: p - ndi.gaussian_filter1d(p, 30, axis=0)
    print('in-band position corr:', [round(np.corrcoef(hp(pt)[:, k], hp(pg)[:, k])[0, 1], 3) for k in range(2)])
    pickle.dump(Wt, open('readout_W.pkl', 'wb'))
