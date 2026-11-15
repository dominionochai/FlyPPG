import numpy as np, time, flyppg as F, fly_conn as C, readout as R
from scipy import ndimage as ndi
def feats_grid(v):
    o = C.bank(R.lum_ds(v)); cols = []
    for n in C.SUB:
        a = o[n]; cols.append(a.mean((1, 2)))
        for (y0, y1) in ((0, 24), (24, 48)):
            for (x0, x1) in ((0, 24), (24, 48)): cols.append(a[:, y0:y1, x0:x1].mean((1, 2)))
        cols.append(a[:, 12:36, 12:36].mean((1, 2)))
    X = np.stack(cols, 1); return X - np.median(X, 0)
if __name__ == '__main__':
    t0 = time.time(); X, Y = [], []
    for i in range(12):
        v, hr = F.make_video(5000 + i, 1, i % 2); pos = F.make_video.pos
        X.append(feats_grid(v)); Y.append(np.diff(pos, axis=0, prepend=pos[:1])); print(i, round(time.time() - t0), flush=True)
    np.savez('tune_feats_grid.npz', X=np.array(X), Y=np.array(Y))
