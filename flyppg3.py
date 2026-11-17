"""FlyPPG v3. (1) surround (background-referenced) divisive illumination normalisation, (2) fast streaming-friendly connectome
T4/T5 bank + LOCAL velocity readout (handles rotation/zoom/non-rigid), (3) harder benchmark. Synthetic video, proxy extractor."""
import numpy as np, cv2, time, sys, os, json, pickle, tracemalloc
from scipy import ndimage as ndi, signal
import flyppg as F, fly_conn as C

H = W = 96; N = 600; FPS = 30; CY = CX = 47.5
YY, XX = np.mgrid[:H, :W].astype(np.float32); RAD = np.hypot(YY - CY, XX - CX)
CELLS = np.array([((i + .5) * 32, (j + .5) * 32) for i in range(3) for j in range(3)], np.float32)   # (y,x) centres of 3x3 grid

# ------------------------------------------------------------------ generator
def sm(r, s, shape=(H, W)):
    a = ndi.gaussian_filter(r.standard_normal(shape), s); return (a / a.std()).astype(np.float32)

def make_video3(seed, tr=1., rot=0., nr=0., il=0., ae=0., fl=0., noise=.006, jpegq=None, sh=1., bgd=0., cl=0., cs=0., two=0.):
    r = np.random.default_rng(seed); hr = r.uniform(55, 120); f = hr / 60; t = np.arange(N) / FPS
    tex = sm(r, 1.3) * .10; bgt = sm(r, 3) * .03
    amap = np.clip(1.15 - RAD / 38 * .8 + .3 * sm(r, 6), .15, 1.5); ph = .5 * sm(r, 8)
    maskf = 1 / (1 + np.exp((RAD - 38) / 1.0))
    base = np.stack([.62 + tex, .50 + tex, .42 + tex], 0).astype(np.float32); bgc = np.array([.15, .17, .21], np.float32)[:, None, None] + bgt
    walk = ndi.gaussian_filter1d(r.standard_normal((N, 2)).cumsum(0), 3, axis=0); walk = walk / (np.abs(walk).max() + 1e-9) * 6 * tr
    bur = np.zeros((N, 2))
    for _ in range(int(6 * tr)): s = r.integers(30, N - 30); bur[s:s + 8] += r.normal(0, 3, 2) * tr
    pos = walk + ndi.gaussian_filter1d(bur, 1.5, axis=0) + tr * 1.2 * np.sin(2 * np.pi * r.uniform(.9, 2.2) * t + r.uniform(0, 6.28))[:, None] * r.normal(1, .3, 2)
    th = rot * (.10 * np.sin(2 * np.pi * .35 * t + r.uniform(0, 6.28)) + .04 * np.sin(2 * np.pi * r.uniform(1, 2) * t + r.uniform(0, 6.28)))
    sc = 1 + rot * .03 * np.sin(2 * np.pi * .25 * t + r.uniform(0, 6.28))
    F1, F2 = np.stack([sm(r, 12), sm(r, 12)]), np.stack([sm(r, 12), sm(r, 12)]); p1, p2 = r.uniform(0, 6.28, 2)
    gain = 1 + il * .18 * np.sin(2 * np.pi * .12 * t + r.uniform(0, 6.28))
    if il:
        for _ in range(3): gain[r.integers(60, N - 60):] *= 1 + r.choice([-1, 1]) * .10
    if ae:
        for _ in range(3):
            s = r.integers(60, N - 90); a = r.choice([-1, 1]) * .08; tt = np.arange(N) - s; gain *= 1 + a * np.where(tt >= 0, np.exp(-np.maximum(tt, 0) / (1.2 * FPS)), 0)
    gain *= 1 + fl * .003 * np.sin(2 * np.pi * r.uniform(1.4, 2.4) * t + r.uniform(0, 6.28))
    gb = 1 + (gain - 1) * sh      # sh=1: lighting shared by face and background; sh=0: face-only lighting (screen glow)
    if bgd:                        # independent lighting changes that only the background sees
        rb = np.random.default_rng(seed + 99); gb = gb * (1 + bgd * .15 * np.sin(2 * np.pi * .2 * t + rb.uniform(0, 6.28)))
        for _ in range(3): gb[rb.integers(40, N - 40):] *= 1 + rb.choice([-1, 1]) * .12 * bgd
    gS = np.ones(N)
    if two:   # two independent lights with different colours: shared ambient (face+bg) and a screen glow only the face sees
        rs = np.random.default_rng(seed + 7); gS = 1 + .12 * np.sin(2 * np.pi * .17 * t + rs.uniform(0, 6.28)) + .003 * np.sin(2 * np.pi * rs.uniform(1.2, 2.6) * t)
        for _ in range(2): gS[rs.integers(60, N - 60):] *= 1 + rs.choice([-1, 1]) * .10
    vid = np.empty((N, H, W, 3), np.float32); U = np.zeros((N, 9, 2))
    for i in range(N):
        c, s_ = np.cos(th[i]), np.sin(th[i]); dy, dx = YY - CY, XX - CX
        d = nr * (1.5 * F1 * np.sin(2 * np.pi * 1.3 * t[i] + p1) + 1.5 * F2 * np.sin(2 * np.pi * .7 * t[i] + p2))
        sy = CY + (c * dy + s_ * dx) / sc[i] - pos[i, 0] + d[0]; sx = CX + (-s_ * dy + c * dx) / sc[i] - pos[i, 1] + d[1]
        pul = 1 + np.array([.0125, .0125 * .2, .0125 * .2])[:, None, None] * amap * np.sin(2 * np.pi * f * t[i] + ph)
        fc = np.stack([ndi.map_coordinates(base[k] * pul[k], [sy, sx], order=1, mode='nearest') for k in range(3)]); m = ndi.map_coordinates(maskf, [sy, sx], order=1, mode='nearest')
        blob = np.exp(-((YY - (CY + 62 * np.sin(.5 * t[i]))) ** 2 + (XX - (CX + 62 * np.cos(.5 * t[i]))) ** 2) / (2 * 10 ** 2))
        mc = (np.array([1., .4, 1.8]) if cs else np.ones(3))[:, None, None]   # cs: lighting changes colour (warm/cool), not just brightness
        if two: mA, mS = np.array([1., .7, .4])[:, None, None], np.array([.4, .9, 1.6])[:, None, None]; fg = 1 + (gain[i] - 1) * mA + (gS[i] - 1) * mS; bgg = 1 + (gb[i] - 1) * mA
        else: fg = 1 + (gain[i] - 1) * mc; bgg = 1 + (gb[i] - 1) * mc
        vid[i] = ((bgc - cl * .10 * blob) * bgg * (1 - m) + fc * fg * m).transpose(1, 2, 0)
        for k, (cyy, cxx) in enumerate(CELLS):   # true content displacement at cell centres = -(src - x)
            dyy, dxx = cyy - CY, cxx - CX; iy, ix = int(cyy), int(cxx)
            U[i, k] = [-((CY + (c * dyy + s_ * dxx) / sc[i] - pos[i, 0] + d[0][iy, ix]) - cyy), -((CX + (-s_ * dyy + c * dxx) / sc[i] - pos[i, 1] + d[1][iy, ix]) - cxx)]
    vid += r.normal(0, noise, vid.shape).astype(np.float32)
    if jpegq:
        for i in range(N):
            ok, b = cv2.imencode('.jpg', np.clip(vid[i] * 255, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, jpegq]); vid[i] = cv2.imdecode(b, 1).astype(np.float32) / 255
    return vid, hr, U

CONDS = {'rigid motion': dict(tr=1.), 'rotation+zoom': dict(tr=.5, rot=1.), 'non-rigid': dict(tr=.5, nr=1.), 'hard motion': dict(tr=1., rot=1., nr=1.),
         'AE+flicker+drift': dict(il=1., ae=1., fl=1., tr=0.), 'everything': dict(tr=1., rot=1., nr=1., il=1., ae=1., fl=1., jpegq=30)}

# ------------------------------------------------------------------ extractor (face ROI, fixed box)
def rppg(v): return v[:, 24:72, 24:72, 1].mean((1, 2))

# ------------------------------------------------------------------ (1) surround divisive normalisation
BG = RAD > 52   # far corners only: face (r=38) + max head shift stays clear of the reference ring
def bg_norm(v):
    g = v[:, BG, :].mean((1, 2)); g = g / np.median(g)
    return v / g[:, None, None, None].astype(np.float32)

def bg_norm_pc(v):
    g = v[:, BG, :].mean(1); g = g / np.median(g, 0); return v / g[:, None, None, :].astype(np.float32)

def surround_gated(v):
    """Apply surround normalisation only when background and face lighting are coupled (guards face-only / bg-only changes)."""
    lg = lambda x: np.log(x + 1e-6); bp = lambda x: ndi.gaussian_filter1d(x, 3) - ndi.gaussian_filter1d(x, 30)
    b = bp(lg(v[:, BG, :].mean((1, 2)))); f = bp(lg(v[:, 24:72, 24:72, :].mean((1, 2, 3))))
    return bg_norm(v) if (np.corrcoef(b, f)[0, 1] > .5) else v

def rppg_pos(v, l=48):
    """POS chrominance rPPG (Wang et al. 2017) on face-ROI mean RGB."""
    c = v[:, 24:72, 24:72, :].mean((1, 2)); hsig = np.zeros(len(c)); P = np.array([[0, 1, -1], [-2, 1, 1]])
    for s0 in range(len(c) - l + 1):
        w = c[s0:s0 + l]; cn = (w / w.mean(0)).T; S = P @ cn; h = S[0] + S[0].std() / (S[1].std() + 1e-9) * S[1]; hsig[s0:s0 + l] += h - h.mean()
    return hsig

# ------------------------------------------------------------------ (2) fast fly bank (cv2 correlation kernels) + local readout
def _kern(offs):
    K = np.zeros((9, 9), np.float32)
    for x, y, w in offs:
        fx, fy = x + 4, y + 4; x0, y0 = int(np.floor(fx)), int(np.floor(fy)); ax, ay = fx - x0, fy - y0
        for yy, xx, ww in ((y0, x0, (1 - ay) * (1 - ax)), (y0, x0 + 1, (1 - ay) * ax), (y0 + 1, x0, ay * (1 - ax)), (y0 + 1, x0 + 1, ay * ax)):
            if 0 <= yy < 9 and 0 <= xx < 9: K[yy, xx] += w * ww
    return K
KERN = {n: (_kern(e), _kern(i)) for n, (e, i) in C.SUB.items()}
def bank_fast(L):
    base = ndi.gaussian_filter1d(L, 45, axis=0, mode='nearest'); c = L / (base + 1e-6) - 1
    pols = {'on': np.maximum(c, 0), 'off': np.maximum(-c, 0)}
    filt = {(p, k): C.tlp(x, C.TAU_F if k == 'f' else C.TAU_S).astype(np.float32) for p, x in pols.items() for k in 'fs'}
    out = {}
    for n, (ke, ki) in KERN.items():
        p = 'on' if n.startswith('T4') else 'off'; fa, sa = filt[(p, 'f')], filt[(p, 's')]; o = np.empty_like(fa)
        for t in range(len(fa)): o[t] = np.maximum(cv2.filter2D(fa[t], -1, ke, borderType=cv2.BORDER_REPLICATE) - C.BETA * cv2.filter2D(sa[t], -1, ki, borderType=cv2.BORDER_REPLICATE), 0)
        out[n] = o
    return out
def lum_ds(v):
    L = v.mean(3); return L.reshape(len(L), 48, 2, 48, 2).mean((2, 4)).astype(np.float32)
def feats_local(v):
    o = bank_fast(lum_ds(v)); X = np.concatenate([o[n].reshape(len(v), 3, 16, 3, 16).mean((2, 4)).reshape(len(v), 9) for n in C.SUB], 1)
    return X - np.median(X, 0)
K_TAPS, LAM = 6, 3e-5
def taps(X, k=K_TAPS): return np.concatenate([np.roll(X, j, 0) for j in range(-k, k + 1)], 1)
def train_readout(n=24, log=print):
    Xs, Ys = [], []; r = np.random.default_rng(0)
    for i in range(n):
        pr = dict(tr=r.uniform(.3, 1), rot=r.uniform(0, 1), nr=r.uniform(0, 1), il=float(i % 2), ae=float(i % 3 == 0), fl=float(i % 2))
        v, hr, U = make_video3(5000 + i, **pr); Xs.append(feats_local(bg_norm(v))); Ys.append(np.diff(U, axis=0, prepend=U[:1]).reshape(N, 18)); log('train clip', i)
    np.savez('train_local.npz', X=np.array(Xs), Y=np.array(Ys)); return Xs, Ys
def fit(Xs, Ys):
    A = np.vstack([np.c_[taps(x), np.ones(len(x))] for x in Xs]); B = np.vstack(Ys)
    return np.linalg.solve(A.T @ A + LAM * np.trace(A.T @ A) / A.shape[1] * np.eye(A.shape[1]), A.T @ B)
def stab_fly_local(v, Wr):
    vn = bg_norm(v); vel = (np.c_[taps(feats_local(vn)), np.ones(len(v))] @ Wr).reshape(len(v), 9, 2)
    pos = np.cumsum(vel, 0); pos = pos - ndi.gaussian_filter1d(pos, 30, axis=0, mode='nearest')
    out = np.empty_like(vn)
    for t in range(len(v)):
        uy = cv2.resize(pos[t, :, 0].reshape(3, 3).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC); ux = cv2.resize(pos[t, :, 1].reshape(3, 3).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        out[t] = cv2.remap(vn[t], XX + ux, YY + uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out

# ------------------------------------------------------------------ classical baselines
def stab_farneback(v):
    g = lambda x: np.clip(x.mean(2) * 255, 0, 255).astype(np.uint8); ref = g(v[:30].mean(0)); out = np.empty_like(v)
    for t in range(len(v)):
        fl = cv2.calcOpticalFlowFarneback(ref, g(v[t]), None, .5, 3, 15, 3, 5, 1.2, 0)
        out[t] = cv2.remap(v[t], XX + fl[..., 0], YY + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out
def stab_phase(v):   # classical global translation by phase correlation
    L = lum_ds(v); ref = L[:30].mean(0); win = np.outer(np.hanning(48), np.hanning(48)); Fr = np.fft.fft2((ref - ref.mean()) * win); out = np.empty_like(v)
    for t in range(len(v)):
        Ft = np.fft.fft2((L[t] - L[t].mean()) * win); c = Fr * np.conj(Ft); c /= np.abs(c) + 1e-9; r = np.real(np.fft.ifft2(c)); y, x = np.unravel_index(r.argmax(), r.shape)
        sub = lambda a, m: .5 * (a[(m - 1) % 48] - a[(m + 1) % 48]) / (a[(m - 1) % 48] - 2 * a[m] + a[(m + 1) % 48] + 1e-12)
        dy = y + sub(r[:, x], y); dx = x + sub(r[y, :], x); dy = dy - 48 if dy > 24 else dy; dx = dx - 48 if dx > 24 else dx
        out[t] = ndi.shift(v[t], (2 * dy, 2 * dx, 0), order=1, mode='nearest')
    return out

def arms(Wr):
    import fly_conn as C0, eval3 as E3
    return {'raw': lambda v: v, 'conventional': F.pre_conventional, 'surround-norm only': bg_norm,
            'classical translation+surround': lambda v: stab_phase(bg_norm(v)), 'classical dense-flow+surround': lambda v: stab_farneback(bg_norm(v)),
            'FlyPPG v3 (surround+local T4/T5)': lambda v: stab_fly_local(v, Wr), 'FlyPPG v2 (step+global T4/T5)': lambda v: F.step_comp(E3.stab_fly2(v))}

def run_eval(n, Wr, tag):
    A = arms(Wr); res = pickle.load(open(f'partial_{tag}.pkl', 'rb')) if os.path.exists(f'partial_{tag}.pkl') else {}
    for c, pr in CONDS.items():
        if any(k[0] == c for k in res): continue
        for a in A: res[(c, a)] = dict(err=[], snr=[])
        for k in range(n):
            v, hr, _ = make_video3(1000 + k, **pr)
            for a, fn in A.items():
                for est, snr in F.window_est((rppg_pos if os.environ.get('EXTRACTOR') == 'pos' else rppg)(fn(v)), hr): res[(c, a)]['err'].append(est - hr); res[(c, a)]['snr'].append(snr)
        pickle.dump(res, open(f'partial_{tag}.pkl', 'wb')); print(c, 'done', flush=True)
    out = {}
    for (c, a), r in res.items():
        e, s = np.array(r['err']), np.array(r['snr']); out[f'{c}|{a}'] = dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())), SNR=float(s.mean()), success=float((np.abs(e) <= 5).mean() * 100), n=len(e))
    v, _, _ = make_video3(7000, **CONDS['everything']); lat = {}
    import eval3 as E3
    for a, fn in A.items():
        E3._c.clear(); t0 = time.perf_counter(); fn(v); lat[a] = (time.perf_counter() - t0) / N * 1000
        E3._c.clear(); tracemalloc.start(); fn(v); lat[a + '|MB'] = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
    out['_latency'] = lat; json.dump(out, open(f'results_{tag}.json', 'w'), indent=1); print('saved', flush=True)

if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'train':
        Xs, Ys = train_readout(); Wr = fit(Xs[:20], Ys[:20]); hp = lambda p: p - ndi.gaussian_filter1d(p, 30, axis=0)
        cs = []
        for i in range(20, 24):
            P = (np.c_[taps(Xs[i]), np.ones(N)] @ Wr).reshape(N, 9, 2); Y = Ys[i].reshape(N, 9, 2)
            cs += [np.corrcoef(hp(np.cumsum(P[:, k, a])), hp(np.cumsum(Y[:, k, a])))[0, 1] for k in range(9) for a in range(2)]
        print('held-out per-cell in-band displacement corr: mean %.3f min %.3f' % (np.mean(cs), np.min(cs)))
        np.save('readout_W3.npy', fit(Xs, Ys))
    elif cmd == 'eval': run_eval(int(sys.argv[2]), np.load('readout_W3.npy'), sys.argv[3] if len(sys.argv) > 3 else 'v3')
