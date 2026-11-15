"""FlyPPG prototype: fly-inspired (T4/T5 + contrast adaptation) preprocessing for rPPG.
NOTE: SmartSpectra is not available in this sandbox. `rppg_proxy` is a stand-in
extractor (ROI green mean -> bandpass -> spectral peak). Swap in SmartSpectra via
the `extractor` hook when running locally."""
import time, json, sys
import numpy as np
from scipy import ndimage as ndi, signal

FPS, N, H, W = 30, 600, 96, 96

# ---------------------------------------------------------------- synthetic data
def make_video(seed, motion=0.0, illum=0.0, noise=0.006):
    r = np.random.default_rng(seed)
    hr = r.uniform(55, 120)
    f = hr / 60
    tex = ndi.gaussian_filter(r.standard_normal((H + 40, W + 40)), 1.3)
    tex = tex / tex.std() * 0.10
    yy, xx = np.mgrid[:H + 40, :W + 40]
    face = ((yy - (H + 40) / 2) ** 2 + (xx - (W + 40) / 2) ** 2) < 38 ** 2
    base = np.where(face, 0.55, 0.18) + tex * face
    t = np.arange(N) / FPS
    pulse = 1 + 0.0025 * np.sin(2 * np.pi * f * t + r.uniform(0, 6.28))
    # motion: smooth random walk + bursts (pixels)
    walk = ndi.gaussian_filter1d(r.standard_normal((N, 2)).cumsum(0), 3, axis=0)
    walk = walk / (np.abs(walk).max() + 1e-9) * 6.0 * motion
    bursts = np.zeros((N, 2))
    for _ in range(int(6 * motion)):
        s = r.integers(30, N - 30)
        bursts[s:s + 8] += r.normal(0, 3, 2) * motion
    pos = walk + ndi.gaussian_filter1d(bursts, 1.5, axis=0)
    pos = pos + motion * 1.2 * np.sin(2 * np.pi * r.uniform(0.9, 2.2) * t + r.uniform(0, 6.28))[:, None] * r.normal(1, .3, 2)
    # illumination: slow drift + steps
    gain = 1 + illum * 0.18 * np.sin(2 * np.pi * 0.12 * t + r.uniform(0, 6.28))
    for _ in range(int(3 * (illum > 0))):
        s = r.integers(60, N - 60)
        gain[s:] *= 1 + r.choice([-1, 1]) * 0.10 * illum
    vid = np.empty((N, H, W, 3), np.float32)
    for i in range(N):
        fr = ndi.shift(base, pos[i], order=1, mode='nearest')[20:20 + H, 20:20 + W]
        for c, (k, a) in enumerate([(1.0, 0.2), (1.0, 1.0), (0.85, 0.15)]):
            ch = fr * k * (1 + (pulse[i] - 1) * a * 5 * (c == 1) + (pulse[i] - 1) * 0.2 * (c != 1))
            vid[i, :, :, c] = ch * gain[i]
    vid += r.normal(0, noise, vid.shape).astype(np.float32)
    return vid, hr

# ---------------------------------------------------------------- preprocessors
def pre_raw(v):
    return v

def pre_conventional(v):
    """Spatial Gaussian + 5-frame temporal mean + lowpass global-brightness normalisation."""
    o = ndi.gaussian_filter(v, (0, 1.5, 1.5, 0))
    o = ndi.uniform_filter1d(o, 5, axis=0)
    g = o.mean((1, 2, 3))
    lp = ndi.gaussian_filter1d(g, 3 * FPS, mode='nearest')
    return o * (lp.mean() / lp)[:, None, None, None]

def fly_motion_energy(L, tau=2.0):
    """T4/T5-style elementary motion detectors on ON/OFF half-wave-rectified contrast.
    Hassenstein-Reichardt correlators in 4 directions; returns per-frame motion map."""
    a = 1 - np.exp(-1 / tau)
    prev = L[0].copy(); lp_on = np.zeros_like(prev); lp_off = np.zeros_like(prev)
    M = np.empty(L.shape, np.float32)
    shifts = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for i in range(len(L)):
        d = L[i] - prev; prev = L[i].copy()
        on, off = np.maximum(d, 0), np.maximum(-d, 0)
        lp_on += a * (on - lp_on); lp_off += a * (off - lp_off)
        m = np.zeros_like(on)
        for dy, dx in shifts:
            for c, lp in ((on, lp_on), (off, lp_off)):
                cs, ls = np.roll(c, (dy, dx), (0, 1)), np.roll(lp, (dy, dx), (0, 1))
                m += np.abs(lp * cs - c * ls)
        M[i] = m
    return M

def pre_fly(v, use_motion=True, use_adapt=True, m0=None, tau_adapt=3.0):
    o = v
    if use_adapt:  # contrast/luminance adaptation: divisive normalisation by slow local mean
        lp = ndi.gaussian_filter(o, (0, 6, 6, 0))
        lp = ndi.gaussian_filter1d(lp, tau_adapt * FPS, axis=0, mode='nearest')
        o = o / (lp + 1e-3) * lp.mean((0, 1, 2), keepdims=True)
    if use_motion:
        L = ndi.gaussian_filter(o.mean(3) / (o.mean() + 1e-6), (0, 1.0, 1.0))
        M = fly_motion_energy(L)
        M = ndi.uniform_filter(M, (1, 5, 5))               # pool over receptive field
        scale = m0 if m0 is not None else 1.5e-5
        w = (1 / (1 + (M / scale) ** 2))[..., None]
        # hold: moving pixels fall back to a slow running reference (pulse-preserving)
        ref = ndi.gaussian_filter1d(o, 0.5 * FPS, axis=0, mode='nearest')
        out = np.empty_like(o)
        run = o[0].copy()
        for i in range(len(o)):
            run = w[i] * o[i] + (1 - w[i]) * run
            out[i] = run
        o = out
    return o

# ---------------------------------------------------------------- extractor + metrics
BAND = (0.7, 3.0)

def rppg_proxy(v):
    roi = v[:, 28:68, 28:68, 1].mean((1, 2))
    return roi

def window_est(sig, hr_true, win=10 * FPS, stride=5 * FPS):
    b, a = signal.butter(3, BAND, 'bandpass', fs=FPS)
    out = []
    for s in range(0, len(sig) - win + 1, stride):
        x = signal.detrend(sig[s:s + win])
        x = signal.filtfilt(b, a, x)
        f, p = signal.welch(x, FPS, nperseg=win, nfft=4096)
        m = (f >= BAND[0]) & (f <= BAND[1])
        fp = f[m][np.argmax(p[m])]
        sig_m = (np.abs(f - fp) < 0.1) | (np.abs(f - 2 * fp) < 0.1)
        snr = 10 * np.log10(p[m & sig_m].sum() / (p[m & ~sig_m].sum() + 1e-18))
        out.append((fp * 60, snr))
    return out

def evaluate(arms, conds, n_vid=8, extractor=rppg_proxy):
    res = {}
    for cname, (mo, il) in conds.items():
        vids = [make_video(1000 + k, mo, il) for k in range(n_vid)]
        for aname, fn in arms.items():
            errs, snrs, lat = [], [], []
            for v, hr in vids:
                t0 = time.perf_counter(); p = fn(v); lat.append((time.perf_counter() - t0) / N * 1000)
                for est, snr in window_est(extractor(p), hr):
                    errs.append(est - hr); snrs.append(snr)
            e = np.array(errs)
            res[(cname, aname)] = dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                                       SNR=np.mean(snrs), success=(np.abs(e) <= 5).mean() * 100,
                                       ms_per_frame=np.mean(lat))
    return res

def step_comp(v):
    """Spatially-uniform ON/OFF transient = illumination step (vs localised T4/T5 edge response).
    Detect via robust threshold on global log-luminance change and cancel the step."""
    g = np.log(v.mean((1, 2, 3)) + 1e-6); dg = np.diff(g, prepend=g[0])
    mad = np.median(np.abs(dg - np.median(dg))) * 1.4826 + 1e-9
    mask = np.abs(dg) > max(8 * mad, 0.02)
    c = np.cumsum(np.where(mask, dg, 0))
    return v * np.exp(-c)[:, None, None, None].astype(np.float32)

def gate_frames(v, m0=8e-4, tw=1.0):
    L = ndi.gaussian_filter(v.mean(3) / (v.mean() + 1e-6), (0, 1, 1))
    g = ndi.uniform_filter(fly_motion_energy(L), (1, 5, 5)).mean((1, 2))
    w = ndi.gaussian_filter1d(1 / (1 + (g / m0) ** 2), tw, mode='nearest')
    bad = w < 0.5; idx = np.arange(len(v)); good = idx[~bad]
    if len(good) < 10 or not bad.any(): return v
    near = good[np.abs(idx[:, None] - good[None]).argmin(1)]
    return np.where(bad[:, None, None, None], v[near], v)

def pre_flyppg(v, step=True, gate=True):
    o = step_comp(v) if step else v
    return gate_frames(o) if gate else o

CONDS = {'static': (0, 0), 'motion': (1, 0), 'illum': (0, 1), 'motion+illum': (1, 1)}
ARMS = {'raw': pre_raw, 'conventional': pre_conventional, 'flyppg': pre_flyppg,
        'fly:T4/T5 gate only': lambda v: pre_flyppg(v, False, True),
        'fly:ON/OFF step only': lambda v: pre_flyppg(v, True, False)}

def table(res):
    lines = [f"{'condition':14}{'arm':16}{'MAE':>7}{'RMSE':>7}{'SNR dB':>8}{'succ%':>7}{'ms/fr':>7}"]
    for (c, a), m in res.items():
        lines.append(f"{c:14}{a:16}{m['MAE']:7.2f}{m['RMSE']:7.2f}{m['SNR']:8.2f}{m['success']:7.1f}{m['ms_per_frame']:7.2f}")
    return "\n".join(lines)

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    r = evaluate(ARMS, CONDS, n)
    print(table(r))
    json.dump({f"{c}|{a}": m for (c, a), m in r.items()}, open('results.json', 'w'), indent=1)
