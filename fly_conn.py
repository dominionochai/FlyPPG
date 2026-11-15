"""Connectome-derived T4/T5 detector bank (weights, signs, hex offsets from flyvis fib25-fib19_v2.2.json).
Assumptions (not from the connectome): fast/slow temporal filter constants, shunting-inhibition form, pixel pitch."""
import json, numpy as np
from scipy import ndimage as ndi
EXC = ('Mi1', 'Tm3', 'Tm1', 'Tm2', 'Tm4', 'Tm9')   # alpha=+1 inputs in the connectome
CJ = json.load(open('connectome/fib25-fib19_v2.2.json'))
PITCH = 1.0   # px (on 2x-downsampled grid) per hex step

def hex2xy(u, v): return PITCH * (u + 0.5 * v), PITCH * (np.sqrt(3) / 2 * v)

def load_cell(name, min_w=0.8):
    exc, inh = [], []
    for e in CJ['edges']:
        if e['tar'] != name or e['src'] in ('T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d'): continue
        for (u, v), w in e['offsets']:
            if w < min_w: continue
            (exc if e['alpha'] > 0 else inh).append((*hex2xy(u, v), float(w)))
    ne = sum(w for *_, w in exc) or 1; ni = sum(w for *_, w in inh) or 1
    return [(x, y, w / ne) for x, y, w in exc], [(x, y, w / ni) for x, y, w in inh]

SUB = {n: load_cell(n) for n in ('T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d')}

def tlp(x, tau):  # causal first-order low-pass along time
    from scipy.signal import lfilter, lfilter_zi
    a = 1 - np.exp(-1 / tau)
    zi = lfilter_zi([a], [1, a - 1])[:, None, None] * x[0][None]
    return lfilter([a], [1, a - 1], x, axis=0, zi=zi)[0]

def sh(x, x_off, y_off):  # sample input at (+x_off, +y_off): out[t,y,x] = in[t,y+y_off,x+x_off]
    return ndi.shift(x, (0, -y_off, -x_off), order=1, mode='nearest')

BETA, TAU_S, TAU_F = 3.0, 3.0, 0.7

def bank(L):
    """L: (T,h,w) luminance -> dict subtype -> (T,h,w). Subtractive-inhibition form: relu(E - beta*I)."""
    base = ndi.gaussian_filter1d(L, 45, axis=0, mode='nearest')
    c = L / (base + 1e-6) - 1                         # divisive contrast adaptation
    pols = {'on': np.maximum(c, 0), 'off': np.maximum(-c, 0)}
    filt = {(p, k): tlp(x, TAU_F if k == 'f' else TAU_S) for p, x in pols.items() for k in 'fs'}
    cache, out = {}, {}
    def get(p, k, x, y):
        key = (p, k, round(x, 2), round(y, 2))
        if key not in cache: cache[key] = sh(filt[(p, k)], x, y)
        return cache[key]
    for n, (exc, inh) in SUB.items():
        p = 'on' if n.startswith('T4') else 'off'
        E = sum(w * get(p, 'f', x, y) for x, y, w in exc)
        I = sum(w * get(p, 's', x, y) for x, y, w in inh)
        out[n] = np.maximum(E - BETA * I, 0)
    return out

def preferred_dirs():
    """Empirical PD per subtype: drift a texture in 8 directions, take argmax (no assumption)."""
    r = np.random.default_rng(0); tex = ndi.gaussian_filter(r.standard_normal((100, 100)), 1.5); tex = 0.5 + 0.1 * tex / tex.std()
    R = {}
    for a in range(0, 360, 45):
        dx, dy = 0.5 * np.cos(np.radians(a)), 0.5 * np.sin(np.radians(a))
        L = np.stack([ndi.shift(tex, (dy * t, dx * t), order=1, mode='nearest')[18:82, 18:82] for t in range(80)]).astype(np.float32)
        o = bank(L); R[a] = {n: o[n][30:].mean() for n in o}
    pd = {}
    for n in SUB:
        best = max(R, key=lambda a: R[a][n]); pd[n] = (np.cos(np.radians(best)), np.sin(np.radians(best)))
    return pd

def ds_signals(Lds):
    """Per-frame global direction-selective output vector (T,2) from the T4/T5 bank (pixel-mean, baseline removed)."""
    pd = PD; o = bank(Lds); v = np.zeros((len(Lds), 2))
    for n, (cx, cy) in pd.items():
        m = o[n].mean((1, 2)); m = m - np.median(m)
        v += np.array([cx, cy])[None] * m[:, None]
    return v
PD = preferred_dirs()
