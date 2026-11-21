"""Apply FlyPPG arms to REAL video files and re-encode each arm identically (control = re-encode only, so compression is never a confound).
Analysis runs at 96x96, corrections are applied at full resolution. Usage: python ss_preprocess.py --in videos --out processed [--arms a,b,c]"""
import os, sys, argparse, subprocess, collections, numpy as np, cv2, imageio_ffmpeg
from scipy import ndimage as ndi
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import flyppg3 as G
A = 96; ARMS = ['control', 'conventional', 'classical', 'surround', 'fly_stab', 'flyppg']

def read_small(path):
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 30; w, h = int(cap.get(3)), int(cap.get(4)); sm = []
    while True:
        ok, f = cap.read()
        if not ok: break
        sm.append(cv2.resize(f, (A, A), interpolation=cv2.INTER_AREA)[:, :, ::-1])
    cap.release(); return np.array(sm, np.float32) / 255, fps, (w, h)

def surround_gain(v):
    """Estimate shared illumination gain from the background ring; otherwise return identity."""
    lg = lambda x: np.log(np.maximum(x, 1e-6))
    bp = lambda x: ndi.gaussian_filter1d(x, 3) - ndi.gaussian_filter1d(x, 30)

    background = v[:, G.BG, :].mean((1, 2))
    foreground = v[:, 24:72, 24:72, :].mean((1, 2, 3))
    background_signal = bp(lg(background.mean(1)))
    foreground_signal = bp(lg(foreground))

    # A constant or near-constant background has undefined correlation. Treat
    # that as insufficient evidence, not permission to normalize the video.
    if np.std(background_signal) < 1e-5 or np.std(foreground_signal) < 1e-5:
        return np.ones(len(v), np.float32)

    correlation = np.corrcoef(background_signal, foreground_signal)[0, 1]
    if not np.isfinite(correlation) or correlation <= .5:
        return np.ones(len(v), np.float32)

    reference = np.median(background, axis=0)
    if np.any(~np.isfinite(reference)) or np.any(reference <= 1e-6):
        return np.ones(len(v), np.float32)

    gain = background / reference
    if not np.all(np.isfinite(gain)):
        return np.ones(len(v), np.float32)
    return gain.astype(np.float32)

def fly_disp(v, Wr, g):
    vn = v / g[:, None, None, None]; vel = (np.c_[G.taps(G.feats_local(vn)), np.ones(len(v))] @ Wr).reshape(len(v), 9, 2)
    pos = np.cumsum(vel, 0); return pos - ndi.gaussian_filter1d(pos, 30, axis=0, mode='nearest')

def farneback_flows(v):
    g8 = lambda x: np.clip(x.mean(2) * 255, 0, 255).astype(np.uint8); ref = g8(v[:30].mean(0))
    return np.stack([cv2.calcOpticalFlowFarneback(ref, g8(v[t]), None, .5, 3, 15, 3, 5, 1.2, 0) for t in range(len(v))]).astype(np.float32)

def process(src, dst, arm, Wr):
    small, fps, (w, h) = read_small(src); T = len(small); gain = np.ones(T, np.float32); field = None
    if arm in ('surround', 'flyppg'): gain = surround_gain(small)
    if arm in ('fly_stab', 'flyppg'): field = ('fly', fly_disp(small, Wr, gain if arm == 'flyppg' else np.ones(T, np.float32)))
    if arm == 'classical': field = ('flow', farneback_flows(small))
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)); sx, sy = w / A, h / A
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
                           '-c:v', 'libx264', '-crf', '12', '-preset', 'fast', '-pix_fmt', 'yuv420p', dst], stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(src); buf = collections.deque(maxlen=5)
    for t in range(T):
        ok, f = cap.read()
        if not ok: break
        f = f.astype(np.float32)
        if arm == 'conventional':
            buf.append(cv2.GaussianBlur(f, (0, 0), 1.5 * sx)); f = np.mean(buf, 0)
        if gain[t] != 1: f = f / gain[t]
        if field is not None:
            if field[0] == 'fly': uy = cv2.resize(field[1][t, :, 0].reshape(3, 3).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) * sy; ux = cv2.resize(field[1][t, :, 1].reshape(3, 3).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC) * sx
            else: fl = cv2.resize(field[1][t], (w, h), interpolation=cv2.INTER_LINEAR); ux, uy = fl[..., 0] * sx, fl[..., 1] * sy
            f = cv2.remap(f, xx + ux, yy + uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ff.stdin.write(np.clip(f, 0, 255).astype(np.uint8).tobytes())
    cap.release(); ff.stdin.close(); ff.wait()

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--in', dest='inp', required=True); ap.add_argument('--out', required=True); ap.add_argument('--arms', default=','.join(ARMS)); a = ap.parse_args()
    Wr = np.load('readout_W3.npy'); vids = sorted(f for f in os.listdir(a.inp) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')))
    for arm in a.arms.split(','):
        os.makedirs(os.path.join(a.out, arm), exist_ok=True)
        for v in vids:
            dst = os.path.join(a.out, arm, os.path.splitext(v)[0] + '.mp4'); print(arm, v, flush=True); process(os.path.join(a.inp, v), dst, arm, Wr)
