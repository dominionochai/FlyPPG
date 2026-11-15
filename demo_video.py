import numpy as np, cv2, flyppg as F, eval2 as E, eval3 as E3
from scipy import signal
# pick the motion+illum clip where raw is worst (most instructive, not cherry-picked toward FlyPPG's favour beyond that)
best = None
for k in range(6):
    v, hr = E.gen(1000 + k, 'motion+illum'); sig = F.rppg_proxy(v)
    e = np.mean([abs(a - hr) for a, _ in F.window_est(sig, hr)])
    if best is None or e > best[0]: best = (e, k, v, hr)
_, k, v, hr = best; fly = F.step_comp(E3.stab_fly2(v)); print('clip seed', 1000 + k, 'true HR', round(hr, 1))
b, a = signal.butter(3, (.7, 3.), 'bandpass', fs=30)
def pulse(x): return signal.filtfilt(b, a, signal.detrend(x))
sr, sf = F.rppg_proxy(v), F.rppg_proxy(fly); pr, pf = pulse(sr), pulse(sf); sc = 3 * pf.std()
Wd, Hd, S = 624, 520, 288
vw = cv2.VideoWriter('demo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (Wd, Hd))
def hr_at(sig, t):
    return F.window_est(sig[t - 300:t], hr)[0][0] if t >= 300 else None
def put(img, s, x, y, sc_=.5, col=(230, 230, 230), th=1): cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc_, col, th, cv2.LINE_AA)
rr, rf = None, None
for t in range(len(v)):
    img = np.full((Hd, Wd, 3), (14, 10, 6), np.uint8)
    for j, (clip, name, col) in enumerate([(v, 'RAW VIDEO', (200, 200, 200)), (fly, 'FLYPPG (illum. step + T4/T5 stabiliser)', (255, 230, 90))]):
        f = np.clip(clip[t][..., ::-1] * 255, 0, 255).astype(np.uint8); f = cv2.resize(f, (S, S), interpolation=cv2.INTER_CUBIC)
        x0 = 16 + j * (S + 16); img[48:48 + S, x0:x0 + S] = f; cv2.rectangle(img, (x0 + 84, 48 + 84), (x0 + 204, 48 + 204), (0, 255, 255), 1)
        put(img, name, x0, 36, .42, col)
    if t % 30 == 0 and t >= 300: rr, rf = hr_at(sr, t), hr_at(sf, t)
    put(img, f'true {hr:5.1f} bpm', 16, 360, .55, (255, 255, 255))
    put(img, f'raw {rr:5.1f}  (err {abs(rr - hr):4.1f})' if rr else 'raw ...', 200, 360, .55, (200, 200, 200))
    put(img, f'FlyPPG {rf:5.1f}  (err {abs(rf - hr):4.1f})' if rf else 'FlyPPG ...', 400, 360, .55, (255, 230, 90))
    for arr, col in [(pr, (170, 170, 170)), (pf, (255, 230, 90))]:
        pts = [(16 + int(i * (Wd - 32) / 150), int(430 - 50 * np.clip(arr[(t - 149 + i) % len(arr)] / sc, -1, 1))) for i in range(150)]
        cv2.polylines(img, [np.array(pts, np.int32)], False, col, 1, cv2.LINE_AA)
    put(img, 'pulse signal from the face ROI (grey raw, yellow FlyPPG)', 16, 392, .4, (150, 170, 190))
    put(img, 'SYNTHETIC clip + proxy extractor (not SmartSpectra) - illustration of the pipeline, not a validation', 16, Hd - 12, .38, (120, 140, 160))
    vw.write(img)
    if t in (120, 300, 480): cv2.imwrite(f'demo_still_{t}.png', img)
vw.release(); print('ok')
