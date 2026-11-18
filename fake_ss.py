"""STAND-IN for SmartSpectra minimal_example (for plumbing tests only): prints the same style of JSON lines using a naive POS-less green-ROI estimate."""
import sys, json, numpy as np, cv2
from scipy import signal
a = dict(x.lstrip('-').split('=', 1) for x in sys.argv[1:] if '=' in x); cap = cv2.VideoCapture(a['input_video_path']); fps = cap.get(5) or 30; g = []
while True:
    ok, f = cap.read()
    if not ok: break
    h, w = f.shape[:2]; g.append(f[h // 3:2 * h // 3, w // 3:2 * w // 3, 1].mean())
g = np.array(g); b, aa = signal.butter(3, (.7, 3.), 'bandpass', fs=fps); x = signal.filtfilt(b, aa, signal.detrend(g))
for s in range(int(12 * fps), len(g) + 1, int(2 * fps)):
    fr, p = signal.welch(x[max(0, s - int(12 * fps)):s], fps, nperseg=min(256, s), nfft=4096); m = (fr > .7) & (fr < 3)
    print('Got edge metrics at', int(s / fps * 1e6), 'microseconds:', json.dumps({'cardio': {'pulse_rate': [{'time': s / fps, 'value': float(fr[m][np.argmax(p[m])] * 60), 'confidence': 1.0}]}}))
print('Done.')
