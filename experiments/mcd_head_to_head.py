"""Same-recording MCD-rPPG head-to-head: FlyPPG vs SmartSpectra vs synchronized PPG."""
import argparse, json, re
from pathlib import Path
import cv2
import numpy as np
from scipy import signal
import flyppg3 as fly

FPS = 24.0
WINDOW = 10.0
STRIDE = 5.0

def pulse_bpm(x, fps):
    x = signal.detrend(np.asarray(x, dtype=float))
    sos = signal.butter(3, [0.7, 3.0], btype="bandpass", fs=fps, output="sos")
    x = signal.sosfiltfilt(sos, x)
    f, p = signal.welch(x, fs=fps, nperseg=len(x), nfft=max(4096, len(x)*4))
    band = (f >= 0.7) & (f <= 3.0)
    return float(f[band][np.argmax(p[band])] * 60)

def load_frames(video):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    frames, box = [], None
    while True:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
        if len(faces):
            x,y,bw,bh = max(faces, key=lambda b:b[2]*b[3]); box=(x,y,bw,bh)
        if box is None:
            side=min(h,w)//2; x=(w-side)//2; y=(h-side)//2; bw=bh=side
        else: x,y,bw,bh=box
        side=int(max(bw,bh)*1.8); cx=x+bw//2; cy=y+bh//2
        xa=max(0,cx-side//2); ya=max(0,cy-side//2)
        xb=min(w,xa+side); yb=min(h,ya+side)
        xa=max(0,xb-side); ya=max(0,yb-side)
        crop=frame[ya:yb,xa:xb]
        if crop.size:
            frames.append(cv2.cvtColor(cv2.resize(crop,(96,96)),cv2.COLOR_BGR2RGB).astype(np.float32)/255)
    cap.release()
    if len(frames)<FPS*30: raise RuntimeError(f"Only {len(frames)} usable frames")
    return np.asarray(frames,dtype=np.float32), float(fps)

def sdk_samples(path):
    data=json.loads(Path(path).read_text())
    out=[]
    def walk(node, callback=None):
        if isinstance(node,dict):
            ts=node.get("timestamp_us", callback)
            for k,v in node.items():
                if re.sub(r"[^a-z]","",k.lower())=="pulserate":
                    for item in (v if isinstance(v,list) else [v]):
                        if isinstance(item,dict):
                            val=next((item[q] for q in ("value","rate","bpm") if isinstance(item.get(q),(int,float))),None)
                            stamp=item.get("timestamp",ts)
                        else: val=item; stamp=ts
                        if isinstance(val,(int,float)): out.append((float(stamp or 0)/1e6,float(val)))
                else: walk(v,ts)
        elif isinstance(node,list):
            for v in node: walk(v,callback)
    walk(data)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video",required=True); ap.add_argument("--reference",required=True)
    ap.add_argument("--smartspectra"); ap.add_argument("--out",required=True)
    ap.add_argument("--weights",default="readout_W3.npy")
    a=ap.parse_args()
    frames,fps=load_frames(a.video)
    if abs(fps-FPS)>0.5: raise RuntimeError(f"Unexpected video FPS: {fps}")
    ref=np.loadtxt(a.reference)[:,0]
    n=min(len(frames),len(ref)); frames=frames[:n]; ref=ref[:n]
    weights=np.load(a.weights)
    fly_frames=fly.stab_fly_local(frames,weights)
    fly_signal=fly.rppg_pos(fly_frames,l=max(2,round(1.6*fps)))
    raw_signal=fly.rppg_pos(frames,l=max(2,round(1.6*fps)))
    ss=sdk_samples(a.smartspectra) if a.smartspectra and Path(a.smartspectra).exists() else []
    rows=[]
    size=round(WINDOW*fps); step=round(STRIDE*fps)
    for start in range(0,n-size+1,step):
        stop=start+size; center=(start+stop-1)/(2*fps)
        truth=pulse_bpm(ref[start:stop],fps)
        raw=pulse_bpm(raw_signal[start:stop],fps)
        fp=pulse_bpm(fly_signal[start:stop],fps)
        ssw=[v for t,v in ss if center-WINDOW/2 <= t <= center+WINDOW/2]
        rows.append({"center_s":center,"reference_bpm":truth,"raw_pos_bpm":raw,
                     "flyppg_bpm":fp,"smartspectra_bpm":float(np.median(ssw)) if ssw else None})
    report={"recording":Path(a.video).name,"fps":fps,"frames_used":n,"duration_s":n/fps,
            "windows_total":len(rows),"arms":{},"windows":rows,
            "reference_method":"dominant 0.7-3.0 Hz frequency of synchronized ppg_sync column 1 per 10s window",
            "flyppg_method":"FlyPPG v3 local motion stabilization followed by POS; same ROI for raw POS control",
            "smartspectra_samples":len(ss),"smartspectra_status":"measured" if ss else "not run: SDK output missing"}
    for arm,key in (("raw_POS","raw_pos_bpm"),("FlyPPG_v3","flyppg_bpm"),("SmartSpectra","smartspectra_bpm")):
        valid=[r for r in rows if r[key] is not None]
        err=np.array([r[key]-r["reference_bpm"] for r in valid])
        report["arms"][arm]={"windows_scored":len(valid),"coverage_pct":100*len(valid)/max(1,len(rows)),
            "mae_bpm":float(np.mean(np.abs(err))) if len(err) else None,
            "rmse_bpm":float(np.sqrt(np.mean(err**2))) if len(err) else None,
            "within_5_bpm_pct_of_all_windows":100*sum(abs(x)<=5 for x in err)/max(1,len(rows))}
    Path(a.out).write_text(json.dumps(report,indent=2))
    print(json.dumps(report["arms"],indent=2))
if __name__=="__main__": main()
