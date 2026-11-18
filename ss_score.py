"""Score SmartSpectra outputs against reference HR. ref.csv columns: video,hr_bpm  (video = file stem). Paired bootstrap vs the control arm (re-encode only).
Usage: python ss_score.py --out ss_out --ref ref.csv [--control control]"""
import os, re, json, argparse, numpy as np, csv
def readings(txt):
    vals = []
    def walk(o, ts):
        if isinstance(o, dict):
            for k, v in o.items():
                if k.lower() in ('pulse_rate', 'pulserate') :
                    for it in (v if isinstance(v, list) else [v]):
                        x = it if isinstance(it, (int, float)) else next((it[q] for q in ('value', 'rate', 'bpm', 'pulse_rate') if isinstance(it, dict) and isinstance(it.get(q), (int, float))), None)
                        if x is not None: vals.append(float(x))
                else: walk(v, ts)
        elif isinstance(o, list):
            for i in o: walk(i, ts)
    for m in re.finditer(r'Got edge metrics at (\d+) microseconds: (\{.*\})', txt):
        try: walk(json.loads(m.group(2)), int(m.group(1)))
        except Exception: pass
    return vals
def estimate(vals):   # median of the later half of readings (rates need ~12 s to appear; early values are least settled)
    return float(np.median(vals[len(vals) // 2:])) if vals else np.nan
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--ref', required=True); ap.add_argument('--control', default='control'); ap.add_argument('--boot', type=int, default=5000); a = ap.parse_args()
ref = {r['video']: float(r['hr_bpm']) for r in csv.DictReader(open(a.ref))}; arms = sorted(os.listdir(a.out)); E = {}
for arm in arms:
    E[arm] = {}
    for v, hr in ref.items():
        p = os.path.join(a.out, arm, v + '.txt'); E[arm][v] = estimate(readings(open(p).read())) - hr if os.path.exists(p) else np.nan
vids = [v for v in ref if all(not np.isnan(E[arm][v]) for arm in arms)]; rng = np.random.default_rng(0)
print(f'videos with readings in ALL arms: {len(vids)}/{len(ref)}\n'); print(f"{'arm':14}{'MAE':>7}{'RMSE':>7}{'<=5bpm%':>9}{'coverage%':>10}{'dMAE vs ctrl [95% CI]':>34}")
for arm in arms:
    e = np.array([E[arm][v] for v in vids]); cov = np.mean([not np.isnan(E[arm][v]) for v in ref]) * 100
    if arm == a.control or not len(vids) or a.control not in E: d = '-'
    else:
        c = np.array([E[a.control][v] for v in vids]); diff = np.abs(e) - np.abs(c); bs = [rng.choice(diff, len(diff)).mean() for _ in range(a.boot)]; d = f'{diff.mean():+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]'
    print(f'{arm:14}{np.abs(e).mean() if len(e) else np.nan:7.2f}{np.sqrt((e ** 2).mean()) if len(e) else np.nan:7.2f}{(np.abs(e) <= 5).mean() * 100 if len(e) else np.nan:9.0f}{cov:10.0f}{d:>34}')
print('\ndMAE < 0 with a CI entirely below 0 = the arm is significantly better than the re-encoded control.')
