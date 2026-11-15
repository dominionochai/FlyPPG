import numpy as np, json, flyppg as F, eval2 as E, eval3 as E3
from scipy import stats
from sklearn.metrics import roc_auc_score
rows = []
for c in ['motion', 'motion+illum', 'compressed(q25)+motion', 'low-light+motion']:
    for k in range(4):
        v, hr = E.gen(2000 + k, c); sf = np.hypot(*E3.fly_motion2(v).T); sc = np.hypot(*np.diff(E.classical_pos(v), axis=0, prepend=0).T)
        sig = F.rppg_proxy(v); win, st = 300, 150
        for s in range(0, len(sig) - win + 1, st):
            est, snr = F.window_est(sig[s:s + win], hr)[0]
            rows.append((abs(est - hr), snr, sf[s:s + win].mean(), sc[s:s + win].mean()))
a = np.array(rows); err, snr, sf, sc = a.T; bad = err > np.percentile(err, 75)
out = dict(n_windows=len(a), spearman_fly_speed_vs_err=float(stats.spearmanr(sf, err)[0]), spearman_classical_speed_vs_err=float(stats.spearmanr(sc, err)[0]),
           spearman_snr_vs_err=float(stats.spearmanr(snr, err)[0]), auc_fly=float(roc_auc_score(bad, sf)), auc_classical=float(roc_auc_score(bad, sc)), auc_snr=float(roc_auc_score(bad, -snr)),
           median_err=float(np.median(err)), p75_err=float(np.percentile(err, 75)), max_err=float(err.max()))
json.dump(out, open('analysis.json', 'w'), indent=1); print(json.dumps(out, indent=1))
