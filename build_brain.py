import numpy as np, json, flyppg as F, readout as R, eval2 as E
from scipy import signal
v, hr = F.make_video(1000, 1, 1); pos = F.make_video.pos
X = R.feats(v); P = E.fly_motion(v); Y = np.diff(pos, axis=0, prepend=pos[:1])
n = lambda a, q=98: np.clip(a / (np.percentile(np.abs(a), q) + 1e-9), 0, 1)
Xn = np.stack([n(np.maximum(X[:, k], 0)) for k in range(8)], 1)
Ld = R.lum_ds(v); lam = n(np.abs(np.diff(Ld, axis=0, prepend=Ld[:1])).mean((1, 2)))
g = np.log(v.mean((1, 2, 3))); dg = np.abs(np.diff(g, prepend=g[0])); S = (dg > max(8 * 1.4826 * np.median(np.abs(dg - np.median(dg))), .02)).astype(float)
S = np.clip(np.convolve(S, np.hanning(9) / np.hanning(9).sum() * 3, 'same'), 0, 1)
V = n(np.hypot(*P.T)); Ve = P[:, 1] / (np.percentile(np.abs(Y[:, 1]), 98) + 1e-9); Vt = Y[:, 1] / (np.percentile(np.abs(Y[:, 1]), 98) + 1e-9)
b, a = signal.butter(3, (.7, 3.), 'bandpass', fs=30)
def pulse(x): y = signal.filtfilt(b, a, signal.detrend(x)); return np.clip(y / (3 * y.std()) * 2, -1, 1)
raw = pulse(F.rppg_proxy(v)); fly = pulse(F.rppg_proxy(F.step_comp(E.stab_fly(v))))
r = lambda a: np.round(a, 2).tolist()
D = dict(T=len(v), X=r(Xn), L=r(lam), S=r(S), V=r(V), Vt=r(np.clip(Vt, -1, 1)), Ve=r(np.clip(Ve, -1, 1)), raw=r(raw), fly=r(fly))
import trimesh, base64
m=trimesh.load('mesh/JRC2018F.ply',process=False); V0=np.asarray(m.vertices,float); c=(m.bounds[0]+m.bounds[1])/2; sc=2.3/(m.bounds[1][0]-m.bounds[0][0])
Vs=np.c_[(V0[:,0]-c[0])*sc, -(V0[:,1]-c[1])*sc, -(V0[:,2]-c[2])*sc].astype(np.float32)
G={'R':0,'cx':[0,0],'sy':[1,1],'sz':[1,1]}; CAP=.95; Rs=[]
for i,sd in enumerate([-1,1]):
    lo=Vs[Vs[:,0]*sd>.62]; xm=np.abs(lo[:,0]).max(); cxs=.5; R=xm-cxs; mid=lo[np.abs(lo[:,0])*1>.7]
    hy=np.percentile(mid[:,1],98)-np.percentile(mid[:,1],2); hz=np.percentile(mid[:,2],98)-np.percentile(mid[:,2],2)
    G['cx'][i]=sd*cxs; G['sy'][i]=float(hy/2*.85/(R*np.sin(CAP))); G['sz'][i]=float(hz/2*.85/(R*np.sin(CAP))); Rs.append(R)
G['R']=float(np.mean(Rs))*.9; D['G']=G
D['M']={'v':base64.b64encode(Vs.tobytes()).decode(),'f':base64.b64encode(np.asarray(m.faces,np.uint16).tobytes()).decode()}
print('G',{k:(np.round(v,3).tolist() if k!='R' else round(v,3)) for k,v in G.items()},'mesh verts',len(Vs))
for tpl,out in [('brain_tpl.html','brain.html'),('brain3d_tpl.html','brain3d.html')]: open(out, 'w').write(open(tpl).read().replace('/*DATA*/', json.dumps(D, separators=(',', ':'))))
print('ok', len(open('brain.html').read()) // 1024, 'KB; step frames', int(S.sum() > 0), 'corr est/true', round(float(np.corrcoef(Ve, Vt)[0, 1]), 2))
