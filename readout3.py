import numpy as np, pickle, flyppg as F, readout as R
from scipy import ndimage as ndi
d=np.load('tune_feats.npz'); X,Y=d['X'],d['Y']
def taps(X,K): return np.concatenate([np.roll(X,k,0) for k in range(-K,K+1)],1)
def fit(idx,K,lam_s):
    A=np.vstack([np.c_[taps(X[i],K),np.ones(len(X[i]))] for i in idx]);B=np.vstack([Y[i] for i in idx])
    return np.linalg.solve(A.T@A+lam_s*np.trace(A.T@A)/A.shape[1]*np.eye(A.shape[1]),A.T@B)
hp=lambda p:p-ndi.gaussian_filter1d(p,30,axis=0)
for K in [8,12,16]:
    for lam_s in [1e-5,1e-4,1e-3]:
        Wt=fit(range(6),K,lam_s);res=[]
        for i in (6,7):
            P=np.c_[taps(X[i],K),np.ones(len(X[i]))]@Wt
            res.append([np.corrcoef(hp(np.cumsum(P,0))[:,k],hp(np.cumsum(Y[i],0))[:,k])[0,1] for k in range(2)])
        print(K,lam_s,np.round(np.mean(res,0),3),flush=True)
