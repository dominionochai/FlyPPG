import numpy as np, flyppg as F
from scipy import ndimage as ndi
def fly_v2(v,m0,tw):
    o=F.pre_fly(v,False,True)
    L=ndi.gaussian_filter(o.mean(3)/(o.mean()+1e-6),(0,1,1))
    g=ndi.uniform_filter(F.fly_motion_energy(L),(1,5,5)).mean((1,2))
    w=ndi.gaussian_filter1d(1/(1+(g/m0)**2),tw,mode='nearest')
    bad=w<0.5; idx=np.arange(len(o)); good=idx[~bad]
    if len(good)<10 or not bad.any(): return o
    near=good[np.abs(idx[:,None]-good[None]).argmin(1)]
    return np.where(bad[:,None,None,None],o[near],o)
if __name__=='__main__':
    conds={'motion':(1,0),'motion+illum':(1,1),'static':(0,0)}
    vids={k:[F.make_video(5000+i,*m) for i in range(3)] for k,m in conds.items()}
    for m0 in [1e-4,2e-4,4e-4,8e-4,1.6e-3]:
        out={}
        for k,vs in vids.items():
            e=[a-hr for v,hr in vs for a,_ in F.window_est(F.rppg_proxy(fly_v2(v,m0,1.0)),hr)]
            out[k]=round(float(np.abs(e).mean()),2)
        print(m0,out)
    for k,vs in vids.items():
        e=[a-hr for v,hr in vs for a,_ in F.window_est(F.rppg_proxy(v),hr)];print('raw',k,round(float(np.abs(e).mean()),2))
