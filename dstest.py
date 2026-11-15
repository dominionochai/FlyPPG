import numpy as np, fly_conn as C, sys, time
from scipy import ndimage as ndi
r=np.random.default_rng(0); tex=ndi.gaussian_filter(r.standard_normal((100,100)),1.5); tex=0.5+0.1*tex/tex.std()
def vid(ang,v=0.5):
    dx,dy=v*np.cos(np.radians(ang)),v*np.sin(np.radians(ang))
    return np.stack([ndi.shift(tex,(dy*t,dx*t),order=1,mode='nearest')[18:82,18:82] for t in range(80)]).astype(np.float32)
V={a:vid(a) for a in range(0,360,45)}
pre={}
for a,L in V.items():
    base=ndi.gaussian_filter1d(L,45,axis=0,mode='nearest'); c=L/(base+1e-6)-1
    pre[a]=(c,c.std()+1e-9)
def run(form,beta,ts,tf=0.7):
    R={}
    for a,(c,s) in pre.items():
        on,off=np.maximum(c,0),np.maximum(-c,0)
        fo,so,ff,sf=C.tlp(on,tf),C.tlp(on,ts),C.tlp(off,tf),C.tlp(off,ts)
        row={}
        for n,(exc,inh) in C.SUB.items():
            f,sl=(fo,so) if n.startswith('T4') else (ff,sf)
            E=sum(w*C.sh(f[30:],x,y) for x,y,w in exc); I=sum(w*C.sh(sl[30:],x,y) for x,y,w in inh)
            o=E/(1+beta*I/s) if form=='shunt' else np.maximum(E-beta*I,0)
            row[n]=o.mean()
        R[a]=row
    out=[]
    for n in C.SUB:
        v=np.array([R[a][n] for a in sorted(R)]); k=v.argmax(); out.append((v[k]-v[(k+4)%8])/(v[k]+v[(k+4)%8]))
    return np.mean(out)
t=time.time()
for form,beta,ts in [('shunt',3,3),('shunt',10,3),('sub',1,3),('sub',3,3),('sub',1,6)]:
    print(form,beta,ts,round(run(form,beta,ts),3),round(time.time()-t),flush=True)
