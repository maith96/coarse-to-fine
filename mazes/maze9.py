import numpy as np
G=9; INF=10**6
def gen(M, density=0.30, seed=0):
    r=np.random.default_rng(seed)
    walls=(r.random((M,G,G))<density)
    gi=r.integers(0,G,M); gj=r.integers(0,G,M)
    walls[np.arange(M),gi,gj]=False
    d=np.full((M,G,G),INF,dtype=np.int32); d[np.arange(M),gi,gj]=0; d[walls]=INF
    for _ in range(6*G*G):
        nd=d.copy()
        nd[:,1:,:]=np.minimum(nd[:,1:,:],d[:,:-1,:]+1); nd[:,:-1,:]=np.minimum(nd[:,:-1,:],d[:,1:,:]+1)
        nd[:,:,1:]=np.minimum(nd[:,:,1:],d[:,:,:-1]+1); nd[:,:,:-1]=np.minimum(nd[:,:,:-1],d[:,:,1:]+1)
        nd[walls]=INF
        if (nd==d).all(): break
        d=nd
    opt=np.zeros((M,G,G,4),dtype=bool)
    opt[:,1:,:,0]=(d[:,:-1,:]==d[:,1:,:]-1); opt[:,:-1,:,1]=(d[:,1:,:]==d[:,:-1,:]-1)
    opt[:,:,1:,2]=(d[:,:,:-1]==d[:,:,1:]-1); opt[:,:,:-1,3]=(d[:,:,1:]==d[:,:,:-1]-1)
    opt &= (d[...,None]<INF)
    return walls,d,opt
