import numpy as np
G=7; INF=10**6
def gen(M, density=0.28, seed=0):
    r=np.random.default_rng(seed)
    walls=(r.random((M,G,G))<density)
    # pick a goal in an open cell
    walls[:,0,0]=False
    gi=r.integers(0,G,M); gj=r.integers(0,G,M)
    walls[np.arange(M),gi,gj]=False
    d=np.full((M,G,G),INF,dtype=np.int32)
    d[np.arange(M),gi,gj]=0
    d[walls]=INF
    for _ in range(4*G*G):
        nd=d.copy()
        nd[:,1:,:]=np.minimum(nd[:,1:,:], d[:,:-1,:]+1)
        nd[:,:-1,:]=np.minimum(nd[:,:-1,:], d[:,1:,:]+1)
        nd[:,:,1:]=np.minimum(nd[:,:,1:], d[:,:,:-1]+1)
        nd[:,:,:-1]=np.minimum(nd[:,:,:-1], d[:,:,1:]+1)
        nd[walls]=INF
        if (nd==d).all(): break
        d=nd
    # optimal-action mask: 0=up,1=down,2=left,3=right (move to neighbour with dist-1)
    opt=np.zeros((M,G,G,4),dtype=bool)
    opt[:,1:,:,0]=(d[:,:-1,:]==d[:,1:,:]-1)
    opt[:,:-1,:,1]=(d[:,1:,:]==d[:,:-1,:]-1)
    opt[:,:,1:,2]=(d[:,:,:-1]==d[:,:,1:]-1)
    opt[:,:,:-1,3]=(d[:,:,1:]==d[:,:,:-1]-1)
    opt &= (d[...,None]<INF)
    return walls, d, opt, gi, gj

if __name__=="__main__":
    w,d,o,gi,gj=gen(20000)
    print("distance histogram over all (maze,cell) pairs, 7x7 density 0.28:")
    tot=0
    for dd in range(1,25):
        n=int((d==dd).sum())
        if dd in (1,2,3,4,6,8,10,12,14,16,18,20): 
            navg=o[d==dd].sum(1).mean() if n>0 else 0
            print(f"  dist {dd:2d}: {n:7d} cells   mean #optimal actions {navg:.2f}  -> random-policy acc {navg/4:.3f}")
        tot+=n
    print("total reachable cells:", tot)
