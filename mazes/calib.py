"""Regime calibration: does a from-scratch policy trained on d<=k actually SOLVE mazes?

The published maze work samples cells at distance EXACTLY d, so a greedy rollout
passes through distances no condition ever trained on -- FINDINGS section 7 rightly
refuses to call that a head-to-head. Training on d<=k instead makes rollout
in-distribution for every condition, and makes the real task (reach the goal) the
metric rather than a one-step classifier on a fixed-distance slice.

This script establishes whether the REGIME exists before any ladder is built:

  d*  = the largest horizon k at which a from-scratch net trained on d<=k still
        rolls out to the goal at >= 50% from held-out starts at distance k.

If d* is well short of the maximum horizon the grid supports, flat training has a
reach limit and a ladder has something to beat. If flat solves everything, there
is no regime at this grid size and the test is not worth running.

Also reports, per section 2, the final train loss at every rung: falling = entropy
floor (ladder viable), pinned at ln4 = void (nothing crosses).

  python calib.py <seconds-budget> [G] [steps]

Uses the corrected optimal-action loss, not lad9.py's (FINDINGS section 11).
"""
import sys, os, json, time, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
torch.set_num_threads(4)
from mazeN import gen, INF

G=int(sys.argv[2]) if len(sys.argv)>2 else 13
STEPS=int(sys.argv[3]) if len(sys.argv)>3 else 3000
DENS={9:0.32, 13:0.30, 17:0.28}[G]
K=[2,4,8,16,24]
BS=128; DELTA=[(-1,0),(1,0),(0,-1),(0,1)]

CACHE=f"ckpt/rg{G}.npz"
os.makedirs("ckpt",exist_ok=True)
if os.path.exists(CACHE):
    z=np.load(CACHE); walls,d,opt=z['w'],z['d'],z['o']
else:
    walls,d,opt=gen(40000,G,density=DENS,seed=1); np.savez_compressed(CACHE,w=walls,d=d,o=opt)
M=walls.shape[0]; SPLIT=int(0.8*M)
tok=np.where(walls,1,0).astype(np.int64); tok[d==0]=2

IDX={}
def index(lo_d, hi_d, lo, hi):
    """cells with lo_d <= dist <= hi_d, in mazes [lo,hi)."""
    key=(lo_d,hi_d,lo,hi)
    if key not in IDX:
        m,i,j=np.nonzero(((d>=lo_d)&(d<=hi_d))[lo:hi]); IDX[key]=(m+lo,i,j)
    return IDX[key]

def batch(idx,bs,rng):
    mi,ii,jj=idx; k=rng.integers(0,len(mi),bs); m,i,j=mi[k],ii[k],jj[k]
    x=tok[m].copy(); x[np.arange(bs),i,j]=3
    return torch.from_numpy(x.reshape(bs,-1)), torch.from_numpy(i*G+j), torch.from_numpy(opt[m,i,j])

class PolNet(nn.Module):
    def __init__(s,nl=3,dm=64):
        super().__init__()
        s.tok=nn.Embedding(4,dm); s.pos=nn.Embedding(G*G,dm)
        s.ls=nn.ModuleList([nn.TransformerEncoderLayer(dm,4,4*dm,dropout=0.0,batch_first=True,
            norm_first=True,activation="gelu") for _ in range(nl)])
        s.ln=nn.LayerNorm(dm); s.hd=nn.Linear(dm,4)
    def forward(s,x,p):
        h=s.tok(x)+s.pos.weight[None]
        for l in s.ls: h=l(h)
        return s.hd(s.ln(h[torch.arange(x.shape[0]),p]))

@torch.no_grad()
def onestep(net,dist,nb=4,bs=512,seed=999):
    te=index(dist,dist,SPLIT,M)
    if len(te[0])<bs: return None,None
    rng=np.random.default_rng(seed); net.eval(); c=t=0; base=0.
    for _ in range(nb):
        x,p,y=batch(te,bs,rng); pr=net(x,p).argmax(-1)
        c+=y[torch.arange(bs),pr].sum().item(); t+=bs; base+=y.float().sum(1).mean().item()/4
    net.train(); return c/t, base/nb

@torch.no_grad()
def rollout(net,dist,n=200,seed=4,policy="net"):
    """greedy rollout from held-out cells at distance exactly `dist`.

    policy="rand"  uniform random action (walks into walls)
    policy="mask"  uniform random among LEGAL moves -- the honest null
    """
    mi,ii,jj=index(dist,dist,SPLIT,M)
    if len(mi)<n: return None
    rng=np.random.default_rng(seed); pick=rng.choice(len(mi),n,replace=False)
    m,i,j=mi[pick].copy(),ii[pick].copy(),jj[pick].copy()
    cap=max(4*dist,30)
    alive=np.ones(n,bool); reached=np.zeros(n,bool); wall=np.zeros(n,bool); looped=np.zeros(n,bool)
    seen=[set() for _ in range(n)]
    if net is not None: net.eval()
    for _ in range(cap):
        at=alive&(d[m,i,j]==0); reached|=at; alive&=~at
        for k in np.nonzero(alive)[0]:
            if (i[k],j[k]) in seen[k]: looped[k]=True; alive[k]=False
            else: seen[k].add((i[k],j[k]))
        idx=np.nonzero(alive)[0]
        if len(idx)==0: break
        if policy=="net":
            x=tok[m[idx]].copy(); x[np.arange(len(idx)),i[idx],j[idx]]=3
            lg=net(torch.from_numpy(x.reshape(len(idx),-1)),
                   torch.from_numpy(i[idx]*G+j[idx])).numpy()
            act=lg.argmax(1)
        elif policy=="rand":
            act=rng.integers(0,4,len(idx))
        else:                                   # legal-move random walk
            act=np.zeros(len(idx),dtype=int)
            for c,k in enumerate(idx):
                ok=[a for a,(di,dj) in enumerate(DELTA)
                    if 0<=i[k]+di<G and 0<=j[k]+dj<G and not walls[m[k],i[k]+di,j[k]+dj]]
                act[c]=rng.choice(ok) if ok else 0
        for c,k in enumerate(idx):
            di,dj=DELTA[act[c]]; ni,nj=i[k]+di,j[k]+dj
            if not(0<=ni<G and 0<=nj<G) or walls[m[k],ni,nj]: wall[k]=True; alive[k]=False
            else: i[k],j[k]=ni,nj
    if net is not None: net.train()
    return dict(reached=float(reached.mean()),wall=float(wall.mean()),loop=float(looped.mean()))

def train(k,BUD,t0):
    ck=f"ckpt/cal{G}_k{k}_n{STEPS}.pt"
    torch.manual_seed(0); net=PolNet()
    o=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.OneCycleLR(o,3e-3,total_steps=STEPS,pct_start=0.15)
    s0=0; curve=[]
    if os.path.exists(ck):
        c=torch.load(ck); net.load_state_dict(c['n']); o.load_state_dict(c['o'])
        sch.load_state_dict(c['s']); s0=c['step']; curve=c['c']
    tr=index(1,k,0,SPLIT); rng=np.random.default_rng(11)
    for _ in range(s0): rng.integers(0,len(tr[0]),BS)
    for s in range(s0+1,STEPS+1):
        x,p,y=batch(tr,BS,rng)
        # -log( sum of probability mass on the optimal actions ).
        # lad9.py multiplies log_softmax by the mask instead, which lets the
        # non-optimal zeros into the logsumexp as exp(0)=1 -- that computes
        # -log(n_nonoptimal + sum p), puts chance at -1.18 rather than ln4, and
        # shrinks the gradient by up to 300x on exactly the cells the net gets
        # most wrong. See FINDINGS section 11.
        loss=-(F.log_softmax(net(x,p),-1).masked_fill(~y,float('-inf'))).logsumexp(-1).mean()
        o.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); o.step(); sch.step()
        if s%200==0 or s==1: curve.append((s,round(loss.item(),4)))
        if time.time()-t0>BUD or s==STEPS:
            torch.save({'n':net.state_dict(),'o':o.state_dict(),'s':sch.state_dict(),
                        'step':s,'c':curve},ck)
            if s<STEPS: print(f"  k={k} {s}/{STEPS} loss {loss.item():.3f}",flush=True); return None
    return net,curve

if __name__=="__main__":
    BUD=float(sys.argv[1]); t0=time.time()
    RES=f"../results/calib{G}.json"
    res=json.load(open(RES)) if os.path.exists(RES) else {}
    if "baseline" not in res:
        res["baseline"]={str(k):{p:rollout(None,k,policy=p) for p in ("rand","mask")} for k in K}
        json.dump(res,open(RES,"w"),indent=1)
        print(f"G={G} baselines (rollout success): " + "  ".join(
            f"d{k}: rand {res['baseline'][str(k)]['rand']['reached']:.3f} "
            f"masked {res['baseline'][str(k)]['mask']['reached']:.3f}" for k in K),flush=True)
    for k in K:
        key=f"k{k}_n{STEPS}"
        if key in res: continue
        if time.time()-t0>BUD: print("PAUSE",flush=True); sys.exit(0)
        out=train(k,BUD,t0)
        if out is None: sys.exit(0)
        net,curve=out
        acc,rb=onestep(net,k)
        r={"train_loss_first":curve[0][1],"train_loss_last":curve[-1][1],
           "onestep_acc":acc,"onestep_random":rb,
           "rollout":{str(h):rollout(net,h) for h in K if h<=k}}
        res[key]=r; json.dump(res,open(RES,"w"),indent=1)
        rr=r["rollout"][str(k)]
        print(f"[{time.time()-t0:5.0f}s] trained d<={k:2d} | loss {curve[0][1]:.3f}->{curve[-1][1]:.3f} "
              f"(ln4={np.log(4):.3f}) | 1-step {acc:.3f} (rand {rb:.3f}) | "
              f"ROLLOUT@d{k} {rr['reached']:.3f} (wall {rr['wall']:.2f} loop {rr['loop']:.2f})",flush=True)
    print("ALLDONE",flush=True)
