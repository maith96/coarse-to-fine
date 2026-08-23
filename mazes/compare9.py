"""Retention + rollout comparison: ancestor chain vs flat controls.

Two questions the ladder never asked:
  1. retention  - one-step accuracy at EVERY horizon, not just the one trained on.
                  The ancestor saw d=2,4,8,16,24; each control saw only its own.
  2. rollout    - does the one-step policy compose into something that solves mazes?
                  Reported raw, and with illegal moves masked out, to separate
                  "doesn't know walls exist" from "doesn't know which way to go".

Both read off finished checkpoints, so neither can be compute-confounded.

Caveat on the rollout: training samples cells at distance EXACTLY d, so a
rollout passes through distances no condition ever trained on.  It is evidence
about what the maze task measures, not a fair head-to-head.  The retention
table is the clean comparison - every cell is a horizon the ancestor trained on.
"""
import json, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
torch.set_num_threads(1)
from lad9 import PolNet, index, ev, SPLIT, M, tok, walls, d, G

DELTA=[(-1,0),(1,0),(0,-1),(0,1)]
NETS=[("anc_d24 (chain, 600+2400)","l9_anc_d24"),
      ("ctrl_d24 (flat, 600)",     "l9_ctrl_d24"),
      ("ctrl_d24 (flat, 3000)",    "sw9_ctrl_d24_n3000_s0")]
HOR=[2,4,8,16,24]

def load(ck):
    net=PolNet(); net.load_state_dict(torch.load(f"ckpt/{ck}.pt")['n']); net.eval(); return net

def rollout(net,dist,n=200,seed=4,cap=70,legal=False):
    rng=np.random.default_rng(seed)
    mi,ii,jj=index(dist,SPLIT,M); pick=rng.choice(len(mi),n,replace=False)
    m=mi[pick].copy(); i=ii[pick].copy(); j=jj[pick].copy()
    alive=np.ones(n,bool); reached=np.zeros(n,bool); wall=np.zeros(n,bool)
    looped=np.zeros(n,bool); steps=np.zeros(n,int); seen=[set() for _ in range(n)]
    for _ in range(cap):
        at_goal=alive&(d[m,i,j]==0); reached|=at_goal; alive&=~at_goal
        for k in np.nonzero(alive)[0]:
            if (i[k],j[k]) in seen[k]: looped[k]=True; alive[k]=False
            else: seen[k].add((i[k],j[k]))
        idx=np.nonzero(alive)[0]
        if len(idx)==0: break
        x=tok[m[idx]].copy(); x[np.arange(len(idx)),i[idx],j[idx]]=3
        with torch.no_grad():
            lg=net(torch.from_numpy(x.reshape(len(idx),-1)),
                   torch.from_numpy(i[idx]*G+j[idx])).numpy()
        if legal:
            for c,k in enumerate(idx):
                for a,(di,dj) in enumerate(DELTA):
                    ni,nj=i[k]+di,j[k]+dj
                    if not(0<=ni<G and 0<=nj<G) or walls[m[k],ni,nj]: lg[c,a]=-1e9
        act=lg.argmax(1)
        for c,k in enumerate(idx):
            di,dj=DELTA[act[c]]; ni,nj=i[k]+di,j[k]+dj
            if not(0<=ni<G and 0<=nj<G) or walls[m[k],ni,nj]: wall[k]=True; alive[k]=False
            else: i[k],j[k]=ni,nj; steps[k]+=1
    return dict(reached=float(reached.mean()), wall=float(wall.mean()),
                loop=float(looped.mean()),
                steps=float(steps[reached].mean()) if reached.any() else None)

OUT={"retention":{}, "random_baseline":{}}
print("== retention: one-step accuracy at every horizon (held-out) ==")
print(f"{'net':28s} " + " ".join(f"d={h:<7d}" for h in HOR))
for name,ck in NETS:
    net=load(ck); row=[]; OUT["retention"][name]={}
    for h in HOR:
        a,b=ev(net,h,nb=6); OUT["retention"][name][h]=float(a)
        OUT["random_baseline"][h]=float(b); row.append(f"{a:.4f}  ")
    print(f"{name:28s} " + " ".join(row))
print(f"{'(random baseline)':28s} " + " ".join(f"{OUT['random_baseline'][h]:.4f}  " for h in HOR))

for dist in (24,8):
    print(f"\n== rollout from held-out cells at d={dist} (200 starts, cap 70) ==")
    print(f"{'net':28s} {'reached':>8} {'wall':>7} {'loop':>7} | masked {'reached':>8} {'loop':>7}")
    OUT[f"rollout_d{dist}"]={}
    for name,ck in NETS:
        net=load(ck); raw=rollout(net,dist); msk=rollout(net,dist,legal=True)
        OUT[f"rollout_d{dist}"][name]={"raw":raw,"legal_masked":msk}
        print(f"{name:28s} {raw['reached']:8.3f} {raw['wall']:7.3f} {raw['loop']:7.3f} |"
              f"       {msk['reached']:8.3f} {msk['loop']:7.3f}")

json.dump(OUT,open("../results/retention9.json","w"),indent=1)
print("\nwrote results/retention9.json")
