import sys, os, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
from gatelm import LM, batch, tgt, tr_ids, va_ids, baseline, DEPTH, CTX, LEVELS
import corpus as CO

LV=LEVELS
STEPS=CO.C["chain_steps"]       # starved: gate showed loss still falling at 400-800

def expand(state, Lold, Lnew, ncls_new):
    """head rows inherited from parent cluster: coarse prediction, uniform within."""
    s=dict(state); d=Lnew-Lold
    par=torch.arange(ncls_new)>>d
    s['hd.weight']=state['hd.weight'][par].clone()
    s['hd.bias']=state['hd.bias'][par].clone()
    return s

def ev(net,L,nb=6,bs=32,seed=7):
    rng=np.random.default_rng(seed); net.eval(); ce=0.; c=t=0
    with torch.no_grad():
        for _ in range(nb):
            x,y=batch(va_ids,bs,rng); lg=net(x); yy=tgt(y,L)
            ce+=F.cross_entropy(lg.reshape(-1,2**L),yy.reshape(-1)).item()
            c+=(lg.argmax(-1)==yy).sum().item(); t+=yy.numel()
    net.train(); return ce/nb, c/t

def train(L, state, key, BUD, t0):
    ncls=2**L; bs=CO.bs(L); ck=CO.ck(f"ch_{key}")
    torch.manual_seed(0); net=LM(ncls)
    if state is not None and not os.path.exists(ck): net.load_state_dict(state)
    o=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.OneCycleLR(o,3e-3,total_steps=STEPS,pct_start=0.15)
    s0=0; zs=None
    if os.path.exists(ck):
        c=torch.load(ck); net.load_state_dict(c['n']); o.load_state_dict(c['o']); sch.load_state_dict(c['s']); s0=c['step']; zs=c['zs']
    else:
        zs=ev(net,L)[0]
    rng=np.random.default_rng(5)
    for _ in range(s0): rng.integers(0,len(tr_ids)-CTX-1,bs)
    for s in range(s0+1,STEPS+1):
        x,y=batch(tr_ids,bs,rng)
        loss=F.cross_entropy(net(x).reshape(-1,ncls), tgt(y,L).reshape(-1))
        o.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); o.step(); sch.step()
        if time.time()-t0>BUD or s==STEPS:
            torch.save({'n':net.state_dict(),'o':o.state_dict(),'s':sch.state_dict(),'step':s,'zs':zs},ck)
            if s<STEPS: print(f"  {key} paused {s}/{STEPS}",flush=True); return None
    ce,acc=ev(net,L); return {'ce':ce,'acc':acc,'zs':zs,'state':net.state_dict()}

if __name__=="__main__":
    BUD=float(sys.argv[1]); t0=time.time()
    CJ=CO.out("chain")
    res=json.load(open(CJ)) if os.path.exists(CJ) else {}
    jobs=[("anc_L1",1,None)]
    for i,L in enumerate(LV[1:],1): jobs.append((f"anc_L{L}",L,LV[i-1]))
    for L in LV[1:]: jobs.append((f"ctrl_L{L}",L,None))
    for key,L,prev in jobs:
        if key in res: continue
        if time.time()-t0>BUD: print("PAUSE",flush=True); sys.exit(0)
        st=None
        if prev is not None:
            st=expand(torch.load(CO.ck(f"ch_anc_L{prev}"))['n'], prev, L, 2**L)
        r=train(L,st,key,BUD,t0)
        if r is None: sys.exit(0)
        r.pop('state')
        res[key]={k:float(v) for k,v in r.items()}
        json.dump(res,open(CJ,"w"))
        b,bce=baseline(L)
        print(f"[{time.time()-t0:4.0f}s] {key:9s} zeroshot CE {r['zs']:.3f} -> final {r['ce']:.4f} "
              f"(marginal {bce:.3f}, gain {bce-r['ce']:+.3f})",flush=True)
    print("ALLDONE",flush=True)
