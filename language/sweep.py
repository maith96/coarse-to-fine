import sys, os, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
from gatelm import LM, batch, tgt, tr_ids, va_ids, baseline, DEPTH, CTX, LEVELS
from chain import expand, ev
import corpus as CO

L=DEPTH; NC=2**L; PREV=LEVELS[-2]; BS=CO.bs(L)
def run(cond, steps, seed, BUD, t0):
    key=f"{cond}_n{steps}_s{seed}"; ck=CO.ck(f"sw_{key}")
    torch.manual_seed(seed); net=LM(NC)
    if cond=="anc" and not os.path.exists(ck):
        net.load_state_dict(expand(torch.load(CO.ck(f"ch_anc_L{PREV}"))['n'],PREV,L,NC))
    o=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.OneCycleLR(o,3e-3,total_steps=steps,pct_start=0.15)
    s0=0
    if os.path.exists(ck):
        c=torch.load(ck); net.load_state_dict(c['n']); o.load_state_dict(c['o']); sch.load_state_dict(c['s']); s0=c['step']
    rng=np.random.default_rng(5+seed)
    for _ in range(s0): rng.integers(0,len(tr_ids)-CTX-1,BS)
    for s in range(s0+1,steps+1):
        x,y=batch(tr_ids,BS,rng)
        loss=F.cross_entropy(net(x).reshape(-1,NC), tgt(y,L).reshape(-1))
        o.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); o.step(); sch.step()
        if time.time()-t0>BUD or s==steps:
            torch.save({'n':net.state_dict(),'o':o.state_dict(),'s':sch.state_dict(),'step':s},ck)
            if s<steps: print(f"  {key} {s}/{steps}",flush=True); return None
    ce,acc=ev(net,L); return ce

if __name__=="__main__":
    BUD=float(sys.argv[1]); t0=time.time()
    SJ=CO.out("sweep")
    res=json.load(open(SJ)) if os.path.exists(SJ) else {}
    jobs=[]
    for n in [600,1200]:
        for c in ["anc","ctrl"]: jobs.append((c,n,0))
    for c in ["anc","ctrl"]: jobs.append((c,300,1))     # seed replication
    for cond,n,sd in jobs:
        k=f"{cond}_{n}_{sd}"
        if k in res: continue
        if time.time()-t0>BUD: print("PAUSE",flush=True); sys.exit(0)
        r=run(cond,n,sd,BUD,t0)
        if r is None: sys.exit(0)
        res[k]=r; json.dump(res,open(SJ,"w"))
        print(f"[{time.time()-t0:4.0f}s] {k:12s} val CE {r:.4f}",flush=True)
    print("ALLDONE",flush=True)
