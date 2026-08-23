"""Budget sweep on the 9x9 maze ladder.

FINDINGS.md #3: the +0.07 ancestor advantage at d=16/24 was measured with the
ancestor holding 5 rungs of compute to the control's 1.  This charges the
curriculum for its own construction, the way language/sweep.py does.

The ancestor arrives at the final rung having already spent CHAIN_COST steps,
so the matched-total comparison is  anc @ 600  vs  ctrl @ 600 + CHAIN_COST.
Everything is reported in steps; wall clock is not comparable across machines.
"""
import sys, os, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
torch.set_num_threads(4)
from lad9 import PolNet, index, batch, ev, SPLIT

ANC={24:"anc_d16", 16:"anc_d8"}     # ancestor checkpoint feeding the final rung
CHAIN_COST={24:2400, 16:1800}       # steps the ancestor spent before it
BS=128

def run(cond, dist, steps, seed, BUD, t0):
    key=f"{cond}_d{dist}_n{steps}_s{seed}"; ck=f"ckpt/sw9_{key}.pt"
    torch.manual_seed(seed); net=PolNet()
    if cond=="anc" and not os.path.exists(ck):
        net.load_state_dict(torch.load(f"ckpt/l9_{ANC[dist]}.pt")['n'])
    o=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.OneCycleLR(o,3e-3,total_steps=steps,pct_start=0.15)
    s0=0
    if os.path.exists(ck):
        c=torch.load(ck); net.load_state_dict(c['n']); o.load_state_dict(c['o'])
        sch.load_state_dict(c['s']); s0=c['step']; curve=c['c']
    else:
        curve=[(0,ev(net,dist)[0])]
    tr=index(dist,0,SPLIT); rng=np.random.default_rng(11+seed)
    for _ in range(s0): rng.integers(0,len(tr[0]),BS)
    for s in range(s0+1,steps+1):
        x,p,y=batch(tr,BS,rng)
        loss=-(F.log_softmax(net(x,p),-1)*y.float()).logsumexp(-1).mean()
        o.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); o.step(); sch.step()
        if s%200==0 or s==steps: curve.append((s,ev(net,dist)[0]))
        if time.time()-t0>BUD or s==steps:
            torch.save({'n':net.state_dict(),'o':o.state_dict(),'s':sch.state_dict(),
                        'step':s,'c':curve},ck)
            if s<steps: print(f"  {key} paused {s}/{steps} acc {curve[-1][1]:.4f}",flush=True); return None
    a,b=ev(net,dist,nb=8)
    return {'curve':curve,'final':a,'base':b,
            'total_steps':steps+(CHAIN_COST[dist] if cond=="anc" else 0)}

if __name__=="__main__":
    BUD=float(sys.argv[1]); t0=time.time()
    res=json.load(open("sweep9.json")) if os.path.exists("sweep9.json") else {}
    # decisive point first: the control at the ancestor's total compute
    jobs=[("ctrl",24,3000,0),("anc",24,3000,0),
          ("ctrl",24,1200,0),("anc",24,1200,0),
          ("ctrl",24,2400,0),("anc",24,2400,0),
          ("ctrl",16,2400,0),("anc",16,2400,0),
          ("ctrl",16,1200,0),("anc",16,1200,0)]
    for cond,dist,n,sd in jobs:
        k=f"{cond}_d{dist}_{n}_{sd}"
        if k in res: continue
        if time.time()-t0>BUD: print("PAUSE",flush=True); sys.exit(0)
        r=run(cond,dist,n,sd,BUD,t0)
        if r is None: sys.exit(0)
        res[k]=r; json.dump(res,open("sweep9.json","w"))
        print(f"[{time.time()-t0:5.0f}s] {k:16s} final {r['final']:.4f} "
              f"(random {r['base']:.3f}, total steps {r['total_steps']})",flush=True)
    print("ALLDONE",flush=True)
