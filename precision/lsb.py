import json, os, sys, time, torch, warnings
warnings.filterwarnings("ignore")
import rg
from rg import LEVEL_BITS, Net, NP, sample
import torch.nn.functional as F

B, EV, SEED = 800, 40, 0
OUT = "results_lsb.json"

def mk_mask(lv):
    m = torch.zeros(NP, dtype=torch.bool); m[NP-LEVEL_BITS[lv]:] = True; return m

def evaluate(net, mask, gen, n=1024, bs=512):
    net.eval(); bitc=exc=tot=0
    with torch.no_grad():
        for _ in range(n//bs):
            x,y = sample(bs,"cpu",gen)
            c = ((net(x)>0).float()[:,mask] == y[:,mask])
            bitc += c.sum().item(); exc += c.all(1).sum().item(); tot += bs
    net.train(); return bitc/(tot*mask.sum().item()), exc/tot

def train(lv, state, seed, steps=B, lr=3e-3, bs=128):
    mask = mk_mask(lv)
    torch.manual_seed(seed); net = Net()
    if state is not None: net.load_state_dict(state)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.15)
    g = torch.Generator(); g.manual_seed(seed*7919+lv)
    curve=[]
    for s in range(1, steps+1):
        x,y = sample(bs,"cpu",g)
        loss = F.binary_cross_entropy_with_logits(net(x)[:,mask], y[:,mask])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
        if s%EV==0 or s==steps:
            ge=torch.Generator(); ge.manual_seed(12345+lv)
            curve.append((s,)+evaluate(net,mask,ge))
    return net, curve

if __name__=="__main__":
    BUDGET=float(sys.argv[1]); t0=time.time()
    res=json.load(open(OUT)) if os.path.exists(OUT) else {}
    jobs=[(f"ctrl_L{lv}",lv,None) for lv in range(6)]
    jobs+=[(f"anc_L{lv}",lv,(f"anc_L{lv-1}" if lv>1 else "ctrl_L0")) for lv in range(1,6)]
    order=["ctrl_L0"]+[f"anc_L{l}" for l in range(1,6)]+[f"ctrl_L{l}" for l in [5,3,4,2,1]]
    jobs.sort(key=lambda j: order.index(j[0]))
    for key,lv,initk in jobs:
        if key in res: continue
        if time.time()-t0>BUDGET:
            print("PAUSE",sum(1 for j in jobs if j[0] not in res),"left",flush=True); sys.exit(0)
        st = torch.load(f"ckpt/lsb_{initk}.pt") if initk else None
        net,curve = train(lv,st,SEED)
        res[key]=curve; torch.save(net.state_dict(),f"ckpt/lsb_{key}.pt")
        json.dump(res,open(OUT,"w"))
        print(f"[{time.time()-t0:4.0f}s] LSB {key} bit {curve[-1][1]:.4f} exact {curve[-1][2]:.4f}",flush=True)
    print("ALLDONE",flush=True)
