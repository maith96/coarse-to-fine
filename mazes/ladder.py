import sys, os, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
from gate import PolNet, index, batch, SPLIT, M

HOR=[2,4,8,12,16]
OUT="ladder.json"

def ev(net, dist, nb=4, bs=512, seed=999):
    tr=index(dist,SPLIT,M); rng=np.random.default_rng(seed)
    net.eval(); c=t=0
    with torch.no_grad():
        for _ in range(nb):
            x,p,y=batch(tr,bs,rng); pr=net(x,p).argmax(-1)
            c+=y[torch.arange(bs),pr].sum().item(); t+=bs
    net.train(); return c/t

def train(dist, state, steps=800, bs=128, lr=3e-3, seed=0, EV=100):
    tr=index(dist,0,SPLIT)
    torch.manual_seed(seed); net=PolNet()
    if state is not None: net.load_state_dict(state)
    o=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.OneCycleLR(o,lr,total_steps=steps,pct_start=0.15)
    rng=np.random.default_rng(seed+11)
    curve=[(0,ev(net,dist))]
    for s in range(1,steps+1):
        x,p,y=batch(tr,bs,rng)
        loss=-(F.log_softmax(net(x,p),-1)*y.float()).logsumexp(-1).mean()
        o.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); o.step(); sch.step()
        if s%EV==0 or s==steps: curve.append((s,ev(net,dist)))
    return net,curve

if __name__=="__main__":
    BUD=float(sys.argv[1]); t0=time.time()
    res=json.load(open(OUT)) if os.path.exists(OUT) else {}
    jobs=[("base_d2",2,None)]
    for i,dd in enumerate(HOR[1:],start=1):
        jobs.append((f"anc_d{dd}",dd,(f"anc_d{HOR[i-1]}" if i>1 else "base_d2")))
    for dd in HOR[1:]: jobs.append((f"ctrl_d{dd}",dd,None))
    for key,dd,ik in jobs:
        if key in res: continue
        if time.time()-t0>BUD:
            print("PAUSE",sum(1 for j in jobs if j[0] not in res),"left",flush=True); sys.exit(0)
        st=torch.load(f"ckpt/ld_{ik}.pt") if ik else None
        net,curve=train(dd,st)
        res[key]=curve; torch.save(net.state_dict(),f"ckpt/ld_{key}.pt")
        json.dump(res,open(OUT,"w"))
        print(f"[{time.time()-t0:4.0f}s] {key:10s} step0 {curve[0][1]:.4f} -> final {curve[-1][1]:.4f}",flush=True)
    print("ALLDONE",flush=True)
