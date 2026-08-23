import sys, json, os, time, torch, warnings; warnings.filterwarnings("ignore")
import torch.nn.functional as F
from grow import GNet
from rg import NP, sample

def run(name, bits, nl, d, steps, bs=128, lr=3e-3, seed=0, resume=None):
    mask=torch.zeros(NP,dtype=torch.bool); mask[bits]=True
    torch.manual_seed(seed); net=GNet(nl,d=d,ff=4*d)
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,lr,total_steps=steps,pct_start=0.15)
    s0=0
    if resume and os.path.exists(resume):
        ck=torch.load(resume); net.load_state_dict(ck['n']); opt.load_state_dict(ck['o'])
        sch.load_state_dict(ck['s']); s0=ck['step']
    g=torch.Generator(); g.manual_seed(seed*104729+1); 
    for _ in range(s0): torch.randint(0,2,(1,),generator=g)   # advance stream
    t0=time.time()
    for s in range(s0+1,steps+1):
        x,y=sample(bs,"cpu",g)
        loss=F.binary_cross_entropy_with_logits(net(x)[:,mask],y[:,mask])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
        if resume and (time.time()-t0>190 or s==steps):
            torch.save({'n':net.state_dict(),'o':opt.state_dict(),'s':sch.state_dict(),'step':s},resume)
            if s<steps: print(f"{name}: paused at {s}/{steps} loss {loss.item():.4f}",flush=True); return None
    ge=torch.Generator(); ge.manual_seed(999); net.eval(); acc=torch.zeros(NP); tot=0
    with torch.no_grad():
        for _ in range(8):
            x,y=sample(512,"cpu",ge); acc+=((net(x)>0).float()==y).float().sum(0); tot+=512
    acc/=tot
    print(f"{name} [{nl}L d{d} {steps}st] trainloss {loss.item():.4f} | "
          f"target-bit acc {acc[bits].mean():.4f} | per-bit "+" ".join(f"{acc[b]:.3f}" for b in bits),flush=True)
    return acc

if __name__=="__main__":
    which=sys.argv[1]
    if which=="a": run("MIDDLE-8 only", list(range(8,16)), 3, 64, 800)
    if which=="b": run("SINGLE bit12 ", [12], 3, 64, 800)
