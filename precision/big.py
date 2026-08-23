import sys, os, time, torch, warnings; warnings.filterwarnings("ignore")
import torch.nn.functional as F
from widthlib import make
W,NL,D,STEPS=8,4,128,800
CK="ckpt/big_w8.pt"
N,smp,NPb=make(W); bit=W
torch.manual_seed(0); net=N(nl=NL,d=D)
opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=0.01)
sch=torch.optim.lr_scheduler.OneCycleLR(opt,2e-3,total_steps=STEPS,pct_start=0.15)
s0=0
if os.path.exists(CK):
    c=torch.load(CK); net.load_state_dict(c['n']); opt.load_state_dict(c['o']); sch.load_state_dict(c['s']); s0=c['step']
g=torch.Generator(); g.manual_seed(1)
for _ in range(s0): smp(128,g)
t0=time.time(); nparam=sum(p.numel() for p in net.parameters())
for s in range(s0+1,STEPS+1):
    x,y=smp(128,g)
    loss=F.binary_cross_entropy_with_logits(net(x)[:,bit],y[:,bit])
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
    if time.time()-t0>185 or s==STEPS:
        torch.save({'n':net.state_dict(),'o':opt.state_dict(),'s':sch.state_dict(),'step':s},CK)
        if s<STEPS: print(f"paused {s}/{STEPS} trainloss {loss.item():.4f}",flush=True); sys.exit(0)
ge=torch.Generator(); ge.manual_seed(9); net.eval(); c=t=0
with torch.no_grad():
    for _ in range(4):
        x,y=smp(512,ge); c+=((net(x)[:,bit]>0).float()==y[:,bit]).sum().item(); t+=512
print(f"BIG NET W=8: {NL}L d={D}, {nparam/1e6:.2f}M params, domain 16384 pairs, {STEPS} steps")
print(f"  final TRAIN loss {loss.item():.4f}  (ln2 = 0.6931)   test acc {c/t:.4f}",flush=True)
