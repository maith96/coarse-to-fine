import torch, torch.nn as nn, warnings, time; warnings.filterwarnings("ignore")
import torch.nn.functional as F

def make(W):
    NPb=2*W; SEQ=2*W+NPb
    class N(nn.Module):
        def __init__(s,nl=3,d=64):
            super().__init__()
            s.tok=nn.Embedding(3,d); s.pos=nn.Embedding(SEQ,d)
            s.ls=nn.ModuleList([nn.TransformerEncoderLayer(d,4,4*d,dropout=0.0,batch_first=True,
                norm_first=True,activation="gelu") for _ in range(nl)])
            s.ln=nn.LayerNorm(d); s.hd=nn.Linear(d,1)
        def forward(s,x):
            q=torch.full((x.shape[0],NPb),2,dtype=torch.long)
            h=s.tok(torch.cat([x,q],1))+s.pos.weight[None]
            for l in s.ls: h=l(h)
            return s.hd(s.ln(h[:,-NPb:])).squeeze(-1)
    def smp(bs,g):
        lo=2**(W-1)
        a=lo+torch.randint(0,lo,(bs,),generator=g); b=lo+torch.randint(0,lo,(bs,),generator=g)
        p=a*b
        si=torch.arange(W-1,-1,-1); so=torch.arange(NPb-1,-1,-1)
        return torch.cat([(a[:,None]>>si)&1,(b[:,None]>>si)&1],1).long(), (((p[:,None]>>so)&1)).float()
    return N, smp, NPb

import sys
for W in [int(v) for v in sys.argv[1].split(",")]:
    NPb=2*W; bit=W                      # middle bit, same convention as bit12 at W=12
    N,smp,_=make(W)
    torch.manual_seed(0); net=N()
    opt=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,3e-3,total_steps=800,pct_start=0.15)
    g=torch.Generator(); g.manual_seed(1); t0=time.time()
    for s in range(800):
        x,y=smp(128,g)
        loss=F.binary_cross_entropy_with_logits(net(x)[:,bit],y[:,bit])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
    ge=torch.Generator(); ge.manual_seed(9); net.eval(); c=t=0
    with torch.no_grad():
        for _ in range(4):
            x,y=smp(512,ge); c+=((net(x)[:,bit]>0).float()==y[:,bit]).sum().item(); t+=512
    dom=(2**(W-1))**2
    print(f"W={W:2d} ({2*W}b product, middle bit {bit})  domain {dom:9d}  "
          f"trainloss {loss.item():.4f}  test acc {c/t:.4f}   [{time.time()-t0:.0f}s]",flush=True)
