import json, os, sys, time, torch, torch.nn as nn, warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
from rg import NP, SEQ, sample, LEVEL_BITS

B, EV, SEED = 800, 40, 0
OUT = "results_grow.json"

class GNet(nn.Module):
    def __init__(self, nl, d=64, H=4, ff=256):
        super().__init__()
        self.tok=nn.Embedding(3,d); self.pos=nn.Embedding(SEQ,d)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(d,H,ff,dropout=0.0,
            batch_first=True,norm_first=True,activation="gelu") for _ in range(nl)])
        self.ln=nn.LayerNorm(d); self.head=nn.Linear(d,1)
    def forward(self,x):
        B_=x.shape[0]
        q=torch.full((B_,NP),2,dtype=torch.long,device=x.device)
        h=self.tok(torch.cat([x,q],1))+self.pos.weight[None]
        for l in self.layers: h=l(h)
        return self.head(self.ln(h[:,-NP:])).squeeze(-1)

def identity_init(layer):
    # pre-LN residual: zeroing both sublayer output projections makes the
    # layer an exact identity, so appending it preserves the ancestor's function
    nn.init.zeros_(layer.self_attn.out_proj.weight); nn.init.zeros_(layer.self_attn.out_proj.bias)
    nn.init.zeros_(layer.linear2.weight); nn.init.zeros_(layer.linear2.bias)

def grow_from(state, nl, seed):
    torch.manual_seed(seed); net=GNet(nl)
    if state is not None:
        net.load_state_dict(state, strict=False)   # old layers copied, new one left fresh
        identity_init(net.layers[-1])
    return net

def mask_msb(lv):
    m=torch.zeros(NP,dtype=torch.bool); m[:LEVEL_BITS[lv]]=True; return m

def ev(net,mask,gen,n=1024,bs=512):
    net.eval(); b=e=t=0
    with torch.no_grad():
        for _ in range(n//bs):
            x,y=sample(bs,"cpu",gen); c=((net(x)>0).float()[:,mask]==y[:,mask])
            b+=c.sum().item(); e+=c.all(1).sum().item(); t+=bs
    net.train(); return b/(t*mask.sum().item()), e/t

def train(lv, state, seed, steps=B, lr=3e-3, bs=128):
    nl=lv+1; mask=mask_msb(lv); net=grow_from(state,nl,seed)
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,lr,total_steps=steps,pct_start=0.15)
    g=torch.Generator(); g.manual_seed(seed*7919+lv); curve=[]
    for s in range(1,steps+1):
        x,y=sample(bs,"cpu",g)
        loss=F.binary_cross_entropy_with_logits(net(x)[:,mask],y[:,mask])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sch.step()
        if s%EV==0 or s==steps:
            ge=torch.Generator(); ge.manual_seed(12345+lv)
            curve.append((s,)+ev(net,mask,ge))
    return net,curve

if __name__=="__main__":
    BUDGET=float(sys.argv[1]); t0=time.time()
    res=json.load(open(OUT)) if os.path.exists(OUT) else {}
    jobs=[("ctrl_L0",0,None)]+[(f"anc_L{l}",l,(f"anc_L{l-1}" if l>1 else "ctrl_L0")) for l in range(1,6)]
    jobs+=[(f"ctrl_L{l}",l,None) for l in [5,4,3,2,1]]
    for key,lv,initk in jobs:
        if key in res: continue
        if time.time()-t0>BUDGET:
            print("PAUSE",sum(1 for j in jobs if j[0] not in res),"left",flush=True); sys.exit(0)
        st=torch.load(f"ckpt/gr_{initk}.pt") if initk else None
        net,curve=train(lv,st,SEED)
        res[key]=curve; torch.save(net.state_dict(),f"ckpt/gr_{key}.pt")
        json.dump(res,open(OUT,"w"))
        print(f"[{time.time()-t0:4.0f}s] GROW {key} ({lv+1}L) bit {curve[-1][1]:.4f} exact {curve[-1][2]:.4f}",flush=True)
    print("ALLDONE",flush=True)
