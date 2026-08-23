import sys, os, json, time, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
from lad9 import PolNet, index, batch, SPLIT, M, G

K=4; DIST=24; STEPS=600

class MacroNet(nn.Module):
    """Trainable encoder proposes subgoals. A FROZEN ancestor policy is invoked
    on each proposed subgoal and its action logits are mixed by proposal weight.
    The new net never learns to navigate - only to delegate."""
    def __init__(s, organelle, dm=64, nl=3):
        super().__init__()
        s.tok=nn.Embedding(4,dm); s.pos=nn.Embedding(G*G,dm)
        s.ls=nn.ModuleList([nn.TransformerEncoderLayer(dm,4,4*dm,dropout=0.0,batch_first=True,
            norm_first=True,activation="gelu") for _ in range(nl)])
        s.ln=nn.LayerNorm(dm); s.sub=nn.Linear(dm,1)
        s.org=organelle
        for p in s.org.parameters(): p.requires_grad=False   # autonomy removed
        s.org.eval()
    def forward(s,x,p):
        B=x.shape[0]
        h=s.tok(x)+s.pos.weight[None]
        for l in s.ls: h=l(h)
        logit=s.sub(s.ln(h)).squeeze(-1)                     # (B,81) subgoal scores
        valid=(x!=1)&(x!=3)                                  # not wall, not own cell
        logit=logit.masked_fill(~valid, -1e9)
        w,idx=torch.softmax(logit,-1).topk(K,dim=-1)         # top-K subgoals + weights
        w=w/(w.sum(-1,keepdim=True)+1e-9)
        # build K variants: true goal erased, subgoal marked
        xr=x.unsqueeze(1).repeat(1,K,1).reshape(B*K,-1).clone()
        gpos=(x==2).float().argmax(-1).unsqueeze(1).repeat(1,K).reshape(-1)
        ar=torch.arange(B*K)
        xr[ar,gpos]=0
        xr[ar,idx.reshape(-1)]=2
        pr=p.unsqueeze(1).repeat(1,K).reshape(-1)
        xr[ar,pr]=3
        with torch.no_grad():
            a=s.org(xr,pr).reshape(B,K,4)                    # organelle called K times
        return (w.unsqueeze(-1)*a).sum(1)                    # gradient flows via w

def ev(net,dist=DIST,nb=8,bs=512,seed=999):
    te=index(dist,SPLIT,M); rng=np.random.default_rng(seed); net.eval(); c=t=0
    with torch.no_grad():
        for _ in range(nb):
            x,p,y=batch(te,bs,rng); pr=net(x,p).argmax(-1)
            c+=y[torch.arange(bs),pr].sum().item(); t+=bs
    net.train(); return c/t

def go(tag, org_ckpt, BUD, t0):
    ck=f"ckpt/macro_{tag}.pt"
    torch.manual_seed(0); org=PolNet()
    if org_ckpt: org.load_state_dict(torch.load(org_ckpt)['n'])
    torch.manual_seed(0); net=MacroNet(org)
    tp=[p for p in net.parameters() if p.requires_grad]
    o=torch.optim.AdamW(tp,lr=3e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.OneCycleLR(o,3e-3,total_steps=STEPS,pct_start=0.15)
    s0=0; curve=[]
    if os.path.exists(ck):
        c=torch.load(ck); net.load_state_dict(c['n']); o.load_state_dict(c['o']); sch.load_state_dict(c['s']); s0=c['step']; curve=c['c']
    else:
        curve=[(0,ev(net,nb=4))]
    tr=index(DIST,0,SPLIT); rng=np.random.default_rng(11)
    for _ in range(s0): rng.integers(0,len(tr[0]),128)
    for s in range(s0+1,STEPS+1):
        x,p,y=batch(tr,128,rng)
        loss=-(F.log_softmax(net(x,p),-1)*y.float()).logsumexp(-1).mean()
        o.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(tp,1.0); o.step(); sch.step()
        if s%150==0: curve.append((s,ev(net,nb=4)))
        if time.time()-t0>BUD or s==STEPS:
            torch.save({'n':net.state_dict(),'o':o.state_dict(),'s':sch.state_dict(),'step':s,'c':curve},ck)
            if s<STEPS: print(f"  {tag} paused {s}/{STEPS} acc {curve[-1][1]:.4f}",flush=True); return None
    a=ev(net); print(f"[{time.time()-t0:4.0f}s] MACRO {tag}: final {a:.4f}",flush=True); return a

if __name__=="__main__":
    BUD=float(sys.argv[2]); t0=time.time()
    tag=sys.argv[1]
    r=go(tag, "ckpt/l9_anc_d8.pt" if tag=="real" else None, BUD, t0)
    if r is not None:
        res=json.load(open("macro.json")) if os.path.exists("macro.json") else {}
        res[tag]=r; json.dump(res,open("macro.json","w"))
