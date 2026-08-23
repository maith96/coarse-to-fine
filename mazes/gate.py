import sys, os, json, time, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
from maze import gen, G, INF

CACHE="ckpt/maze60k.npz"
if os.path.exists(CACHE):
    z=np.load(CACHE); walls,d,opt=z['w'],z['d'],z['o']
else:
    walls,d,opt,_,_=gen(60000,seed=0)
    np.savez_compressed(CACHE,w=walls,d=d,o=opt)
M=walls.shape[0]; SPLIT=int(0.8*M)
tok=np.where(walls,1,0).astype(np.int64)
tok[d==0]=2                                     # goal marker

def index(dist, lo, hi):
    m,i,j=np.nonzero((d==dist)[lo:hi]); return m+lo,i,j

def batch(idx, bs, rng):
    mi,ii,jj=idx; k=rng.integers(0,len(mi),bs)
    m,i,j=mi[k],ii[k],jj[k]
    x=tok[m].copy(); x[np.arange(bs),i,j]=3
    pos=(i*G+j)
    return (torch.from_numpy(x.reshape(bs,-1)), torch.from_numpy(pos),
            torch.from_numpy(opt[m,i,j]))

class PolNet(nn.Module):
    def __init__(s,nl=3,dm=64):
        super().__init__()
        s.tok=nn.Embedding(4,dm); s.pos=nn.Embedding(G*G,dm)
        s.ls=nn.ModuleList([nn.TransformerEncoderLayer(dm,4,4*dm,dropout=0.0,
            batch_first=True,norm_first=True,activation="gelu") for _ in range(nl)])
        s.ln=nn.LayerNorm(dm); s.hd=nn.Linear(dm,4)
    def forward(s,x,p):
        h=s.tok(x)+s.pos.weight[None]
        for l in s.ls: h=l(h)
        return s.hd(s.ln(h[torch.arange(x.shape[0]),p]))

def run(dist, steps=800, bs=128, lr=3e-3, seed=0):
    tr=index(dist,0,SPLIT); te=index(dist,SPLIT,M)
    if len(tr[0])<200: return None
    torch.manual_seed(seed); net=PolNet()
    opt_=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.OneCycleLR(opt_,lr,total_steps=steps,pct_start=0.15)
    rng=np.random.default_rng(seed)
    for s in range(steps):
        x,p,y=batch(tr,bs,rng)
        logits=net(x,p)
        loss=-(F.log_softmax(logits,-1)*y.float()).logsumexp(-1).mean()  # any optimal action
        opt_.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt_.step(); sch.step()
    net.eval(); rng2=np.random.default_rng(999); c=t=0; base=0.0
    with torch.no_grad():
        for _ in range(8):
            x,p,y=batch(te,512,rng2)
            pred=net(x,p).argmax(-1)
            c+=y[torch.arange(512),pred].sum().item(); t+=512
            base+=y.float().sum(1).mean().item()/4
    return c/t, base/8, len(tr[0]), len(te[0]), loss.item()

if __name__=="__main__":
    OUT="gate.json"; res=json.load(open(OUT)) if os.path.exists(OUT) else {}
    t0=time.time()
    for dist in [int(v) for v in sys.argv[1].split(",")]:
        r=run(dist)
        if r is None: print(f"horizon {dist}: too few samples"); continue
        acc,base,ntr,nte,tl=r; res[str(dist)]=r; json.dump(res,open(OUT,"w"))
        print(f"horizon d={dist:2d} | held-out acc {acc:.4f} | random-policy {base:.4f} | "
              f"lift {acc-base:+.4f} | trainloss {tl:.3f} | n_train {ntr}  [{time.time()-t0:.0f}s]",flush=True)
