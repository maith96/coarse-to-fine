import sys, os, json, time, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
from maze9 import gen, G

CACHE="ckpt/m9.npz"
if os.path.exists(CACHE):
    z=np.load(CACHE); walls,d,opt=z['w'],z['d'],z['o']
else:
    walls,d,opt=gen(120000,density=0.32,seed=1); np.savez_compressed(CACHE,w=walls,d=d,o=opt)
M=walls.shape[0]; SPLIT=int(0.8*M)
tok=np.where(walls,1,0).astype(np.int64); tok[d==0]=2
IDX={}
def index(dist,lo,hi):
    k=(dist,lo,hi)
    if k not in IDX:
        m,i,j=np.nonzero((d==dist)[lo:hi]); IDX[k]=(m+lo,i,j)
    return IDX[k]
def batch(idx,bs,rng):
    mi,ii,jj=idx; k=rng.integers(0,len(mi),bs); m,i,j=mi[k],ii[k],jj[k]
    x=tok[m].copy(); x[np.arange(bs),i,j]=3
    return torch.from_numpy(x.reshape(bs,-1)), torch.from_numpy(i*G+j), torch.from_numpy(opt[m,i,j])

class PolNet(nn.Module):
    def __init__(s,nl=3,dm=64):
        super().__init__()
        s.tok=nn.Embedding(4,dm); s.pos=nn.Embedding(G*G,dm)
        s.ls=nn.ModuleList([nn.TransformerEncoderLayer(dm,4,4*dm,dropout=0.0,batch_first=True,
            norm_first=True,activation="gelu") for _ in range(nl)])
        s.ln=nn.LayerNorm(dm); s.hd=nn.Linear(dm,4)
    def forward(s,x,p):
        h=s.tok(x)+s.pos.weight[None]
        for l in s.ls: h=l(h)
        return s.hd(s.ln(h[torch.arange(x.shape[0]),p]))

def ev(net,dist,nb=4,bs=512,seed=999):
    te=index(dist,SPLIT,M); rng=np.random.default_rng(seed); net.eval(); c=t=0; base=0.
    with torch.no_grad():
        for _ in range(nb):
            x,p,y=batch(te,bs,rng); pr=net(x,p).argmax(-1)
            c+=y[torch.arange(bs),pr].sum().item(); t+=bs; base+=y.float().sum(1).mean().item()/4
    net.train(); return c/t, base/nb

STEPS=600
def run(key,dist,initk,res,t0,BUD):
    ck=f"ckpt/l9_{key}.pt"; s0=0
    torch.manual_seed(0); net=PolNet()
    o=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.OneCycleLR(o,3e-3,total_steps=STEPS,pct_start=0.15)
    if os.path.exists(ck):
        c=torch.load(ck); net.load_state_dict(c['n']); o.load_state_dict(c['o']); sch.load_state_dict(c['s']); s0=c['step']; curve=c['c']
    else:
        if initk: net.load_state_dict(torch.load(f"ckpt/l9_{initk}.pt")['n'])
        curve=[(0,ev(net,dist)[0])]
    tr=index(dist,0,SPLIT); rng=np.random.default_rng(11); 
    for _ in range(s0): rng.integers(0,len(tr[0]),128)
    for s in range(s0+1,STEPS+1):
        x,p,y=batch(tr,128,rng)
        loss=-(F.log_softmax(net(x,p),-1)*y.float()).logsumexp(-1).mean()
        o.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); o.step(); sch.step()
        if s%100==0 or s==STEPS: curve.append((s,ev(net,dist)[0]))
        if time.time()-t0>BUD or s==STEPS:
            torch.save({'n':net.state_dict(),'o':o.state_dict(),'s':sch.state_dict(),'step':s,'c':curve},ck)
            if s<STEPS: print(f"  {key} paused {s}/{STEPS} acc {curve[-1][1]:.4f}",flush=True); return False
    a,b=ev(net,dist,nb=8); res[key]={'curve':curve,'final':a,'base':b}
    json.dump(res,open("lad9.json","w"))
    print(f"[{time.time()-t0:4.0f}s] {key:10s} zs {curve[0][1]:.4f} -> final {a:.4f}  (random {b:.3f})",flush=True)
    return True

if __name__=="__main__":
    BUD=float(sys.argv[1]); t0=time.time()
    res=json.load(open("lad9.json")) if os.path.exists("lad9.json") else {}
    HOR=[2,4,8,16,24]
    jobs=[("base_d2",2,None)]+[(f"anc_d{h}",h,(f"anc_d{HOR[i-1]}" if i>1 else "base_d2")) for i,h in enumerate(HOR[1:],1)]
    jobs+=[(f"ctrl_d{h}",h,None) for h in [24,16,8,4]]
    for key,dd,ik in jobs:
        if key in res: continue
        if time.time()-t0>BUD: print("PAUSE",flush=True); sys.exit(0)
        if not run(key,dd,ik,res,t0,BUD): sys.exit(0)
    print("ALLDONE",flush=True)
