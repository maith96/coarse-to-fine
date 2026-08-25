"""Can 3 layers implement d steps of function composition?

Architecture is held at the project's language model: 96 dimensions, 3 layers,
4 heads. Only the task depth varies. Data is generated on the fly, so there is
no finite training set to overfit -- the only held-out object is a fixed 20% of
ordered function PAIRS, which never appear composed during training.

Loss is taken on the ANSWER token only. That is the most generous setting for
the architecture: the definitions are random and unpredictable, so training to
predict them would only dilute the gradient. If it fails here it fails.

  python train.py <depth> [steps]
"""
import os, sys, time, json, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
torch.set_num_threads(int(os.environ.get('NT','4')))  # depths run as
# parallel single-thread processes; unlike the ctx-64 Shakespeare net this one
# does benefit from threads (124 vs 408 ms/step), but 4 procs x 1 thread wins.
from gen import held_pairs, encode, TOK, V, CONSTS, FUNCS
# GEN=2 selects the depth-invariant generator; see gen2.py for why it exists.
if os.environ.get('GEN','1')=='2': from gen2 import make
else: from gen import make

GEN=os.environ.get('GEN','1')          # which generator produced the data
NDEF=int(os.environ.get('NDEF','20'))
LR=float(os.environ.get('LR','3e-3'))
CTX=int(os.environ.get('CTX','140')); DM=96; NL=3; NH=4; BS=64
DEPTH=int(sys.argv[1]); STEPS=int(sys.argv[2]) if len(sys.argv)>2 else 20000
HELD=held_pairs()
CONST_IDS=torch.tensor([TOK[c] for c in CONSTS])

class LM(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb=nn.Embedding(V,DM); s.pos=nn.Embedding(CTX,DM)
        s.ls=nn.ModuleList([nn.TransformerEncoderLayer(DM,NH,4*DM,dropout=0.0,
            batch_first=True,norm_first=True,activation="gelu") for _ in range(NL)])
        s.ln=nn.LayerNorm(DM); s.hd=nn.Linear(DM,V)
    def forward(s,x):
        h=s.emb(x)+s.pos.weight[None,:x.shape[1]]
        m=nn.Transformer.generate_square_subsequent_mask(x.shape[1])
        for l in s.ls: h=l(h,src_mask=m,is_causal=True)
        return s.hd(s.ln(h))

def batch(bs, rng, split):
    """returns padded token ids, the index of the last prompt token, and answers."""
    X=np.zeros((bs,CTX),dtype=np.int64); pos=np.zeros(bs,dtype=np.int64)
    Y=np.zeros(bs,dtype=np.int64)
    for i in range(bs):
        t,ans,*_=make(DEPTH,rng,HELD,split,n_defs=NDEF)
        ids=encode(t)
        assert len(ids)<CTX, f"sequence {len(ids)} exceeds CTX {CTX}"
        X[i,:len(ids)]=ids; pos[i]=len(ids)-1; Y[i]=TOK[ans]
    return torch.from_numpy(X), torch.from_numpy(pos), torch.from_numpy(Y)

@torch.no_grad()
def evaluate(net, split, n=1000, seed=99):
    rng=np.random.default_rng(seed); net.eval(); c=0; t=0
    for _ in range(0, n, BS):
        x,p,y=batch(BS,rng,split)
        lg=net(x)[torch.arange(BS),p]
        lg=lg[:,CONST_IDS]                      # answer is always a constant
        c+=(CONST_IDS[lg.argmax(-1)]==y).sum().item(); t+=BS
    net.train(); return c/t

if __name__=="__main__":
    torch.manual_seed(0); net=LM()
    npar=sum(p.numel() for p in net.parameters())
    print(f"depth {DEPTH} | {npar/1e6:.2f}M params {NL}L {DM}d {NH}h | ctx {CTX} ndef {NDEF} lr {LR}")
    o=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.OneCycleLR(o,LR,total_steps=STEPS,pct_start=0.1)
    rng=np.random.default_rng(1); t0=time.time(); hist=[]
    for s in range(1,STEPS+1):
        x,p,y=batch(BS,rng,"train")
        loss=F.cross_entropy(net(x)[torch.arange(BS),p], y)
        o.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); o.step(); sch.step()
        if s%1000==0 or s==STEPS:
            tr=evaluate(net,"train",512); te=evaluate(net,"test",512)
            hist.append((s,round(loss.item(),4),round(tr,4),round(te,4)))
            print(f"  {s:6d}  loss {loss.item():.3f}  seen-pairs {tr:.3f}  "
                  f"HELD-OUT {te:.3f}   ({time.time()-t0:.0f}s)",flush=True)
    tr=evaluate(net,"train",2000); te=evaluate(net,"test",2000)
    print(f"\nFINAL depth {DEPTH}: seen-pair compositions {tr:.4f}   "
          f"held-out compositions {te:.4f}")
    os.makedirs("ckpt",exist_ok=True)
    torch.save({'n':net.state_dict()},f"ckpt/compose_g{GEN}_d{DEPTH}_n{NDEF}_lr{LR}.pt")
    R=json.load(open("../results/compose.json")) if os.path.exists("../results/compose.json") else {}
    R[f"g{GEN}_d{DEPTH}_n{NDEF}_lr{LR}"]={"steps":STEPS,"params":npar,"seen":tr,"held":te,"hist":hist}
    json.dump(R,open("../results/compose.json","w"),indent=1)
