"""Where does the composition fail? Decompose every wrong answer into a named mode.

For a depth-d query  fd OF ... OF f1 OF x0  with true chain x0 -f1-> x1 ... -fd-> xd,
the prediction is attributed to the FIRST matching bucket:

  correct          xd
  stopped at k     x_k for k<d  -- resolved the first k steps then emitted the
                                   intermediate value instead of continuing
  outer-fn match   value of some definition whose function is fd but whose
                   argument is not x_{d-1}   (the strongest single-cue heuristic)
  query-arg match  value of some definition whose argument is x0 but whose
                   function is not f1
  other-in-context some other value present in the definitions
  off-context      a constant that appears nowhere
"""
import os, sys, json, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get('NT','4')))
from gen import make, held_pairs, encode, TOK, V, CONSTS
sys.argv=[sys.argv[0],'1']          # train.py reads depth from argv at import
import train as T

def load(depth, ndef, lr):
    net=T.LM(); ck=torch.load(f"ckpt/compose_d{depth}_n{ndef}_lr{lr}.pt",map_location="cpu")
    net.load_state_dict(ck['n']); net.eval(); return net

@torch.no_grad()
def diag(net, depth, ndef, split, n=2000, seed=7):
    rng=np.random.default_rng(seed); HELD=held_pairs()
    CID=torch.tensor([TOK[c] for c in CONSTS])
    keys=["correct"]+[f"stopped at {k}" for k in range(depth)]+ \
         ["outer-fn match","query-arg match","other-in-context","off-context"]
    cnt={k:0 for k in keys}; tot=0
    for _ in range(n//64):
        X=np.zeros((64,T.CTX),dtype=np.int64); pos=np.zeros(64,dtype=np.int64); meta=[]
        for i in range(64):
            t,ans,defs,fs,xs=make(depth,rng,HELD,split,n_defs=ndef)
            ids=encode(t); X[i,:len(ids)]=ids; pos[i]=len(ids)-1
            meta.append((ans,defs,fs,xs))
        lg=net(torch.from_numpy(X))[torch.arange(64),torch.from_numpy(pos)][:,CID]
        pred=[CONSTS[j] for j in lg.argmax(-1).tolist()]
        for p,(ans,defs,fs,xs) in zip(pred,meta):
            tot+=1
            if p==ans: cnt["correct"]+=1; continue
            hit=None
            for k in range(depth):                       # x_0 .. x_{d-1}
                if p==xs[k]: hit=f"stopped at {k}"; break
            if hit is None:
                if any(f==fs[-1] and y==p for f,x,y in defs): hit="outer-fn match"
                elif any(x==xs[0] and y==p for f,x,y in defs): hit="query-arg match"
                elif any(y==p for _,_,y in defs): hit="other-in-context"
                else: hit="off-context"
            cnt[hit]+=1
    return cnt, tot

if __name__=="__main__":
    ndef=int(os.environ.get('NDEF','12')); lr=os.environ.get('LR','0.0005')
    for depth in (1,2,3):
        net=load(depth,ndef,lr)
        for split in ("train","test"):
            c,tot=diag(net,depth,ndef,split)
            tag={"train":"seen pairs","test":"HELD-OUT "}[split]
            print(f"depth {depth}  {tag}  (n={tot})")
            for k,v in c.items():
                if v: print(f"     {k:<18s} {v/tot:.3f}")
        print()
