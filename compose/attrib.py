"""What ROW is the model actually reading when it is wrong?

diag.py showed the errors are not truncated chains -- the intermediate value is
almost never emitted. So the question becomes which definition the wrong answer
was copied out of, described relative to the true chain
    x0 -f1-> x1 -f2-> ... -fd-> xd.
Each wrong prediction is attributed to the first bucket that matches:

  skip-inner        the value of the definition (f_outer, x0): the outer function
                    applied directly to the QUERY argument, ignoring the nesting
  outer-fn/wrong-x  function is f_outer, argument is not x_{d-1}
  inner-fn/wrong-x  function is f_1, argument is not x_0
  chain-crossed     function AND argument both occur in the chain, but not paired
                    as the chain pairs them -- the conjunctive-binding signature
  fn-in-chain       function is a chain function, argument is unrelated
  arg-in-chain      argument is a chain argument, function is unrelated
  unrelated-row     a value from a definition sharing nothing with the chain
  off-context       a constant appearing in no definition
"""
import os, sys, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); torch.set_num_threads(int(os.environ.get('NT','1')))
from gen import make, held_pairs, encode, TOK, CONSTS
sys.argv=[sys.argv[0],'1']
import train as T

# a wrong answer copied from a row keyed on chain argument x_k tells us how far
# the pointer chase actually got: x_{d-1} means every step but the last succeeded.
KEYS=["correct","intermediate","echo query-arg"]+ \
     [f"row keyed on x{k}, wrong fn" for k in range(6)]+ \
     ["row keyed on x_last, RIGHT fn (skip)","fn-in-chain only","unrelated-row","off-context"]

def bucket(p, ans, defs, fs, xs):
    if p==ans: return "correct"
    if p in xs[1:-1]: return "intermediate"
    if p==xs[0]: return "echo query-arg"
    rows=[(f,x) for f,x,y in defs if y==p]
    if not rows: return "off-context"
    d=len(fs)
    # the correct final lookup is (f_d, x_{d-1}); a row on x_{d-1} with the right
    # function but a different value cannot happen, so RIGHT fn here means the
    # model matched the outer function against some OTHER argument entirely.
    for k in range(d-1,-1,-1):                  # prefer the deepest chain arg
        if any(x==xs[k] for f,x in rows): return f"row keyed on x{k}, wrong fn"
    if any(f==fs[-1] for f,x in rows): return "row keyed on x_last, RIGHT fn (skip)"
    if any(f in set(fs) for f,x in rows): return "fn-in-chain only"
    return "unrelated-row"

@torch.no_grad()
def run(net, depth, ndef, split, n=1984, seed=7):
    rng=np.random.default_rng(seed); H=held_pairs()
    CID=torch.tensor([TOK[c] for c in CONSTS]); cnt={k:0 for k in KEYS}; tot=0
    for _ in range(n//64):
        X=np.zeros((64,T.CTX),dtype=np.int64); pos=np.zeros(64,dtype=np.int64); meta=[]
        for i in range(64):
            t,ans,defs,fs,xs=make(depth,rng,H,split,n_defs=ndef)
            ids=encode(t); X[i,:len(ids)]=ids; pos[i]=len(ids)-1
            meta.append((ans,defs,fs,xs))
        pr=net(torch.from_numpy(X))[torch.arange(64),torch.from_numpy(pos)][:,CID].argmax(-1)
        for j,m in zip(pr.tolist(),meta): cnt[bucket(CONSTS[j],*m)]+=1; tot+=1
    return cnt,tot

if __name__=="__main__":
    ndef=int(os.environ.get('NDEF','12')); lr=os.environ.get('LR','0.0005')
    for depth in (2,3):
        net=T.LM(); net.load_state_dict(torch.load(
            f"ckpt/compose_d{depth}_n{ndef}_lr{lr}.pt",map_location="cpu")['n']); net.eval()
        c,tot=run(net,depth,ndef,"test")
        print(f"--- depth {depth}, n_defs {ndef}, held-out queries (n={tot}) ---")
        for k in KEYS:
            if c[k]: print(f"    {k:<18s} {c[k]/tot:.3f}")
        print()
