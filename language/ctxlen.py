"""How much of the 64-token context does the level-13 model actually use?

Truncate to the last k tokens and compare against the full window on the SAME
held-out positions. Paired, because the unpaired version is worthless here: over
200 positions it appeared to show 16 tokens beating 64 by 0.10 nats, which the
paired test at n=600 dissolves to -0.007 +- 0.018.

  python ctxlen.py [n]
"""
import sys, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
from gatelm import va_ids, CTX
from predict import best_ckpt, load, nextdist
import numpy as np

if __name__=="__main__":
    N=int(sys.argv[1]) if len(sys.argv)>1 else 600
    ck,_=best_ckpt(); net=load(ck)
    code=np.load("vocabtree.npz")['code'].astype(np.int64)
    rng=np.random.default_rng(23); K=(4,8,16,32,64)
    nll={k:[] for k in K}; hit={k:[] for k in K}
    for _ in range(N):
        i=int(rng.integers(0,len(va_ids)-CTX-1))
        c=list(va_ids[i:i+CTX]); tl=int(code[int(va_ids[i+CTX])])
        for k in K:
            q=nextdist(net,c[-k:]).numpy().astype(np.float64)
            nll[k].append(-np.log(max(q[tl],1e-12))); hit[k].append(int(q.argmax()==tl))
    for k in K: nll[k]=np.array(nll[k]); hit[k]=np.array(hit[k])
    print(f"paired against the full 64-token context, n={N} held-out positions")
    print(f"  {'context':>9} {'CE':>7} {'acc':>6} | {'paired diff':>12} {'SE':>7} {'t':>6}")
    for k in K:
        if k==64:
            print(f"  {k:>5} tok {nll[k].mean():>7.3f} {hit[k].mean():>6.3f} | {'--':>12}")
            continue
        d=nll[k]-nll[64]; se=d.std(ddof=1)/np.sqrt(N)
        print(f"  {k:>5} tok {nll[k].mean():>7.3f} {hit[k].mean():>6.3f} | "
              f"{d.mean():>+12.4f} {se:>7.4f} {d.mean()/se:>6.2f}")
    print("\n  positive = that context is worse than the full window; |t|>2 significant")
