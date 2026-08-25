"""Deep analysis of one next-word prediction from the best level-13 net.

predict.py shows what the model says. This asks how it says it:

  1. shape        entropy, where the probability mass actually sits
  2. hierarchy    the 8192-way distribution aggregated up the cluster tree, so
                  you can see at which level of coarse-graining the model commits
                  and where its uncertainty lives -- the one analysis this
                  project's vocabulary makes possible
  3. context      which words the context moves most, against the unigram prior
  4. ablation     how much of the 64-token window is load-bearing, measured by
                  masking all but the last k tokens (positions held fixed, so
                  this is information content and not a position shift)

  python analyse.py                 # a held-out passage
  python analyse.py "romeo:\\nbut soft,"
"""
import sys, os, re, json, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
from gatelm import LM, tgt, tr_ids, va_ids, CTX, DEPTH, V
from predict import best_ckpt, load, leaf2w, words, encode, word, show, nextdist

ck,rec=best_ckpt(); net=load(ck)
print(f"model {ck}  (recorded val CE {rec:.4f})\n")
z=np.load("vocabtree.npz"); code=z['code'].astype(np.int64)
uni=np.bincount(tr_ids,minlength=V).astype(np.float64)+0.5; uni/=uni.sum()

# ---- pick the context -------------------------------------------------------
if len(sys.argv)>1:
    ctx=encode(sys.argv[1].replace("\\n","\n"))[-CTX:]; true=None
else:
    i=int(np.random.default_rng(3).integers(0,len(va_ids)-CTX-1))
    ctx=list(va_ids[i:i+CTX]); true=int(va_ids[i+CTX])
print("CONTEXT (last 20 tokens):")
print("  ..." + " ".join(show(str(words[t])) for t in ctx[-20:]))
if true is not None: print(f"TRUE NEXT WORD: '{show(str(words[true]))}'\n")

p=nextdist(net,ctx).numpy().astype(np.float64)
pw=np.zeros(V)                       # level 13 is injective: leaf -> word
for leaf,w in leaf2w.items(): pw[w]=p[leaf]

# ---- 1. shape ---------------------------------------------------------------
H=-(p[p>0]*np.log(p[p>0])).sum()
srt=np.sort(pw)[::-1]; cum=np.cumsum(srt)
print("1. DISTRIBUTION SHAPE")
print(f"   entropy {H:.3f} nats  (uniform over 8000 = {np.log(8000):.3f}, "
      f"corpus unigram = {-(uni*np.log(uni)).sum():.3f})")
print(f"   -> effective vocabulary e^H = {np.exp(H):.0f} words")
print(f"   top-1 {srt[0]:.3f}   top-5 {cum[4]:.3f}   top-10 {cum[9]:.3f}   top-50 {cum[49]:.3f}")
for frac in (0.5,0.9,0.99):
    print(f"   words needed for {frac:.0%} of the mass: {int(np.searchsorted(cum,frac))+1}")

# ---- 2. hierarchical decomposition -----------------------------------------
print("\n2. WHERE THE MODEL COMMITS, LEVEL BY LEVEL")
print(f"   {'level':>5} {'clusters':>9} {'entropy':>8} {'top cluster p':>14} {'p(true)':>8}  top cluster contents")
for L in range(1,DEPTH+1):
    agg=np.zeros(2**L)
    np.add.at(agg, np.arange(2**DEPTH)>>(DEPTH-L), p)
    hL=-(agg[agg>0]*np.log(agg[agg>0])).sum()
    top=int(agg.argmax())
    mem=[str(words[w]) for w in np.where((code>>(DEPTH-L))==top)[0][:5]]
    pt=f"{agg[code[true]>>(DEPTH-L)]:.3f}" if true is not None else "  -  "
    print(f"   {L:>5} {2**L:>9} {hL:>8.3f} {agg[top]:>14.3f} {pt:>8}  "
          + " ".join(show(m) for m in mem))

# ---- 3. what the context contributes ---------------------------------------
print("\n3. WHAT THE CONTEXT ADDS, vs the corpus unigram prior")
lr=np.log(np.maximum(pw,1e-12)/uni)
print(f"   KL(model || unigram) = {(pw*lr)[pw>0].sum():.3f} nats")
up=np.argsort(-(pw*lr)); dn=np.argsort(pw*lr)
print("   most promoted by context:")
for w in up[:6]:
    print(f"     {show(str(words[w])):>12}  p {pw[w]:.4f}  unigram {uni[w]:.4f}  x{pw[w]/uni[w]:7.1f}")
print("   most suppressed (of words the prior likes):")
for w in [w for w in dn if uni[w]>0.004][:4]:
    print(f"     {show(str(words[w])):>12}  p {pw[w]:.4f}  unigram {uni[w]:.4f}  x{pw[w]/uni[w]:7.2f}")

# ---- 4. how much context is load-bearing ------------------------------------
print("\n4. HOW MUCH OF THE 64-TOKEN WINDOW IS USED")
print("   (all but the last k tokens replaced by <unk>; positions unchanged)")
print(f"   {'k':>4} {'top-1':>12} {'p(top-1)':>9} {'p(true)':>8} {'KL from full':>13}")
for k in (1,2,4,8,16,32,64):
    m=list(ctx); 
    if k<len(m): m[:len(m)-k]=[0]*(len(m)-k)
    q=nextdist(net,m).numpy().astype(np.float64)
    kl=(p*np.log(np.maximum(p,1e-12)/np.maximum(q,1e-12))).sum()
    t=int(q.argmax()); pt=f"{q[code[true]]:.3f}" if true is not None else "  -  "
    print(f"   {k:>4} {show(word(t) or '?'):>12} {q[t]:>9.3f} {pt:>8} {kl:>13.3f}")
