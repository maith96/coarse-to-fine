import re, numpy as np, collections, json, math
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
import corpus as CO

txt=open(CO.C["txt"]).read().lower()
toks=re.findall(CO.C["tok"], txt)
cnt=collections.Counter(toks)
V=min(CO.C["vmax"], len(cnt)+1)                  # +1 for UNK; small corpora keep every type
words=[w for w,_ in cnt.most_common(V-1)]
w2i={w:i+1 for i,w in enumerate(words)}          # 0 = UNK
ids=np.array([w2i.get(t,0) for t in toks],dtype=np.int32)
DEPTH=CO.C["depth"] or max(1,math.ceil(math.log2(V)))
print(f"corpus {CO.NAME}  tokens {len(ids)}  vocab {V}  depth {DEPTH}  UNK rate {(ids==0).mean():.3f}")

# PPMI co-occurrence, window 4
W=4
rows=[];cols=[]
for off in range(1,W+1):
    rows.append(ids[:-off]); cols.append(ids[off:])
    rows.append(ids[off:]);  cols.append(ids[:-off])
r=np.concatenate(rows); c=np.concatenate(cols)
Cm=coo_matrix((np.ones(len(r),dtype=np.float32),(r,c)),shape=(V,V)).tocsr()
tot=Cm.sum(); rs=np.asarray(Cm.sum(1)).ravel(); cs=np.asarray(Cm.sum(0)).ravel()
Cc=Cm.tocoo()
pmi=np.log(np.maximum(Cc.data*tot/(rs[Cc.row]*cs[Cc.col]+1e-9),1e-12))
pmi=np.maximum(pmi,0)
P=coo_matrix((pmi.astype(np.float32),(Cc.row,Cc.col)),shape=(V,V)).tocsr()
U,S,_=svds(P,k=min(64,V-1))
E=U*S
E=E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-9)
print("embeddings", E.shape)

# balanced recursive 2-means -> DEPTH-bit hierarchical code (prefix property exact)
code=np.zeros(V,dtype=np.int32)
def bisect(idx, depth, rng):
    if depth==DEPTH or len(idx)<=1: return
    X=E[idx]
    ctr=X[rng.choice(len(idx),2,replace=False)]
    for _ in range(12):
        a=((X-ctr[0])**2).sum(1); b=((X-ctr[1])**2).sum(1)
        lab=(b<a).astype(int)
        # balance: force a near-even split by ranking the margin
        m=a-b; k=len(idx)//2
        lab=np.zeros(len(idx),dtype=int); lab[np.argsort(-m)[:len(idx)-k]]=1
        if lab.sum()==0 or lab.sum()==len(idx): break
        ctr=np.stack([X[lab==0].mean(0),X[lab==1].mean(0)])
    code[idx]|= (lab<<(DEPTH-1-depth))
    bisect(idx[lab==0],depth+1,rng); bisect(idx[lab==1],depth+1,rng)
bisect(np.arange(V),0,np.random.default_rng(0))
vocab=np.array(["<unk>"]+words)
np.savez(CO.TREE, ids=ids, code=code, words=vocab, V=V, DEPTH=DEPTH, CTX=CO.C["ctx"])
print(f"wrote {CO.TREE}")
for L in CO.levels(DEPTH):
    cl=code>>(DEPTH-L); print(f"  level {L:2d}: {2**L:5d} buckets, {len(set(cl.tolist())):5d} used")
# show a coarse cluster's contents as a sanity check
SH=min(4,DEPTH); cl4=code>>(DEPTH-SH)
for b in [0,7%(2**SH)]:
    ws=[str(vocab[i]) for i in np.where(cl4==b)[0][:14]]
    print(f"  cluster{b}: "+" ".join(ws).replace("\n","\\n"))
