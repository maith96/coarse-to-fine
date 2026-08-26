"""Hierarchical vocabulary tree over the hip-hop corpus.

Faithful port of language/vocab.py: PPMI co-occurrence -> SVD -> recursive
balanced 2-means, giving a DEPTH-bit code with an exact prefix property
(level k cluster of word w is code[w] >> (DEPTH-k)).

Extra here: a per-level token-mass entropy readout, because the whole
"the coarse rungs learn the template" claim stands or falls on whether the
coarse levels carry any information at all.
"""
import re, numpy as np, collections
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

txt = open("hiphop.txt", encoding="utf-8", errors="replace").read().lower()
toks = re.findall(r"[a-z']+|[.,!?;:\n]", txt)
cnt = collections.Counter(toks)
V = 8000
words = [w for w, _ in cnt.most_common(V - 1)]
w2i = {w: i + 1 for i, w in enumerate(words)}          # 0 = UNK
ids = np.array([w2i.get(t, 0) for t in toks], dtype=np.int32)
print(f"tokens {len(ids)}  vocab {V}  UNK rate {(ids==0).mean():.3f}  "
      f"newline rate {(ids==w2i[chr(10)]).mean():.3f}")

# PPMI co-occurrence, window 4
W = 4
rows = []; cols = []
for off in range(1, W + 1):
    rows.append(ids[:-off]); cols.append(ids[off:])
    rows.append(ids[off:]);  cols.append(ids[:-off])
r = np.concatenate(rows); c = np.concatenate(cols)
Cm = coo_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(V, V)).tocsr()
tot = Cm.sum(); rs = np.asarray(Cm.sum(1)).ravel(); cs = np.asarray(Cm.sum(0)).ravel()
Cc = Cm.tocoo()
pmi = np.log(np.maximum(Cc.data * tot / (rs[Cc.row] * cs[Cc.col] + 1e-9), 1e-12))
pmi = np.maximum(pmi, 0)
P = coo_matrix((pmi.astype(np.float32), (Cc.row, Cc.col)), shape=(V, V)).tocsr()
U, S, _ = svds(P, k=64)
E = U * S
E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
print("embeddings", E.shape)

# balanced recursive 2-means -> 13-bit hierarchical code
DEPTH = 13
code = np.zeros(V, dtype=np.int32)
def bisect(idx, depth, rng):
    if depth == DEPTH or len(idx) <= 1: return
    X = E[idx]
    ctr = X[rng.choice(len(idx), 2, replace=False)]
    for _ in range(12):
        a = ((X - ctr[0])**2).sum(1); b = ((X - ctr[1])**2).sum(1)
        m = a - b; k = len(idx) // 2
        lab = np.zeros(len(idx), dtype=int); lab[np.argsort(-m)[:len(idx) - k]] = 1
        if lab.sum() == 0 or lab.sum() == len(idx): break
        ctr = np.stack([X[lab == 0].mean(0), X[lab == 1].mean(0)])
    code[idx] |= (lab << (DEPTH - 1 - depth))
    bisect(idx[lab == 0], depth + 1, rng); bisect(idx[lab == 1], depth + 1, rng)
bisect(np.arange(V), 0, np.random.default_rng(0))
np.savez("vocabtree.npz", ids=ids, code=code, words=np.array(["<unk>"] + words))

# how much information does each rung actually carry, in token mass?
freq = np.bincount(ids, minlength=V).astype(np.float64); freq /= freq.sum()
print(f"{'level':>5} {'buckets':>8} {'used':>6} {'H(mass)':>8} {'lnN':>6} {'frac':>6}")
for L in range(1, DEPTH + 1):
    cl = code >> (DEPTH - L)
    m = np.bincount(cl, minlength=2**L, weights=freq)
    nz = m[m > 0]; H = float(-(nz * np.log(nz)).sum())
    print(f"{L:5d} {2**L:8d} {len(nz):6d} {H:8.3f} {np.log(2**L):6.2f} "
          f"{H/np.log(2**L):6.3f}")

cl4 = code >> (DEPTH - 4)
allw = np.array(["<unk>"] + words)
for b in range(8):
    w = [str(x) for x in allw[np.where(cl4 == b)[0][:12]]]
    print(f"  L4 cluster{b}: " + " ".join(x.replace("\n", "\\n") for x in w))
