import numpy as np, torch, warnings; warnings.filterwarnings("ignore")
import torch.nn.functional as F
from gatelm import LM, tr_ids, va_ids, CTX, DEPTH, LEVELS
import corpus as CO
z=np.load(CO.TREE); code=z['code'].astype(np.int64); words=z['words']
nets={}
for L in LEVELS:
    n=LM(2**L); n.load_state_dict(torch.load(CO.ck(f"ch_anc_L{L}"))['n']); n.eval(); nets[L]=n

def cluster_words(L, cid, k=8):
    m=np.where((code>>(DEPTH-L))==cid)[0]
    return [str(words[i]) for i in m[:k]]

rng=np.random.default_rng(4)
for trial in range(3):
    i=rng.integers(0,len(va_ids)-CTX-1)
    ctx=va_ids[i:i+CTX]; nxt=va_ids[i+CTX]
    print("="*78)
    print("CONTEXT: ..."+" ".join(str(words[t]) for t in ctx[-14:]).replace("\n","\\n"))
    print(f"TRUE NEXT WORD: '{words[nxt]}'")
    x=torch.from_numpy(ctx[None].astype(np.int64))
    for L in LEVELS:
        with torch.no_grad(): lg=nets[L](x)[0,-1]
        p=F.softmax(lg,-1); top=p.argmax().item()
        true_c=code[nxt]>>(DEPTH-L)
        hit="HIT " if top==true_c else "miss"
        print(f"  L{L:2d} ({2**L:5d} buckets) {hit} p(true)={p[true_c].item():.3f}  "
              f"predicted bucket -> {', '.join(cluster_words(L,top,6))}")
    with torch.no_grad(): lg=nets[LEVELS[-1]](x)[0,-1]
    tp=torch.softmax(lg,-1).topk(6)
    leaf2w={}
    for w in range(len(code)): leaf2w.setdefault(code[w],w)
    print("  full-vocab top-6:", ", ".join(
        f"'{words[leaf2w[c.item()]]}'({v:.2f})" if c.item() in leaf2w else f"?({v:.2f})"
        for c,v in zip(tp.indices,tp.values)))
