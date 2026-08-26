"""Sample lyrics three ways: flat, cascade, and template-locked.

  flat      one 8192-way softmax, sample a leaf, emit its word.
  cascade   walk the tree: pick the coarse bucket first (the "template"), then
            refine inside it. Because each rung is a separate decision you get a
            knob the flat model does not have -- a per-rung temperature. Cold
            coarse + hot fine means "stay on the template, be surprising inside
            it"; the reverse means "wander semantically, use safe words".
  locked    take the coarse cluster path of a real verse and force the cascade
            to follow it, resampling only the rungs below. Same skeleton, new
            words -- which is the sharpest way to see what the coarse rungs are
            actually holding onto.

usage: python generate.py [N] [seed] [ntokens]
"""
import sys, os, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
from gatelm import LM, tr_ids, va_ids, DEPTH, CTX

LV = [1, 4, 8, 12, 13]
N    = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
NTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 120

z = np.load("vocabtree.npz"); code = z['code'].astype(np.int64); words = z['words']
leaf2w = np.full(1 << DEPTH, -1, dtype=np.int64)
for w, c in enumerate(code): leaf2w[c] = w
LIVE = torch.from_numpy(leaf2w >= 0)          # 8000 of 8192 leaves are real words

def load(tag, L, steps):
    ck = f"ckpt/cas_{tag}_L{L}_n{steps}_s{SEED}.pt"
    if not os.path.exists(ck): raise SystemExit(f"missing {ck} -- run cascade.py first")
    net = LM(2**L); net.load_state_dict(torch.load(ck)['n']); net.eval(); return net

nets = {L: load("cas", L, N) for L in LV}
flat = load("flat", 13, N * len(LV))

def detok(ws):
    out = []
    for w in ws:
        if w == "\n": out.append("\n")
        elif w in ".,!?;:": out.append(w)
        else: out.append(" " + w)
    return "".join(out).replace("\n ", "\n").strip()

def sample_from(logits, temp, mask=None):
    lg = logits / max(temp, 1e-6)
    if mask is not None: lg = lg.masked_fill(~mask, -1e30)
    return torch.multinomial(F.softmax(lg, -1), 1).item()

def gen_flat(ctx, ntok, temp=0.9):
    x = list(ctx)
    for _ in range(ntok):
        inp = torch.tensor([x[-CTX:]], dtype=torch.long)
        with torch.no_grad(): lg = flat(inp)[0, -1]
        leaf = sample_from(lg, temp, LIVE)
        x.append(int(leaf2w[leaf]))
    return x[len(ctx):]

def gen_cascade(ctx, ntok, temps, lock=None):
    """temps: per-rung temperature, same length as LV.
    lock: optional list-of-(step -> forced bucket at rung index li)."""
    x = list(ctx)
    for t in range(ntok):
        inp = torch.tensor([x[-CTX:]], dtype=torch.long)
        with torch.no_grad(): lgs = {L: nets[L](inp)[0, -1] for L in LV}
        par = None
        for i, L in enumerate(LV):
            if lock is not None and i < len(lock) and t < len(lock[i]) and lock[i][t] >= 0:
                par = lock[i][t]; continue
            if i == 0:
                # every level-1 bucket is live, no mask needed
                par = sample_from(lgs[L], temps[i])
            else:
                d = L - LV[i-1]
                idx = torch.arange(par << d, (par + 1) << d)
                sub = lgs[L][idx]
                m = LIVE.reshape(-1, 1 << (DEPTH - L)).any(-1)[idx] if L < DEPTH else LIVE[idx]
                if not m.any(): m = None
                par = int(idx[sample_from(sub, temps[i], m)])
        w = leaf2w[par]
        x.append(int(w) if w >= 0 else 0)
    return x[len(ctx):]

def show(title, toks):
    print("\n" + "-" * 70); print(title); print("-" * 70)
    print(detok([str(words[t]) for t in toks]))

if __name__ == "__main__":
    rng = np.random.default_rng(4)
    i = rng.integers(0, len(va_ids) - CTX - NTOK - 1)
    ctx = va_ids[i:i+CTX]
    print("=" * 70); print("PROMPT (held-out):"); print("=" * 70)
    print(detok([str(words[t]) for t in ctx[-24:]]))

    show("FLAT  (single 8192-way head, temp 0.9)", gen_flat(ctx, NTOK, 0.9))
    show("CASCADE  uniform temp 0.9 at every rung",
         gen_cascade(ctx, NTOK, [0.9]*5))
    show("CASCADE  cold template / hot detail  (0.6,0.6,0.9,1.1,1.2)",
         gen_cascade(ctx, NTOK, [0.6, 0.6, 0.9, 1.1, 1.2]))
    show("CASCADE  hot template / cold detail  (1.3,1.3,1.0,0.7,0.6)",
         gen_cascade(ctx, NTOK, [1.3, 1.3, 1.0, 0.7, 0.6]))

    # template-locked: reuse the real continuation's L1+L4 path, resample below
    real = va_ids[i+CTX : i+CTX+NTOK]
    lock = [[int(code[w] >> (DEPTH - L)) for w in real] if L in (1, 4) else []
            for L in LV]
    show("REAL continuation", list(real))
    show("TEMPLATE-LOCKED  (real L1+L4 path, rungs 8/12/13 resampled)",
         gen_cascade(ctx, NTOK, [0.9]*5, lock=lock))
