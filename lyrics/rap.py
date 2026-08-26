"""Prompt the model and print a structured verse and hook.

Uses the flat L13 model, which §4/§5 of the README established is the better
generator at matched compute -- the cascade's only surviving advantage is
control, which is what the template-locked hook below demonstrates.

Two decoding conveniences the experiments deliberately did not use, because they
would have muddied the CE comparison but are plainly right for reading output:
nucleus sampling, and masking <unk> (3.7% of the corpus, so an unmasked sample
is peppered with holes that say nothing about the model).

usage: python rap.py "your opening line" [nlines] [seed]
"""
import sys, re, os, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
from gatelm import LM, DEPTH, CTX

LV = [1, 4, 8, 12, 13]
z = np.load("vocabtree.npz"); code = z['code'].astype(np.int64); words = z['words']
w2i = {str(w): i for i, w in enumerate(words)}
leaf2w = np.full(1 << DEPTH, -1, dtype=np.int64)
for w, c in enumerate(code): leaf2w[c] = w
LIVE = torch.from_numpy(leaf2w >= 0).clone()
LIVE[code[0]] = False                     # never emit <unk>
NL = w2i["\n"]

def load(tag, L, steps, seed=0):
    ck = f"ckpt/cas_{tag}_L{L}_n{steps}_s{seed}.pt"
    net = LM(2**L); net.load_state_dict(torch.load(ck)['n']); net.eval(); return net

def encode(s):
    toks = re.findall(r"[a-z']+|[.,!?;:\n]", s.lower())
    miss = [t for t in toks if t not in w2i]
    if miss: print(f"[out of vocab, mapped to <unk>: {' '.join(miss)}]", file=sys.stderr)
    return [w2i.get(t, 0) for t in toks]

def detok(ids):
    out = []
    for t in ids:
        w = str(words[t])
        if w == "\n": out.append("\n")
        elif w in ".,!?;:": out.append(w)
        else: out.append(" " + w)
    return "\n".join(l.strip() for l in "".join(out).split("\n"))

def pick(logits, temp, top_p, mask):
    lg = (logits / max(temp, 1e-6)).masked_fill(~mask, -1e30)
    p = F.softmax(lg, -1)
    sp, si = torch.sort(p, descending=True)
    keep = (torch.cumsum(sp, -1) - sp) < top_p        # always keeps the argmax
    sp = sp * keep
    return int(si[torch.multinomial(sp / sp.sum(), 1)])

def gen_lines(net, ctx, nlines, temp=0.9, top_p=0.92, maxlen=13, minlen=4):
    x = list(ctx); got = []; cur = []
    while len(got) < nlines and len(x) - len(ctx) < nlines * (maxlen + 2) + 40:
        inp = torch.tensor([x[-CTX:]], dtype=torch.long)
        with torch.no_grad(): lg = net(inp)[0, -1].clone()
        m = LIVE.clone()
        if len(cur) < minlen: m[int(code[NL])] = False
        leaf = int(code[NL]) if len(cur) >= maxlen else pick(lg, temp, top_p, m)
        w = int(leaf2w[leaf]); x.append(w)
        if w == NL:
            if cur: got.append(cur); cur = []
        else: cur.append(w)
    if cur and len(got) < nlines: got.append(cur)
    return got

def locked_line(nets, ctx, template, temp=1.0, top_p=0.95):
    """Resample a line under a fixed L4 cluster path -- same skeleton, new words."""
    x = list(ctx); out = []
    for w_true in template:
        c4 = int(code[w_true] >> (DEPTH - 4))
        inp = torch.tensor([x[-CTX:]], dtype=torch.long)
        with torch.no_grad(): lgs = {L: nets[L](inp)[0, -1] for L in LV if L > 4}
        par = c4
        for i, L in enumerate([8, 12, 13]):
            Lp = 4 if L == 8 else (8 if L == 12 else 12)
            d = L - Lp
            idx = torch.arange(par << d, (par + 1) << d)
            sub = lgs[L][idx]
            m = LIVE.reshape(-1, 1 << (DEPTH - L)).any(-1)[idx] if L < DEPTH else LIVE[idx]
            if not m.any(): m = torch.ones_like(idx, dtype=torch.bool)
            par = int(idx[pick(sub, temp, top_p, m)])
        w = int(leaf2w[par]); w = w if w >= 0 else int(w_true)
        out.append(w); x.append(w)
    return out

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "i came up from nothing"
    NLINES = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    torch.manual_seed(seed); np.random.seed(seed)

    flat = load("flat", 13, 3000)
    ctx = encode(prompt.rstrip() + "\n")

    print("=" * 62); print("PROMPT"); print("=" * 62)
    print(prompt.strip())

    print("\n" + "=" * 62); print(f"VERSE  ({NLINES} bars, flat model, temp 0.90 / top-p 0.92)")
    print("=" * 62)
    for l in gen_lines(flat, ctx, NLINES, temp=0.90):
        print(detok(l))

    print("\n" + "=" * 62); print("HOOK  (4 bars, temp 0.75 -- hooks are simpler than verses)")
    print("=" * 62)
    hook = gen_lines(flat, ctx, 4, temp=0.75, maxlen=9, minlen=4)
    for l in hook: print(detok(l))

    nets = {L: load("cas", L, 600) for L in LV}
    print("\n" + "=" * 62)
    print("HOOK, TEMPLATE-LOCKED  (same L4 skeleton as above, words resampled)")
    print("=" * 62)
    for l in hook:
        print(detok(locked_line(nets, ctx, l)))
