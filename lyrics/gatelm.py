"""Gate test: does the coarse-to-fine axis hold on hip-hop lyrics?

Train an independent LM at each rung L (predict the 2^L-way cluster of the next
token) and check that (a) train loss falls and (b) the information gain over the
marginal grows with granularity. A rung where the model cannot beat the unigram
marginal is a void in the axis, and the ladder above it means nothing.

Architecture is byte-identical to language/gatelm.py so the hip-hop numbers sit
next to the Shakespeare ones. Sizes are overridable via env for the generator.
"""
import sys, os, json, time, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("LM_THREADS", "4")))

z = np.load("vocabtree.npz")
ids = z['ids'].astype(np.int64); code = z['code'].astype(np.int64)
words = z['words']
DEPTH = 13; V = 8000
CTX = int(os.environ.get("LM_CTX", "64"))
DM  = int(os.environ.get("LM_DM",  "96"))
NL  = int(os.environ.get("LM_NL",  "3"))

n = len(ids); SP = int(0.9 * n)
tr_ids, va_ids = ids[:SP], ids[SP:]

def tgt(x, L): return torch.from_numpy(code[x] >> (DEPTH - L))

class LM(nn.Module):
    def __init__(s, ncls, dm=None, nl=None):
        super().__init__()
        dm = DM if dm is None else dm; nl = NL if nl is None else nl
        s.emb = nn.Embedding(V, dm); s.pos = nn.Embedding(CTX, dm)
        s.ls = nn.ModuleList([nn.TransformerEncoderLayer(dm, 4, 4*dm, dropout=0.0,
            batch_first=True, norm_first=True, activation="gelu") for _ in range(nl)])
        s.ln = nn.LayerNorm(dm); s.hd = nn.Linear(dm, ncls)
    def forward(s, x):
        h = s.emb(x) + s.pos.weight[None, :x.shape[1]]
        m = nn.Transformer.generate_square_subsequent_mask(x.shape[1])
        for l in s.ls: h = l(h, src_mask=m, is_causal=True)
        return s.hd(s.ln(h))

def batch(arr, bs, rng):
    i = rng.integers(0, len(arr) - CTX - 1, bs)
    x = np.stack([arr[k:k+CTX] for k in i]); y = np.stack([arr[k+1:k+CTX+1] for k in i])
    return torch.from_numpy(x), y

def evaluate(net, L, nb=6, bs=32, seed=7):
    rng = np.random.default_rng(seed); net.eval(); c = t = 0; ce = 0.
    with torch.no_grad():
        for _ in range(nb):
            x, y = batch(va_ids, bs, rng); lg = net(x); yy = tgt(y, L)
            p = lg.argmax(-1); c += (p == yy).sum().item(); t += p.numel()
            ce += F.cross_entropy(lg.reshape(-1, 2**L), yy.reshape(-1)).item()
    net.train(); return c / t, ce / nb

def baseline(L):
    cl = code[tr_ids] >> (DEPTH - L); pr = np.bincount(cl, minlength=2**L) + 1.0
    pr /= pr.sum(); vc = code[va_ids] >> (DEPTH - L)
    return (vc == np.bincount(cl).argmax()).mean(), float(-np.log(pr[vc]).mean())

CFG = {1:(800,64), 4:(800,64), 8:(800,64), 12:(400,24), 13:(400,16)}
def run(L, lr=3e-3, BUD=180, t0=0):
    steps, bs = CFG[L]
    ncls = 2**L; ck = f"ckpt/lm_L{L}.pt"; os.makedirs("ckpt", exist_ok=True)
    torch.manual_seed(0); net = LM(ncls)
    o = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(o, lr, total_steps=steps, pct_start=0.15)
    s0 = 0; hist = []
    if os.path.exists(ck):
        c = torch.load(ck); net.load_state_dict(c['n']); o.load_state_dict(c['o'])
        sch.load_state_dict(c['s']); s0 = c['step']; hist = c['h']
    rng = np.random.default_rng(3)
    for _ in range(s0): rng.integers(0, len(tr_ids) - CTX - 1, bs)
    for s in range(s0+1, steps+1):
        x, y = batch(tr_ids, bs, rng)
        loss = F.cross_entropy(net(x).reshape(-1, ncls), tgt(y, L).reshape(-1))
        o.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); o.step(); sch.step()
        if s % 200 == 0 or s == 1: hist.append((s, round(loss.item(), 4)))
        if time.time() - t0 > BUD or s == steps:
            torch.save({'n':net.state_dict(),'o':o.state_dict(),'s':sch.state_dict(),
                        'step':s,'h':hist}, ck)
            if s < steps:
                print(f"  L{L} paused {s}/{steps} loss {loss.item():.3f}", flush=True); return None
    acc, vce = evaluate(net, L); b, bce = baseline(L)
    print(f"L{L:2d} ({ncls:5d} cl) | val CE {vce:.4f} vs marginal {bce:.4f} = {bce-vce:+.4f} nats | "
          f"acc {acc:.4f}/maj {b:.4f} | train {hist[0][1]:.3f}->{hist[-1][1]:.3f} | lnN {np.log(ncls):.2f}", flush=True)
    return {'acc':acc, 'vce':vce, 'base':float(b), 'bce':bce, 'hist':hist}

if __name__ == "__main__":
    t0 = time.time(); BUD = float(sys.argv[2])
    res = json.load(open("gatelm.json")) if os.path.exists("gatelm.json") else {}
    for L in [int(v) for v in sys.argv[1].split(",")]:
        if str(L) in res: continue
        r = run(L, BUD=BUD, t0=t0)
        if r is None: sys.exit(0)
        res[str(L)] = r; json.dump(res, open("gatelm.json", "w"))
    print("DONE", flush=True)
