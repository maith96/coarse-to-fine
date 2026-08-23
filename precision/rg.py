"""
Bit-precision crux: is there a real coarse-graining operator R on computation?

Task: p = a * b, a,b are 12-bit unsigned -> p is 24 bits.
Level k trains on the top 2^k bits of p.  Level k+1's bit set strictly contains
level k's, so R(level k+1) == level k exactly (prefix property, by construction).

Architecture is IDENTICAL at every level; only the loss mask changes.
So R acts purely as a mask on the objective. Nothing else differs.

Ancestor condition: level k+1 initialized from trained level k weights.
Control  condition: level k+1 initialized fresh.
"""
import math, json, time, argparse
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(1)

NB = 12          # operand bits
NP = 2 * NB      # product bits = 24
SEQ = NB + NB + NP
LEVEL_BITS = [1, 2, 4, 8, 16, NP]   # top-k bits per level


def sample(bs, device, gen):
    # NORMALIZED operands: both have the MSB set, so a,b in [2^11, 2^12).
    # => product always in [2^22, 2^24), i.e. the leading bits of the 24-bit
    # field are informative rather than almost-always-zero. This is exactly
    # "multiply two normalized mantissas"; level k = 2^k bits of precision.
    lo = 2 ** (NB - 1)
    a = lo + torch.randint(0, lo, (bs,), generator=gen, device=device)
    b = lo + torch.randint(0, lo, (bs,), generator=gen, device=device)
    p = a * b
    sh_in = torch.arange(NB - 1, -1, -1, device=device)
    sh_out = torch.arange(NP - 1, -1, -1, device=device)
    abits = (a[:, None] >> sh_in) & 1
    bbits = (b[:, None] >> sh_in) & 1
    pbits = (p[:, None] >> sh_out) & 1
    return torch.cat([abits, bbits], 1).long(), pbits.float()


class Net(nn.Module):
    def __init__(self, d=64, L=3, H=4, ff=256):
        super().__init__()
        self.tok = nn.Embedding(3, d)          # 0,1 = input bits; 2 = output query
        self.pos = nn.Embedding(SEQ, d)
        layer = nn.TransformerEncoderLayer(d, H, ff, dropout=0.0,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, L)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.d = d

    def forward(self, x, return_acts=False):
        B = x.shape[0]
        q = torch.full((B, NP), 2, dtype=torch.long, device=x.device)
        t = self.tok(torch.cat([x, q], 1)) + self.pos.weight[None]
        if return_acts:
            acts, h = [], t
            for lyr in self.enc.layers:
                h = lyr(h)
                acts.append(h.detach())
            return self.head(self.ln(h[:, -NP:])).squeeze(-1), acts
        h = self.enc(t)
        return self.head(self.ln(h[:, -NP:])).squeeze(-1)


def evaluate(net, mask, device, gen, n=1024, bs=512):
    net.eval()
    bitc = exc = tot = 0
    with torch.no_grad():
        for _ in range(n // bs):
            x, y = sample(bs, device, gen)
            pr = (net(x) > 0).float()[:, mask]
            corr = (pr == y[:, mask])
            bitc += corr.sum().item()
            exc += corr.all(1).sum().item()
            tot += bs
    net.train()
    nb = mask.sum().item()
    return bitc / (tot * nb), exc / tot


def train_level(level, init_state, steps, device, seed, lr=3e-3, bs=128,
                eval_every=None, log=None):
    k = LEVEL_BITS[level]
    mask = torch.zeros(NP, dtype=torch.bool, device=device)
    mask[:k] = True                      # top-k bits (MSB first)
    torch.manual_seed(seed)
    net = Net().to(device)
    if init_state is not None:
        net.load_state_dict(init_state)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps,
                                              pct_start=0.15)
    g = torch.Generator(device=device); g.manual_seed(seed * 7919 + level)
    ge = torch.Generator(device=device); ge.manual_seed(12345 + level)  # fixed eval stream
    eval_every = eval_every or max(steps // 30, 1)
    curve = []
    if log is not None:
        pb, ex = evaluate(net, mask, device, ge)
        curve.append((0, pb, ex))
    for s in range(1, steps + 1):
        x, y = sample(bs, device, g)
        logits = net(x)
        loss = F.binary_cross_entropy_with_logits(logits[:, mask], y[:, mask])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sch.step()
        if s % eval_every == 0 or s == steps:
            ge2 = torch.Generator(device=device); ge2.manual_seed(12345 + level)
            pb, ex = evaluate(net, mask, device, ge2)
            curve.append((s, pb, ex))
            if log:
                print(f"  {log} L{level}({k}b) step {s:5d} loss {loss.item():.4f} "
                      f"bit {pb:.4f} exact {ex:.4f}", flush=True)
    return net, curve


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    dev = "cpu"
    if a.bench:
        t0 = time.time()
        train_level(5, None, 50, dev, 0, eval_every=10**9)
        print("50 steps in", round(time.time() - t0, 2), "s")

def pilot():
    for lv, st in [(5, 1200), (2, 600)]:
        train_level(lv, None, st, "cpu", 0, log="pilot")
