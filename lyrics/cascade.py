"""Does the ladder learn the template? Cascade decoding vs a flat model.

The repo so far used the ladder only as an *initialisation* and then threw the
coarse rungs away. The lyrics theory is different: keep them, and factorise the
next-word distribution across rungs

    P(leaf) = P_1(c1) . P_4(c4|c1) . P_8(c8|c4) . P_12(c12|c8) . P_13(c13|c12)

where each factor is that rung's own model, restricted to the subtree the
coarser rung picked and renormalised. The supports tile exactly, so this is a
normalised distribution over the same 8192 leaves as the flat model and its
held-out CE is directly comparable.

The sharp test is the per-rung decomposition. A flat L13 model *also* induces a
distribution at every rung (sum its softmax within each bucket), and that
telescopes to exactly its own CE. So we can ask, rung by rung:

    does a model that only ever had to predict the template predict the template
    better than a full-vocabulary model does?

If "coarse training learns the template" means anything, the answer is yes at
the coarse rungs. Compute is charged honestly: the cascade trains 5 models, so
the flat control gets the cascade's whole step budget.
"""
import sys, os, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
from gatelm import LM, batch, tgt, tr_ids, va_ids, DEPTH, CTX

LV = [1, 4, 8, 12, 13]
BS = 16                      # identical for every member, so steps are comparable
EVB, EVBS = 20, 32           # eval batches / batch size -> 40960 held-out positions

def train_one(L, steps, seed, tag, BUD, t0):
    """One member of the cascade, or the flat control when L==13 and tag=='flat'."""
    ncls = 2**L; ck = f"ckpt/cas_{tag}_L{L}_n{steps}_s{seed}.pt"
    os.makedirs("ckpt", exist_ok=True)
    torch.manual_seed(seed); net = LM(ncls)
    o = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(o, 3e-3, total_steps=steps, pct_start=0.15)
    s0 = 0; spent = 0.0
    if os.path.exists(ck):
        c = torch.load(ck); net.load_state_dict(c['n']); o.load_state_dict(c['o'])
        sch.load_state_dict(c['s']); s0 = c['step']; spent = c['spent']
        if s0 >= steps: return net, spent
    rng = np.random.default_rng(11 + seed)
    for _ in range(s0): rng.integers(0, len(tr_ids) - CTX - 1, BS)
    tw = time.time()
    for s in range(s0+1, steps+1):
        x, y = batch(tr_ids, BS, rng)
        loss = F.cross_entropy(net(x).reshape(-1, ncls), tgt(y, L).reshape(-1))
        o.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); o.step(); sch.step()
        if time.time() - t0 > BUD or s == steps:
            torch.save({'n':net.state_dict(),'o':o.state_dict(),'s':sch.state_dict(),
                        'step':s, 'spent':spent + time.time() - tw}, ck)
            if s < steps:
                print(f"  {tag} L{L} n{steps} paused {s}/{steps}", flush=True); return None, None
    return net, spent + time.time() - tw

def rung_nats(nets, flat):
    """Per-rung nats for the cascade and for the flat model's induced hierarchy.

    Both columns telescope to a total CE over the same 8192 leaves, so the
    difference at each rung is exactly what the dedicated rung model buys.
    """
    rng = np.random.default_rng(7)
    cas = np.zeros(len(LV)); flt = np.zeros(len(LV)); npos = 0
    for n in nets.values(): n.eval()
    flat.eval()
    with torch.no_grad():
        for _ in range(EVB):
            x, y = batch(va_ids, EVBS, rng)
            true = {L: tgt(y, L).reshape(-1) for L in LV}
            npos += true[13].numel()
            # --- cascade: each rung's own model, renormalised inside the parent
            for i, L in enumerate(LV):
                lg = nets[L](x).reshape(-1, 2**L)
                if i == 0:
                    cas[i] += -F.log_softmax(lg, -1).gather(
                        1, true[L][:, None]).sum().item()
                else:
                    Lp = LV[i-1]; d = L - Lp
                    idx = (true[Lp][:, None] << d) + torch.arange(1 << d)
                    sub = F.log_softmax(lg.gather(1, idx), -1)
                    cas[i] += -sub.gather(1, (true[L] - (true[Lp] << d))[:, None]).sum().item()
                del lg
            # --- flat: marginalise its own softmax into the same buckets
            lp13 = F.log_softmax(flat(x).reshape(-1, 8192), -1)
            prev = None
            for i, L in enumerate(LV):
                # log P_flat(c_L) by summing leaf mass inside each level-L bucket
                m = torch.logsumexp(lp13.reshape(-1, 2**L, 8192 >> L), -1)
                cur = m.gather(1, true[L][:, None]).squeeze(1)
                flt[i] += -(cur - (prev if prev is not None else 0)).sum().item()
                prev = cur
                del m
            del lp13
    for n in nets.values(): n.train()
    flat.train()
    return cas / npos, flt / npos

if __name__ == "__main__":
    BUD = float(sys.argv[1]); t0 = time.time()
    NS = [int(v) for v in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["300","600","1200"])]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    res = json.load(open("cascade.json")) if os.path.exists("cascade.json") else {}
    for N in NS:
        key = f"n{N}_s{seed}"
        if key in res: continue
        nets = {}; cost = 0.0
        for L in LV:
            net, sp = train_one(L, N, seed, "cas", BUD, t0)
            if net is None: sys.exit(0)
            nets[L] = net; cost += sp
        flat, fsp = train_one(13, N * len(LV), seed, "flat", BUD, t0)
        if flat is None: sys.exit(0)
        cas, flt = rung_nats(nets, flat)
        res[key] = {'N': N, 'cascade_rungs': cas.tolist(), 'flat_rungs': flt.tolist(),
                    'cascade_ce': float(cas.sum()), 'flat_ce': float(flt.sum()),
                    'cascade_sec': cost, 'flat_sec': fsp,
                    'cascade_steps': N * len(LV), 'flat_steps': N * len(LV)}
        json.dump(res, open("cascade.json", "w"))
        print(f"\n=== {N} steps/rung  (cascade {N*len(LV)} steps / {cost:.0f}s   "
              f"flat {N*len(LV)} steps / {fsp:.0f}s)", flush=True)
        print(f"{'rung':>6} {'cascade':>9} {'flat':>9} {'delta':>9}")
        for i, L in enumerate(LV):
            lab = f"L{LV[i-1]}->{L}" if i else f"->L{L}"
            print(f"{lab:>6} {cas[i]:9.4f} {flt[i]:9.4f} {flt[i]-cas[i]:+9.4f}")
        print(f"{'TOTAL':>6} {cas.sum():9.4f} {flt.sum():9.4f} {flt.sum()-cas.sum():+9.4f}"
              f"   ppl {np.exp(cas.sum()):.1f} vs {np.exp(flt.sum()):.1f}", flush=True)
    print("ALLDONE", flush=True)
