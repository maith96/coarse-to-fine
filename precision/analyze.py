import json, torch, warnings
warnings.filterwarnings("ignore")
from rg import LEVEL_BITS, Net, evaluate, NP
res = json.load(open("results.json"))
S = 0
print("level  bits | ctrl_final(bit/exact) | anc_final(bit/exact) | anc@step40 | tau_bit tau_ex")
print("-"*100)
rows=[]
for lv in range(1, len(LEVEL_BITS)):
    c = res[f"ctrl_s{S}_L{lv}"]; a = res[f"anc_s{S}_L{lv}"]
    cb, ce = c[-1][1], c[-1][2]
    B = c[-1][0]
    def first_reach(curve, idx, target):
        for st, pb, ex in curve:
            if (pb if idx==1 else ex) >= target: return st
        return None
    sb = first_reach(a,1,cb); se = first_reach(a,2,ce)
    tb = B/sb if sb else float('inf')
    te = B/se if se else float('inf')
    rows.append((lv, LEVEL_BITS[lv], cb, ce, a[-1][1], a[-1][2], a[0][1], a[0][2], tb, te))
    print(f"L{lv}   {LEVEL_BITS[lv]:2d}b | {cb:.4f}/{ce:.4f}     | {a[-1][1]:.4f}/{a[-1][2]:.4f}     | "
          f"{a[0][1]:.4f}/{a[0][2]:.4f} | {tb:6.2f}  {te:6.2f}")

# true zero-shot: ancestor L(n-1) checkpoint evaluated under level n mask
print("\nZERO-SHOT (ancestor weights, never trained on this level's extra bits):")
print("level | zs_bit  ctrl_step40_bit  ctrl_final_bit  majority_bit")
g = torch.Generator(); g.manual_seed(999)
import rg
for lv in range(1, len(LEVEL_BITS)):
    prev = f"anc_s{S}_L{lv-1}" if lv>1 else f"ctrl_s{S}_L0"
    net = Net(); net.load_state_dict(torch.load(f"ckpt/{prev}.pt"))
    mask = torch.zeros(NP, dtype=torch.bool); mask[:LEVEL_BITS[lv]] = True
    ge = torch.Generator(); ge.manual_seed(12345+lv)
    zb, ze = evaluate(net, mask, "cpu", ge)
    c = res[f"ctrl_s{S}_L{lv}"]
    print(f"L{lv} {LEVEL_BITS[lv]:2d}b | {zb:.4f}   {c[0][1]:.4f}          {c[-1][1]:.4f}")
json.dump(rows, open("tau.json","w"))
