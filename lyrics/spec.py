"""Specialisation control: is a coarse specialist better at the template than a
generalist with the same number of updates?

cascade.py charges the cascade honestly -- 5 models, so the flat control gets
the whole 5N budget -- and the flat model wins. That conflates two things. This
strips the compute out: flat L13 trained for N steps, exactly what each cascade
member got, then compared rung by rung against the specialists.

If training only on the 2-way task buys anything at all, it shows up here.
"""
import sys, os, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from cascade import LV, train_one, rung_nats

if __name__ == "__main__":
    BUD = float(sys.argv[1]); t0 = time.time()
    NS = [int(v) for v in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["300","600","1200"])]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    res = json.load(open("spec.json")) if os.path.exists("spec.json") else {}
    for N in NS:
        key = f"n{N}_s{seed}"
        if key in res: continue
        nets = {}
        for L in LV:
            net, _ = train_one(L, N, seed, "cas", BUD, t0)
            if net is None: sys.exit(0)
            nets[L] = net
        flat, _ = train_one(13, N, seed, "spec", BUD, t0)
        if flat is None: sys.exit(0)
        cas, flt = rung_nats(nets, flat)
        res[key] = {'N': N, 'cascade_rungs': cas.tolist(), 'flat_rungs': flt.tolist(),
                    'cascade_ce': float(cas.sum()), 'flat_ce': float(flt.sum())}
        json.dump(res, open("spec.json", "w"))
        print(f"\n=== {N} steps per model, both sides (specialists vs one generalist)", flush=True)
        print(f"{'rung':>7} {'special':>9} {'general':>9} {'delta':>9}")
        for i, L in enumerate(LV):
            lab = f"L{LV[i-1]}->{L}" if i else f"->L{L}"
            print(f"{lab:>7} {cas[i]:9.4f} {flt[i]:9.4f} {flt[i]-cas[i]:+9.4f}")
        print(f"{'TOTAL':>7} {cas.sum():9.4f} {flt.sum():9.4f} {flt.sum()-cas.sum():+9.4f}", flush=True)
    print("ALLDONE", flush=True)
