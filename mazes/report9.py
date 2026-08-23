"""Print the matched-compute table for the 9x9 maze ladder."""
import json, os

lad=json.load(open("lad9.json")); sw=json.load(open("sweep9.json")) if os.path.exists("sweep9.json") else {}
CHAIN={24:2400,16:1800}

print("\n== reproduction check: lad9.py at 600 steps ==")
ref=json.load(open("../results/lad9.json"))
for k in sorted(ref):
    if k in lad:
        print(f"  {k:10s} published {ref[k]['final']:.4f}  rerun {lad[k]['final']:.4f}  "
              f"delta {lad[k]['final']-ref[k]['final']:+.4f}")

for dist in (24,16):
    print(f"\n== d={dist}: final-rung budget sweep ==")
    a600=lad.get(f"anc_d{dist}"); c600=lad.get(f"ctrl_d{dist}")
    print(f"  {'budget':>7} {'anc':>8} {'ctrl':>8} {'gap':>8}   anc total / ctrl total")
    rows=[(600, a600['final'] if a600 else None, c600['final'] if c600 else None)]
    for n in (1200,2400,3000):
        a=sw.get(f"anc_d{dist}_{n}_0"); c=sw.get(f"ctrl_d{dist}_{n}_0")
        if a or c: rows.append((n, a['final'] if a else None, c['final'] if c else None))
    for n,a,c in rows:
        g=f"{a-c:+.4f}" if (a is not None and c is not None) else "     -"
        print(f"  {n:7d} {a if a else float('nan'):8.4f} {c if c else float('nan'):8.4f} {g:>8}"
              f"   {n+CHAIN[dist]:5d} / {n:5d}")
    # the decisive line: ancestor at its published budget vs control at matched TOTAL compute
    tot=600+CHAIN[dist]
    c_m=sw.get(f"ctrl_d{dist}_{tot}_0")
    if a600 and c_m:
        print(f"\n  MATCHED TOTAL COMPUTE ({tot} steps each):")
        print(f"    anc  600 steps + {CHAIN[dist]} chain = {tot}: {a600['final']:.4f}")
        print(f"    ctrl {tot} steps flat              : {c_m['final']:.4f}")
        print(f"    curriculum advantage: {a600['final']-c_m['final']:+.4f}   "
              f"(published, starved baseline: {a600['final']-c600['final']:+.4f})")
