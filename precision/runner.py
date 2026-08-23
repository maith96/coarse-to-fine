import json, os, sys, time, torch, warnings
warnings.filterwarnings("ignore")
from rg import train_level, LEVEL_BITS

B = 800
EV = 40
SEEDS = [0]
OUT = "results.json"
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 200.0
os.makedirs("ckpt", exist_ok=True)
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
t0 = time.time()

jobs = []
for s in SEEDS:
    for lv in range(len(LEVEL_BITS)):
        jobs.append((f"ctrl_s{s}_L{lv}", lv, s, None))
    for lv in range(1, len(LEVEL_BITS)):
        prev = f"anc_s{s}_L{lv-1}" if lv > 1 else f"ctrl_s{s}_L0"
        jobs.append((f"anc_s{s}_L{lv}", lv, s, prev))

for key, lv, seed, initk in jobs:
    if key in res:
        continue
    if time.time() - t0 > BUDGET:
        print("PAUSE", sum(1 for j in jobs if j[0] not in res), "left", flush=True)
        sys.exit(0)
    state = torch.load(f"ckpt/{initk}.pt") if initk else None
    if initk and initk not in res:
        continue
    net, curve = train_level(lv, state, B, "cpu", seed, eval_every=EV)
    res[key] = curve
    torch.save(net.state_dict(), f"ckpt/{key}.pt")
    json.dump(res, open(OUT, "w"))
    print(f"[{time.time()-t0:5.0f}s] {key} bit {curve[-1][1]:.4f} exact {curve[-1][2]:.4f}", flush=True)
print("ALLDONE" if all(j[0] in res for j in jobs) else "PARTIAL", flush=True)
