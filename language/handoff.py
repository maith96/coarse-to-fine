"""Isolate the final handoff: same ancestor, four ways of expanding its head.

multiseed.py's ancp arm differs from ancu in two places at once (the chain that
built the L12 parent, and the L12->L13 expansion). This holds the parent fixed
at ancu_L12 and varies only the expansion, so the final handoff is the only
thing being measured.

The prior is estimated from train counts, and at L12->L13 those counts are thin
-- 61% of children have <5 tokens. alpha is the additive smoothing on that
estimate; alpha -> large recovers the uniform expansion.

  python handoff.py <seconds-budget> [nseeds]
"""
import sys, os, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
from gatelm import LM
from multiseed import expand, child_logprior, train, ce_full, ce_repo, TOP

ALPHAS=[1.0, 20.0, 200.0]
STEPS=300

def expand_a(state, alpha):
    s=expand(state,12,TOP)
    s['hd.bias']=s['hd.bias']+child_logprior(12,TOP,alpha=alpha)
    return s

if __name__=="__main__":
    BUD=float(sys.argv[1]); NS=int(sys.argv[2]) if len(sys.argv)>2 else 5
    t0=time.time(); RES="../results/handoff.json"
    res=json.load(open(RES)) if os.path.exists(RES) else {}
    for seed in range(NS):
        par=torch.load(f"ckpt/ms_s{seed}/ancu_L12.pt")['n']
        for a in ALPHAS:
            key=f"s{seed}/prior_a{a:g}"
            if key in res: continue
            if time.time()-t0>BUD: print("PAUSE",flush=True); sys.exit(0)
            r=train(TOP, expand_a(par,a), f"hf_{key}", STEPS, seed, BUD, t0)
            if r is None: sys.exit(0)
            res[key]={k:float(v) for k,v in r.items() if k!='state'}
            json.dump(res,open(RES,"w"),indent=1)
            print(f"[{time.time()-t0:5.0f}s] {key:18s} zs {r['zs_full']:6.3f} -> "
                  f"CE {r['ce_full']:.4f}",flush=True)
    print("ALLDONE",flush=True)
