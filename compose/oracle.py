"""Two numbers the depth curve cannot be read without.

(1) ARGUMENT ORACLE. The attribution says almost every wrong answer is copied
    from a row keyed on the correct FINAL argument x_{d-1}. So the interesting
    baseline is not a single-cue heuristic over the whole context but: given
    that you have already found x_{d-1}, guess uniformly among the rows keyed on
    it. That is the score of a model that does the pointer chase perfectly and
    the final function match not at all.

(2) Single-cue heuristic floors at whatever n_defs is being run.
"""
import os, numpy as np
from gen import held_pairs, CONSTS, FUNCS, NC
GEN=os.environ.get('GEN','1')
if GEN=='2': from gen2 import make
else:        from gen import make


ndef=int(os.environ.get('NDEF','12')); N=4000
rng=np.random.default_rng(11); H=held_pairs()
print(f"n_defs={ndef}   chance=1/20=0.050   (n={N} per cell)\n")
print(f"{'depth':>5} {'arg-oracle':>11} {'rows on x_last':>15} {'outer-fn':>9} {'query-arg':>10} {'last-def':>9}")
for d in (1,2,3,4):
    orc=[]; nrow=[]; ofm=0; qam=0; ld=0
    for _ in range(N):
        t,ans,defs,fs,xs=make(d,rng,H,"train",n_defs=ndef)
        rows=[y for f,x,y in defs if x==xs[-2]]          # keyed on the final argument
        nrow.append(len(rows)); orc.append(rows[rng.integers(0,len(rows))]==ans)
        m=[y for f,x,y in defs if f==fs[-1]]; ofm+= bool(m) and m[rng.integers(0,len(m))]==ans
        m2=[y for f,x,y in defs if x==xs[0]]; qam+= bool(m2) and m2[rng.integers(0,len(m2))]==ans
        ld+= defs[-1][2]==ans
    print(f"{d:>5} {np.mean(orc):>11.3f} {np.mean(nrow):>15.2f} {ofm/N:>9.3f} {qam/N:>10.3f} {ld/N:>9.3f}")
