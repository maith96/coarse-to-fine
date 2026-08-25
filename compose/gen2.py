"""Depth-invariant generator. The first generator had a flaw that would have
faked a depth curve.

In gen.py the distractor budget was split evenly across chain positions, so the
number of definitions competing for the FINAL lookup shrank as depth grew:
7.0 rows at depth 1, 3.2 at depth 2, 2.2 at depth 3. An "already found the right
argument, now guess the function" strategy therefore scores 0.19 / 0.35 / 0.50 --
it climbs with depth purely because the last lookup gets less ambiguous. Any
accuracy-vs-depth curve measured on that generator is partly an artefact of its
own baseline moving underneath it. At n_defs=6 it is fatal: depth 3 leaves 1.05
rows on the final argument, so the task is solvable without reading functions at all.

Here the local ambiguity of every lookup is pinned, independent of depth:
  * exactly KX competing rows share the final argument   -> argument-oracle = 1/(KX+1)
  * exactly KF competing rows share the outer function   -> outer-fn oracle  = 1/(KF+1)
  * every intermediate step gets one argument-competitor and one function-competitor
  * all distractor values are drawn OUTSIDE the chain, so a wrong row is always a
    wrong answer and never accidentally rejoins the correct chain
Total definitions are held at n_defs for every depth, so context length is fixed too.
"""
import numpy as np
from gen import NF, NC, FUNCS, CONSTS, SPECIAL, VOCAB, TOK, V, held_pairs, encode

KX = KF = 4          # competitors on the final argument / the outer function

def make(depth, rng, held, split, n_defs=18):
    need = depth + KX + KF + 2*(depth-1)
    assert n_defs >= need, f"depth {depth} needs n_defs>={need}"
    for _ in range(400):
        fs=[FUNCS[i] for i in rng.integers(0,NF,depth)]
        xs=list(rng.permutation(CONSTS)[:depth+1])
        pairs={(fs[i+1],fs[i]) for i in range(depth-1)}
        outer=(fs[-1],fs[-2]) if depth>=2 else None
        if split=="train" and (pairs & held): continue
        if split=="test" and depth>=2 and outer not in held: continue
        chainset=set(xs)
        pool=[c for c in CONSTS if c not in chainset]      # safe distractor values
        defs=[(fs[i],xs[i],xs[i+1]) for i in range(depth)]
        keys={(f,x) for f,x,_ in defs}
        def val(): return pool[rng.integers(0,len(pool))]
        def add(f,x):
            if (f,x) in keys: return False
            keys.add((f,x)); defs.append((f,x,val())); return True
        def tries(fn, k):
            got=0
            for _ in range(200):
                if got==k: break
                got += fn()
            return got==k
        fa, xa = fs[-1], xs[-2]                            # the final lookup
        ok  = tries(lambda: add(FUNCS[rng.integers(0,NF)], xa), KX)   # share final arg
        ok &= tries(lambda: add(fa, CONSTS[rng.integers(0,NC)]), KF)  # share outer fn
        for i in range(depth-1):                           # ambiguity at every step
            ok &= tries(lambda: add(FUNCS[rng.integers(0,NF)], xs[i]), 1)
            ok &= tries(lambda: add(fs[i], CONSTS[rng.integers(0,NC)]), 1)
        if not ok: continue
        while len(defs)<n_defs:                            # unrelated filler
            add(FUNCS[rng.integers(0,NF)], CONSTS[rng.integers(0,NC)])
        rng.shuffle(defs)
        toks=[]
        for f,x,y in defs: toks += [f,"OF",x,"IS",y,"."]
        toks += ["QUERY",":"]
        for f in reversed(fs): toks += [f,"OF"]
        toks += [xs[0],"ANSWER",":"]
        return toks, xs[-1], defs, fs, xs
    raise RuntimeError("generator failed")

if __name__=="__main__":
    rng=np.random.default_rng(0); H=held_pairs(); N=4000
    for d in (1,2,3,4):
        t,ans,defs,fs,xs=make(d,rng,H,"train")
        print(f"--- depth {d}: {len(t)+1} tokens, {len(defs)} definitions ---")
        print("  chain: "+xs[0]+" "+" ".join(f"-{f}-> {x}" for f,x in zip(fs,xs[1:])))
    print(f"\nfloors, n_defs=18, n={N}   (chance = 1/20 = 0.050)")
    print(f"{'depth':>5} {'arg-oracle':>11} {'outer-fn':>9} {'query-arg':>10} {'last-def':>9}")
    for d in (1,2,3,4):
        o=[];ofm=0;qam=0;ld=0
        for _ in range(N):
            t,ans,defs,fs,xs=make(d,rng,H,"train")
            r=[y for f,x,y in defs if x==xs[-2]]; o.append(r[rng.integers(0,len(r))]==ans)
            m=[y for f,x,y in defs if f==fs[-1]]; ofm+= m[rng.integers(0,len(m))]==ans
            m2=[y for f,x,y in defs if x==xs[0]]; qam+= m2[rng.integers(0,len(m2))]==ans
            ld+= defs[-1][2]==ans
        print(f"{d:>5} {np.mean(o):>11.3f} {ofm/N:>9.3f} {qam/N:>10.3f} {ld/N:>9.3f}")
