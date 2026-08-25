"""Nested function resolution: generator, held-out split, and heuristic baselines.

  f3 OF a7 IS a2 .   g9 OF a2 IS a5 .   ...distractors...
  QUERY : g9 OF f3 OF a7 ANSWER : a5

Function tables are resampled per example, so nothing is memorisable -- the model
must read the bindings out of its own context. Two structural guarantees make
single-cue heuristics fail, and they are what decide whether the result means
anything:

  * every chain function also appears in a DIFFERENT definition, so matching on
    the function alone is ambiguous;
  * every chain argument also appears in a DIFFERENT definition, so matching on
    the argument alone is ambiguous.

The model therefore has to match a (function, argument) CONJUNCTION, and then
carry the result forward as a pointer to the next step.
"""
import numpy as np

NF, NC = 20, 20
FUNCS=[f"f{i}" for i in range(NF)]
CONSTS=[f"a{i}" for i in range(NC)]
SPECIAL=["OF","IS","QUERY","ANSWER",".",":","<pad>"]
VOCAB=SPECIAL+FUNCS+CONSTS
TOK={w:i for i,w in enumerate(VOCAB)}
V=len(VOCAB)

def held_pairs(frac=0.2, seed=0):
    """a fixed 20% of ordered function pairs, never composed during training."""
    rng=np.random.default_rng(seed)
    allp=[(a,b) for a in FUNCS for b in FUNCS]
    idx=rng.permutation(len(allp))[:int(frac*len(allp))]
    return {allp[i] for i in idx}

def make(depth, rng, held, split, n_defs=20):
    """one example. split='train' avoids held pairs; 'test' requires one.

    The definition count is FIXED at n_defs for every depth, so a failure at
    depth 4 cannot be blamed on a longer context. The distractor budget is spent
    on definitions that share a chain function or a chain argument, which is what
    forces a conjunctive match rather than a single-cue one.
    """
    for _ in range(400):
        fs=[FUNCS[i] for i in rng.integers(0,NF,depth)]
        xs=list(rng.permutation(CONSTS)[:depth+1])
        pairs={(fs[i+1],fs[i]) for i in range(depth-1)}
        outer=(fs[-1],fs[-2]) if depth>=2 else None
        if split=="train" and (pairs & held): continue
        if split=="test" and depth>=2 and outer not in held: continue
        defs=[(fs[i],xs[i],xs[i+1]) for i in range(depth)]
        keys={(f,x) for f,x,_ in defs}
        def add(f,x,y):
            if (f,x) in keys: return False
            keys.add((f,x)); defs.append((f,x,y)); return True
        budget=n_defs-depth
        share_f=budget//2; share_x=budget-share_f
        for k in range(share_f):                    # share a chain FUNCTION
            f=fs[k%depth]
            for _ in range(40):
                x2=CONSTS[rng.integers(0,NC)]
                if x2!=xs[k%depth] and add(f,x2,CONSTS[rng.integers(0,NC)]): break
        for k in range(share_x):                    # share a chain ARGUMENT
            x=xs[k%depth]
            for _ in range(40):
                f2=FUNCS[rng.integers(0,NF)]
                if f2!=fs[k%depth] and add(f2,x,CONSTS[rng.integers(0,NC)]): break
        while len(defs)<n_defs:                     # top up if collisions blocked some
            if not add(FUNCS[rng.integers(0,NF)],CONSTS[rng.integers(0,NC)],
                       CONSTS[rng.integers(0,NC)]): continue
        rng.shuffle(defs)
        toks=[]
        for f,x,y in defs: toks += [f,"OF",x,"IS",y,"."]
        toks += ["QUERY",":"]
        for f in reversed(fs): toks += [f,"OF"]
        toks += [xs[0],"ANSWER",":"]
        return toks, xs[-1], defs, fs, xs
    raise RuntimeError("generator failed")

def encode(toks): return [TOK[t] for t in toks]

if __name__=="__main__":
    rng=np.random.default_rng(0); held=held_pairs()
    for d in (1,2,3,4):
        t,ans,defs,fs,xs=make(d,rng,held,"train")
        print(f"--- depth {d}  ({len(t)+1} tokens, {len(defs)} definitions) ---")
        print(" "+" ".join(t)+" "+ans)
        print(f"    chain: {xs[0]} " + " ".join(f"-{f}-> {x}" for f,x in zip(fs,xs[1:])))
    # ---- heuristic baselines: how far do single cues get you? ----
    print("\nheuristic accuracy over 4000 examples (chance = 1/20 = 0.050)")
    for d in (1,2,3,4):
        n=4000; hits={"random const":0,"random const in ctx":0,
                      "outer-function match":0,"last definition":0,"query-arg match":0}
        for _ in range(n):
            t,ans,defs,fs,xs=make(d,rng,held,"train")
            hits["random const"]+= CONSTS[rng.integers(0,NC)]==ans
            ctx=[y for _,_,y in defs]
            hits["random const in ctx"]+= ctx[rng.integers(0,len(ctx))]==ans
            m=[y for f,x,y in defs if f==fs[-1]]
            hits["outer-function match"]+= (m[rng.integers(0,len(m))]==ans) if m else 0
            hits["last definition"]+= defs[-1][2]==ans
            m2=[y for f,x,y in defs if x==xs[0]]
            hits["query-arg match"]+= (m2[rng.integers(0,len(m2))]==ans) if m2 else 0
        print(f"  depth {d}: " + "  ".join(f"{k} {v/n:.3f}" for k,v in hits.items()))
