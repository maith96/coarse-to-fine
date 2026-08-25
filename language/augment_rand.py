"""Entity-randomised training corpus: the same augmentation, but no fixed facts.

§6 held out QA pairs and paraphrased the questions, and both arms still scored
zero, because the facts were constant across every document -- memorisation
remained a sufficient strategy for the entire training set. Here each document
gets its own cast, colours, times and quantities (randomize.py), so no answer is
predictable from the question alone and the narrative in context is the only
thing that determines it.

The split is the one augment.py already chose, by block/question index, so the
held-out questions are the same 15 as §6.

Evaluation uses assignments never seen in training, in two sets:
  held   the 15 held-out questions, fresh cast -- new question AND new facts
  seen   15 trained question types, fresh cast -- does it read the context?
and evalqa.py's 'cold' prompt (question, no narrative) is the control that
separates reading from guessing.

  python augment_rand.py [n_variants] [seed]
"""
import sys, json, random, re
from parse_marine import parse
from augment import qforms, aforms
import randomize as R

NVAR=int(sys.argv[1]) if len(sys.argv)>1 else 24
SEED=int(sys.argv[2]) if len(sys.argv)>2 else 0
PER =sys.argv[3] if len(sys.argv)>3 else "doc"      # 'doc' or 'chunk': how often the cast changes
CHUNK=4
OUT ="marine_rand" if PER=="doc" else "marine_randc"

def cast(blocks, a):
    """one assignment applied to every narrative, question and answer"""
    return [{"par":b["par"],"narr":[R.apply(s,a) for s in b["narr"]],
             "qa":[{"q":R.apply(qa["q"],a),"a":R.apply(qa["a"],a)} for qa in b["qa"]]}
            for b in blocks]

if __name__=="__main__":
    rng=random.Random(SEED); blocks=parse()
    SP=json.load(open("marine_split.json"))
    held_idx=[tuple(h["idx"]) for h in SP["held"]]
    seen_idx=[tuple(t["idx"]) for t in SP["train_qa"]]
    heldset=set(held_idx)

    # ---- training documents, one cast each ----
    docs=[]; train_casts=[]
    for v in range(NVAR):
        a=R.assign(rng, idx=v); train_casts.append(tuple(sorted((k,str(x)) for k,x in a.items())))
        bl=cast(blocks,a)
        order=list(range(len(bl))); rng.shuffle(order); doc=[]
        for bi in order:
            keep=[qi for qi in range(len(bl[bi]["qa"])) if (bi,qi) not in heldset]
            rng.shuffle(keep)
            for c in range(0,len(keep),CHUNK):
                # in 'chunk' mode every re-anchored narrative brings a new cast, so the
                # same question recurs with a different answer a few hundred tokens
                # apart and no single answer can be memorised for it
                if PER=="chunk":
                    a2=R.assign(rng); train_casts.append(tuple(sorted((k,str(x)) for k,x in a2.items())))
                    b=cast(blocks,a2)[bi]
                else: b=bl[bi]
                doc += b["narr"]                             # (re-)anchor
                for qi in keep[c:c+CHUNK]:
                    qa=b["qa"][qi]
                    doc.append(f"{rng.choice(qforms(qa['q']))} {rng.choice(aforms(qa['a']))}")
        docs.append(" ".join(doc))
    open(f"{OUT}.txt","w").write("\n\n".join(docs)+"\n")

    # ---- evaluation sets, casts drawn well away from the training ones ----
    # every trained question type once, and every held-out question under REPS
    # different casts -- 15 items was too few to separate a real context effect
    # from noise, since each item states only one or two randomised values
    REPS=4
    erng=random.Random(10_000+SEED); evalsets={}
    for name,idxs in (("held",held_idx*REPS),("seen",seen_idx)):
        items=[]
        for bi,qi in idxs:
            a=R.assign(erng)
            if tuple(sorted((k,str(x)) for k,x in a.items())) in train_casts:
                a=R.assign(erng)                              # vanishingly rare, but check
            b=cast(blocks,a)[bi]; qa=b["qa"][qi]
            # does this answer actually move with the cast? if not, the question is
            # answerable without reading anything, and it is scored separately
            alt=cast(blocks,R.assign(erng))[bi]["qa"][qi]["a"]
            items.append({"idx":[bi,qi],"par":b["par"],"narr":" ".join(b["narr"]),
                          "q":qa["q"],"a":qa["a"],"forms":qforms(qa["q"]),
                          "varies":alt!=qa["a"],
                          "cast":{k:str(x) for k,x in a.items()}})
        evalsets[name]=items
    json.dump({"n_variants":NVAR,"seed":SEED,"chunk":CHUNK,
               "held":evalsets["held"],"train_qa":evalsets["seen"]},
              open(f"{OUT}_split.json","w"),indent=1)

    # ---- one vocabulary over training text and both evaluation sets ----
    ev=" ".join(f"{it['narr']} {' '.join(it['forms'])} {it['a']}"
                for s in evalsets.values() for it in s)
    open(f"{OUT}_union.txt","w").write(open(f"{OUT}.txt").read()+"\n\n"+ev+"\n")
    wc=lambda p: len(re.findall(r"\S+",open(p).read()))
    print(f"{NVAR} documents, cast changes per {PER} ({len(set(train_casts))} distinct casts); {len(seen_idx)} QA pairs trained, "
          f"{len(held_idx)} held out")
    for p in [f"{OUT}.txt",f"{OUT}_union.txt"]: print(f"  {p:24s} {wc(p):6d} words")
    for n,s_ in evalsets.items():
        print(f"  {n:5s}: {sum(it['varies'] for it in s_)}/{len(s_)} answers move with the cast")
    print(f"\nsame question, two training documents:")
    for v in (0,1):
        a=R.assign(random.Random(0), idx=v); bl=cast(blocks,a)
        print(f"  doc {v}: {bl[0]['qa'][0]['q']}  ->  {bl[0]['qa'][0]['a']}")
    print("\none evaluation item (cast never trained on):")
    it=evalsets["held"][0]
    print(f"  narrative: {it['narr'][:150]} ...")
    print(f"  Q: {it['q']}\n  A: {it['a']}")
