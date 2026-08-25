"""Build the held-out QA split and the paraphrase-augmented training corpus.

The original marine run scored a model on the last 10% of *tokens* -- content it
had never seen in any form -- which measures nothing but overfitting. This holds
out QA *pairs* instead: the narrative that answers them stays in training, so a
model that has learned to read the passage can answer them and a model that has
memorised strings cannot.

Writes three files plus the split:
  marine_base.txt   original text, held-out QA pairs deleted        (control)
  marine_aug.txt    NVAR shuffled variants, questions paraphrased   (treatment)
  marine_union.txt  base + aug + held-out, for building ONE vocabulary tree that
                    both conditions share, so the A/B is not confounded by vocab
  marine_split.json the held-out pairs with the narrative that answers them

  python augment.py [n_heldout] [n_variants] [seed]
"""
import sys, json, random, re
from parse_marine import parse, sentences

NHELD=int(sys.argv[1]) if len(sys.argv)>1 else 15
NVAR =int(sys.argv[2]) if len(sys.argv)>2 else 24
SEED =int(sys.argv[3]) if len(sys.argv)>3 else 0
CHUNK=4        # QA pairs per repetition of the narrative

# Why CHUNK exists: a question is only answerable by reading if the narrative
# holding the answer is still inside the context window. Block 3 runs 18 QA
# pairs -- ~450 tokens -- past its narrative, so the later ones are out of reach
# at any context this model can afford. Re-anchoring the narrative every CHUNK
# questions makes every training example extractive.

def lower1(s): return s[0].lower()+s[1:] if s else s

def qforms(q):
    """Paraphrases of one question. Wrappers apply to any question; the rest key
    off the wh-word. All of them reuse corpus words plus a handful of new ones."""
    inner=lower1(q.rstrip("?").strip())
    out=[q, f"Question: {q}", f"Do you know {inner}?", f"Can you say {inner}?",
         f"Tell me {inner}.", f"I want to know {inner}."]
    SUB=[("Who ","Which person "), ("Whose ","Which person's "), ("Where ","In which place "),
         ("When ","At what time "), ("Why ","For what reason "), ("How many ","What number of "),
         ("How old ","What age "), ("Which ","What ")]
    for pre,rep in SUB:
        if q.startswith(pre): out.append(rep+q[len(pre):]); break
    return out

def aforms(a): return [a, a, a, f"The answer is: {a}"]

if __name__=="__main__":
    rng=random.Random(SEED); blocks=parse()
    # ---- split: hold out extractive pairs, spread across blocks ----
    pool=[(bi,qi) for bi,b in enumerate(blocks) for qi,qa in enumerate(b["qa"]) if qa["cover"]>=0.6]
    rng.shuffle(pool)
    held=[]; per={}
    for bi,qi in pool:                      # at most 2 per block first, then fill
        if len(held)>=NHELD: break
        if per.get(bi,0) < 2: held.append((bi,qi)); per[bi]=per.get(bi,0)+1
    for bi,qi in pool:
        if len(held)>=NHELD: break
        if (bi,qi) not in held: held.append((bi,qi))
    heldset=set(held)

    # ---- control corpus: original text minus the held-out pairs ----
    base=[]
    for bi,b in enumerate(blocks):
        base += b["narr"]
        base += [f"{qa['q']} {qa['a']}" for qi,qa in enumerate(b["qa"]) if (bi,qi) not in heldset]
    open("marine_base.txt","w").write(" ".join(base)+"\n")

    # ---- treatment corpus: NVAR variants, blocks and questions reordered ----
    docs=[]
    for v in range(NVAR):
        order=list(range(len(blocks))); rng.shuffle(order); doc=[]
        for bi in order:
            b=blocks[bi]; doc += b["narr"]
            keep=[qi for qi in range(len(b["qa"])) if (bi,qi) not in heldset]
            rng.shuffle(keep)
            for c in range(0,len(keep),CHUNK):
                if c: doc += b["narr"]                       # re-anchor
                for qi in keep[c:c+CHUNK]:
                    qa=b["qa"][qi]
                    doc.append(f"{rng.choice(qforms(qa['q']))} {rng.choice(aforms(qa['a']))}")
        docs.append(" ".join(doc))
    open("marine_aug.txt","w").write("\n\n".join(docs)+"\n")

    # ---- vocabulary source: everything, so both conditions share one tree ----
    heldtxt=" ".join(f"{f} {qa['a']}" for bi,qi in held
                     for qa in [blocks[bi]["qa"][qi]] for f in qforms(qa["q"]))
    open("marine_union.txt","w").write(open("marine_aug.txt").read()+"\n\n"+
                                       " ".join(base)+"\n\n"+heldtxt+"\n")
    # ---- the split itself ----
    split={"n_heldout":NHELD,"n_variants":NVAR,"seed":SEED,"chunk":CHUNK,
           "held":[{"par":blocks[bi]["par"],"narr":" ".join(blocks[bi]["narr"]),
                    "q":blocks[bi]["qa"][qi]["q"],"a":blocks[bi]["qa"][qi]["a"],
                    "cover":blocks[bi]["qa"][qi]["cover"],
                    "forms":qforms(blocks[bi]["qa"][qi]["q"])} for bi,qi in held],
           "train_qa":[{"par":b["par"],"narr":" ".join(b["narr"]),"q":qa["q"],"a":qa["a"]}
                       for bi,b in enumerate(blocks) for qi,qa in enumerate(b["qa"])
                       if (bi,qi) not in heldset]}
    json.dump(split,open("marine_split.json","w"),indent=1)
    wc=lambda p: len(re.findall(r"\S+",open(p).read()))
    print(f"held out {len(held)} of 75 QA pairs across {len(per)} blocks; "
          f"{len(split['train_qa'])} pairs remain in training")
    for p in ["marine_base.txt","marine_aug.txt","marine_union.txt"]: print(f"  {p:20s} {wc(p):6d} words")
    print("\nheld-out questions:")
    for h in split["held"]: print(f"  [par {h['par']} cover {h['cover']:.2f}] {h['q']}  ->  {h['a']}")
