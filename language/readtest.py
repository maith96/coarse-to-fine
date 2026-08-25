"""Does the model take its answer from the context, or from its weights?

Exact match and F1 are too blunt for this: an answer can be wholly correct about
the cast and still miss on wording. This scores only the randomised values --
names, colours, times, quantities -- that appear in the gold answer.

For each evaluation item, with the narrative present and with it removed:
  hit    the answer carries the value this document's cast assigns
  wrong  it carries a different value from the same pool -- a value belonging to
         some other document, which is what memorisation looks like
  none   neither

If the model reads, hits collapse when the narrative is taken away. If it has
memorised the casts, the two conditions look alike.

  CORPUS=marine_rand python readtest.py
"""
import json, re, numpy as np, warnings; warnings.filterwarnings("ignore")
from qa import gen, words, detok, CTX
from evalqa import enc, answer
import os, corpus as CO, randomize as R

SP=json.load(open(CO.SPLIT))
POOLVALS={k:[str(v) for v in vs] for k,vs in R.POOL.items()}

def slots_in(text, castv, exclude=""):
    """(slot, correct value) for every randomised slot this answer states and the
    question does not. A value already present in the question can be copied from
    it, which says nothing about whether the narrative was read, so it is dropped."""
    out=[]
    for k in POOLVALS:
        v=castv.get(k); pat=rf"(?<![\w-]){re.escape(v or chr(0))}(?![\w-])"
        if v and re.search(pat,text) and not re.search(pat,exclude): out.append((k,v))
    return out

def tally(items, label):
    T={c:[0,0,0] for c in ("ctx","cold")}; detail=[]
    for it in items:
        sl=slots_in(it["a"], it["cast"], it["q"])
        if not sl: continue
        P={"ctx":f"{it['narr']} {it['q']}", "cold":it["q"]}
        row={"q":it["q"],"gold":it["a"],"slots":[k for k,_ in sl]}
        for c,p in P.items():
            pred=answer(p); out=" ".join(pred).lower()
            for k,v in sl:
                others=[o for o in POOLVALS[k] if o!=v]
                if re.search(rf"(?<![\w-]){re.escape(v.lower())}(?![\w-])", out): T[c][0]+=1
                elif any(re.search(rf"(?<![\w-]){re.escape(o.lower())}(?![\w-])", out) for o in others): T[c][1]+=1
                else: T[c][2]+=1
            row[c]=detok(pred)
        detail.append(row)
    n=sum(T["ctx"])
    print(f"\n{label}: {len(detail)} items, {n} values stated in the answer but not in the question")
    print(f"  {'':16s} {'hit':>6s} {'wrong-cast':>11s} {'absent':>7s}")
    for c in ("ctx","cold"):
        h,w,a=T[c]
        print(f"  {'with narrative' if c=='ctx' else 'no narrative':16s} {h/n:6.3f} {w/n:11.3f} {a/n:7.3f}")
    return T, detail

if __name__=="__main__":
    th,dh=tally(SP["train_qa"], "SEEN question types, cast never trained on")
    tl,dl=tally(SP["held"],     "HELD-OUT questions, cast never trained on")
    print("\nexamples (seen question types):")
    for r in dh[:5]:
        print(f"  Q     {r['q']}\n  gold  {r['gold']}\n  ctx   {r['ctx']}\n  cold  {r['cold']}\n")
    json.dump({"seen":{c:th[c] for c in th},"held":{c:tl[c] for c in tl},
               "detail_seen":dh,"detail_held":dl}, open(CO.out("readtest"+(f"_s{os.environ['SEED']}" if os.environ.get("SEED","0")!="0" else "")),"w"), indent=1)
