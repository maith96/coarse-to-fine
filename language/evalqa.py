"""Score a model on QA pairs it was never trained on.

Held-out pairs were removed from training but the narrative answering them was
not, so a model that reads can answer and a model that memorised strings cannot.
Three prompt formats per question:
  ctx   narrative + the question as originally written
  para  narrative + a paraphrase of it ("do you know ...?")
  cold  the question alone, no narrative

Exact match is the strict metric; token F1 gives partial credit so a model that
is finding the right region of the passage is visible even when it is wrong.

  CORPUS=marine_aug python evalqa.py
"""
import sys, re, json, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
from qa import gen, words, detok, CTX, ABBR
import corpus as CO

z=np.load(CO.TREE); w2i={str(w):i for i,w in enumerate(z['words'])}
TOK=CO.C["tok"]; SP=json.load(open("marine_split.json"))

def enc(s): return np.array([w2i.get(t,0) for t in re.findall(TOK,s.lower())])
def toks(s): return re.findall(TOK,s.lower())

def f1(pred,gold):
    if not pred or not gold: return 0.0
    common=0; g=list(gold)
    for p in pred:
        if p in g: g.remove(p); common+=1
    if not common: return 0.0
    pr=common/len(pred); rc=common/len(gold)
    return 2*pr*rc/(pr+rc)

def answer(prompt):
    got=gen(enc(prompt)[-CTX:]); return [str(words[t]) for t in got]

def score(items, para_form=2):
    out={k:[0,0.] for k in ("ctx","para","cold")}; rows=[]
    for it in items:
        gold=toks(it["a"]); r={"q":it["q"],"gold":it["a"]}
        forms=it.get("forms")
        P={"ctx": f"{it['narr']} {it['q']}",
           "para": f"{it['narr']} {forms[para_form]}" if forms else f"{it['narr']} {it['q']}",
           "cold": it["q"]}
        for k,p in P.items():
            pred=answer(p)
            out[k][0]+= pred==gold; out[k][1]+= f1(pred,gold)
            r[k]=detok(pred)
        rows.append(r)
    n=len(items)
    return {k:{"exact":v[0]/n,"f1":v[1]/n} for k,v in out.items()}, rows

if __name__=="__main__":
    held=SP["held"]; seen=SP["train_qa"][:len(held)]
    hs,hr=score(held); ss,_=score(seen)
    print(f"corpus {CO.NAME}   {len(held)} held-out pairs, {len(seen)} seen pairs for reference\n")
    print(f"{'prompt':6s} {'HELD-OUT exact':>15s} {'f1':>7s}   {'SEEN exact':>11s} {'f1':>7s}")
    for k in ("ctx","para","cold"):
        print(f"{k:6s} {hs[k]['exact']:15.3f} {hs[k]['f1']:7.3f}   {ss[k]['exact']:11.3f} {ss[k]['f1']:7.3f}")
    print()
    for r in hr[:6]:
        print("Q    ",r["q"]); print("gold ",r["gold"])
        for k in ("ctx","para","cold"): print(f"{k:5s}",r[k])
        print()
    json.dump({"corpus":CO.NAME,"held":hs,"seen":ss,"rows":hr},open(CO.out("evalqa"),"w"),indent=1)
