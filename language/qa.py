"""Ask the fitted model every question the passage asks.

Two probes, because they measure different things:
  ctx  -- the question is fed with the narrative that precedes it, as in
          training. The answer may be copied from context.
  cold -- the question alone, at position 0. Nothing to copy from, so a
          correct answer has to come out of the weights.

  python qa.py [n_shown]
"""
import sys, re, json, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
from gatelm import LM
import corpus as CO

import glob, os
# MODEL pins a specific checkpoint; without it, the corpus's seed-0 model
M=torch.load(os.environ.get("MODEL") or sorted(glob.glob(f"models/{CO.NAME}_L*.pt"))[-1])
DEPTH=M['DEPTH']; CTX=M['CTX']; NC=M['ncls']; code=M['code'].numpy(); words=M['words']
net=LM(NC); net.load_state_dict(M['n']); net.eval()
leaf2w={int(c):i for i,c in enumerate(code)}      # exact at the finest level
z=np.load(CO.TREE); ids=z['ids'].astype(np.int64)
ABBR={"dr","mr","mrs","st"}                        # a '.' after these is not a sentence end

def detok(ts):
    s=" ".join(ts)
    s=re.sub(r" ([.,!?;:'—])", r"\1", s); s=re.sub(r"([$—]) ", r"\1", s)
    return s.replace(" - ","-").strip()

def qa_pairs():
    """(question tokens, answer tokens) for every '?' in the corpus."""
    out=[]; qm=[i for i,t in enumerate(ids) if words[t]=="?"]
    for j,q in enumerate(qm):
        st=qm[j-1]+1 if j else 0
        ques=ids[st:q+1]
        a=[]
        for k in range(q+1,len(ids)):
            a.append(ids[k])
            if words[ids[k]]=="." and (len(a)<2 or words[a[-2]] not in ABBR): break
        out.append((np.array(ques[-CTX:]), np.array(a)))
    return out

def gen(prompt, maxn=48):
    seq=list(prompt); got=[]
    for _ in range(maxn):
        x=torch.from_numpy(np.array(seq[-CTX:])[None])
        with torch.no_grad(): leaf=net(x)[0,-1].argmax().item()
        w=leaf2w.get(leaf)
        if w is None: break
        seq.append(w); got.append(w)
        if words[w]=="." and (len(got)<2 or words[got[-2]] not in ABBR): break
    return got

if __name__=="__main__":
    show=int(sys.argv[1]) if len(sys.argv)>1 else 8
    pairs=qa_pairs(); qm=[i for i,t in enumerate(ids) if words[t]=="?"]
    score={"ctx":[0,0,0],"cold":[0,0,0]}           # exact, token-hits, token-total
    rows=[]
    for (ques,ans),qi in zip(pairs,qm):
        gold=[str(words[t]) for t in ans]
        got={"ctx":gen(ids[max(0,qi+1-CTX):qi+1]), "cold":gen(ques)}
        row={"q":detok([str(words[t]) for t in ques]),"gold":detok(gold)}
        for k,g in got.items():
            pred=[str(words[t]) for t in g]
            score[k][0]+= pred==gold
            score[k][1]+= sum(a==b for a,b in zip(pred,gold)); score[k][2]+=len(gold)
            row[k]=detok(pred)
        rows.append(row)
    n=len(rows)
    for r in rows[:show]:
        print("Q   ", r["q"]); print("gold", r["gold"])
        for k in ("ctx","cold"):
            print(f"{k:4s} {r[k]}" + ("" if r[k]==r["gold"] else "   <-- differs"))
        print()
    print(f"{n} questions in the passage")
    for k in ("ctx","cold"):
        e,h,t=score[k]
        print(f"  {k:4s}  exact answers {e}/{n} = {e/n:.3f}   token acc {h/t:.3f}")
    json.dump({"n":n,"exact":{k:score[k][0]/n for k in score},
               "token_acc":{k:score[k][1]/score[k][2] for k in score},"rows":rows},
              open(CO.out("qa"),"w"),indent=1)
