"""Ask the fitted model a question.

  python ask.py "who is marcus chen?"
  python ask.py                      # then type questions, one per line

Words outside the 347-token vocabulary are shown, since they enter as <unk> and
the answer is not about them. CORPUS picks the model (default marine).
"""
import sys, re, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
from qa import gen, words, detok, CTX
import corpus as CO

z=np.load(CO.TREE); w2i={str(w):i for i,w in enumerate(z['words'])}

def ask(q):
    ts=re.findall(CO.C["tok"], q.lower())
    ids=np.array([w2i.get(t,0) for t in ts][-CTX:])
    oov=[t for t,i in zip(ts,ids) if i==0]
    return detok([str(words[t]) for t in gen(ids)]), oov

if __name__=="__main__":
    qs=[" ".join(sys.argv[1:])] if len(sys.argv)>1 else (l.strip() for l in sys.stdin)
    for q in qs:
        if not q: continue
        a,oov=ask(q)
        print(f"Q: {q}\nA: {a}" + (f"\n   (not in vocabulary: {', '.join(oov)})" if oov else "") + "\n")
