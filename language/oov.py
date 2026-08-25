"""What the model does with nouns that are not in its corpus.

The marine vocabulary is closed -- 347 types, UNK rate 0.000 -- so <unk> is a
token the model owns an embedding for and has never once seen in training. A
novel noun lands there. Each probe is paired with the in-vocabulary question it
was derived from, so the question is not "does it know dolphins" but "does
swapping the noun for an unknown one change the answer at all".

  python oov.py
"""
import re, json, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
import torch.nn.functional as F
from qa import net, leaf2w, words, detok, CTX, ABBR
import corpus as CO

z=np.load(CO.TREE); w2i={str(w):i for i,w in enumerate(z['words'])}
TOK=CO.C["tok"]

PAIRS=[
 ("who is the 34-year-old dolphin?",            "who is the 34-year-old marine biologist?"),
 ("who owns the blue electric kayak?",          "who owns the blue electric pickup truck?"),
 ("who photographed the penguin?",              "who photographed the harbor seal?"),
 ("what did marcus use to clean the propeller?","what did marcus use to clean the terminals?"),
 ("who is dr. rodriguez?",                      "who is dr. samir patel?"),
 ("where is the lighthouse located?",           "where is the research station located?"),
 ("who accompanied priya to the station?",      "who accompanied elena to the station?"),
 ("what did the earthquake damage?",            "what was damaged by the storm?"),
]
SOLO=[  # no in-vocabulary twin: every content word is new
 "who fed the dolphin at the aquarium?",
 "what did the astronaut repair?",
]

def enc(q):
    ts=re.findall(TOK,q.lower())
    return np.array([w2i.get(t,0) for t in ts]), ts

def gen_p(prompt, maxn=48):
    """greedy decode, also returning the probability taken at each step."""
    seq=list(prompt); got=[]; ps=[]
    for _ in range(maxn):
        x=torch.from_numpy(np.array(seq[-CTX:])[None])
        with torch.no_grad(): p=F.softmax(net(x)[0,-1],-1)
        leaf=int(p.argmax()); w=leaf2w.get(leaf)
        if w is None: break
        ps.append(float(p[leaf])); seq.append(w); got.append(w)
        if words[w]=="." and (len(got)<2 or words[got[-2]] not in ABBR): break
    return got, ps

def show(q, tag):
    ids_,ts=enc(q); oov=[t for t,i in zip(ts,ids_) if i==0]
    got,ps=gen_p(ids_)
    ans=detok([str(words[t]) for t in got])
    print(f"{tag} {q}")
    print(f"     seen as: {detok([str(words[i]) for i in ids_])}"
          + (f"   [new: {', '.join(oov)}]" if oov else "   [all in vocabulary]"))
    print(f"     -> {ans}")
    print(f"        mean p {np.mean(ps):.3f}  min p {np.min(ps):.3f}")
    return ans

if __name__=="__main__":
    rows=[]; same=0
    for novel, ctrl in PAIRS:
        a=show(novel,"NEW "); b=show(ctrl,"twin"); 
        eq = a==b; same+=eq
        print(f"     answers identical: {'yes' if eq else 'NO'}\n")
        rows.append({"novel":novel,"ctrl":ctrl,"a_novel":a,"a_ctrl":b,"same":eq})
    for q in SOLO:
        a=show(q,"NEW "); print(); rows.append({"novel":q,"ctrl":None,"a_novel":a,"same":None})
    print(f"{same}/{len(PAIRS)} novel-noun questions returned exactly the twin's answer")
    json.dump(rows, open(CO.out("oov"),"w"), indent=1)
