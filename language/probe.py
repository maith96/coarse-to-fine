"""Qualitative probe: questions the passage never asks, in words it does use.

The QA score in qa.py is recall of memorised strings. This asks what happens
just outside that set -- rephrased questions, and one question with no answer
in the passage at all. Every prompt is checked for out-of-vocabulary words
first, since an unknown word would enter as UNK and tell us nothing.

  python probe.py
"""
import numpy as np, re, warnings; warnings.filterwarnings("ignore")
from qa import gen, words, detok, CTX
import corpus as CO
z=np.load(CO.TREE)
w2i={str(w):i for i,w in enumerate(z['words'])}
tok=CO.C["tok"]

ASK=[
 "who drank black coffee?",
 "what did marcus clean with vinegar?",
 "who brought the coffee?",
 "where is the coastal research station?",
 "how many hours of monitoring data were lost?",
 "who did elena drive to the station with?",
 "what did margaret holt inspect?",
 "who repaired the seal?",                 # no answer in the passage
]
for q in ASK:
    ts=re.findall(tok,q.lower()); oov=[t for t in ts if t not in w2i]
    if oov: print(f"Q  {q}\n   skipped, out of vocabulary: {oov}\n"); continue
    p=np.array([w2i[t] for t in ts][-CTX:])
    print(f"Q  {q}\nA  {detok([str(words[t]) for t in gen(p)])}\n")
