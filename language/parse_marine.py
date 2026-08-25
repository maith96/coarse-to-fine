"""Split marine.txt into blocks of (narrative sentences, the QA run after them).

A sentence ending in '?' is a question and the sentence after it is its answer;
anything else is narrative. Abbreviations ('Dr.') are not sentence boundaries.
Blocks matter because a question is only answerable from the narrative that
precedes it -- paragraph 1 introduces more narrative halfway through its QA run,
so permuting questions across that boundary would make some unanswerable.
"""
import re, json

ABBR=re.compile(r"\b(?:Dr|Mr|Mrs|Ms|St)\.$")
STOP=set("the a an of to in for and or is was were are be been at by on it its "
         "that which who what when where why how did do does had have has with "
         "from as they them their he she her his would will can could".split())

def sentences(par):
    out=[]; buf=""
    for piece in re.split(r"(?<=[.?])\s+", par.strip()):
        buf = (buf+" "+piece).strip() if buf else piece
        if ABBR.search(buf): continue          # 'Dr.' -- keep accumulating
        out.append(buf); buf=""
    if buf: out.append(buf)
    return out

def content(s):
    return {w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in STOP}

def parse(path="marine.txt"):
    """-> [ {par, narr:[sent], qa:[{q,a,cover}]} ], one entry per block."""
    blocks=[]
    for pi,par in enumerate(p for p in open(path).read().split("\n\n") if p.strip()):
        ss=sentences(par); cur=None; i=0
        while i < len(ss):
            if ss[i].endswith("?") and i+1 < len(ss):
                cur["qa"].append({"q":ss[i],"a":ss[i+1]}); i+=2
            else:
                if cur is None or cur["qa"]:                  # narrative starts a new block
                    cur={"par":pi,"narr":[],"qa":[]}; blocks.append(cur)
                cur["narr"].append(ss[i]); i+=1
    for b in blocks:
        nc=content(" ".join(b["narr"]))
        for qa in b["qa"]:
            ac=content(qa["a"])-content(qa["q"])              # what the answer adds
            qa["cover"]=round(len(ac&nc)/max(1,len(ac)),3)    # ...and how much of it the narrative holds
    return blocks

if __name__=="__main__":
    bs=parse(); nq=sum(len(b["qa"]) for b in bs)
    print(f"{len(bs)} blocks, {nq} QA pairs, {sum(len(b['narr']) for b in bs)} narrative sentences")
    for b in bs:
        cv=[qa["cover"] for qa in b["qa"]]
        print(f"  par {b['par']}: {len(b['narr'])} narr, {len(b['qa']):2d} qa, "
              f"answer-in-narrative coverage mean {sum(cv)/len(cv):.2f} "
              f"(>=0.6: {sum(c>=0.6 for c in cv)}/{len(cv)})")
    ex=[qa for b in bs for qa in b["qa"] if qa["cover"]>=0.6]
    lo=sorted((qa for b in bs for qa in b["qa"]), key=lambda x:x["cover"])[:4]
    print(f"\n{len(ex)}/{nq} pairs are extractive (coverage >= 0.6). Least extractive:")
    for qa in lo: print(f"  {qa['cover']:.2f}  {qa['q'][:70]} -> {qa['a'][:60]}")
