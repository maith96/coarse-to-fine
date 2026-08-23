"""Run a trained level-13 net as a word-level LM.

Level 13 is injective on this vocabulary -- 8000 words into 8000 distinct
leaves -- so the cluster head is a word head and the net samples plain text.

  python predict.py                      # best L13 ckpt by recorded val CE
  python predict.py ckpt/ch_anc_L13.pt   # a specific one
  python predict.py "romeo:\nbut soft,"  # own prompt (anything not *.pt)
"""
import sys, os, re, json, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
from gatelm import LM, tgt, batch, va_ids, CTX, DEPTH, V

L=13; NC=2**L
z=np.load("vocabtree.npz"); code=z['code'].astype(np.int64); words=z['words']
leaf2w={}
for w in range(V): leaf2w.setdefault(int(code[w]), w)
w2i={str(w):i for i,w in enumerate(words)}

def best_ckpt():
    """lowest recorded L13 val CE across the sweep and the 300-step chain."""
    cand={}
    if os.path.exists("sweep.json"):
        for k,v in json.load(open("sweep.json")).items():
            c,n,s=k.split("_"); cand[f"ckpt/sw_{c}_n{n}_s{s}.pt"]=v
    if os.path.exists("chain.json"):
        for k,v in json.load(open("chain.json")).items():
            if k.endswith("L13"): cand[f"ckpt/ch_{k}.pt"]=v['ce']
    cand={k:v for k,v in cand.items() if os.path.exists(k)}
    if not cand: sys.exit("no L13 checkpoint found -- run chain.py / sweep.py first")
    return min(cand.items(), key=lambda kv: kv[1])

def load(ck):
    net=LM(NC); net.load_state_dict(torch.load(ck)['n']); net.eval(); return net

def encode(s): return [w2i.get(t,0) for t in re.findall(r"[a-z']+|[.,!?;:\n]", s.lower())]
def show(t): return "\\n" if t=="\n" else t
def word(leaf): w=leaf2w.get(int(leaf)); return str(words[w]) if w is not None else None

@torch.no_grad()
def nextdist(net, ctx):
    return F.softmax(net(torch.tensor(ctx[-CTX:],dtype=torch.long)[None])[0,-1],-1)

def topk(net, prompt, k=8):
    p=nextdist(net, encode(prompt)); v,i=p.topk(k)
    return [f"{show(word(li)) or '<empty>'} {pv:.3f}" for pv,li in zip(v.tolist(),i.tolist())]

@torch.no_grad()
def gen(net, prompt, n=40, temp=0.8, seed=0):
    g=torch.Generator().manual_seed(seed); ctx=encode(prompt); out=""
    for _ in range(n):
        p=nextdist(net,ctx)
        li=int(torch.multinomial(torch.softmax(torch.log(p+1e-12)/temp,-1),1,generator=g))
        w=leaf2w.get(li)
        if w is None: continue
        t=str(words[w]); ctx.append(w)
        out += t if t in ".,!?;:\n" else " "+t
    return out

@torch.no_grad()
def teacher_forced(net, seed=11, tail=24):
    """where the loss actually lives: structure is cheap, content words are not."""
    i=int(np.random.default_rng(seed).integers(0,len(va_ids)-CTX-1))
    ctx=va_ids[i:i+CTX+1]
    p=F.softmax(net(torch.tensor(ctx[:-1],dtype=torch.long)[None])[0],-1)
    yy=tgt(ctx[1:],L); nll=-torch.log(p[torch.arange(CTX),yy]+1e-12)
    print(f"{'context':>14} {'TRUE next':>12} {'top-1':>12} {'p(true)':>8} {'nll':>6}")
    for k in range(CTX-tail,CTX):
        print(f"{show(str(words[ctx[k]])):>14} {show(str(words[ctx[k+1]])):>12} "
              f"{show(word(p[k].argmax())) or '-':>12} {p[k,yy[k]].item():8.3f} {nll[k].item():6.2f}")
    print(f"passage CE {nll.mean():.4f} -> ppl {np.exp(nll.mean().item()):.1f}")

@torch.no_grad()
def val_ce(net, nb=6, bs=32, seed=7):
    rng=np.random.default_rng(seed); ce=0.
    for _ in range(nb):
        x,y=batch(va_ids,bs,rng)
        ce+=F.cross_entropy(net(x).reshape(-1,NC), tgt(y,L).reshape(-1)).item()
    return ce/nb

PROMPTS=["first citizen:\nbefore we proceed any further, hear me",
         "romeo:\nbut soft, what light through yonder",
         "king richard:\na horse, a horse, my kingdom for a",
         "to be, or not to"]

if __name__=="__main__":
    a=[v for v in sys.argv[1:]]
    ck=next((v for v in a if v.endswith(".pt")), None)
    pr=[v for v in a if not v.endswith(".pt")]
    if ck is None: ck,rec=best_ckpt(); print(f"best by recorded val CE: {ck} ({rec:.4f})")
    net=load(ck)
    print(f"{ck}  {sum(p.numel() for p in net.parameters())/1e6:.2f}M params  "
          f"leaves {len(leaf2w)}/{NC} for {V} words")
    prompts=[p.replace("\\n","\n") for p in pr] or PROMPTS

    print("\n== next-word distribution ==")
    for p in prompts:
        print(f"\n...{p[-58:]!r}\n  " + " | ".join(topk(net,p)))
    print("\n== continuation (temp 0.8) ==")
    for p in prompts:
        print(f"\n{p!r}\n  ->" + gen(net,p).replace("\n","\n     "))
    if not pr:
        print("\n== held-out passage, teacher-forced ==")
        teacher_forced(net)
        ce=val_ce(net); print(f"\nval CE {ce:.4f} -> ppl {np.exp(ce):.1f}  (uniform = {V})")
