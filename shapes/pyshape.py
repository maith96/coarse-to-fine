"""Same net, same 1200 steps, same two minutes -- Python instead of Shakespeare.

Python has something Shakespeare does not: a provably NON-LOCAL dependency.
Which bracket closes is fixed by an opener that may be arbitrarily far back, and
tracking it needs a stack. So this is the section-6 distance probe transposed to
a structure no bigram can represent.
"""
import re, sys, time, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
torch.set_num_threads(4)

TOK=re.compile(r'"{3}[\s\S]*?"{3}|\'{3}[\s\S]*?\'{3}|"[^"\n]*"|\'[^\'\n]*\''
               r'|\#[^\n]*|[A-Za-z_]\w*|\d+\.?\d*|\n|[ ]{4}|[^\s]')
def lex(t):
    out=[]
    for m in TOK.finditer(t):
        s=m.group(0)
        if s[0] in '"\'': out.append("<STR>")
        elif s[0]=='#':   out.append("<CMT>")
        elif s=='    ':   out.append("<IND>")
        else:             out.append(s)
    return out

toks=lex(open("py_corpus.txt",encoding="utf-8",errors="ignore").read())
import collections
cnt=collections.Counter(toks); V=8000
vocab=[w for w,_ in cnt.most_common(V-1)]
w2i={w:i+1 for i,w in enumerate(vocab)}; words=np.array(["<unk>"]+vocab)
ids=np.array([w2i.get(t,0) for t in toks],dtype=np.int64)
SP=int(0.9*len(ids)); tr,va=ids[:SP],ids[SP:]
print(f"tokens {len(ids):,}  vocab {V}  UNK {(ids==0).mean():.3f}")
print("top 20:", " ".join(repr(w) for w in vocab[:20]))

CTX=64
class LM(nn.Module):
    def __init__(s,n,dm=96,nl=3):
        super().__init__()
        s.emb=nn.Embedding(V,dm); s.pos=nn.Embedding(CTX,dm)
        s.ls=nn.ModuleList([nn.TransformerEncoderLayer(dm,4,4*dm,dropout=0.0,
            batch_first=True,norm_first=True,activation="gelu") for _ in range(nl)])
        s.ln=nn.LayerNorm(dm); s.hd=nn.Linear(dm,n)
    def forward(s,x):
        h=s.emb(x)+s.pos.weight[None,:x.shape[1]]
        m=nn.Transformer.generate_square_subsequent_mask(x.shape[1])
        for l in s.ls: h=l(h,src_mask=m,is_causal=True)
        return s.hd(s.ln(h))

def batch(a,bs,rng):
    i=rng.integers(0,len(a)-CTX-1,bs)
    return (torch.from_numpy(np.stack([a[k:k+CTX] for k in i])),
            torch.from_numpy(np.stack([a[k+1:k+CTX+1] for k in i])))

STEPS=1200; BS=16
torch.manual_seed(0); net=LM(V)
o=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=0.01)
sch=torch.optim.lr_scheduler.OneCycleLR(o,3e-3,total_steps=STEPS,pct_start=0.15)
rng=np.random.default_rng(5); t0=time.time()
for s in range(1,STEPS+1):
    x,y=batch(tr,BS,rng)
    loss=F.cross_entropy(net(x).reshape(-1,V),y.reshape(-1))
    o.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); o.step(); sch.step()
    if s%400==0: print(f"  step {s} loss {loss.item():.3f}  ({time.time()-t0:.0f}s)",flush=True)
net.eval(); print(f"trained in {time.time()-t0:.0f}s")
torch.save({'n':net.state_dict(),'w':words},"pyshape.pt")

@torch.no_grad()
def ce_full(a,bs=32):
    st=[k*CTX for k in range((len(a)-1)//CTX)]; tot=0.;n=0
    for i in range(0,len(st),bs):
        ch=st[i:i+bs]
        x=torch.from_numpy(np.stack([a[k:k+CTX] for k in ch]))
        y=torch.from_numpy(np.stack([a[k+1:k+CTX+1] for k in ch]))
        tot+=F.cross_entropy(net(x).reshape(-1,V),y.reshape(-1),reduction='sum').item(); n+=y.numel()
    return tot/n
print(f"val CE {ce_full(va):.4f}  -> perplexity {np.exp(ce_full(va)):.1f}")

def render(ts):
    s=""
    for t in ts:
        w=str(words[t])
        if w=="\n": s+="\n"
        elif w=="<IND>": s+="    "
        elif w=="<STR>": s+=" '...'"
        elif w=="<CMT>": s+=" # ..."
        elif w in "().,:[]{}": s+=w
        else: s+=" "+w
    return s
@torch.no_grad()
def gen(prompt,n,temp,seed):
    c=[w2i.get(t,0) for t in lex(prompt)]; g=torch.Generator().manual_seed(seed); out=[]
    for _ in range(n):
        p=torch.softmax(net(torch.tensor(c[-CTX:])[None])[0,-1],-1)
        if temp<=.01: k=int(p.argmax())
        else: k=int(torch.multinomial(torch.softmax(torch.log(p+1e-12)/temp,-1),1,generator=g))
        out.append(k); c.append(k)
    return render(out)
for lbl,t,s in [("t0.7 seed0",0.7,0),("t0.7 seed1",0.7,1),("t1.0 seed0",1.0,0)]:
    print(f"\n### SAMPLE {lbl}\n"+gen("def ",70,t,s))
