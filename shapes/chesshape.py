"""Same net, same 1200 steps -- chess move sequences instead of Shakespeare.

Chess has what Python's brackets only hinted at: legality depends on the ENTIRE
move history, because the history determines the board. No window of any fixed
size is sufficient. This is the sharpest available gap between looking right and
being right, and python-chess adjudicates it exactly.
"""
import numpy as np, torch, torch.nn as nn, time, collections, warnings, chess
warnings.filterwarnings("ignore"); import torch.nn.functional as F
torch.set_num_threads(4)

games=[l.split() for l in open("chess_corpus.txt")]
toks=[t for g in games for t in g]
cnt=collections.Counter(toks); V=4096
vocab=[w for w,_ in cnt.most_common(V-1)]
w2i={w:i+1 for i,w in enumerate(vocab)}; words=np.array(["<unk>"]+vocab)
ids=np.array([w2i.get(t,0) for t in toks],dtype=np.int64)
NG=len(games); SPG=int(0.9*NG)
cut=sum(len(g) for g in games[:SPG])
tr,va=ids[:cut],ids[cut:]
print(f"{NG} games, {len(ids):,} tokens, vocab {V} (distinct {len(cnt)}), UNK {(ids==0).mean():.3f}")
print(f"train {len(tr):,}  val {len(va):,}  held-out games {NG-SPG}")

CTX=64
class LM(nn.Module):
    def __init__(s,n,dm=96):
        super().__init__()
        s.emb=nn.Embedding(V,dm); s.pos=nn.Embedding(CTX,dm)
        s.ls=nn.ModuleList([nn.TransformerEncoderLayer(dm,4,4*dm,dropout=0.0,
            batch_first=True,norm_first=True,activation="gelu") for _ in range(3)])
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
    if s%400==0: print(f"  step {s} loss {loss.item():.3f} ({time.time()-t0:.0f}s)",flush=True)
net.eval(); print(f"trained in {time.time()-t0:.0f}s")

@torch.no_grad()
def ce(a,bs=32):
    st=[k*CTX for k in range((len(a)-1)//CTX)]; tot=0.;n=0
    for i in range(0,len(st),bs):
        ch=st[i:i+bs]
        x=torch.from_numpy(np.stack([a[k:k+CTX] for k in ch]))
        y=torch.from_numpy(np.stack([a[k+1:k+CTX+1] for k in ch]))
        tot+=F.cross_entropy(net(x).reshape(-1,V),y.reshape(-1),reduction='sum').item(); n+=y.numel()
    return tot/n
c=ce(va); print(f"val CE {c:.4f}  perplexity {np.exp(c):.1f}")

@torch.no_grad()
def top1(seq):
    c=[w2i.get(t,0) for t in seq][-CTX:] or [0]
    return str(words[int(net(torch.tensor(c)[None])[0,-1].argmax())])

# ---------- A. teacher-forced legality: real prefix, is the model's move legal? ----------
held=games[SPG:]
uni_top=vocab[0]                                  # position-blind baseline: always play the modal move
print("\nA. TEACHER-FORCED LEGALITY  (real game prefix of length k, model's top-1 move)")
print(f"{'ply k':>7} {'positions':>10} {'model legal':>12} {'unigram-top1':>13} {'random SAN':>11}")
rng2=np.random.default_rng(1)
for k in (1,2,4,8,16,32,64,100):
    ok=0;ub=0;rb=0;n=0
    for g in held:
        if len(g)<=k: continue
        b=chess.Board()
        try:
            for m in g[:k]: b.push_san(m)
        except Exception: continue
        n+=1
        for cand,acc in ((top1(g[:k]),'m'),(uni_top,'u'),(vocab[rng2.integers(0,300)],'r')):
            try: b.parse_san(cand); good=True
            except Exception: good=False
            if acc=='m': ok+=good
            elif acc=='u': ub+=good
            else: rb+=good
        if n>=150: break
    if n: print(f"{k:>7} {n:>10} {ok/n:>12.3f} {ub/n:>13.3f} {rb/n:>11.3f}")

# ---------- B. free rollout: how long before the first illegal move? ----------
@torch.no_grad()
def rollout(seed,temp=0.6,cap=160):
    g=torch.Generator().manual_seed(seed); b=chess.Board(); seq=[]
    for ply in range(cap):
        c=[w2i.get(t,0) for t in seq][-CTX:] or [0]
        p=torch.softmax(net(torch.tensor(c)[None])[0,-1],-1)
        p[len(words):]=0; p=p/p.sum()
        t=str(words[int(torch.multinomial(torch.softmax(torch.log(p+1e-12)/temp,-1),1,generator=g))])
        if t=="<eog>": return ply,"game ended"
        try: b.push_san(t)
        except Exception: return ply,t
        seq.append(t)
    return cap,"reached cap"
print("\nB. FREE ROLLOUT  (sample from the empty board, stop at first illegal move)")
L=[rollout(s)[0] for s in range(40)]
L=np.array(L)
print(f"  first illegal move at ply: median {np.median(L):.0f}  mean {L.mean():.1f}  max {L.max()}")
print(f"  reached ply 10: {(L>=10).mean():.0%}   ply 20: {(L>=20).mean():.0%}   ply 40: {(L>=40).mean():.0%}")
n,why=rollout(0)
print(f"\n  example: survived {n} plies, then played {why!r}")
