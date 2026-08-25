"""Elo estimate for the 58-second chess model.

Two regimes, because they answer different questions:

  UNASSISTED  the model's own token is played. It plays an illegal move at
              median ply 6, which under any real rules is an immediate forfeit.
  MASKED      the model's distribution is restricted to legal moves. This
              removes board-tracking entirely and measures only whatever chess
              judgment is left in the move distribution.

Opponents, weakest first, to bracket it.
"""
import numpy as np, torch, torch.nn as nn, collections, warnings, chess, chess.engine, shutil, random
warnings.filterwarnings("ignore"); import torch.nn.functional as F
torch.set_num_threads(2)
SF=shutil.which("stockfish") or "/usr/games/stockfish"

games=[l.split() for l in open("chess_corpus.txt")]
toks=[t for g in games for t in g]
cnt=collections.Counter(toks); V=4096
vocab=[w for w,_ in cnt.most_common(V-1)]
w2i={w:i+1 for i,w in enumerate(vocab)}; words=np.array(["<unk>"]+vocab)
ids=np.array([w2i.get(t,0) for t in toks],dtype=np.int64)
cut=sum(len(g) for g in games[:int(0.9*len(games))]); tr=ids[:cut]
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
torch.manual_seed(0); net=LM(V)
o=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=0.01)
sch=torch.optim.lr_scheduler.OneCycleLR(o,3e-3,total_steps=1200,pct_start=0.15)
rng=np.random.default_rng(5)
for s in range(1200):
    i=rng.integers(0,len(tr)-CTX-1,16)
    x=torch.from_numpy(np.stack([tr[k:k+CTX] for k in i]))
    y=torch.from_numpy(np.stack([tr[k+1:k+CTX+1] for k in i]))
    loss=F.cross_entropy(net(x).reshape(-1,V),y.reshape(-1))
    o.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); o.step(); sch.step()
net.eval(); print("model retrained (1200 steps)\n")

@torch.no_grad()
def model_move(board, hist, masked=True, temp=0.3, gen=None):
    c=[w2i.get(t,0) for t in (["<eog>"]+hist)][-CTX:]
    p=torch.softmax(net(torch.tensor(c)[None])[0,-1],-1).numpy()
    if not masked:
        return str(words[int(p.argmax())])
    legal={board.san(m):m for m in board.legal_moves}
    sc={s:p[w2i[s]] for s in legal if s in w2i}
    if not sc or max(sc.values())<=0: return board.san(random.choice(list(board.legal_moves)))
    return max(sc,key=sc.get)

def greedy_move(board):
    VAL={chess.PAWN:1,chess.KNIGHT:3,chess.BISHOP:3,chess.ROOK:5,chess.QUEEN:9,chess.KING:0}
    best=None;bv=-99
    for m in board.legal_moves:
        v=0
        if board.is_capture(m):
            pc=board.piece_at(m.to_square); v=VAL[pc.piece_type] if pc else 1
        board.push(m)
        if board.is_checkmate(): v+=100
        board.pop()
        if v>bv: bv=v; best=m
    return board.san(best)

def rand_move(board): return board.san(random.choice(list(board.legal_moves)))

def play(opponent, masked, n=40, cap=100, seed=0):
    """model plays both colours alternately; returns model score in [0,1]."""
    random.seed(seed); tot=0.0
    eng=None
    if isinstance(opponent,int):
        eng=chess.engine.SimpleEngine.popen_uci(SF)
        eng.configure({"UCI_LimitStrength":True,"UCI_Elo":opponent,"Threads":1,"Hash":16})
    adj=chess.engine.SimpleEngine.popen_uci(SF); adj.configure({"Threads":1,"Hash":16})
    for g in range(n):
        b=chess.Board(); hist=[]; model_white=(g%2==0); res=None
        for ply in range(cap):
            if b.is_game_over(): break
            mine=(b.turn==chess.WHITE)==model_white
            if mine:
                san=model_move(b,hist,masked=masked)
                try: b.push_san(san)
                except Exception:
                    res=0.0; break            # illegal -> forfeit
            else:
                if eng: san=b.san(eng.play(b,chess.engine.Limit(depth=6)).move)
                elif opponent=="random": san=rand_move(b)
                else: san=greedy_move(b)
                b.push_san(san)
            hist.append(san)
        if res is None:
            if b.is_game_over():
                r=b.result()
                res=1.0 if (r=="1-0")==model_white and r!="1/2-1/2" else 0.5 if r=="1/2-1/2" else 0.0
            else:                              # adjudicate by engine eval
                sc=adj.analyse(b,chess.engine.Limit(depth=8))["score"].white().score(mate_score=10000)
                sc=sc if model_white else -sc
                res=1.0 if sc>150 else 0.0 if sc<-150 else 0.5
        tot+=res
    adj.quit()
    if eng: eng.quit()
    return tot/n

def elo_diff(score):
    s=min(max(score,1e-3),1-1e-3)
    return -400*np.log10(1/s-1)


import json
def se(n): return (0.25/n)**0.5
R={}
print("UNASSISTED (own tokens; illegal move = forfeit)")
u=play("random",masked=False,n=30); R['unassisted_vs_random']=u
print(f"  vs random legal mover     score {u:.3f}")
print("\nMASKED (restricted to legal moves; judgment only), n=100")
for opp,lab,n in (("random","random legal mover",100),("greedy","1-ply material greedy",100)):
    s_=play(opp,masked=True,n=n); R[lab]=s_
    print(f"  vs {lab:24s} score {s_:.3f} +- {se(n):.3f}   Elo diff {elo_diff(s_):+.0f}")
s_=play(1320,masked=True,n=20); R['sf1320']=s_
print(f"  vs {'Stockfish 16 @ Elo 1320':24s} score {s_:.3f}")
json.dump(R,open("elo.json","w"),indent=1)
