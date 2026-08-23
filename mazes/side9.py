"""Same held-out inputs, every net's output side by side."""
import sys, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
torch.set_num_threads(4)
from lad9 import PolNet, index, SPLIT, M, tok, walls, d, G

ACT=["UP","DOWN","LEFT","RIGHT"]; DELTA=[(-1,0),(1,0),(0,-1),(0,1)]
NETS=[("anc  (chain 600+2400)","l9_anc_d24"),
      ("ctrl (flat 600)",      "l9_ctrl_d24"),
      ("ctrl (flat 3000)",     "sw9_ctrl_d24_n3000_s0")]

def load(ck):
    n=PolNet(); n.load_state_dict(torch.load(f"ckpt/{ck}.pt")['n']); n.eval(); return n
nets=[(nm,load(ck)) for nm,ck in NETS]

def legal(m,i,j,a):
    di,dj=DELTA[a]; ni,nj=i+di,j+dj
    return 0<=ni<G and 0<=nj<G and not walls[m,ni,nj]

def probs(net,m,i,j):
    x=tok[m].copy(); x[i,j]=3
    with torch.no_grad():
        return F.softmax(net(torch.from_numpy(x.reshape(1,-1)),torch.tensor([i*G+j])),-1)[0].numpy()

DIST=int(sys.argv[1]) if len(sys.argv)>1 else 24
N=int(sys.argv[2]) if len(sys.argv)>2 else 5
mi,ii,jj=index(DIST,SPLIT,M)
rng=np.random.default_rng(7)

print(f"### held-out mazes at d={DIST}   A=agent  G=goal  #=wall\n")
for k in rng.choice(len(mi),N,replace=False):
    m,i,j=mi[k],ii[k],jj[k]; good=d[m] # noqa
    opt_acts=[a for a in range(4) if legal(m,i,j,a) and d[m,i+DELTA[a][0],j+DELTA[a][1]]==d[m,i,j]-1]
    grid=[]
    for r in range(G):
        grid.append("".join("#" if walls[m,r,c] else "G" if d[m,r,c]==0
                            else "A" if (r,c)==(i,j) else "." for c in range(G)))
    print(f"maze {m}  agent=({i},{j})  dist={d[m,i,j]}  optimal: {'/'.join(ACT[a] for a in opt_acts)}")
    body=[f"  {g}" for g in grid]
    rows=[f"{'net':22s} {'UP':>6} {'DOWN':>6} {'LEFT':>6} {'RIGHT':>6}   pick"]
    for nm,net in nets:
        p=probs(net,m,i,j); b=int(p.argmax())
        mark="CORRECT" if b in opt_acts else ("WRONG->WALL" if not legal(m,i,j,b) else "WRONG")
        rows.append(f"{nm:22s} "+" ".join(f"{p[a]:6.2f}" for a in range(4))+f"   {ACT[b]:5s} {mark}")
    for n_ in range(max(len(body),len(rows))):
        L=body[n_] if n_<len(body) else " "*11
        R=rows[n_] if n_<len(rows) else ""
        print(f"{L:14s}{R}")
    print()

# aggregate over many cells
n=min(600,len(mi)); pick=rng.choice(len(mi),n,replace=False)
print(f"### aggregate over {n} held-out cells at d={DIST}")
print(f"{'net':22s} {'correct':>8} {'picks a wall':>13} {'mean top-prob':>14}")
picks={}
for nm,net in nets:
    c=w=0.; tp=[]
    pk=[]
    for k in pick:
        m,i,j=mi[k],ii[k],jj[k]; p=probs(net,m,i,j); b=int(p.argmax()); pk.append(b); tp.append(p[b])
        oa=[a for a in range(4) if legal(m,i,j,a) and d[m,i+DELTA[a][0],j+DELTA[a][1]]==d[m,i,j]-1]
        c+= b in oa; w+= not legal(m,i,j,b)
    picks[nm]=np.array(pk)
    print(f"{nm:22s} {c/n:8.3f} {w/n:13.3f} {np.mean(tp):14.3f}")
ks=list(picks)
print(f"\n agreement  anc vs ctrl600 {np.mean(picks[ks[0]]==picks[ks[1]]):.3f}"
      f"   anc vs ctrl3000 {np.mean(picks[ks[0]]==picks[ks[2]]):.3f}"
      f"   ctrl600 vs ctrl3000 {np.mean(picks[ks[1]]==picks[ks[2]]):.3f}")
for nm in ks:
    b=np.bincount(picks[nm],minlength=4)/n
    print(f" {nm:22s} action mix  "+"  ".join(f"{ACT[a]} {b[a]:.2f}" for a in range(4)))
