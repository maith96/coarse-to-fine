"""Fit the finest rung on the WHOLE corpus and save a self-contained model.

chain.py holds out the last 10% to measure transfer; that split is what the
ladder is scored on. This script is the deliverable instead of the measurement:
it trains on every token so the saved model has seen the entire passage, and
packs the vocabulary tree into the checkpoint so qa.py needs nothing else.

  python fit.py <budget_seconds> [steps] [scratch|chain]
"""
import sys, os, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
from gatelm import LM, tgt, DEPTH, V, CTX, LEVELS
import corpus as CO
from chain import expand

z=np.load(CO.TREE); ALL=z['ids'].astype(np.int64); words=z['words']
L=DEPTH; NC=2**L; BS=CO.bs(L)
STEPS=int(sys.argv[2]) if len(sys.argv)>2 else 3000
INIT=sys.argv[3] if len(sys.argv)>3 else "chain"
SEED=int(os.environ.get("SEED","0"))        # replication: same data, different init
SFX ="" if SEED==0 else f"_s{SEED}"
MODEL=f"models/{CO.NAME}{SFX}_L{L}.pt"

def batch_all(bs,rng):
    i=rng.integers(0,len(ALL)-CTX-1,bs)
    x=np.stack([ALL[k:k+CTX] for k in i]); y=np.stack([ALL[k+1:k+CTX+1] for k in i])
    return torch.from_numpy(x), y

def fit_acc(net,nb=24,bs=32,seed=11):
    """next-token accuracy at leaf granularity over the corpus it was fit on."""
    rng=np.random.default_rng(seed); net.eval(); c=t=0; ce=0.
    with torch.no_grad():
        for _ in range(nb):
            x,y=batch_all(bs,rng); lg=net(x); yy=tgt(y,L)
            ce+=F.cross_entropy(lg.reshape(-1,NC),yy.reshape(-1)).item()
            c+=(lg.argmax(-1)==yy).sum().item(); t+=yy.numel()
    net.train(); return c/t, ce/nb

if __name__=="__main__":
    BUD=float(sys.argv[1]); t0=time.time(); ck=CO.ck(f"fit_L{L}{SFX}")
    torch.manual_seed(SEED); net=LM(NC)
    par=sum(p.numel() for p in net.parameters())
    if INIT=="chain" and not os.path.exists(ck):
        src=CO.ck(f"ch_anc_L{L}")
        if os.path.exists(src): net.load_state_dict(torch.load(src)['n']); print(f"warm start <- {src}")
        else: print("no chain ancestor; starting from scratch")
    o=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.OneCycleLR(o,3e-3,total_steps=STEPS,pct_start=0.15)
    s0=0; hist=[]
    if os.path.exists(ck):
        c=torch.load(ck); net.load_state_dict(c['n']); o.load_state_dict(c['o']); sch.load_state_dict(c['s']); s0=c['step']; hist=c['h']
    print(f"corpus {CO.NAME}  tokens {len(ALL)}  vocab {V}  leaves {NC}  params {par/1e3:.0f}k  "
          f"steps {STEPS} bs {BS} ctx {CTX}",flush=True)
    rng=np.random.default_rng(17)
    for _ in range(s0): rng.integers(0,len(ALL)-CTX-1,BS)
    for s in range(s0+1,STEPS+1):
        x,y=batch_all(BS,rng)
        loss=F.cross_entropy(net(x).reshape(-1,NC), tgt(y,L).reshape(-1))
        o.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); o.step(); sch.step()
        if s%250==0 or s==1: hist.append((s,round(loss.item(),4))); print(f"  {s:5d}/{STEPS} loss {loss.item():.4f}",flush=True)
        if time.time()-t0>BUD or s==STEPS:
            torch.save({'n':net.state_dict(),'o':o.state_dict(),'s':sch.state_dict(),'step':s,'h':hist},ck)
            if s<STEPS: print(f"paused {s}/{STEPS}",flush=True); sys.exit(0)
    acc,ce=fit_acc(net)
    os.makedirs("models",exist_ok=True)
    torch.save({'n':net.state_dict(),'corpus':CO.NAME,'V':V,'DEPTH':DEPTH,'CTX':CTX,'ncls':NC,
                'code':torch.from_numpy(z['code'].astype(np.int64)),
                'words':[str(w) for w in words],'steps':STEPS,'init':INIT,
                'fit_acc':acc,'fit_ce':ce,'hist':hist}, MODEL)
    print(f"fit next-token acc {acc:.4f}  CE {ce:.4f}  -> saved {MODEL} "
          f"({os.path.getsize(MODEL)/1e6:.1f} MB)",flush=True)
    json.dump({'corpus':CO.NAME,'level':L,'steps':STEPS,'init':INIT,'params':par,
               'fit_acc':acc,'fit_ce':ce,'hist':hist}, open(CO.out(f"fit{SFX}"),"w"))
