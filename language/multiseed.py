"""Multi-seed replication of the language ladder, with a corrected head expansion.

Three changes over chain.py / sweep.py:

  * the whole chain is rebuilt per seed. sweep.py reseeds only the final rung's
    batch order -- for cond="anc" the net is overwritten by ch_anc_L12, so the
    ancestor was byte-identical across "seeds" while the control got a fresh
    init as well. That comparison is asymmetric; this one is not.

  * expand_prior() adds log p(child|parent) to the inherited head bias. The
    uniform expansion in chain.py omits it, which costs 0.65-1.0 nats of
    zero-shot CE at every handoff. Both arms are run so the difference is
    measured rather than assumed.

  * val CE is also computed over the whole validation split at a fixed stride,
    not 6 sampled batches (~12k tokens). Same protocol for every arm, and no
    eval-sampling noise on top of the seed noise we are trying to measure.

  python multiseed.py <seconds-budget> [nseeds]     # resumable, run until ALLDONE
"""
import sys, os, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
from gatelm import LM, batch, tgt, tr_ids, va_ids, CTX, DEPTH

z=np.load("vocabtree.npz"); code=z['code'].astype(np.int64)
LV=[1,4,8,12]; TOP=13
BS={1:64,4:64,8:64,12:24,13:16}
STEPS=300
BUDGETS=[300,600,1200]

def expand(state, Lold, Lnew):
    """chain.py's expansion: parent row copied to every child, uniform within."""
    s=dict(state); par=torch.arange(2**Lnew)>>(Lnew-Lold)
    s['hd.weight']=state['hd.weight'][par].clone()
    s['hd.bias']=state['hd.bias'][par].clone()
    return s

def child_logprior(Lold, Lnew, alpha=1.0):
    """log p(child|parent) from the training split, additively smoothed."""
    d=Lnew-Lold
    cf=np.bincount(code[tr_ids]>>(DEPTH-Lnew), minlength=2**Lnew).astype(np.float64)+alpha
    par=np.arange(2**Lnew)>>d
    cp=np.bincount(par, weights=cf, minlength=2**Lold)
    return torch.tensor(np.log(cf/cp[par]), dtype=torch.float32)

def expand_prior(state, Lold, Lnew):
    """as above, plus the within-cluster log marginal the uniform version drops.

    Softmax is global, but every parent has the same number of children, so
    adding a term that sums to 1 within each parent leaves the coarse marginal
    exactly as the parent predicted it and fills in the split below."""
    s=expand(state, Lold, Lnew)
    s['hd.bias']=s['hd.bias']+child_logprior(Lold,Lnew)
    return s

EXP={'u':expand, 'p':expand_prior}

@torch.no_grad()
def ce_full(net, L, bs=32):
    """deterministic sweep of the whole val split at stride CTX."""
    net.eval(); starts=[k*CTX for k in range((len(va_ids)-1)//CTX)]
    tot=0.; n=0
    for i in range(0,len(starts),bs):
        ch=starts[i:i+bs]
        x=np.stack([va_ids[k:k+CTX] for k in ch]); y=np.stack([va_ids[k+1:k+CTX+1] for k in ch])
        lg=net(torch.from_numpy(x))
        tot+=F.cross_entropy(lg.reshape(-1,2**L), tgt(y,L).reshape(-1), reduction='sum').item()
        n+=y.size
    net.train(); return tot/n

@torch.no_grad()
def ce_repo(net, L, nb=6, bs=32, seed=7):
    """chain.py's sampled protocol, kept so numbers stay comparable."""
    rng=np.random.default_rng(seed); net.eval(); ce=0.; c=t=0
    for _ in range(nb):
        x,y=batch(va_ids,bs,rng); lg=net(x); yy=tgt(y,L)
        ce+=F.cross_entropy(lg.reshape(-1,2**L),yy.reshape(-1)).item()
        c+=(lg.argmax(-1)==yy).sum().item(); t+=yy.numel()
    net.train(); return ce/nb, c/t

def train(L, state, key, steps, seed, BUD, t0):
    ncls=2**L; bs=BS[L]; ck=f"ckpt/ms_{key}.pt"
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    torch.manual_seed(seed); net=LM(ncls)
    if state is not None and not os.path.exists(ck): net.load_state_dict(state)
    o=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.OneCycleLR(o,3e-3,total_steps=steps,pct_start=0.15)
    s0=0; zs=None
    if os.path.exists(ck):
        c=torch.load(ck); net.load_state_dict(c['n']); o.load_state_dict(c['o'])
        sch.load_state_dict(c['s']); s0=c['step']; zs=c['zs']
    else:
        zs=ce_full(net,L)
    rng=np.random.default_rng(5+seed)
    for _ in range(s0): rng.integers(0,len(tr_ids)-CTX-1,bs)
    for s in range(s0+1,steps+1):
        x,y=batch(tr_ids,bs,rng)
        loss=F.cross_entropy(net(x).reshape(-1,ncls), tgt(y,L).reshape(-1))
        o.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); o.step(); sch.step()
        if time.time()-t0>BUD or s==steps:
            torch.save({'n':net.state_dict(),'o':o.state_dict(),'s':sch.state_dict(),
                        'step':s,'zs':zs},ck)
            if s<steps: print(f"  {key} {s}/{steps}",flush=True); return None
    cr,acc=ce_repo(net,L)
    return {'ce_full':ce_full(net,L),'ce_repo':cr,'acc':acc,'zs_full':zs,
            'state':net.state_dict()}

if __name__=="__main__":
    BUD=float(sys.argv[1]); NS=int(sys.argv[2]) if len(sys.argv)>2 else 5
    t0=time.time()
    RES="../results/multiseed.json"
    res=json.load(open(RES)) if os.path.exists(RES) else {}
    def done(k,r):
        res[k]={x:(float(v) if x!='state' else None) for x,v in r.items() if x!='state'}
        json.dump(res,open(RES,"w"),indent=1)
        print(f"[{time.time()-t0:5.0f}s] {k:22s} zs {r['zs_full']:6.3f} -> "
              f"CE {r['ce_full']:.4f} (repo protocol {r['ce_repo']:.4f})",flush=True)

    for seed in range(NS):
        # ---- ancestor chains: uniform and prior-corrected, shared L1 root ----
        def rung(arm,L):                            # L1 has no handoff: one shared root
            return f"s{seed}/anc{'u' if L==1 else arm}_L{L}"
        for arm in ['u','p']:
            prev=None
            for L in LV:
                key=rung(arm,L)
                if key in res: prev=L; continue
                if time.time()-t0>BUD: print("PAUSE",flush=True); sys.exit(0)
                st=None if prev is None else EXP[arm](
                    torch.load(f"ckpt/ms_{rung(arm,prev)}.pt")['n'], prev, L)
                r=train(L,st,key,STEPS,seed,BUD,t0)
                if r is None: sys.exit(0)
                done(key,r); prev=L
        # ---- final rung at level 13: three arms x three budgets ----
        for n in BUDGETS:
            for arm in ['ancu','ancp','ctrl']:
                key=f"s{seed}/{arm}_n{n}"
                if key in res: continue
                if time.time()-t0>BUD: print("PAUSE",flush=True); sys.exit(0)
                st=None
                if arm!='ctrl':
                    st=EXP[arm[-1]](torch.load(f"ckpt/ms_{rung(arm[-1],12)}.pt")['n'],12,TOP)
                r=train(TOP,st,key,n,seed,BUD,t0)
                if r is None: sys.exit(0)
                done(key,r)
    print("ALLDONE",flush=True)
