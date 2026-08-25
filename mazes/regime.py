"""The regime test: horizon ladder vs flat control, where flat training cannot reach.

calib.py established the regime (FINDINGS section 12): on 13x13 with the corrected
loss, a from-scratch policy rolls out at 0.890 (d<=2) and 0.565 (d<=4) but collapses
to 0.105 (d<=8) and 0.000 (d<=16, d<=24), and 5x the compute does not move it. That
leaves ~20 horizon units the flat control cannot cross -- the first setting in this
project where a curriculum has something to beat that its control cannot reach.

Pre-registered before running (FINDINGS section 12):

  ladder    d<=2 -> 4 -> 8 -> 16 -> 24, weight inheritance, 1200 steps per rung
  control   flat d<=24 at the ladder's TOTAL step count (6000), not the per-rung one
  metric    d* = largest horizon with rollout success >= 0.5 from held-out starts
  seeds     n=5, per-seed signs reported
  win       d*(ladder) >= d*(flat) + 4 in >=4/5 seeds
  null      |delta d*| <= 2
  kill      flat >= ladder

Both arms are evaluated at every rung boundary, so the result is a reach-vs-compute
curve rather than a single point (section 3).

Deviation from the section 12 plan, on compute grounds: the ladder uses the
published horizon set [2,4,8,16,24] rather than a 9-rung one. This makes the ladder
coarser, i.e. weaker, so it cannot inflate the effect. d* resolution comes from the
evaluation grid, which is finer than either.

Each rung gets a fresh AdamW and OneCycleLR, as lad9.py does -- so the ladder takes
5 LR cycles to the control's 1. That is a property of the method being tested, kept
deliberately rather than fixed here.

  python regime.py <seconds-budget> [nseeds]
"""
import sys, os, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
sys.argv=[sys.argv[0], sys.argv[1] if len(sys.argv)>1 else '0', '13', '2000']
from calib import PolNet, index, batch, rollout, onestep, G, SPLIT, M

RUNGS=[2,4,8,16,24]; R=1200; TOTAL=R*len(RUNGS)
EVAL=[2,3,4,6,8,12,16,20,24]
BS=128; THRESH=0.5

def dstar(net):
    """largest horizon whose greedy rollout still reaches the goal >= THRESH."""
    roll={}
    for h in EVAL:
        r=rollout(net,h,n=200)
        if r is not None: roll[h]=r
    ok=[h for h in EVAL if h in roll and roll[h]['reached']>=THRESH]
    return (max(ok) if ok else 0), {str(h):roll[h] for h in roll}

def fit(net, k, steps, seed, tag, BUD, t0, sched_total=None):
    """train `net` on cells at distance <= k up to step `steps`; resumable.

    sched_total is the OneCycle horizon, which is not always `steps`: the control
    is one continuous 6000-step run that merely stops at each rung boundary to be
    evaluated, so its schedule must span 6000 regardless of where we pause."""
    ck=f"ckpt/rg_{tag}.pt"
    o=torch.optim.AdamW(net.parameters(),lr=3e-3,weight_decay=0.01)
    sch=torch.optim.lr_scheduler.OneCycleLR(o,3e-3,total_steps=sched_total or steps,
                                            pct_start=0.15)
    s0=0; curve=[]
    if os.path.exists(ck):
        c=torch.load(ck); net.load_state_dict(c['n']); o.load_state_dict(c['o'])
        sch.load_state_dict(c['s']); s0=c['step']; curve=c['c']
        if s0>=steps: return net,curve,True
    tr=index(1,k,0,SPLIT); rng=np.random.default_rng(11+seed)
    for _ in range(s0): rng.integers(0,len(tr[0]),BS)
    for s in range(s0+1,steps+1):
        x,p,y=batch(tr,BS,rng)
        loss=-(F.log_softmax(net(x,p),-1).masked_fill(~y,float('-inf'))).logsumexp(-1).mean()
        o.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); o.step(); sch.step()
        if s%200==0 or s==1: curve.append((s,round(loss.item(),4)))
        if time.time()-t0>BUD or s==steps:
            torch.save({'n':net.state_dict(),'o':o.state_dict(),'s':sch.state_dict(),
                        'step':s,'c':curve},ck)
            if s<steps: print(f"  {tag} {s}/{steps}",flush=True); return net,curve,False
    return net,curve,True

if __name__=="__main__":
    BUD=float(sys.argv[1]); NS=int(sys.argv[2]) if len(sys.argv)>2 else 5
    t0=time.time(); RES="../results/regime13.json"
    res=json.load(open(RES)) if os.path.exists(RES) else {}
    def save(): json.dump(res,open(RES,"w"),indent=1)

    for seed in range(NS):
        # ---------- ladder: inherit weights rung to rung ----------
        torch.manual_seed(seed); net=PolNet()
        for i,k in enumerate(RUNGS):
            cum=(i+1)*R; key=f"s{seed}/lad_k{k}"
            tag=f"s{seed}_lad_k{k}"
            if key in res:
                net.load_state_dict(torch.load(f"ckpt/rg_{tag}.pt")['n']); continue
            if time.time()-t0>BUD: print("PAUSE",flush=True); save(); sys.exit(0)
            net,curve,fin=fit(net,k,R,seed,tag,BUD,t0)
            if not fin: save(); sys.exit(0)
            ds,roll=dstar(net)
            res[key]={"cum_steps":cum,"dstar":ds,"loss":[curve[0][1],curve[-1][1]],
                      "rollout":roll}
            save()
            print(f"[{time.time()-t0:5.0f}s] s{seed} ladder rung d<={k:2d} "
                  f"(cum {cum}) loss {curve[0][1]:.3f}->{curve[-1][1]:.3f}  d*={ds}",flush=True)
        # ---------- control: flat d<=24, checkpointed at every rung boundary ----------
        torch.manual_seed(seed); cnet=PolNet()   # same init as the ladder: paired
        for i in range(len(RUNGS)):
            cum=(i+1)*R; key=f"s{seed}/ctrl_n{cum}"
            tag=f"s{seed}_ctrl"
            if key in res: continue
            if time.time()-t0>BUD: print("PAUSE",flush=True); save(); sys.exit(0)
            # one continuous 6000-step run; stop at each boundary to evaluate
            cnet,curve,fin=fit(cnet,24,cum,seed,tag,BUD,t0,sched_total=TOTAL)
            if not fin: save(); sys.exit(0)
            ds,roll=dstar(cnet)
            res[key]={"cum_steps":cum,"dstar":ds,"loss":[curve[0][1],curve[-1][1]],
                      "rollout":roll}
            save()
            print(f"[{time.time()-t0:5.0f}s] s{seed} control  n={cum:5d}      "
                  f"loss {curve[0][1]:.3f}->{curve[-1][1]:.3f}  d*={ds}",flush=True)
    print("ALLDONE",flush=True)
