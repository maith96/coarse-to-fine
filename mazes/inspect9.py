"""Look at what a trained 9x9 policy actually outputs.

One-step accuracy hides whether the policy composes: a net can pick a correct
first action 56% of the time and still never reach the goal.  This prints
per-cell predictions on held-out mazes and then rolls the greedy policy out.
"""
import sys, numpy as np, torch, warnings
warnings.filterwarnings("ignore"); import torch.nn.functional as F
torch.set_num_threads(1)                      # be polite to the running sweep
from lad9 import PolNet, index, batch, SPLIT, M, tok, walls, d, opt, G

ACT=["UP","DOWN","LEFT","RIGHT"]; DELTA=[(-1,0),(1,0),(0,-1),(0,1)]

def load(ck):
    net=PolNet(); net.load_state_dict(torch.load(f"ckpt/{ck}.pt")['n']); net.eval(); return net

def render(m,i,j,pred=None):
    out=[]
    for r in range(G):
        row=""
        for c in range(G):
            if walls[m,r,c]: row+="#"
            elif d[m,r,c]==0: row+="G"
            elif (r,c)==(i,j): row+="A"
            else: row+="."
        out.append(row)
    return out

def show(net,dist,n=4,seed=3):
    rng=np.random.default_rng(seed)
    mi,ii,jj=index(dist,SPLIT,M)
    pick=rng.choice(len(mi),n,replace=False)
    for k in pick:
        m,i,j=mi[k],ii[k],jj[k]
        x=tok[m].copy(); x[i,j]=3
        with torch.no_grad():
            lg=net(torch.from_numpy(x.reshape(1,-1)), torch.tensor([i*G+j]))
            p=F.softmax(lg,-1)[0].numpy()
        best=int(p.argmax()); good=opt[m,i,j]
        grid=render(m,i,j)
        probs="  ".join(f"{ACT[a]:5s}{p[a]:.2f}{'*' if good[a] else ' '}" for a in range(4))
        print(f"\n  maze {m}  agent=({i},{j})  true dist to goal = {d[m,i,j]}")
        for r,line in enumerate(grid):
            tag=""
            if r==1: tag=f"   optimal: {[ACT[a] for a in range(4) if good[a]]}"
            if r==2: tag=f"   model:   {ACT[best]}  {'CORRECT' if good[best] else 'WRONG'}"
            if r==3: tag=f"   {probs}"
            print(f"    {line}{tag}")

def rollout(net,dist,n=200,seed=4,cap=80):
    """Greedy rollout from held-out start cells: does the policy reach the goal?"""
    rng=np.random.default_rng(seed)
    mi,ii,jj=index(dist,SPLIT,M)
    pick=rng.choice(len(mi),n,replace=False)
    reached=0; optimal=0; stuck=0; loop=0; lens=[]
    for k in pick:
        m,i,j=mi[k],ii[k],jj[k]; d0=d[m,i,j]; seen=set(); steps=0
        while steps<cap:
            if d[m,i,j]==0: reached+=1; lens.append(steps); optimal+= (steps==d0); break
            if (i,j) in seen: loop+=1; break
            seen.add((i,j))
            x=tok[m].copy(); x[i,j]=3
            with torch.no_grad():
                a=int(net(torch.from_numpy(x.reshape(1,-1)),torch.tensor([i*G+j])).argmax())
            di,dj=DELTA[a]; ni,nj=i+di,j+dj
            if not (0<=ni<G and 0<=nj<G) or walls[m,ni,nj]: stuck+=1; break
            i,j=ni,nj; steps+=1
        else: loop+=1
    print(f"\n  rollout from {n} held-out cells at d={dist} (cap {cap} steps):")
    print(f"    reached goal      {reached/n:.3f}")
    print(f"    optimal path      {optimal/n:.3f}")
    print(f"    walked into wall  {stuck/n:.3f}")
    print(f"    looped / timeout  {loop/n:.3f}")
    if lens: print(f"    mean steps when reached {np.mean(lens):.1f}  (optimal would be {dist})")

if __name__=="__main__":
    ck=sys.argv[1] if len(sys.argv)>1 else "sw9_ctrl_d24_n3000_s0"
    dist=int(sys.argv[2]) if len(sys.argv)>2 else 24
    print(f"=== {ck} at horizon d={dist} ===")
    net=load(ck)
    show(net,dist)
    rollout(net,dist)
