import json, torch, warnings
warnings.filterwarnings("ignore")
from rg import LEVEL_BITS, Net, NP, sample
S=0
def perpos(ck, n=4096, bs=512):
    net=Net(); net.load_state_dict(torch.load(f"ckpt/{ck}.pt")); net.eval()
    g=torch.Generator(); g.manual_seed(777); acc=torch.zeros(NP); tot=0
    with torch.no_grad():
        for _ in range(n//bs):
            x,y=sample(bs,"cpu",g); pr=(net(x)>0).float()
            acc+=(pr==y).float().sum(0); tot+=bs
    return acc/tot
print("Decomposition: accuracy on OLD bits (inherited) vs NEW bits (first seen at this level)")
print("lvl  newbits |  ctrl_new  anc_new  |  ctrl_old  anc_old  | anc_ZEROSHOT_new")
print("-"*82)
for lv in range(1,len(LEVEL_BITS)):
    old=LEVEL_BITS[lv-1]; new=LEVEL_BITS[lv]
    sl=slice(old,new)
    c=perpos(f"ctrl_s{S}_L{lv}"); a=perpos(f"anc_s{S}_L{lv}")
    prev=f"anc_s{S}_L{lv-1}" if lv>1 else f"ctrl_s{S}_L0"
    z=perpos(prev)
    print(f"L{lv} {old:2d}->{new:2d} |  {c[sl].mean():.4f}   {a[sl].mean():.4f}  |  "
          f"{c[:old].mean():.4f}   {a[:old].mean():.4f}  |   {z[sl].mean():.4f}")
