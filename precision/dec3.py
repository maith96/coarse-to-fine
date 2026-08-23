import torch, warnings; warnings.filterwarnings("ignore")
from grow import GNet
from rg import NP, sample, LEVEL_BITS
def perpos(ck,nl,n=4096,bs=512):
    net=GNet(nl); net.load_state_dict(torch.load(f"ckpt/{ck}.pt")); net.eval()
    g=torch.Generator(); g.manual_seed(777); acc=torch.zeros(NP); tot=0
    with torch.no_grad():
        for _ in range(n//bs):
            x,y=sample(bs,"cpu",g); acc+=((net(x)>0).float()==y).float().sum(0); tot+=bs
    return acc/tot
print("GROWING ladder (MSB-first, +1 identity-init layer per rung, matched-capacity controls)")
print("lvl  new  |  ctrl_new  anc_new    delta  | anc_ZEROSHOT | ctrl_old  anc_old | all24 ctrl/anc")
print("-"*104)
for lv in range(3,6):
    old,new=LEVEL_BITS[lv-1],LEVEL_BITS[lv]; sl=slice(old,new)
    c=perpos(f"gr_ctrl_L{lv}",lv+1); a=perpos(f"gr_anc_L{lv}",lv+1)
    z=perpos(f"gr_anc_L{lv-1}",lv)
    print(f"L{lv} {old:2d}->{new:2d} |  {c[sl].mean():.4f}   {a[sl].mean():.4f}   {(a[sl].mean()-c[sl].mean()):+.4f} |"
          f"    {z[sl].mean():.4f}    |  {c[:old].mean():.4f}   {a[:old].mean():.4f} | {c.mean():.4f}/{a.mean():.4f}")
