import torch, warnings; warnings.filterwarnings("ignore")
from rg import Net, NP, sample, LEVEL_BITS
def perpos(ck,n=4096,bs=512):
    net=Net(); net.load_state_dict(torch.load(f"ckpt/{ck}.pt")); net.eval()
    g=torch.Generator(); g.manual_seed(777); acc=torch.zeros(NP); tot=0
    with torch.no_grad():
        for _ in range(n//bs):
            x,y=sample(bs,"cpu",g); acc+=((net(x)>0).float()==y).float().sum(0); tot+=bs
    return acc/tot
print("LSB-FIRST ladder. New bits at level lv = positions [NP-bits(lv), NP-bits(lv-1))")
print("lvl  new | ctrl_new  anc_new   delta  | anc_ZEROSHOT_new | ctrl_old  anc_old")
print("-"*88)
for lv in range(1,6):
    hi=NP-LEVEL_BITS[lv-1]; lo=NP-LEVEL_BITS[lv]; sl=slice(lo,hi); oldsl=slice(hi,NP)
    c=perpos(f"lsb_ctrl_L{lv}"); a=perpos(f"lsb_anc_L{lv}")
    prev=f"lsb_anc_L{lv-1}" if lv>1 else "lsb_ctrl_L0"
    z=perpos(prev)
    d=(a[sl].mean()-c[sl].mean()).item()
    print(f"L{lv} {LEVEL_BITS[lv-1]:2d}->{LEVEL_BITS[lv]:2d} | {c[sl].mean():.4f}   {a[sl].mean():.4f}  "
          f"{d:+.4f} |     {z[sl].mean():.4f}       | {c[oldsl].mean():.4f}   {a[oldsl].mean():.4f}")
