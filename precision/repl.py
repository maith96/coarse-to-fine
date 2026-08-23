import sys, json, torch, warnings; warnings.filterwarnings("ignore")
from rg import train_level, LEVEL_BITS, Net, NP, sample
lv=int(sys.argv[1]); init=sys.argv[2]; seed=int(sys.argv[3]); tag=sys.argv[4]
st=None if init=="none" else torch.load(f"ckpt/{init}.pt")
net,curve=train_level(lv,st,800,"cpu",seed,eval_every=40)
torch.save(net.state_dict(),f"ckpt/{tag}.pt")
old,new=LEVEL_BITS[lv-1],LEVEL_BITS[lv]
g=torch.Generator(); g.manual_seed(777); acc=torch.zeros(NP); tot=0
net.eval()
with torch.no_grad():
    for _ in range(8):
        x,y=sample(512,"cpu",g); acc+=((net(x)>0).float()==y).float().sum(0); tot+=512
acc/=tot
print(f"{tag}: L{lv} new-bits({old}->{new}) acc {acc[old:new].mean():.4f} | old-bits {acc[:old].mean():.4f}")
