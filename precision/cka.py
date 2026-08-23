import torch, warnings; warnings.filterwarnings("ignore")
from rg import Net, sample, LEVEL_BITS
def gram_cka(X,Y):
    X=X-X.mean(0,keepdim=True); Y=Y-Y.mean(0,keepdim=True)
    hsic=(X.T@Y).norm()**2
    return (hsic/((X.T@X).norm()*(Y.T@Y).norm())).item()
def acts(ck,x):
    n=Net(); n.load_state_dict(torch.load(f"ckpt/{ck}.pt")); n.eval()
    with torch.no_grad(): _,a=n(x,return_acts=True)
    return [h.reshape(h.shape[0],-1) for h in a]
g=torch.Generator(); g.manual_seed(31337); x,_=sample(1024,"cpu",g)
print("CKA between INDEPENDENTLY-trained controls (no shared weights).")
print("Tests scale-invariant motifs: does a 4-bit solver look like a 24-bit solver?\n")
print("pair        layer1 layer2 layer3")
A={lv:acts(f"ctrl_s0_L{lv}",x) for lv in range(6)}
for i,j in [(2,3),(2,4),(2,5),(3,5),(4,5),(0,5),(1,4)]:
    v=[gram_cka(A[i][k],A[j][k]) for k in range(3)]
    print(f"L{i} vs L{j}    "+"  ".join(f"{q:.3f}" for q in v))
print("\nBaseline: same-level nets differ only by training; chain vs control at L5")
B=acts("anc_s0_L5",x)
print("ctrlL5 vs ancL5  "+"  ".join(f"{gram_cka(A[5][k],B[k]):.3f}" for k in range(3)))
