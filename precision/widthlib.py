import torch, torch.nn as nn, warnings, time; warnings.filterwarnings("ignore")
import torch.nn.functional as F

def make(W):
    NPb=2*W; SEQ=2*W+NPb
    class N(nn.Module):
        def __init__(s,nl=3,d=64):
            super().__init__()
            s.tok=nn.Embedding(3,d); s.pos=nn.Embedding(SEQ,d)
            s.ls=nn.ModuleList([nn.TransformerEncoderLayer(d,4,4*d,dropout=0.0,batch_first=True,
                norm_first=True,activation="gelu") for _ in range(nl)])
            s.ln=nn.LayerNorm(d); s.hd=nn.Linear(d,1)
        def forward(s,x):
            q=torch.full((x.shape[0],NPb),2,dtype=torch.long)
            h=s.tok(torch.cat([x,q],1))+s.pos.weight[None]
            for l in s.ls: h=l(h)
            return s.hd(s.ln(h[:,-NPb:])).squeeze(-1)
    def smp(bs,g):
        lo=2**(W-1)
        a=lo+torch.randint(0,lo,(bs,),generator=g); b=lo+torch.randint(0,lo,(bs,),generator=g)
        p=a*b
        si=torch.arange(W-1,-1,-1); so=torch.arange(NPb-1,-1,-1)
        return torch.cat([(a[:,None]>>si)&1,(b[:,None]>>si)&1],1).long(), (((p[:,None]>>so)&1)).float()
    return N, smp, NPb

