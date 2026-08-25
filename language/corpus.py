"""Which corpus a language run uses. Select with the CORPUS env var.

Everything corpus-shaped lives here -- text file, tokeniser, context, ladder
rungs, per-rung budgets -- so a second corpus costs one entry and no edits
elsewhere. Default "shake" reproduces the original numbers byte for byte:
its tag is empty, so its checkpoints and json keep their old names.
"""
import os

REG = {
 "shake": dict(
    txt="shake.txt", tag="", ctx=64, vmax=8000, depth=13,
    tok=r"[a-z']+|[.,!?;:\n]",
    levels=[1,4,8,12,13], chain_steps=300,
    gate={1:(800,64),4:(800,64),8:(800,64),12:(400,24),13:(400,16)},
    bs={1:64,4:64,8:64,12:24,13:16}, bs_default=16),
 # 1.8k tokens of marine-station narrative interleaved with its own QA pairs.
 # Two orders of magnitude smaller than shake, so: keep every type (no UNK
 # cutoff), let the depth follow the vocabulary, and run wider batches.
 "marine": dict(
    txt="marine.txt", tag="marine_", ctx=64, vmax=8000, depth=None,
    tok=r"[^\W\d_]+(?:'[^\W\d_]+)?|\d+|[^\s\w]",
    levels=[1,3,5,7], chain_steps=300,
    gate={}, gate_default=(800,32),
    bs={}, bs_default=32),
}

NAME = os.environ.get("CORPUS", "shake")
if NAME not in REG:
    raise SystemExit(f"unknown CORPUS {NAME!r}; have {sorted(REG)}")
C = REG[NAME]
TAG = C["tag"]
TREE = f"vocabtree_{NAME}.npz" if TAG else "vocabtree.npz"

def ck(name):   return f"ckpt/{TAG}{name}.pt"
def out(name):  return f"{name}_{NAME}.json" if TAG else f"{name}.json"
def gate(L):    return C["gate"].get(L, C.get("gate_default"))
def bs(L):      return C["bs"].get(L, C["bs_default"])
def levels(depth):
    """Ladder rungs, always ending at the finest level the tree supports."""
    ls=[l for l in C["levels"] if l < depth]+[depth]
    return sorted(set(ls))
