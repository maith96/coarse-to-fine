# Hip-hop lyrics: the ladder as a *decoder*, not an initialiser

The theory under test: the language ladder learns the **template** at its coarse
rungs, so keeping those rungs at generation time — pick the semantic bucket
first, then refine inside it — should make a better lyrics generator than one
flat softmax.

**Headline: no. The flat model learns the template for free, and the cascade
costs 5× to reproduce it.** One real capability survives, and it is not a
likelihood win — see [Per-rung control](#what-actually-survives) below.

Corpus: 1.53M tokens of US rap lyrics, 36 artists — 5.3× the Shakespeare corpus
the rest of the repo runs on, and a genuinely different register.

---

## 1. The vocabulary tree finds hip-hop semantics

Same PPMI → SVD → balanced recursive bisection as `language/vocab.py`, rebuilt on
lyrics. The level-4 clusters are sharply domain-specific — two of them
(ad-libs, spelled-out letters) are registers that do not exist in Shakespeare:

| L4 cluster | contents |
|---|---|
| 0 | `two drop top beat ride gun hands light straight full pop dope` |
| 1 | `<unk> \n , the i you a and to my in it` |
| 2 | `alive smile hurt reason cry save someone else knows friend grow` |
| 3 | `fear beautiful child moment learn feels tears freedom control mirror` |
| 4 | `la gang ice na town ay y c'mon bounce south boom bam` |
| 5 | `. g o l d e b c mac p eazy double` |
| 6 | `smoke weed hundred six smoking bag drink fat lit pack coke lean` |
| 7 | `car gold red blue green bought clothes hair whip fresh window swag` |

Note cluster 1: `\n` is pooled with the function words, not isolated. The
bisection balances by word **type**, so the coarse rungs carry far less token
mass than their width suggests — level 1 holds 0.168 of a possible 0.693 nats,
24%. This is the limitation the root README already flags, and it means the
"template" is not cleanly separated out at the top of the tree.

## 2. Gate test — the axis is connected (`gatelm.py`)

Information gain over the unigram marginal grows monotonically with granularity,
so no rung is a void:

| level | clusters | val CE | marginal | gain | acc / majority |
|---|---|---|---|---|---|
| 1 | 2 | 0.1625 | 0.1714 | +0.0089 | 0.958 / 0.959 |
| 4 | 16 | 0.9001 | 0.9968 | +0.0967 | 0.798 / 0.792 |
| 8 | 256 | 2.7822 | 3.0580 | +0.2758 | 0.355 / 0.257 |
| 12 | 4096 | 4.9577 | 5.4112 | +0.4535 | 0.159 / 0.107 |
| 13 | 8192 | 5.3742 | 5.9057 | +0.5315 | 0.150 / 0.105 |

Gains are about 60% of the Shakespeare figures at matched rungs (+0.53 vs +0.874
at L13): 36 artists are a harder target than one playwright.

## 3. The ladder replicates — and so does its caveat (`chain.py`)

Ancestor-initialised chain vs from-scratch controls, 300 steps per rung:

| rung | ancestor | control | delta |
|---|---|---|---|
| 4 | 0.9178 | 0.9194 | +0.002 |
| 8 | 2.8256 | 2.8641 | +0.039 |
| 12 | 4.8722 | 5.0271 | +0.155 |
| 13 | 5.2221 | 5.4723 | **+0.250** |

The Shakespeare run gave +0.262 at the same rung and budget. The pattern
transfers to a new domain and a 5× corpus almost exactly — including, as below,
the part where it dissolves once compute is charged.

## 4. The actual test: cascade decoding (`cascade.py`)

Keep every rung and factorise the next-word distribution across them:

```
P(leaf) = P₁(c₁) · P₄(c₄|c₁) · P₈(c₈|c₄) · P₁₂(c₁₂|c₈) · P₁₃(c₁₃|c₁₂)
```

each factor being that rung's own model, restricted to the subtree the coarser
rung chose and renormalised. The supports tile exactly, so this is a normalised
distribution over the same 8192 leaves and its held-out CE is directly
comparable to the flat model's.

The comparison is per-rung, because a flat L13 model *also* induces a
distribution at every rung — sum its softmax inside each bucket — and that
telescopes to exactly its own CE (verified to 6 decimals). So we can ask, rung
by rung: does a model that only ever had to predict the template predict the
template better?

Charged honestly — the cascade trains 5 models, so the flat control gets the
cascade's whole step budget:

**300 steps/rung** (cascade 1500 steps / 90s · flat 1500 steps / 148s)

| rung | cascade | flat | delta |
|---|---|---|---|
| →L1 | 0.1676 | 0.1632 | −0.0044 |
| L1→4 | 0.7749 | 0.7529 | −0.0221 |
| L4→8 | 1.9820 | 1.8898 | −0.0922 |
| L8→12 | 2.1577 | 1.9256 | −0.2321 |
| L12→13 | 0.4178 | 0.3325 | −0.0853 |
| **total** | **5.5000** | **5.0639** | **−0.4361** |

**600 steps/rung** (cascade 3000 steps / 167s · flat 3000 steps / 310s)

| rung | cascade | flat | delta |
|---|---|---|---|
| →L1 | 0.1662 | 0.1609 | −0.0052 |
| L1→4 | 0.7649 | 0.7366 | −0.0284 |
| L4→8 | 1.9441 | 1.8383 | −0.1058 |
| L8→12 | 2.0607 | 1.8332 | −0.2274 |
| L12→13 | 0.3720 | 0.3068 | −0.0652 |
| **total** | **5.3079** | **4.8759** | **−0.4320** |

Perplexity 202 vs 131. The flat model wins at **every rung including the
coarsest** — the one the cascade was supposed to own. The gap is stable across a
doubling of budget, and the seconds column understates it: the flat model pays
more per step (it always carries the 8192-way head, while four of the five
cascade members carry a tiny one), so matching on wall clock rather than steps
would hand the cascade ~86% more steps. The cascade gained 0.192 nats for its
last doubling, so even spending all of that it lands near 5.14 against the flat
model's 4.876 — the sign does not change.

## 5. Why — the specialisation control (`spec.py`)

Two things were confounded above: specialisation and compute. Strip the compute
out by giving the flat model the same N steps each cascade member got:

| rung | specialist | generalist | delta (300 steps) | delta (600 steps) |
|---|---|---|---|---|
| →L1 | 0.1676 | 0.1679 | +0.0003 | −0.0006 |
| L1→4 | 0.7749 | 0.7809 | +0.0059 | +0.0026 |
| L4→8 | 1.9820 | 1.9818 | −0.0002 | −0.0068 |
| L8→12 | 2.1577 | 2.1505 | −0.0071 | −0.0122 |
| L12→13 | 0.4178 | 0.4178 | −0.0000 | +0.0000 |
| **total** | | | **−0.0011** | **−0.0170** |

**A model trained only on the 2-way template task is not better at the template
than one trained on the full 8192-way vocabulary.** Every rung is inside
±0.012 nats. Training on the coarse task buys nothing on the coarse task.

This is what the algebra predicts. If each rung model were perfect it would
equal the true marginal of the same conditional, and the product would telescope
back to exactly the flat model's distribution. The factorisation is a
*reparametrisation*, not extra information — its only possible advantage is
optimisation, and empirically there isn't one. So all that survives of the
cascade is its 5× bill, which is exactly the 0.43 nats in §4.

The theory's premise — the coarse rungs hold something the fine model lacks —
is the part that fails. The fine model learns the template for free, as a
by-product of learning the words.

## 6. What actually survives

The hierarchy is worthless as a likelihood decomposition, but it is not worthless
as an *interface*. Splitting one decision into five gives two knobs a flat
softmax cannot expose (`generate.py`):

**Per-rung temperature.** Cold coarse + hot fine reads as "stay on the template,
be surprising inside it"; the reverse wanders semantically while using safe
words. The two are audibly different, and the failure modes differ in the right
direction — hot-template output breaks lines in the wrong places while keeping
plausible words, cold-template output keeps line structure and swaps vocabulary.

**Template locking.** Take the coarse cluster path of a real verse, force the
cascade to follow it, resample only rungs 8/12/13. Same skeleton, new words:

```
real     getting <unk> in <unk> and tastes
         ain't no telling where this felon is headin', just in case
         keep a shell at the tip of your <unk>, clear the space
         your brain was a terrible thing to waste

locked   smoke the <unk> <unk> its <unk> cover dome, slay with his words, di
         when i can't play joker ho, unlike fame
         alleys in my name the eyes in hotels and a stop
```

Line lengths and comma placement carry over; the words do not. That is a real
control surface for a lyrics tool — but it is bought at 5× compute and 0.43 nats
of quality, and the same effect is reachable from a flat model by masking its
softmax to a chosen cluster, which costs nothing. **The cascade is the expensive
way to get it.**

Sample quality tracks CE exactly: the flat model's output is the most coherent of
the four conditions in `samples.txt`, the uniform-temperature cascade the least.

---

## Reproducing

```bash
cd lyrics && pip install -r ../requirements.txt
python fetch.py && python vocab.py
python gatelm.py 1,4,8,12,13 3600     # gate test
python chain.py 99999                 # ladder vs controls
python cascade.py 99999 300,600 0     # cascade vs flat, matched total compute
python spec.py 700 300,600 0          # specialisation control
python generate.py 600 0 100          # samples
```

Trailing integers are seconds budgets; scripts checkpoint and resume. Run from
inside this directory (modules import by bare name). Everything here is
**single-seed** — the repo's measured single-seed noise floor is ~±0.08, so the
+0.25 ladder delta in §3 is real but the ±0.012 numbers in §5 are only evidence
of *absence* of an effect, not a measurement of one. The 0.43 nat cascade gap is
5× that floor and replicates across two budgets.

The corpus is copyrighted lyrics and is **not committed** — `fetch.py` pulls it
from [fpaupier/RapLyrics-Scraper](https://github.com/fpaupier/RapLyrics-Scraper)
(MIT-licensed scraper; lyrics scraped from Genius), the same way the Shakespeare
work downloads `shake.txt`.

## Limitations

- Single seed everywhere. §5's null is the load-bearing claim and deserves 3–5
  seeds; it would be ~10 minutes of compute.
- The 1200-steps/rung point of the §4 sweep was cut for time. Both completed
  budgets agree to 0.004 nats, so the trend is flat, but the crossing behaviour
  the Shakespeare sweep found at higher budgets is untested here.
- The bisection balances by word type, so `\n` is not isolated at the coarse
  rungs (§1) and the template is not cleanly addressable. A token-mass-balanced
  tree would test the theory on its strongest terms; it needs more than 13 levels
  to keep one word per leaf, so it is a real change, not a flag.
- 96-dim, 3 layers, 64-token context — about 6 lines of context, enough for
  couplet structure and not for verse structure. Samples are correspondingly
  local.
