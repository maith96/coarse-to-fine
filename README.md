# Coarse-to-fine curricula: a negative result

Testing whether a general problem solver can be grown by training on
progressively refined versions of one task — the intuition being that simple
problems are hard problems "highly abstracted", the way an amoeba's problem is
a coarse-grained version of ours.

**Headline: the curriculum does not beat flat training at matched compute.**
Three ladders across two domains. The one apparent win dissolved under a budget
sweep.

What survives is methodology, not mechanism — see [FINDINGS.md](FINDINGS.md).

---

## The theory being tested

Define a coarse-graining operator `R` that blurs a problem's states, goals and
actions. The claim is that for real problem families
`R(hard problem) ≈ easy problem`, so a general solver is a system whose
circuitry has converged to the **fixed point of `R`** — the motifs that survive
blurring at every scale.

That yields a training scheme: take the hard problem, coarse-grain until
trivial, train there, then **refine**. Not a sequence of different tasks ordered
by difficulty — one task, zoomed.

Measured by `τ` = (steps for a from-scratch control to reach threshold) /
(steps for an ancestor-initialised net). `τ` rising with depth confirms;
flat kills it.

---

## Experiments

### 1. Precision on multiplication (`precision/`) — refuted, then voided

Multiply two normalised 12-bit mantissas; level `k` trains on the top `2^k` bits
of the 24-bit product. `R` is exactly a mask on the loss — the architecture is
byte-identical at every rung, so nothing but the objective changes.

Transfer measured on the bits each level *newly* introduces:

| rung | fixed capacity (MSB) | fixed capacity (LSB) | growing (MSB) |
|---|---|---|---|
| L3 (4→8) | +0.105 | −0.023 | +0.067 |
| L4 (8→16) | +0.001 | +0.003 | +0.003 |
| L5 (16→24) | −0.115 | −0.227 | −0.085 |

Zero-shot accuracy on new bits was **0.49–0.55 (chance) at every rung, in every
condition**. One positive rung out of thirteen.

Growth (one identity-initialised layer per rung, function-preserving handoff)
did not rescue it, closing the capacity confound.

**Then the domain turned out to be pathological.** The middle bits are
unlearnable by gradient descent — a parity-type obstruction:

| operand width W | domain size | train loss | test acc |
|---|---|---|---|
| 4 | 64 | 0.0002 | 1.000 |
| 6 | 1,024 | 0.5335 | 0.683 |
| 8 | 16,384 | 0.6888 | 0.511 |
| 12 | 4.2M | 0.6931 | 0.500 |

A 0.80M-parameter net, on a domain of only 16,384 pairs seen ~6× over, reached
train loss **0.6944** — it could not fit data it had fully observed. Capacity
failure looks like overfitting; this is the opposite. The successes at W=4–5 are
memorisation, not algorithm.

So six of the rungs were never a test of `R` at all: the axis is **partitioned
by a region no method can cross**. Verdict: not refuted, untested.

### 2. Shortest-path with horizon (`mazes/`) — the axis works

Grid mazes, exact BFS distances, predict the first optimal action at horizon
`d`. Gate test first (7×7, held-out mazes, baseline *measured* — it is 0.31–0.35,
not 0.25, because cells often have several optimal actions):

| horizon | acc | random | lift |
|---|---|---|---|
| 2 | 0.858 | 0.332 | +0.526 |
| 8 | 0.668 | 0.348 | +0.320 |
| 16 | 0.669 | 0.309 | +0.360 |

Smooth degradation, no cliff. **Connected axis.**

Ladder on 7×7: every delta within ±0.006 — both conditions saturate, advantage
absorbed. On 9×9 with horizons to 24, where 600 steps is not enough:

| horizon | ctrl | anc | delta |
|---|---|---|---|
| 4 | 0.7402 | 0.7383 | −0.002 |
| 8 | 0.6165 | 0.6516 | +0.035 |
| 16 | 0.5076 | 0.5793 | +0.072 |
| 24 | 0.4548 | 0.5239 | +0.069 |

Zero-shot lift was +0.20 to +0.41 — large, unlike multiplication.
**Caveat that later proved decisive: the ancestor had 5 rungs of compute to the
control's 1.**

### 3. Endosymbiotic distillation (`mazes/macro.py`) — failed

Freeze the horizon-8 policy, expose it as a callable macro-action; the new net
learns only *where to delegate*, via top-K subgoal proposals mixed by weight.

| condition at d=24 | acc |
|---|---|
| random policy | 0.318 |
| macro-net, **random** frozen organelle | 0.353 |
| macro-net, **real** frozen organelle | 0.404 |
| flat control | 0.455 |
| ancestor chain, fine-tuned | 0.524 |

The organelle carries real competence (0.404 vs 0.353 for the null). But the
architecture destroys more than it contributes, and the learning curves are flat
from step 0 — gradient reaches the proposer only through the mixture weights, and
when candidates produce similar outputs that gradient vanishes. **Plain sequential
fine-tuning beat the elaborate mechanism by 0.12.**

### 4. Language (`language/`) — the win, and its collapse

Hierarchically cluster an 8k word vocabulary by distributional similarity (PPMI →
SVD → recursive balanced bisection), giving an exact prefix property. Level `k`
predicts the `2^k`-way cluster. Clusters are semantically coherent
(`blood eyes face crown tears sorrow grief sun light`).

Gate — train loss falls at every rung, information gain over the marginal *grows*
with granularity. Entropy floor, not a void:

| level | clusters | train loss | val CE | marginal | gain |
|---|---|---|---|---|---|
| 1 | 2 | 0.817 → 0.131 | 0.144 | 0.159 | +0.015 |
| 8 | 256 | 5.724 → 2.332 | 2.318 | 2.917 | +0.599 |
| 13 | 8192 | 9.186 → 4.683 | 4.795 | 5.669 | +0.874 |

Chain vs matched controls at 300 steps — the ladder wins, growing with
granularity (−0.26 nats at full vocab, perplexity 101.9 vs 132.6).

**Then the budget sweep at level 13:**

| final-rung budget | anc CE | ctrl CE | gap |
|---|---|---|---|
| 300 | 4.6245 | 4.8863 | **+0.262** |
| 600 | 4.5720 | 4.6430 | **+0.071** |
| 1200 | 4.5613 | 4.5548 | **−0.007** |

Both converge to ~4.56. Same destination, reached sooner. Replicated on a second
seed at 300 steps (+0.363).

**At matched total compute the ladder loses.** The chain cost ~218s of wall clock
(L1 2s, L4 52s, L8 57s, L12 53s, L13 54s) to reach 4.6245. The flat control
reaches **4.5548 in ~216s** by training 1200 steps.

---

## Reproducing

```bash
pip install -r requirements.txt
cd language && curl -sL -o shake.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python vocab.py && python gatelm.py 1,4,8,12,13 999
python chain.py 9999 && python sweep.py 9999
```

Run each script from inside its own directory (modules import by bare name).
Scripts are resumable and time-budgeted — the trailing integer is a seconds
budget, after which they checkpoint and exit. All figures above come from
`results/`.

Everything ran on **one CPU core with 3 GB RAM**, which is why models are 1–2M
parameters and corpora are tiny.

---

## Limitations

- Decisive comparisons are **single-seed** except where noted.
- Language work is 290k tokens of Shakespeare, a 96-dim 3-layer model, 64-token
  context. The *pattern* has three independent confirmations; the magnitudes
  would not survive a real corpus.
- The vocabulary bisection balances by word **type**, not token mass, so level 1
  is 96/4 and the coarse rungs carry little information. Fixing this should
  steepen the ladder — untested.
- The head expansion inherits the parent row without the within-cluster log
  marginal, making zero-shot CE *worse* than predicting corpus frequencies
  (5.137 vs 5.072 at L12). A few lines would fix it — untested.
