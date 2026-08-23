# Coarse-to-fine curricula: a negative result

Testing whether a general problem solver can be grown by training on
progressively refined versions of one task — the intuition being that simple
problems are hard problems "highly abstracted", the way an amoeba's problem is
a coarse-grained version of ours.

**Headline: the curriculum does not beat flat training at matched compute.**
Three ladders across two domains. Every apparent win has now dissolved — two
under a budget sweep, one under a reinstall.

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

| horizon | ctrl | anc | delta | rerun delta (2nd build) |
|---|---|---|---|---|
| 4 | 0.7402 | 0.7383 | −0.002 | −0.022 |
| 8 | 0.6165 | 0.6516 | +0.035 | +0.032 |
| 16 | 0.5076 | 0.5793 | +0.072 | **−0.013** |
| 24 | 0.4548 | 0.5239 | +0.069 | +0.048 |

Zero-shot lift was +0.20 to +0.41 — large, unlike multiplication.
**The caveat proved decisive twice over, and neither surviving column means
what it looks like:**

*Compute.* The ancestor had 5 rungs to the control's 1. Charged for its own
construction (`sweep9.py`), the d=24 advantage inverts — flat control at the
ancestor's own 3000-step total reaches **0.5588** against the chain's 0.5149,
and both converge to ~0.557 with the chain paying 1.8× for it.

*Reproducibility.* Rerunning the identical code and seeds on a second torch
build moves rungs by up to 0.055 and **flips the sign at d=16**. The
single-seed error bar here is ~±0.08 — wider than any effect in the table.

*What the metric measures.* Cells are sampled at distance exactly `d`, so
these are one-step classifiers on a fixed-distance slice, not policies. Rolled
out greedily, every net reaches the goal **0.000** of the time (`compare9.py`).

Verdict: the axis is connected, and nothing built on it survived.

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

Both converge to ~4.56. Same destination, reached sooner.

**Now with error bars** — five seeds, the whole chain rebuilt per seed
(`multiseed.py`; the old replication reseeded only the control, see FINDINGS §9):

| final-rung budget | gap (ctrl − anc) | 95% CI | seeds positive |
|---|---|---|---|
| 300 | **+0.2845** ± 0.019 | [+0.261, +0.308] | 5/5 |
| 600 | **+0.0833** ± 0.022 | [+0.057, +0.110] | 5/5 |
| 1200 | −0.0119 ± 0.020 | [−0.037, +0.013] | 3/5 |

The decay is a measurement, not a single point, and the crossing is real.

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
python multiseed.py 1500 5   # 5-seed replication + corrected head expansion (~75 min)
python handoff.py 1500 5     # final-handoff smoothing sweep (~9 min)
python predict.py            # run the best L13 net as a word-level LM
```

Maze audit (regenerates the 9×9 chain first — `ckpt/` is gitignored):

```bash
cd mazes && python lad9.py 7000 && python sweep9.py 30000
python compare9.py && python inspect9.py sw9_ctrl_d24_n3000_s0 24
```

Run each script from inside its own directory (modules import by bare name).
Scripts are resumable and time-budgeted — the trailing integer is a seconds
budget, after which they checkpoint and exit. All figures above come from
`results/`.

The original figures ran on **one CPU core with 3 GB RAM**, which is why models
are 1–2M parameters and corpora are tiny. The rerun columns come from a 4-core
i5-8265U on the same torch 2.13 — and see §4 of FINDINGS: **do not expect these
numbers to reproduce to better than ~±0.08 on a different build**, even at
identical seeds.

---

## Limitations

- Maze comparisons are **single-seed**, and the measured build-to-build noise
  floor there (~±0.08 on maze deltas) is wider than most effects claimed for
  that domain. Multi-seed replication in mazes is still the first thing this
  project needs. The language ladder now has n=5 (FINDINGS §9), where the
  seed-to-seed floor is ~±0.02 — the ±0.08 figure is a maze number and does not
  transfer.
- Language work is 290k tokens of Shakespeare, a 96-dim 3-layer model, 64-token
  context. The *pattern* has three independent confirmations; the magnitudes
  would not survive a real corpus.
- The language crossing point is no longer single-seed: −0.0119 ± 0.020 over
  five seeds, a 95% CI of [−0.037, +0.013] that straddles zero. "Both converge
  to ~4.56" and the crossing itself both survive.
- The vocabulary bisection balances by word **type**, not token mass, so level 1
  is 96/4 and the coarse rungs carry little information. Fixing this should
  steepen the ladder — untested.
- The head expansion inherits the parent row without the within-cluster log
  marginal, making zero-shot CE *worse* than predicting corpus frequencies
  (5.137 vs 5.072 at L12). ~~A few lines would fix it — untested.~~ Now tested
  (FINDINGS §10): supplying `log p(child|parent)` improves zero-shot by up to
  1.1 nats, exactly as predicted, and makes the trained model **0.04–0.05 nats
  worse** at level 13 on every seed. A better handoff is not a better ladder.
