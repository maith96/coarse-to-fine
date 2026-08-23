# Findings

What I would defend from this project, separated from what it set out to show.
The mechanism failed; the methodology is the deliverable.

> **Revision, 2026-08-23.** The maze ladder has now had the budget sweep that
> §3 said it needed, and the whole ladder was rerun on a second machine. Both
> changed the conclusions: §3 gains a second confirmation, §4 is new, and §8 is
> a retraction of what the previous version called the one live direction.

---

## 1. The gate test

**Before building any ladder, train independent models from scratch at every
rung and confirm each one learns.** Cost: a fraction of the ladder itself.

This is the step whose absence cost most of the project. Multiplication's
middle bits are unlearnable by gradient descent, so six rungs of the precision
ladder were never testing coarse-graining — they were testing whether transfer
can bootstrap something no method can learn. The gate would have caught it on
day one.

## 2. Void vs entropy floor

Two failure modes look identical if you only watch accuracy, and they have
opposite implications.

| | signature | meaning |
|---|---|---|
| **Void** | train loss pinned at chance (exactly `ln N`) | axis partitioned, no ladder can cross |
| **Entropy floor** | train loss falls, test accuracy plateaus above chance | just the Bayes limit, ladder still viable |

Multiplication at W=8: 0.80M parameters, a 16,384-pair domain seen ~6× over,
train loss **0.6944** vs `ln 2 = 0.6931`. It could not fit data it had fully
observed — that is not capacity (capacity failure overfits), it is a
gradient-signal failure of parity type.

Language at level 13: train loss **9.186 → 4.683** against `ln N = 9.01`.
Traction everywhere.

**Watch train loss, not accuracy, to tell them apart.**

## 3. Matched-compute accounting

**A curriculum must be charged for its own construction.** Violated twice,
caught twice, and it inverted the headline result in *both* domains.

Language at level 13: the ladder looked like −0.26 nats until the sweep showed
the gap decaying to zero by 1200 steps, at which point the flat control matched
the chain's *total* wall clock and beat it outright.

Mazes at d=24, swept the same way. The ancestor arrives at the final rung
having already spent 2400 steps on `base_d2 → anc_d4 → anc_d8 → anc_d16`, so
matched total compute is anc@600 against ctrl@3000:

| condition | final-rung steps | total steps | acc |
|---|---|---|---|
| ancestor chain | 600 | 3000 | 0.5149 |
| flat control, **starved** (as published) | 600 | 600 | 0.4668 |
| flat control, **matched total** | 3000 | 3000 | **0.5588** |
| ancestor chain, same final-rung budget | 3000 | 5400 | 0.5569 |

The published +0.069 becomes **−0.044** once the control is paid the same
compute. Both conditions converge to ~0.557; the ancestor gets there having
spent 1.8× as much. Unlike language, the ladder does not even arrive sooner.

**Corollary:** any curriculum evaluated only under a starved baseline will look
like it works. Sweep the budget or the result means nothing. Two domains, two
inversions, no exceptions so far.

## 4. Establish the noise floor before believing a delta

New, and it subsumes a lot of the above.

Rerunning `lad9.py` completely unchanged — same code, same seeds, same
torch 2.13, verified identical mazes and bit-identical evaluation through
step 100 — on a different wheel (Windows MKL 2026.1 / AVX2 rather than the
original pip build) moved individual rungs by up to 0.055:

| horizon | published Δ (anc − ctrl) | rerun Δ | shift |
|---|---|---|---|
| 4 | −0.002 | −0.022 | −0.020 |
| 8 | +0.035 | +0.032 | −0.004 |
| 16 | **+0.072** | **−0.013** | **−0.084** |
| 24 | +0.069 | +0.048 | −0.021 |

**The d=16 result changes sign.** It was the largest effect in the maze
section, and it does not survive a reinstall — no compute argument required.

Thread count is *not* the mechanism: 1, 2 and 4 threads are bit-identical to
12 decimals over 40 steps. The drift is between builds, and it is invisible
until it flips an argmax — which is why it shows up on the two rungs whose
learning curves were still climbing (`base_d2`, `anc_d16`) and not on the
converged ones (`anc_d8` reproduces to 0.0002).

So the honest error bar on any single-seed delta in this project is about
**±0.08**, which is larger than every effect it ever claimed except the
language gap at 300 steps. A quantized metric on a mid-transition learning
curve is a variance amplifier: sub-ULP differences move *when* the transition
happens, and a fixed-step readout turns that into a large apparent gap.

**Measure the noise floor first — rerun one cell on a second machine, or with
a second seed — and refuse to interpret anything smaller.**

## 5. Reachable ≠ representable

A transformer can *represent* multiplication — the weights could be written by
hand. The obstruction is that the solution is **unreachable by gradient descent
from random init**. Different claim, different fix: not a bigger model, but a
different training signal (intermediate supervision on carries, curriculum over
carry depth).

## 6. Base-rate artifacts — the error log

Seven instances. All the same failure: reasoning from the salient hypothesis
instead of the null. The first four were caught by computing the null in code;
the rest were not, because they were never written down as numbers.

1. **All-zeros sampler.** Level 2 hit "95% exact at step 20" — the top bits were
   almost always zero under the original magnitude sampling. Fixed by
   normalising operands.
2. **τ = 20× at L4.** Spectacular, and pure old-bit retention. Zero-shot on the
   *new* bits was chance at every rung. Decomposing old vs new killed it.
3. **Level-1 negative lift.** Val accuracy 0.9622 vs majority 0.9629 on a rung
   whose train loss had fallen 0.817 → 0.131 (below `ln 2`). Clusters were 96/4
   by token mass; accuracy was the wrong metric. Cross-entropy vs the marginal
   was the right one.
4. **Compute-confounded ladder win.** Section 3 above, now twice over.
5. **Two GPT-2 predictions, both wrong.** Predicted a *Tempest* continuation
   from `miranda : \n certainly , sir , i can .` — got Space Station 13 server
   logs, because the escaped-newline serialisation matched chat-log format far
   more strongly than the content matched Shakespeare. Reformatted properly,
   predicted Prospero — got Star Trek, because `MIRANDA:` in caps is generic
   screenplay format and the line contains no archaic English. Register
   prediction held; content prediction failed twice.
6. **Single-seed deltas read as measurements.** Every decisive comparison was
   run once, and the run-to-run floor was never established. Section 4: it is
   ±0.08, and one headline effect sits under it.
7. **A result generalised from the domain already declared untestable.**
   Section 8.

**The discipline that works in code does not automatically transfer to prose
predictions — or to deciding which numbers deserve belief.** If a claim is
worth making, write down the null first.

## 7. What the theory looks like after contact

The abstraction relation looked real: short-horizon nets transfer zero-shot to
long horizons (+0.20 to +0.41 lift); vocabulary clusters are semantically
coherent. What fails is converting that relation into a training schedule.

The transfer claim needs narrowing, though. The maze task samples cells at
distance *exactly* `d`, so what transfers is a **one-step classifier on a
fixed-distance slice**, not a policy. Rolled out greedily from held-out cells,
every net — ancestor and control alike — reaches the goal **0.000** of the
time, walking into a wall ~73% of the time and looping the rest. Masking
illegal moves converts the crashes into 100% looping rather than into
solutions. (The rollout is off-distribution for every condition, since no net
ever saw the intermediate distances, so it is not a head-to-head; it is
evidence about what the metric ever meant.)

The likely reason the schedule fails, visible in the generation samples:
**coarse levels teach the high-frequency structure a flat model acquires
cheaply anyway.** A 1.9M-parameter model learns "a speaker name goes after a
blank line" — and so does the control, within a few hundred steps. Neither
touches the residual (*which* name), where all the remaining entropy lives.

GPT-2 XL illustrates the same split at 800× the scale: perfect Elizabethan
register, invented characters (`CONSTANTINE`, `ARIES`) from a near-verbatim
*Tempest* cue. Style is the coarse level and scales beautifully; specific
identity is the fine level and does not.

## 8. Retention — retracted

The previous version of this document called retention "the one effect never
compute-confounded" and the one live direction left: ancestor-initialised nets
held ancestral competence better than independently trained ones, +0.09 to
+0.10 at L4, and it came free with plain sequential initialisation.

**It does not replicate in mazes, and its evidential base was the pathological
domain.** Retention was only ever measured in `precision/` — where §1 and §2
establish that the middle rungs are unlearnable, so the +0.09 was measured
between two models that were both stuck. Measured in the domain with a
confirmed connected axis, one-step accuracy at every horizon:

| net | d=2 | d=4 | d=8 | d=16 | d=24 |
|---|---|---|---|---|---|
| ancestor chain (trained on **all** of these) | 0.3079 | 0.3034 | 0.3659 | 0.5251 | 0.5205 |
| flat control, 600 (saw only d=24) | 0.3809 | 0.3877 | 0.3945 | 0.4596 | 0.4619 |
| flat control, 3000 (saw only d=24) | 0.2969 | 0.3031 | 0.3460 | 0.4948 | 0.5632 |
| random baseline | 0.3266 | 0.3297 | 0.3335 | 0.3223 | 0.3184 |

The ancestor is **at or below chance on the rungs it trained on**, and the
control that never saw d=2 or d=4 beats it there. That is catastrophic
forgetting, not retention.

What remains is the reframing, not the evidence for it: "accumulate capability
without eroding it" is a better-posed problem than "train faster", and it is an
open one. But this project produced no result supporting it, and one against.

## 9. Multi-seed replication — language, with error bars

The gap §4 said was missing. Five seeds, the **whole chain rebuilt per seed**,
full-val deterministic evaluation (`language/multiseed.py`):

| final-rung budget | ancestor | control | gap (ctrl − anc) | 95% CI (t₄) | seeds positive |
|---|---|---|---|---|---|
| 300 | 4.7166 | 5.0010 | **+0.2845** ± 0.019 | [+0.261, +0.308] | 5/5 |
| 600 | 4.6666 | 4.7499 | **+0.0833** ± 0.022 | [+0.057, +0.110] | 5/5 |
| 1200 | 4.6450 | 4.6332 | −0.0119 ± 0.020 | [−0.037, +0.013] | 3/5 |

**The decay is real and the crossing is real.** The 300- and 600-step gaps are
far outside seed noise; the 1200-step gap straddles zero. This confirms §3 with
a measurement rather than a single point: the ladder arrives sooner and lands in
the same place.

Two corrections to how this project has been measuring things:

**The published seed replication was asymmetric.** In `sweep.py`,
`torch.manual_seed(seed)` runs before the net is built, but for `cond="anc"` the
net is then overwritten by `ch_anc_L12`. So across "seeds" the ancestor was
byte-identical and only its batch order changed, while the control got a fresh
init as well. The +0.262 / +0.363 pair was two draws from an asymmetric
comparison, not a replication.

**The language noise floor is not the maze noise floor.** Seed-to-seed sd on a
single arm is 0.009–0.023 and on the gap ~0.02 — roughly 4× tighter than the
±0.08 that §4 measured across *builds* in mazes. §4's floor is a statement about
mazes and about reinstallation, and it should not be applied to the language
numbers unchanged. Effects of 0.05 nats are interpretable here; they are not
there.

One null worth recording: replacing the sampled 6×32-batch evaluation with a
deterministic sweep of the entire validation split changed the gap sd not at all
(0.0190 vs 0.0196 at 300 steps). Evaluation sampling was never a contributor.
The full-val protocol is kept because it is deterministic, not because it was
needed.

## 10. The head expansion — fixed, and it costs more than it returns

§4's limitation list called the missing within-cluster log marginal "a few lines
would fix it — untested". Tested now.

`expand()` copies the parent row to every child, so the inherited head is uniform
within each cluster. Adding `log p(child|parent)` to the bias supplies exactly
the missing term, and — since every parent has the same number of children — it
leaves the coarse marginal the parent predicted *exactly* unchanged while filling
in the split below. It does what it was designed to do:

| handoff | zero-shot CE, uniform | + log p(child\|parent) | predicted if exact |
|---|---|---|---|
| L1→L4 | 2.218 | **1.209** | 1.217 |
| L4→L8 | 3.745 | **2.634** | 2.708 |
| L8→L12 | 5.190 | **4.545** | 4.485 |
| L12→L13 | 4.978 | **4.890** | 4.751 |

Zero-shot lands on the predicted value to within 0.06 nats at the coarse rungs.
**And the final model is worse:** +0.039 / +0.042 / +0.051 nats at 300 / 600 /
1200 steps, every seed, every budget.

Where it helps and where it does not tracks how well the prior is estimated:

| handoff | children/parent | median train count/child | children with <5 | Δ trained CE |
|---|---|---|---|---|
| L1→L4 | 8 | 1793 | 0.0% | **−0.025** |
| L4→L8 | 16 | 112 | 0.0% | **−0.026** |
| L8→L12 | 16 | 7 | 32.5% | +0.004 |
| L12→L13 | 2 | 3 | 61.2% | +0.032 |

At the final rung the empirical prior is estimated from a median of 3 tokens per
child, and 81 leaves absent from training carry 1.7% of validation tokens.
Holding the parent fixed at `ancu_L12` and varying only the smoothing
(`language/handoff.py`, n=5) shows no interior optimum — the best amount of prior
at L13 is none:

| final expansion | val CE | vs uniform |
|---|---|---|
| uniform (`chain.py`) | 4.7166 | — |
| prior, α=1 | 4.7488 | +0.032 |
| prior, α=20 | 4.7228 | +0.006 |
| prior, α=200 (≈ uniform) | 4.7168 | +0.000 |

So the correction helps where counts are dense and the expansion is wide, and
the rung that decides the headline is neither. **A better zero-shot handoff is
not a better ladder** — the 1.0-nat improvement at the coarse handoffs is gone
within 300 steps of training, and the thin-count damage at the final rung is not.
Another instance of §6: the salient hypothesis was that a defect this large must
be suppressing the effect; the null was that 300 steps of gradient descent
reaches the same place regardless of where the head bias started, and the null
won.


---

## Status of the evidence

| claim | status |
|---|---|
| gate test; void vs entropy floor | holds |
| matched-compute accounting | confirmed in both domains |
| ±0.08 single-seed noise floor | measured across two builds, one seed each — **mazes only** |
| language seed noise ~±0.02 on the gap | measured, n=5, chain rebuilt per seed |
| maze d=16 advantage | **withdrawn** — sign flips across builds |
| maze d=24 advantage | **withdrawn** — inverts under matched compute |
| language ladder win | withdrawn previously; the crossing is now **confirmed at n=5** |
| corrected head expansion helps | **refuted** — fixes zero-shot, costs 0.04–0.05 nats trained |
| retention | **withdrawn** — contradicted where cleanly testable |

Not yet done: multi-seed replication **in mazes**. Language now has n=5 (§9);
the maze ±0.08 floor is still a second *build*, not a second seed, which is the
weaker instrument. Remaining budget-sweep cells (d=24 at 1200/2400, the whole d=16
block) are still running; they affect curve shape, not the verdicts above.

Reproduce with `mazes/sweep9.py` (budget sweep), `mazes/compare9.py`
(retention and rollout), `mazes/inspect9.py` (per-maze predictions),
`language/multiseed.py` (§9 and §10), `language/handoff.py` (§10 smoothing).
Raw numbers in `results/lad9_rerun.json`, `results/sweep9.json`,
`results/retention9.json`, `results/multiseed.json`, `results/handoff.json`.
