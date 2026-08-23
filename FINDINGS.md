# Findings

What I would defend from this project, separated from what it set out to show.
The mechanism failed; the methodology is the deliverable.

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

**A curriculum must be charged for its own construction.** Violated twice here
before being caught, and it inverted the headline result.

The language ladder looked like −0.26 nats until the sweep showed the gap
decaying to zero by 1200 steps, at which point the flat control matched the
chain's *total* wall clock and beat it outright. The 9×9 maze result (+0.07)
carries the same unexamined asymmetry — the ancestor had five rungs of compute
to the control's one — and I would not now claim it without a budget sweep.

**Corollary:** any curriculum evaluated only under a starved baseline will look
like it works. Sweep the budget or the result means nothing.

## 4. Reachable ≠ representable

A transformer can *represent* multiplication — the weights could be written by
hand. The obstruction is that the solution is **unreachable by gradient descent
from random init**. Different claim, different fix: not a bigger model, but a
different training signal (intermediate supervision on carries, curriculum over
carry depth).

## 5. Base-rate artifacts — the error log

Five instances. All the same failure: reasoning from the salient hypothesis
instead of the null. The first four were caught by computing the null in code;
the fifth was not, because it was never written down as a number.

1. **All-zeros sampler.** Level 2 hit "95% exact at step 20" — the top bits were
   almost always zero under the original magnitude sampling. Fixed by
   normalising operands.
2. **τ = 20× at L4.** Spectacular, and pure old-bit retention. Zero-shot on the
   *new* bits was chance at every rung. Decomposing old vs new killed it.
3. **Level-1 negative lift.** Val accuracy 0.9622 vs majority 0.9629 on a rung
   whose train loss had fallen 0.817 → 0.131 (below `ln 2`). Clusters were 96/4
   by token mass; accuracy was the wrong metric. Cross-entropy vs the marginal
   was the right one.
4. **Compute-confounded ladder win.** Section 3 above.
5. **Two GPT-2 predictions, both wrong.** Predicted a *Tempest* continuation
   from `miranda : \n certainly , sir , i can .` — got Space Station 13 server
   logs, because the escaped-newline serialisation matched chat-log format far
   more strongly than the content matched Shakespeare. Reformatted properly,
   predicted Prospero — got Star Trek, because `MIRANDA:` in caps is generic
   screenplay format and the line contains no archaic English. Register
   prediction held; content prediction failed twice.

**The discipline that works in code does not automatically transfer to
prose predictions.** If a claim is worth making, write down the null first.

## 6. What the theory looks like after contact

The abstraction relation is **real**: short-horizon policies transfer zero-shot
to long horizons (+0.20 to +0.41 lift); vocabulary clusters are semantically
coherent. What fails is converting that relation into a training schedule.

The likely reason, visible in the generation samples: **coarse levels teach the
high-frequency structure a flat model acquires cheaply anyway.** A 1.9M-parameter
model learns "a speaker name goes after a blank line" — and so does the control,
within a few hundred steps. Neither touches the residual (*which* name), where
all the remaining entropy lives.

GPT-2 XL illustrates the same split at 800× the scale: perfect Elizabethan
register, invented characters (`CONSTANTINE`, `ARIES`) from a near-verbatim
*Tempest* cue. Style is the coarse level and scales beautifully; specific
identity is the fine level and does not.

## 7. The one effect never compute-confounded

**Retention.** Ancestor-initialised nets held ancestral competence better than
independently trained ones in nearly every rung of all three precision ladders,
strongest where the new task was hardest (+0.09 to +0.10 at L4, where neither
condition could learn anything new).

It costs nothing extra to measure, so no compute asymmetry can explain it. And
it appeared with **plain sequential initialisation** — no freezing, no
distillation, no organelle machinery. The elaborate mechanism was defending
something that comes for free.

If there is a live direction left, it is this one: reframe from "train faster"
to **"accumulate capability without eroding it"** — closer to what evolution
actually solved, and an open problem at any scale.
