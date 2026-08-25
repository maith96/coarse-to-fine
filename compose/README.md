# compose/ — nested function resolution

Can 3 layers implement *d* steps of function composition?

```
f6 OF a4 IS a7 .   f11 OF a7 IS a5 .   ...distractors...
QUERY : f11 OF f6 OF a4 ANSWER : a5
```

Function tables are **resampled per example**, so nothing is memorisable — the
model has to read the bindings out of its own context, hold the intermediate
`a7` as a pointer it never emits, and use it as the key for the next lookup.
The architecture is held fixed at the project's language model (3 layers, 96
dimensions, 4 heads) and only the composition depth varies, which turns the
"one layer ≈ one step of pointer chasing" story into a testable prediction.

## What the runs actually found

The predicted failure mode — chases a step or two, then runs out of layers and
emits the intermediate — **did not happen**. The observed failure is elsewhere,
and the diagnostics below are what located it.

**1. The first bottleneck was haystack size, not depth.** At 20 definitions per
example nothing learned at any depth or learning rate tried: depth 1 sat at
chance for 4000 steps with the loss essentially flat (2.806 → 2.805 at lr 3e-3,
and no better at 5e-4 or 1e-4). At 6 definitions the same model learned readily.
Depth 1 is a single conjunctive lookup with no composition in it at all, so this
is a property of retrieval over a long context, not of chaining. Learning rate
mattered secondarily but sharply — at 12 definitions, lr 5e-4 reached 0.92 while
lr 1.5e-3 stalled at 0.25.

**2. At 12 definitions the depth curve collapses after one step.**

| depth | held-out | strongest single-cue floor | argument oracle |
|---|---|---|---|
| 1 | **0.920** | 0.208 | 0.191 |
| 2 | 0.405 | 0.353 | 0.352 |
| 3 | 0.499 | 0.459 | 0.502 |

Depth 1 clears its floor by a factor of four. Depths 2 and 3 sit within a few
points of theirs, and depth 3 lands *exactly* on the argument oracle.

**3. The pointer chase is not what fails.** Every wrong answer was attributed to
the definition it was copied from, described relative to the true chain. At
depth 2, 41.4% are correct and **58.4% come from a row keyed on `x1` — the
correct final argument — but carrying the wrong function.** Together that is
99.8% of predictions landing on the right row-key. The intermediate value itself
is emitted essentially never (0.000). At depth 3, correct (0.505) plus rows
keyed on the correct final argument `x2` (0.368) covers 87%, with only 4.5%
landing on an earlier chain argument.

So the model finds the right key at every step. What it cannot do is the final
conjunctive match: having located `x_{d-1}`, pick the row that also matches the
outer function. It then guesses among the rows sharing that argument, which is
precisely the argument oracle it scores.

That is a strange result next to depth 1, where the *same* conjunctive match
succeeds 92% of the time. The difference between the two cases is not the number
of competing rows — depth 1 has *more* (7.0 vs 3.2) — it is that at depth 1 the
argument is a literal token in the prompt, while at depth 2 it is a value the
model computed. **The binding fails when one of the two keys is internal rather
than literal.** Depth is close to free; conjunctive matching against a computed
key is not.

## A generator flaw that would have faked the depth curve

`gen.py` splits the distractor budget evenly across chain positions, so the
number of definitions competing for the *final* lookup shrinks as depth grows:
7.0 rows at depth 1, 3.2 at depth 2, 2.2 at depth 3. The argument oracle
therefore *climbs with depth on its own* — 0.19 / 0.35 / 0.50 — so a rising
accuracy-vs-depth curve on this generator is partly an artefact of its baseline
moving underneath it. At 6 definitions it is fatal: depth 3 leaves 1.05 rows on
the final argument, making the task solvable without reading functions at all.
A sweep launched at that setting was killed rather than reported.

`gen2.py` pins the local ambiguity of every lookup independent of depth: exactly
`KX=4` competing rows share the final argument, exactly `KF=4` share the outer
function, every intermediate step gets one argument- and one function-competitor,
and all distractor values are drawn outside the chain so a wrong row can never
accidentally rejoin the correct one. Total definitions are fixed, so context
length is fixed too. Measured floors are then flat (4000 examples each,
chance = 0.05):

| depth | argument oracle | outer-fn match | query-arg match | last definition |
|---|---|---|---|---|
| 1 | 0.180 | 0.192 | 0.202 | 0.058 |
| 2 | 0.189 | 0.189 | 0.000 | 0.054 |
| 3 | 0.197 | 0.200 | 0.000 | 0.053 |
| 4 | 0.196 | 0.201 | 0.000 | 0.052 |

Any accuracy difference across depth measured on `gen2` is a real difference.

## Generalisation split

A fixed 20% of ordered function pairs is held out. Training examples never
compose a held-out pair; test examples require the outermost pair to be one.
Depth 1 has no pairs, so there train and test coincide by construction — the
identical numbers in the depth-1 row are that, not a bug. At depths 2 and 3 the
seen/held-out gap is within noise (0.399/0.405 and 0.511/0.499), so memorisation
of pair-specific circuits is not what is being measured either way.

## Method notes

Loss is taken on the **ANSWER token only** — the most generous setting for the
architecture, since the definitions are random and training to predict them
would only dilute the gradient. Data is generated on the fly, so there is no
finite training set to overfit. Evaluation is restricted to constant tokens,
and reported accuracies use 2048 examples (s.e. ≈ 0.011).

## Running

```bash
python gen2.py                       # sample examples + measure the floors
GEN=2 NDEF=18 CTX=132 LR=5e-4 python train.py <depth> [steps]
python diag.py                       # failure modes: is the chain truncated?
python attrib.py                     # which row was the wrong answer copied from?
python oracle.py                     # argument oracle + single-cue floors
```

Depths run best as parallel single-threaded processes
(`for d in 1 2 3 4; do NT=1 ... python train.py $d 25000 & done`). This net
*does* benefit from threading (124 vs 408 ms/step at 4 threads) unlike the
ctx-64 language model, because the context is longer — but four processes on
four cores still wins on wall clock.
