# compose/ — nested function resolution

Can 3 layers implement *d* steps of function composition?

```
f6 OF a4 IS a7 .   f11 OF a7 IS a5 .   ...18 distractors...
QUERY : f11 OF f6 OF a4 ANSWER : a5
```

Function tables are **resampled per example**, so nothing is memorisable — the
model has to read the bindings out of its own context, hold the intermediate
`a7` as a pointer it never emits, and use it as the key for the next lookup.
The architecture is held fixed at the project's language model (3 layers, 96
dimensions, 4 heads) and only the composition depth varies, which turns the
"one layer ≈ one step of pointer chasing" story into a testable prediction.

## Two design decisions that decide whether the result means anything

**Distractors must defeat single-cue matching.** The first generator gave the
`outer-function match` heuristic **0.52** — with one distractor per chain
function it is a coin flip, so a model scoring 60% would look impressive and
mean nothing. The distractor budget is now spent on definitions that share a
chain function or a chain argument, so neither cue alone suffices and the model
must match a (function, argument) *conjunction*.

Measured heuristic floors, 4000 examples each (chance = 0.05):

| depth | random const | in-context const | outer-function match | last definition | query-arg match |
|---|---|---|---|---|---|
| 1 | 0.049 | 0.093 | **0.140** | 0.097 | 0.132 |
| 2 | 0.051 | 0.100 | **0.227** | 0.094 | 0.038 |
| 3 | 0.045 | 0.088 | **0.323** | 0.093 | 0.038 |
| 4 | 0.049 | 0.081 | **0.319** | 0.085 | 0.033 |

`outer-function match` is the floor any claimed success must clear.

**Context length is fixed across depths.** Every example carries exactly 20
definitions (~130 tokens) whatever the depth, so a failure at depth 4 cannot be
blamed on a longer context — only on more composition steps.

## Generalisation split

A fixed 20% of ordered function pairs is held out. Training examples never
compose a held-out pair; test examples require the outermost pair to be one.
This tests that composition generalises rather than pair-specific circuits being
memorised. Depth 1 has no pairs, so there train and test coincide.

## Method notes

Loss is taken on the **ANSWER token only** — the most generous setting for the
architecture, since the definitions are random and training to predict them
would only dilute the gradient. Data is generated on the fly, so there is no
finite training set to overfit.

## Running

```bash
python gen.py                    # sample examples + measure the heuristic floors
python train.py <depth> [steps]  # one model per depth; writes ../results/compose.json
```

Depths run best as parallel single-threaded processes
(`for d in 1 2 3 4; do NT=1 python train.py $d 15000 & done`). Note this net
*does* benefit from threading (124 vs 408 ms/step at 4 threads) unlike the
ctx-64 language model, because the context is longer — but four processes on
four cores still wins on wall clock.
