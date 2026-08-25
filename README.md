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

Both converge to ~4.56. Same destination, reached sooner. Replicated on a second
seed at 300 steps (+0.363).

**At matched total compute the ladder loses.** The chain cost ~218s of wall clock
(L1 2s, L4 52s, L8 57s, L12 53s, L13 54s) to reach 4.6245. The flat control
reaches **4.5548 in ~216s** by training 1200 steps.

### 5. A second corpus: the marine QA passage (`language/`, `CORPUS=marine`) — memorisation, not a test

The language pipeline is now corpus-parameterised (`language/corpus.py`;
`CORPUS=shake` is the default and reproduces §4 unchanged). The second entry is
`marine.txt` — a 1.8k-token narrative about a storm-damaged research station,
interleaved with 75 of its own comprehension questions and their answers.

Same machinery, rescaled to the corpus: every type is kept (347 types, no UNK
cutoff), so the tree is 9 levels deep and its leaves are **one word each** —
level 9 is the vocabulary itself. Rungs are 1, 3, 5, 7, 9. Clusters come out
coherent (`would of array data if lose happened time more than first severing`).

Gate test — and the first thing worth reporting:

| level | clusters | train loss | val CE | marginal | gain |
|---|---|---|---|---|---|
| 1 | 2 | 0.749 → 0.013 | 2.584 | 0.735 | **−1.849** |
| 3 | 8 | 2.257 → 0.034 | 5.845 | 2.059 | **−3.786** |
| 5 | 32 | 3.597 → 0.039 | 7.271 | 3.617 | **−3.653** |
| 7 | 128 | 5.001 → 0.043 | 8.600 | 5.188 | **−3.412** |
| 9 | 512 | 6.379 → 0.045 | 9.203 | 5.934 | **−3.270** |

The sign is inverted against every row of §4. Train loss falls to ~0.04 at every
rung — a 0.42M-parameter net on 1.8k tokens memorises the stream outright — while
held-out CE lands 3+ nats *worse* than predicting the marginal. The 181-token
held-out tail is the passage's last paragraph, content the model never saw and
cannot infer from 1.6k tokens.

The ladder inherits that, at 300 steps a rung:

| level | ctrl CE | anc CE | delta | anc zero-shot |
|---|---|---|---|---|
| 3 | 5.207 | 5.461 | +0.254 | 3.61 |
| 5 | 6.566 | 8.228 | +1.661 | 6.85 |
| 7 | 7.914 | 9.863 | +1.949 | 9.61 |
| 9 | 8.539 | 11.752 | +3.213 | 11.25 |

The ancestor loses at every rung, and loses *more* the finer the rung — the
opposite of §4's headline before its budget sweep. Nothing here is evidence
against `R`: with a corpus this size the ancestor's extra rungs buy extra passes
over 1.6k tokens, so the chain is simply the more thoroughly overfitted of the
two, and held-out CE is measuring that and nothing else. **A corpus this small
cannot test the hypothesis.** It can only test capacity — which is what the rest
of the run does deliberately.

`fit.py` drops the held-out split and fits level 9 on the whole passage
(3000 steps, warm-started from the chain's finest ancestor), then saves a
self-contained model — weights plus the vocabulary tree — to
`language/models/marine_L9.pt` (1.7 MB, the one `.pt` the repo ships):

| | |
|---|---|
| next-token accuracy over the corpus | **0.983** |
| CE | 0.039 |
| parameters | 0.42M |

`qa.py` then asks it all 75 questions the passage asks, two ways — `ctx`, the
question with its preceding narrative, where the answer could be copied from
context; and `cold`, the question alone at position 0, where it cannot:

| probe | exact answers | token accuracy |
|---|---|---|
| ctx | **74 / 75** | 0.999 |
| cold | **74 / 75** | 0.999 |

Cold matches ctx exactly, so the facts are in the weights, not being copied.
The single miss is the last question — *what happened last in the passage?* —
where the model continues into the longer narrative form of the same sentence,
which is a real ambiguity in the text (that clause appears twice).

`ask.py` is the one-line way to use the saved model — `python ask.py "how old is
Elena?"` → *elena is 34 years old*. All twelve passage questions tried this way
come back correct, and a light rephrasing ("who is Elena's assistant?") survives.

`probe.py` marks the boundary of what that means. Questions the passage never
asks, in words it does use, mostly retrieve a topical but wrong memorised
sentence:

```
Q  what did marcus clean with vinegar?   A  marcus got the vinegar from the station's kitchen.
Q  where is the coastal research station? A  the research station located near crescent bay.
Q  who drank black coffee?               A  marcus harbor seal when he should have been helping.
Q  how many hours of monitoring data were lost?  A  the storm severed the primary data transmission line.
```

`oov.py` goes one step further out, to nouns the corpus never contains. The
vocabulary is closed and the UNK rate is 0.000, so `<unk>` is a token the model
holds an embedding for and has never seen in training; a new noun lands there.
Each probe is paired with the in-vocabulary question it was derived from:

| asked | seen as | answered |
|---|---|---|
| who owns the blue electric **kayak**? | `who owns the blue electric <unk>?` | elena owns the blue electric **pickup truck**. |
| who photographed the **penguin**? | `who photographed the <unk>?` | marcus photographed the **harbor seal**. |
| what did marcus use to clean the **propeller**? | `... clean the <unk>?` | marcus used vinegar to clean the **terminals**. |
| who accompanied **priya** to the station? | `who accompanied <unk> to the station?` | marcus chen accompanied **elena** to the station. |

Five of eight returned *exactly* the twin question's answer: the unknown noun is
ignored and the surrounding frame indexes the memorised sentence. Where the
answer hinges on the noun itself there is no graceful degradation — `who is the
34-year-old <unk>?` returns Margaret Holt rather than Elena. With the frame
unfamiliar too, retrieval goes unrelated (`what did the <unk> <unk>?` → *the
station resumed data transmission at 10:47 am*).

**Confidence does not fall.** Those answers decode at mean token probability
0.90–0.98 against 1.000 for real questions — a gap far too narrow to threshold
on. The model has no representation for *out of distribution*; it is nearly as
certain when it fabricates as when it recalls.

Near-perfect recall of 75 memorised QA pairs, no reliable recombination one step
outside them, and no signal at all that it has left the corpus. That is the
honest description of a 0.42M-parameter transformer fitted to 1.8k tokens, and
it is what the saved model is good for.

### 6. Fixing the metric: a held-out QA split, and paraphrase augmentation

§5's split was the last 10% of *tokens* — a paragraph the model had never seen
in any form — which is why its val CE sits 3 nats worse than the marginal and
its ladder table means nothing. This replaces it with a split that can
distinguish memorising from reading.

**The split.** `augment.py` holds out 15 of the 75 QA pairs, chosen from the 56
whose answers are recoverable from the narrative (answer content words present
in the block that precedes them; `parse_marine.py` scores this), spread across
all 7 paragraphs. The narrative stays in training — only the question and its
answer are deleted. A model that has learned to read the passage can answer
them; a model that has memorised strings cannot.

**The augmentation.** 24 document variants, blocks shuffled, questions within a
block shuffled, each question rendered in one of 6–7 paraphrase forms
(`do you know …?`, `tell me …`, `which person …`, `for what reason …`) and each
answer in one of two. 1.8k tokens becomes 55k. One further change matters: a
question is only answerable by reading if the narrative is still in the context
window, and block 3 runs 18 QA pairs past its narrative, so the narrative is
**re-anchored every 4 questions** and the context is widened to 192. Both arms
share one vocabulary tree built over the union of all text, so the A/B is not
confounded by vocabulary.

Two models, 4000 steps each, both from scratch, identical context and batch:
`marine_base` (original text, held-out pairs deleted) and `marine_aug`.

| prompt | base exact | **aug exact** | base F1 | **aug F1** |
|---|---|---|---|---|
| **held-out**, narrative + question | 0.000 | **0.000** | 0.332 | **0.365** |
| **held-out**, narrative + paraphrase | 0.000 | **0.000** | 0.328 | **0.358** |
| **held-out**, question alone | 0.000 | **0.000** | 0.357 | **0.400** |
| seen pairs, narrative + question | 0.667 | **0.933** | 0.798 | **0.993** |
| seen pairs, narrative + paraphrase | 0.667 | **0.933** | 0.798 | **0.993** |
| seen pairs, question alone | 1.000 | **0.933** | 1.000 | **0.985** |

**The metric now works, and both arms score zero on it.** Not one held-out
question is answered correctly by either model. Token F1 of 0.33–0.40 is the
signature of retrieving a topically adjacent memorised sentence, which is
exactly what the outputs show — *Where did the seal appear?* returns "marcus
used his smartphone to photograph the seal", *Who photographed the harbor seal?*
returns "margaret holt is the director of the ocean preservation foundation".

**The augmentation bought the thing it can buy, and not the other.** Look at the
seen-pair rows: the control is perfect (1.000) when a question is presented
exactly as trained and drops to 0.667 when the same question is preceded by its
narrative — it has memorised a token sequence, and moving the question breaks it.
The augmented model is 0.93 in all three formats. So augmentation delivered
invariance to question form and position. It delivered no ability to answer a
question it had not been trained on.

**Why, and it is not the model size.** Every training answer in this design is
*also* memorisable — the facts are constant across all 24 variants, so
memorisation stays a sufficient strategy for every single training example, and
it is the strategy gradient descent finds first. Nothing in the data ever
punishes a model for failing to read. To force extraction the facts themselves
have to vary: randomise the entities per document (Elena/Priya, 34/41,
blue/green) so no fixed answer survives, and reading the context becomes the
only strategy that fits the training set. That is the experiment this one
implies, and it is untested.

### 7. Randomising the facts: reading finally appears, and diversity is the knob

§6 ended with a diagnosis rather than a result: facts were constant across every
document, so memorisation stayed sufficient for the whole training set and
nothing ever pushed the model to read. `randomize.py` removes that option. Each
document gets its own cast — Elena/Priya/Ingrid/Rosa, blue/green/silver/crimson,
34/41/29/52, every clock time, quantity and place name — applied consistently to
narrative, questions and answers. Four facts are *derived* and are recomputed
rather than substituted, or the passage would contradict itself: Margaret's
arrival (resume time + offset, meridiem included), the counterfactual hours lost
(stated loss + how much earlier the line severed), "the third cup" and "Three
people" (cups + 1). 60 assignments were checked for arithmetic consistency and
for surviving originals.

The split is §6's, unchanged. Evaluation uses casts never trained on, and the
metric is **slot accuracy**: of the randomised values a gold answer states and
its question does *not* contain, does the model produce the value this document's
cast assigns (`hit`), a value from another document (`wrong-cast`), or nothing
(`absent`)? Values copyable from the question are excluded — they prove nothing.
`cold` (question, no narrative) is the control: if the model reads, hits must
collapse when the narrative is removed.

Two arms, same corpus size (~55k tokens), same 4000 steps, differing only in how
often the cast changes — once per document (12 distinct casts, since the
balanced selector cycles) or once per re-anchored block of 4 questions (468):

| | slot hit **with** narrative | slot hit **without** | exact answer with | without |
|---|---|---|---|---|
| 12 casts | 13/54 = 0.241 | 11/54 = 0.204 | 0.086 | 0.086 |
| **468 casts** | **33/54 = 0.611** | 18/54 = 0.333 | **0.400** | 0.200 |

*(trained question types, casts never seen, n=35 items)*

**Cast diversity is the binding constraint, not model size.** At 12 casts the
narrative might as well not be there — 0.241 against 0.204, and whole answers
correct 8.6% of the time either way. At 468 casts the same architecture on the
same token budget doubles its hit rate when the narrative is present, halves
wrong-cast fills (0.556 → 0.315), and gets 40% of whole answers exactly right
against 20% without context. Reading emerged only once memorisation stopped
paying:

```
Q     Who accompanied Rosa to the station?      gold  Dmitri Adeyemi accompanied Rosa ...
  with narrative   dmitri adeyemi accompanied rosa to the station.      <- correct cast
  no narrative     kwame silva accompanied rosa to the station.         <- another document's
```

What has *not* moved is generalisation to unseen questions. Held-out questions
under fresh casts stay at **0.000 exact** in both arms; their slot hits rise with
context (13/59 vs 3/59) but 71% of the time the answer does not state the value
at all. The model learned to bind entities it was trained to ask about. It did
not learn to answer a question it has never been asked.

**Copy distance is not a confound here**, though it would be an obvious one. Both
arms re-anchor the narrative every 4 questions regardless of when the cast
changes, and the generated corpora bear that out: 552 narrative anchors each,
question-to-anchor distance median 106 words against 107, 90th percentile 312
against 315. The arms differ in cast diversity and in nothing else measurable.

---

## Reproducing

```bash
pip install -r requirements.txt
cd language && curl -sL -o shake.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python vocab.py && python gatelm.py 1,4,8,12,13 999
python chain.py 9999 && python sweep.py 9999
```

Marine corpus (§5) — `marine.txt` is in the repo, so there is nothing to fetch.
`CORPUS` selects everything; checkpoints and json are tagged by it, so the two
corpora coexist:

```bash
cd language
export CORPUS=marine
python vocab.py && python gatelm.py all 999 && python chain.py 9999
python fit.py 999 3000 chain          # full-corpus fit -> models/marine_L9.pt
python qa.py 8 && python probe.py     # 75 passage questions, then paraphrases
python oov.py                         # nouns the corpus does not contain
```

The whole marine run is about six minutes on four CPU cores. `fit.py`'s model
ships with the repo, so `qa.py` and `probe.py` run without retraining.

The held-out QA experiment (§6) builds its own corpora and trains both arms —
about twelve minutes:

```bash
cd language && python augment.py            # split + 24 augmented variants
for c in marine_aug marine_base; do
  CORPUS=$c python vocab.py && CORPUS=$c python fit.py 900 4000 scratch
  CORPUS=$c python evalqa.py
done
```

Entity randomisation (§7):

```bash
cd language
python augment_rand.py 24 0 chunk        # 468 casts; 'doc' gives the 12-cast arm
CORPUS=marine_randc python vocab.py
CORPUS=marine_randc python fit.py 900 4000 scratch
CORPUS=marine_randc python readtest.py   # slot accuracy, context present vs removed
python randomize.py 7                    # inspect one randomised cast
```

`vocab.py` now pins the ARPACK start vector. Without it `svds` seeds itself
randomly and the vocabulary tree differs between runs of the same script —
worth knowing before comparing any two trees.

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

- Decisive comparisons are **single-seed** except where noted, and the measured
  single-seed noise floor (~±0.08 on maze deltas) is wider than most effects
  claimed here. Multi-seed replication is the first thing this project needs;
  none of it has been done.
- Language work is 290k tokens of Shakespeare, a 96-dim 3-layer model, 64-token
  context. The *pattern* has three independent confirmations; the magnitudes
  would not survive a real corpus.
- The language crossing point (anc/ctrl at 1200 steps, −0.007) is one seed, and
  −0.007 is far inside the noise floor above. "Both converge to ~4.56" survives
  that; the exact crossing does not.
- The vocabulary bisection balances by word **type**, not token mass, so level 1
  is 96/4 and the coarse rungs carry little information. Fixing this should
  steepen the ladder — untested.
- The marine corpus (§5) is 1.8k tokens against a 0.42M-parameter model. Its
  ladder table is reported for completeness only — at that ratio held-out CE
  measures overfitting, not transfer, so §5 says nothing either way about `R`.
  Its saved model recalls the passage; it does not generalise past it.
- The head expansion inherits the parent row without the within-cluster log
  marginal, making zero-shot CE *worse* than predicting corpus frequencies
  (5.137 vs 5.072 at L12). A few lines would fix it — untested.
