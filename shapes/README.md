# shapes/ — the same net, three domains

An offshoot of the language work. §4 of the language analysis found that what a
tiny model learns is bounded by **what recurs**, not by form versus meaning. That
predicts which domains a two-minute model will capture — and, more usefully,
where it must fail: on structure a short window cannot fake.

Three domains, identical architecture (96-dim, 3 layers, 64-token context,
1.9M parameters), identical 1200 steps, corpora matched to within 4%.

| domain | tokens | train | perplexity | non-local structure | how far it holds |
|---|---|---|---|---|---|
| Python source | 319k | 91s | **10.2** | bracket identity | ~7 tokens |
| Shakespeare | 290k | 130s | 103 | verb agreement | **1 token** |
| Chess (TCEC) | 300k | 58s | 211 | move legality | ~6 plies |

**Perplexity measures how much shape a domain has. The distance tests measure
how much substance that shape can fake.** They are unrelated axes.

## The findings

**Python** tracks the most recent bracket robustly — 0.99 separation between
`(` and `[` when adjacent, still 0.71 at seven tokens, decaying by fifteen.
Compare Shakespeare's verb agreement, which falls from 0.98 to 0.25 the instant
*one* token intervenes: that is a bigram, this is not. But it is not a stack
either — only 2 of 20 200-token samples end with brackets balanced.

**Chess** is the sharpest case, because legality depends on the entire history
and `python-chess` adjudicates it exactly. Top-1 legality falls from 0.987 at
ply 1 to 0.300 by ply 100 (random-SAN null: 0.08–0.13). In free play the median
game dies at ply 6. The longest legal game was a real Sicilian —
`1.e4 c5 2.Nf3 Nc6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 b6 6.Be2 g6` — and then it played
`e4`, with its own pawn already there. **It replays moves it has already made.**
There is no board in the model, only a distribution over moves that appear in
openings.

**Elo** (`chesselo.py`, results in `../results/chess_elo.json`): playing its own
tokens it scores **0.000** against a uniformly random legal mover over 30 games —
it forfeits, so it has no rating. Masked to legal moves, isolating judgment from
board-tracking: +85 Elo over random (0.620 ± 0.050, n=100), level with a 1-ply
material-greedy bot (0.500 ± 0.050, n=100), and 0/20 against Stockfish 16 at its
calibrated floor of 1320. Those middle rows are non-transitive — greedy itself
beats random 0.850, i.e. +301 — because the model cannot see the board and so
never punishes a random opponent for hanging a queen. Roughly **350–600 Elo with
the legality crutch**, wide band, and unrated without it.

## The mechanism

Distance-robust structure is learned **when the corpus makes it obligatory**. In
Shakespeare `thou` is always adjacent to its verb, so a bigram sufficed and a
bigram is what was learned. In Python brackets are routinely separated by
variable material, so a bigram fails — and something better appeared. Same
parameters, same two minutes. It is a property of the data, not of capacity.

## Running it

```bash
pip install chess                    # for the chess domain
python fetch_python.py && python pyshape.py
python fetch_chess.py  && python chesshape.py
apt-get install stockfish && python chesselo.py
```

Corpora and the ~100MB TCEC download are gitignored — the fetch scripts
regenerate them. `fetch_python.py` needs no network; `fetch_chess.py` pulls from
the TCEC master archive on GitHub.

Full write-up with samples: [`../docs/shape-of-shakespeare.html`](../docs/shape-of-shakespeare.html).
