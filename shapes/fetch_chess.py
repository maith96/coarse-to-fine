"""Build the chess corpus: TCEC engine games -> SAN move sequences.

Downloads the TCEC master archive (~100MB of PGN, gitignored) and parses it down
to a 300k-token move corpus matched in size to the Shakespeare one, so the
three-domain comparison is controlled.

  pip install chess
  python fetch_chess.py
"""
import io, os, glob, random, subprocess, collections
import chess, chess.pgn
B="https://raw.githubusercontent.com/TCEC-Chess/tcecgames/master/master-archive"
ROUNDS=["Round32","Round16","Octofinal","Quarterfinal","Quaterfinal",
        "Semifinal","Final","Bronze"]
names=[f"TCEC_Cup_{n}_{r}.pgn" for n in range(1,17) for r in ROUNDS]
names+= [f"TCEC_Match_{n}.pgn" for n in range(1,6)]
for f in names:
    if os.path.exists("pg_"+f): continue
    r=subprocess.run(["curl","-sS","-f","-m","90","-o","pg_"+f,f"{B}/{f}"],
                     capture_output=True)
    if r.returncode!=0 and os.path.exists("pg_"+f): os.remove("pg_"+f)
print(f"{len(glob.glob('pg_*.pgn'))} PGN files on disk")

CAP=300_000
files=sorted(glob.glob("pg_*.pgn")); random.Random(0).shuffle(files)
games=[]; plies=0
for f in files:
    if plies>=CAP: break
    h=io.StringIO(open(f,encoding="utf-8",errors="ignore").read())
    while plies<CAP:
        try: g=chess.pgn.read_game(h)
        except Exception: break
        if g is None: break
        b=g.board(); mv=[]
        try:
            for m in g.mainline_moves(): mv.append(b.san(m)); b.push(m)
        except Exception: continue
        if len(mv)>=20: games.append(mv); plies+=len(mv)+1
with open("chess_corpus.txt","w") as fh:
    for g in games: fh.write(" ".join(g)+" <eog>\n")
c=collections.Counter(m for g in games for m in g)
print(f"{len(games)} games, {plies:,} tokens, {len(c)} distinct moves -> chess_corpus.txt")
