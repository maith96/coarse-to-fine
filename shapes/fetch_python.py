"""Build the Python corpus from locally installed source. No network needed.

Writes py_corpus.txt (~2MB, gitignored -- regenerate rather than commit).
"""
import glob, os, random
ROOTS=["/usr/lib/python3.11","/usr/local/lib/python3.11/site-packages"]
files=[]
for r in ROOTS:
    files+=[f for f in glob.glob(r+"/**/*.py",recursive=True)
            if os.path.getsize(f)>2000 and "test" not in f.lower()]
random.Random(0).shuffle(files)
buf=[];n=0
for f in files:
    try: t=open(f,encoding="utf-8",errors="ignore").read()
    except Exception: continue
    if "def " not in t: continue
    buf.append(t); n+=len(t)
    if n>2_000_000: break
open("py_corpus.txt","w").write("\n".join(buf))
print(f"{len(buf)} files, {n:,} chars -> py_corpus.txt")
