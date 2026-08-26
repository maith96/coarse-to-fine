"""Download the hip-hop corpus. Lyrics are copyrighted; nothing here is committed.

Source: fpaupier/RapLyrics-Scraper (MIT-licensed scraper, lyrics scraped from
Genius). 36 US rap artists, one file per artist, songs separated by blank lines.
"""
import os, sys, urllib.parse, urllib.request

BASE = ("https://raw.githubusercontent.com/fpaupier/RapLyrics-Scraper/"
        "master/lyrics_US")
ARTISTS = ["A$AP Ant", "A$AP Rocky", "Action Bronson", "André 3000", "Bas",
    "Big L", "Chance The Rapper", "Childish Gambino", "Common",
    "CunninLynguists", "Deniro Farrar", "Drake", "Earl Sweatshirt", "Eazy-E",
    "Eminem", "Ice Cube", "Immortal Technique", "Isaiah Rashad", "J Cole",
    "Jay-z", "Joey Bada", "Kanye West", "Kendrick Lamar", "Lil Wayne", "Logic",
    "Lupe Fiasco", "Mac Miller", "Montana of 300", "NF", "Nas", "Pusha-T",
    "Royce Da 59", "Scarface", "Talib Kweli", "Tyler The Creator",
    "the notorious big"]

os.makedirs("raw", exist_ok=True)
parts = []
for a in ARTISTS:
    fn = f"raw/{a}_lyrics.txt"
    if not os.path.exists(fn):
        url = f"{BASE}/{urllib.parse.quote(a + '_lyrics.txt')}"
        with urllib.request.urlopen(url, timeout=60) as r, open(fn, "wb") as f:
            f.write(r.read())
        print(f"  fetched {a}", flush=True)
    parts.append(open(fn, encoding="utf-8", errors="replace").read())

txt = "\n".join(parts)
open("hiphop.txt", "w", encoding="utf-8").write(txt)
print(f"hiphop.txt  {len(txt)} chars  {len(ARTISTS)} artists")
