"""Bring PennSound readings into the pipeline.

PennSound (writing.upenn.edu/pennsound) offers poets' readings as MP3s for noncommercial, educational use;
rights stay with the poets. Keep data/pennsound/ to yourself and share only generated output.

  python pennsound.py download Ginsberg [--max-files 6]   # MP3 links from the poet's author page
  python pennsound.py chunk [--minutes 15]                # long readings -> 30 s clips per poet

Instead of downloading you can drop MP3s into data/pennsound/raw/<Poet>/ by hand.
Readings are long and start with an introduction, so `chunk` skips each file's first 45 s and samples
30 s chunks spread over every file, up to the same number of minutes per poet.
Then: python extract.py pennsound --per-source && python analyze.py pennsound
"""
import argparse
import random
import re
import subprocess
import time
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urljoin

ROOT = Path(__file__).parent
PAGE = "https://writing.upenn.edu/pennsound/x/{}.php"
HEADERS = {"User-Agent": "poetry-intonation/0.1 (research script; small sample, rate-limited)"}


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60).read()


def download(poet, page, max_files, seed, dataset):
    page = page or PAGE.format(poet)
    html = fetch(page).decode("utf-8", "replace")
    links = sorted({urljoin(page, h) for h in re.findall(r'href="([^"]+\.mp3)"', html, re.I)})
    print(f"{len(links)} mp3 links on {page}")
    out = ROOT / "data" / dataset / "raw" / poet
    out.mkdir(parents=True, exist_ok=True)
    for url in random.Random(seed).sample(links, min(max_files, len(links))):
        dest = out / unquote(url.rsplit("/", 1)[1])
        if not dest.exists():
            dest.write_bytes(fetch(url))
            print("  ", dest.name)
            time.sleep(2)  # be gentle with the archive


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return float(out) if out else 0.0


def chunk(minutes, length, skip_start, seed, dataset, sr=16000):
    raw = ROOT / "data" / dataset / "raw"
    audio = ROOT / "data" / dataset / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for poet_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        candidates = []
        for f in sorted(x for x in poet_dir.iterdir() if x.suffix.lower() in {".mp3", ".m4a", ".wav", ".flac", ".mpga"}):
            d = duration(f)
            start = skip_start if d > skip_start + 2 * length else 0
            # files shorter than one chunk still give one (shorter) clip
            candidates += [(f, s) for s in range(int(start), max(int(d - length), int(start) + 1), length)]
        picks = rng.sample(candidates, min(len(candidates), int(minutes * 60 // length)))
        for f, s in picks:
            stem = re.sub(r"[^A-Za-z0-9]+", "-", f.stem).strip("-")
            dest = audio / f"{poet_dir.name}__{stem}__{s:05d}.wav"
            if not dest.exists():
                subprocess.run(["ffmpeg", "-v", "quiet", "-ss", str(s), "-t", str(length), "-i", str(f),
                                "-ac", "1", "-ar", str(sr), str(dest)], check=True)
        print(f"{poet_dir.name}: {len(picks)} chunks of {length}s from {len(candidates)} candidates")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pennsound")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("poet", help="folder name, and the author page name when --page is omitted")
    d.add_argument("--page", help="author page URL (default writing.upenn.edu/pennsound/x/<poet>.php)")
    d.add_argument("--max-files", type=int, default=6)
    d.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("chunk")
    c.add_argument("--minutes", type=float, default=15, help="per poet")
    c.add_argument("--length", type=int, default=30, help="chunk seconds")
    c.add_argument("--skip-start", type=float, default=45, help="skip introductions")
    c.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "download":
        download(a.poet, a.page, a.max_files, a.seed, a.dataset)
    else:
        chunk(a.minutes, a.length, a.skip_start, a.seed, a.dataset)


if __name__ == "__main__":
    main()
