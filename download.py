"""Download a random sample of recitations from the Ganjoor poetry audio dataset.

Usage: python download.py [n_files] [seed]
Files are spread evenly over the dataset's shard folders so no single source dominates.
"""
import json
import random
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = "Reza2kn/ganjoor-chizzled-poetry-review-audio"
API = f"https://huggingface.co/api/datasets/{REPO}"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
OUT = Path(__file__).parent / "data" / "audio"


def main(n=200, seed=0):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.load(urllib.request.urlopen(API))
    by_shard = defaultdict(list)
    for s in meta["siblings"]:
        name = s["rfilename"]
        if name.endswith(".m4a"):
            by_shard[name.split("/")[1]].append(name)

    rng = random.Random(seed)
    per_shard = max(1, n // len(by_shard))
    picks = []
    for shard in sorted(by_shard):
        picks += rng.sample(by_shard[shard], min(per_shard, len(by_shard[shard])))
    picks = picks[:n]

    for i, name in enumerate(picks):
        dest = OUT / f"{name.split('/')[1]}__{Path(name).name}"
        if dest.exists():
            continue
        urllib.request.urlretrieve(RESOLVE + name, dest)
        print(f"[{i + 1}/{len(picks)}] {dest.name}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))
