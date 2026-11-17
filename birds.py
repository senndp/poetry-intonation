"""Download bird song from the CC0 iNaturalist part of SaiPavankumar22/bird-sounds and cut it into clips.

Usage: python birds.py [--per-species 25] [--minutes 6] Turdus_merula Fringilla_coelebs ...
Raw files -> data/birds/raw/<Species>/, clips -> data/birds/audio/ (32 kHz, birds sing up to ~8 kHz)
Then: python extract.py birds --profile bird --per-source
"""
import argparse
import json
import random
import time
import urllib.request
from pathlib import Path

import pennsound

REPO = "SaiPavankumar22/bird-sounds"
API = f"https://huggingface.co/api/datasets/{REPO}"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
ROOT = Path(__file__).parent
DEFAULT = ["Turdus_merula", "Turdus_philomelos", "Luscinia_megarhynchos", "Fringilla_coelebs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("species", nargs="*", default=DEFAULT)
    ap.add_argument("--per-species", type=int, default=25, help="recordings to download per species")
    ap.add_argument("--minutes", type=float, default=6, help="audio per species after chunking")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    files = [s["rfilename"] for s in json.load(urllib.request.urlopen(API))["siblings"]]
    rng = random.Random(a.seed)
    for sp in a.species:
        mine = sorted(f for f in files if f.startswith(f"audio/inat/{sp}/"))
        out = ROOT / "data" / "birds" / "raw" / sp
        out.mkdir(parents=True, exist_ok=True)
        for name in rng.sample(mine, min(a.per_species, len(mine))):
            dest = out / Path(name).name
            if not dest.exists():
                urllib.request.urlretrieve(RESOLVE + name, dest)
                time.sleep(0.2)
        print(f"{sp}: {len(list(out.iterdir()))} recordings")
    # birds sing in short bursts, so shorter clips and no introduction to skip
    pennsound.chunk(a.minutes, 15, 0, a.seed, "birds", sr=32000)


if __name__ == "__main__":
    main()
