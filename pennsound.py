"""Workflow step 0 (poet recordings): cut long poetry readings into clips.
birds.py reuses chunk() for the bird recordings.

Put a poet's readings (MP3s saved from PennSound, archive.org, ...) in data/pennsound/raw/<Poet>/.
Readings rarely may be redistributed, so data/ stays out of git.

  python pennsound.py [--minutes 15]

Readings start with an introduction, so each file's first 45 s is skipped; 30 s chunks are sampled
across every file, up to the same number of minutes per poet.
Then: python extract.py pennsound --per-source
"""
import argparse
import random
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent


def duration(path):
    """Length of an audio file in seconds (0 if ffprobe can't read it)."""
    out = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return float(out) if out else 0.0


def chunk(minutes, length, skip_start, seed, dataset, sr=16000):
    """Cut every file in data/<dataset>/raw/<Name>/ into mono clips of `length` s at `sr` Hz.

    Clips are sampled at random over all of a name's files, up to `minutes` per name, and saved as
    data/<dataset>/audio/<Name>__<file>__<start s>.wav (the folder name becomes the clip's source).
    """
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
    """Command line: chunk all poets in data/<dataset>/raw/."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pennsound")
    ap.add_argument("--minutes", type=float, default=15, help="per poet")
    ap.add_argument("--length", type=int, default=30, help="chunk seconds")
    ap.add_argument("--skip-start", type=float, default=45, help="skip introductions")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    chunk(a.minutes, a.length, a.skip_start, a.seed, a.dataset)


if __name__ == "__main__":
    main()
