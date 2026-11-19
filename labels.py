"""Workflow step 2 (labels): turn intonation units into labels for the Markov chain.

Every unit becomes a symbolic state  LEVEL-SHAPE-LENGTH:
  LEVEL  L / M / H      mean pitch below -2 st, within +-2 st, above +2 st of the reader's median
  SHAPE  fall / flat / rise   end-start below -1.5 st, within, above +1.5 st
  LENGTH s / l          shorter / longer than 250 ms (≈ median)
Gaps become pause tokens:  '|' short pause (120-450 ms), '||' phrase break (>450 ms or clip end).
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
SHORT_PAUSE, PHRASE_BREAK = 0.12, 0.45


def state_of(u):
    """Label of one unit, e.g. 'M-fall-l' (thresholds in the module docstring)."""
    level = "L" if u["level"] < -2 else "H" if u["level"] > 2 else "M"
    delta = u["end"] - u["start"]
    shape = "fall" if delta < -1.5 else "rise" if delta > 1.5 else "flat"
    return f"{level}-{shape}-{'s' if u['dur'] < 0.25 else 'l'}"


def tokenize(clip):
    """Clip -> list of (token, unit-or-None)."""
    seq = []
    for u in clip["units"]:
        seq.append((state_of(u), u))
        gap = u["gap"]
        if gap is None or gap > PHRASE_BREAK:
            seq.append(("||", None))
        elif gap > SHORT_PAUSE:
            seq.append(("|", None))
    return seq


def load_clips(dataset):
    """All clips of a dataset as written by extract.py."""
    return json.loads((ROOT / "data" / dataset / "units.json").read_text())
