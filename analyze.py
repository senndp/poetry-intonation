"""Find relations between intonation units.

Every unit becomes a symbolic state  LEVEL-SHAPE-LENGTH:
  LEVEL  L / M / H      mean pitch below -2 st, within +-2 st, above +2 st of the reader's median
  SHAPE  fall / flat / rise   end-start below -1.5 st, within, above +1.5 st
  LENGTH s / l          shorter / longer than 250 ms (≈ median)
Gaps become pause tokens:  '|' short pause (120-450 ms), '||' phrase break (>450 ms or clip end).

Usage: python analyze.py [dataset]   ->  analysis/<dataset>/*.png + summary.txt
With 2-10 sources (e.g. poets) it also compares them -> analysis/<dataset>/sources.png
python analyze.py compare ganjoor pennsound birds   ->  analysis/compare/ (datasets side by side;
a dataset with more than 10 sources counts as one source)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
SHORT_PAUSE, PHRASE_BREAK = 0.12, 0.45
LEVELS, SHAPES, LENGTHS = ["L", "M", "H"], ["fall", "flat", "rise"], ["s", "l"]
STATES = [f"{a}-{b}-{c}" for a in LEVELS for b in SHAPES for c in LENGTHS] + ["|", "||"]


def state_of(u):
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
    return json.loads((ROOT / "data" / dataset / "units.json").read_text())


def phrases(seq):
    """Split a token sequence at phrase breaks into lists of units."""
    out, cur = [], []
    for tok, u in seq:
        if tok == "||":
            if cur:
                out.append(cur)
            cur = []
        elif u is not None:
            cur.append(u)
    return out


def neighbour_jumps(seqs):
    """Pitch jump (next start - previous end) between units inside a phrase."""
    jumps = []
    for s in seqs:
        us = [u for _, u in s if u is not None]
        jumps += [b["start"] - a["end"] for a, b in zip(us, us[1:]) if a["gap"] is not None and a["gap"] < PHRASE_BREAK]
    return np.array(jumps)


def declination(seqs):
    """Mean level at 10 positions through phrases of >=3 units, and the linear drift start->end."""
    bins = np.linspace(0, 1, 11)
    pos_level = defaultdict(list)
    for p in (p for s in seqs for p in phrases(s) if len(p) >= 3):
        for i, u in enumerate(p):
            pos_level[np.digitize(i / (len(p) - 1), bins[1:-1])].append(u["level"])
    curve = [np.mean(pos_level[b]) if pos_level[b] else np.nan for b in range(10)]
    x = np.linspace(0, 1, 10)
    ok = ~np.isnan(curve)
    return curve, np.polyfit(x[ok], np.array(curve)[ok], 1)[0]


def compare_sources(clips, out):
    """One row of intonation statistics per source, plus a comparison figure."""
    by = defaultdict(list)
    for c in clips:
        by[c["source"]].append(c)
    rows = []
    for src, cs in sorted(by.items()):
        seqs = [tokenize(c) for c in cs]
        us = [u for c in cs for u in c["units"]]
        gaps = np.array([u["gap"] for u in us if u["gap"] is not None])
        shapes = Counter(state_of(u).split("-")[1] for u in us)
        curve, drift = declination(seqs)
        rows.append({
            "source": src, "minutes": sum(c["duration"] for c in cs) / 60, "units": len(us),
            "median_hz": np.median([c["median_hz"] for c in cs]),
            "range": np.std([u["level"] for u in us]),  # how far the voice spreads around its centre
            "jump": np.mean(np.abs(neighbour_jumps(seqs))),
            "rate": len(us) / sum(c["duration"] for c in cs),  # voiced units per second
            "pauses": np.mean(gaps > SHORT_PAUSE),
            "rise": shapes["rise"] / len(us), "fall": shapes["fall"] / len(us),
            "dur": np.median([u["dur"] for u in us]) * 1000,
            "drift": drift, "curve": curve,
            # average unit shape with each unit's own mean removed
            "shape": np.mean([np.array(u["contour"]) - np.mean(u["contour"]) for u in us], 0),
        })

    lines = ["", "Per source:",
             f"  {'source':22s} {'min':>5s} {'Hz':>5s} {'range':>6s} {'jump':>5s} {'units/s':>7s} "
             f"{'pauses':>6s} {'rise':>5s} {'fall':>5s} {'dur ms':>6s} {'drift':>6s}"]
    for r in rows:
        lines.append(f"  {r['source'][:22]:22s} {r['minutes']:5.1f} {r['median_hz']:5.0f} {r['range']:6.2f} "
                     f"{r['jump']:5.2f} {r['rate']:7.2f} {r['pauses']:6.0%} {r['rise']:5.0%} {r['fall']:5.0%} "
                     f"{r['dur']:6.0f} {r['drift']:+6.2f}")
    lines.append("  range = sd of unit level (st), jump = mean |pitch jump| between units (st),"
                 " drift = declination over a phrase (st)")

    names = [r["source"] for r in rows]
    colors = plt.cm.tab10(np.arange(len(rows)))
    fig, axs = plt.subplots(2, 4, figsize=(20, 8.5))
    panels = [("range", "Pitch range (sd, st)"), ("jump", "Mean jump between units (st)"),
              ("rate", "Voiced units per second"), ("pauses", "Share of gaps that are pauses"),
              ("rise", "Share of rising units"), ("dur", "Median unit duration (ms)")]
    for ax, (key, title) in zip(axs.flat, panels):
        ax.bar(range(len(rows)), [r[key] for r in rows], color=colors)
        ax.set_xticks(range(len(rows)), names, rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
    for r, col in zip(rows, colors):
        axs[1, 2].plot(np.linspace(0, 1, 10), r["curve"], "o-", color=col, label=r["source"])
        axs[1, 3].plot(r["shape"], "o-", color=col, label=r["source"])
    axs[1, 2].set_title("Declination through a phrase (st)")
    axs[1, 3].set_title("Average unit shape (st, own mean removed)")
    axs[1, 3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sources.png", dpi=110)
    plt.close(fig)
    return lines


def main(dataset="ganjoor"):
    OUT = ROOT / "analysis" / dataset
    OUT.mkdir(parents=True, exist_ok=True)
    clips = load_clips(dataset)
    seqs = [tokenize(c) for c in clips]
    units = [u for c in clips for u in c["units"]]
    lines = [f"{len(clips)} clips, {len(units)} voiced units, "
             f"{sum(c['duration'] for c in clips) / 60:.1f} minutes of recitation", ""]

    # --- state frequencies and first-order transitions
    counts = Counter(t for s in seqs for t, _ in s)
    idx = {s: i for i, s in enumerate(STATES)}
    T = np.zeros((len(STATES), len(STATES)))
    for s in seqs:
        for (a, _), (b, _) in zip(s, s[1:]):
            T[idx[a], idx[b]] += 1
    P = T / np.maximum(T.sum(1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(P, cmap="magma", vmin=0, vmax=min(0.5, P.max()))
    ax.set_xticks(range(len(STATES)), STATES, rotation=90, fontsize=7)
    ax.set_yticks(range(len(STATES)), STATES, fontsize=7)
    ax.set_xlabel("next state")
    ax.set_ylabel("current state")
    ax.set_title("Transition probabilities P(next | current)")
    fig.tight_layout()
    fig.savefig(OUT / "transitions.png", dpi=120)
    plt.close(fig)

    lines.append("Most frequent states:")
    lines += [f"  {s:12s} {n:6d}  ({n / sum(counts.values()):.1%})" for s, n in counts.most_common(10)]

    # --- pitch movement between neighbouring units
    jumps, levels = neighbour_jumps(seqs), []
    for s in seqs:
        us = [u for _, u in s if u is not None]
        levels += [b["level"] - a["level"] for a, b in zip(us, us[1:]) if a["gap"] is not None and a["gap"] < PHRASE_BREAK]
    levels = np.array(levels)
    corr = np.corrcoef([u["level"] for u in units[:-1]], [u["level"] for u in units[1:]])[0, 1]
    lines += ["", "Pitch movement between neighbouring units (within a phrase):",
              f"  jump end->next start: median {np.median(jumps):+.2f} st, "
              f"{np.mean(np.abs(jumps) < 1):.0%} within 1 st, {np.mean(np.abs(jumps) > 4):.0%} leap over 4 st",
              f"  level change: mean {levels.mean():+.2f} st, sd {levels.std():.2f} st",
              f"  level autocorrelation (lag 1): {corr:.2f}"]

    # --- declination and phrase-final behaviour
    all_phr = [p for s in seqs for p in phrases(s) if len(p) >= 3]
    final_shapes, other_shapes = Counter(), Counter()
    for p in all_phr:
        for i, u in enumerate(p):
            (final_shapes if i == len(p) - 1 else other_shapes)[state_of(u).split("-")[1]] += 1
    decl, slope = declination(seqs)
    lines += ["", f"Phrases (>=3 units): {len(all_phr)}, mean {np.mean([len(p) for p in all_phr]):.1f} units",
              f"  declination: pitch drifts {slope:+.2f} st from phrase start to end",
              "  shape of phrase-final unit vs others:"]
    for sh in SHAPES:
        lines.append(f"    {sh:5s} final {final_shapes[sh] / sum(final_shapes.values()):5.1%}"
                     f"   elsewhere {other_shapes[sh] / sum(other_shapes.values()):5.1%}")
    lengthening = (np.mean([p[-1]["dur"] for p in all_phr]) / np.mean([u["dur"] for p in all_phr for u in p[:-1]]))
    lines.append(f"  final-unit lengthening: x{lengthening:.2f} duration")

    # --- rhythm
    durs = np.array([u["dur"] for u in units])
    gaps = np.array([u["gap"] for u in units if u["gap"] is not None])
    lines += ["", "Rhythm:", f"  unit duration median {np.median(durs) * 1000:.0f} ms",
              f"  gaps: {np.mean(gaps < SHORT_PAUSE):.0%} tight, "
              f"{np.mean((gaps >= SHORT_PAUSE) & (gaps <= PHRASE_BREAK)):.0%} short pause, "
              f"{np.mean(gaps > PHRASE_BREAK):.0%} phrase break"]

    # --- frequent trigrams of sounding states
    tri = Counter()
    for s in seqs:
        toks = [t for t, _ in s]
        tri.update(zip(toks, toks[1:], toks[2:]))
    lines += ["", "Most frequent 3-state patterns:"] + [f"  {' > '.join(k)}  ({n})" for k, n in tri.most_common(8)]

    # --- overview figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].hist(jumps, bins=np.arange(-12, 12.5, 0.5), color="#c5507a")
    axs[0, 0].set_title("Pitch jump between neighbouring units (st)")
    axs[0, 1].plot(np.linspace(0, 1, 10), decl, "o-", color="#e8a33d")
    axs[0, 1].set_title("Declination: mean level vs position in phrase")
    axs[0, 1].set_xlabel("position in phrase (0 = start, 1 = end)")
    axs[0, 1].set_ylabel("st re. speaker median")
    axs[1, 0].hist(durs * 1000, bins=60, range=(0, 900), color="#5b8bd1")
    axs[1, 0].set_title("Unit duration (ms)")
    for c in np.array([u["contour"] for u in units])[np.random.default_rng(0).choice(len(units), 400)]:
        axs[1, 1].plot(c, color="#7a5bd1", alpha=0.08)
    axs[1, 1].plot(np.mean([u["contour"] for u in units], 0), color="black", lw=2)
    axs[1, 1].set_title("Unit contour shapes (400 random + mean)")
    fig.tight_layout()
    fig.savefig(OUT / "overview.png", dpi=120)
    plt.close(fig)

    n_sources = len({c["source"] for c in clips})
    if 2 <= n_sources <= 10:
        lines += compare_sources(clips, OUT)

    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def compare(datasets):
    clips = []
    for ds in datasets:
        cs = load_clips(ds)
        lump = len({c["source"] for c in cs}) > 10
        clips += [dict(c, source=ds if lump else c["source"]) for c in cs]
    out = ROOT / "analysis" / "compare"
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"Comparing {', '.join(datasets)}"] + compare_sources(clips, out)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1:2] == ["compare"]:
        compare(sys.argv[2:])
    else:
        main(*sys.argv[1:2])
