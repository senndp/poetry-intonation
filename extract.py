"""Translate recitations into intonation units ("bird-song" style transcription).

For every clip:
  1. track f0 with pYIN (10 ms hop)
  2. convert to semitones relative to the reader's median pitch (speaker normalisation:
     per clip by default, per source with --per-source, e.g. one baseline per poet)
  3. cut the track into voiced units (~syllables) separated by unvoiced gaps
  4. describe each unit: level, start/end pitch, slope, duration, loudness, gap after,
     and a fixed-length contour shape (8 points)
  5. draw the clip as pitch strokes on a dark canvas, like the video's visualisations

Clips live in data/<dataset>/audio/ as <source>__<name>.(m4a|mp3|wav); clips with too little
voicing (applause, music, silence) are skipped.

Profiles: 'speech' (default, pYIN 60-500 Hz) and 'bird' (strongest spectral peak 1-10 kHz, short notes;
pYIN locks onto its range limits on noisy field recordings of birdsong).

Usage: python extract.py [dataset] [--profile speech|bird] [--per-source] [--limit N]
Writes data/<dataset>/units.json and data/<dataset>/pictures/*.png
"""
import argparse
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).parent
AUDIO_EXT = {".m4a", ".mp3", ".wav"}
FRAME_S = 0.01  # 10 ms analysis frames in every profile
SYLL_DIP_DB = 3.0  # a loudness dip this deep inside a voiced run marks a syllable boundary
CONTOUR_PTS = 8
VOICE_GATE_DB = 25  # frames this far below the clip's loud level are pauses, even if pYIN hears pitch (hum)
PROFILES = {
    # MIN_UNIT_S: shorter voiced runs are noise; BRIDGE_S: shorter unvoiced gaps are bridged;
    # MIN_VOICED: clips voiced less than this are mostly applause, music or silence;
    # DIP_SMOOTH: frames of loudness smoothing before looking for syllable dips; SPAN: picture range in st
    # PEAK_DB: in the 'peak' tracker a frame counts as song when its peak stands this far above that
    # frequency's background level (median over the clip)
    "speech": dict(SR=16000, TRACKER="pyin", FMIN=60, FMAX=500, FRAME_LEN=1024, MIN_UNIT_S=0.06,
                   BRIDGE_S=0.03, MIN_VOICED=0.15, DIP_SMOOTH=7, SPAN=12),
    "bird": dict(SR=32000, TRACKER="peak", FMIN=1000, FMAX=10000, FRAME_LEN=1024, PEAK_DB=15,
                 MIN_UNIT_S=0.02, BRIDGE_S=0.01, MIN_VOICED=0.03, DIP_SMOOTH=3, SPAN=24),
}


def configure(profile):
    """Set the module-level analysis constants (also run in every worker process)."""
    globals().update(PROFILES[profile], HOP=int(PROFILES[profile]["SR"] * FRAME_S))


configure("speech")


def paths(dataset):
    """audio dir, f0 cache dir (so re-segmenting skips the slow pYIN pass), pictures dir, units file"""
    d = ROOT / "data" / dataset
    return d / "audio", d / "f0", d / "pictures", d / "units.json"


def source_of(path):
    return path.name.split("__")[0]


def load(path):
    raw = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def voiced_runs(voiced):
    """Return (start, end) frame indices of voiced runs, bridging tiny gaps."""
    runs, start = [], None
    for i, v in enumerate(np.append(voiced, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    bridge = int(BRIDGE_S / FRAME_S)
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= bridge:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [r for r in merged if (r[1] - r[0]) * FRAME_S >= MIN_UNIT_S]


def syllables(a, b, db, dip_db=None):
    """Split a voiced run at loudness dips; returns (start, end) pieces of at least MIN_UNIT_S."""
    seg = np.convolve(np.pad(db[a:b], DIP_SMOOTH // 2, mode="edge"), np.ones(DIP_SMOOTH) / DIP_SMOOTH, mode="valid")
    dips, _ = find_peaks(-seg, prominence=dip_db or SYLL_DIP_DB)
    min_len = int(MIN_UNIT_S / FRAME_S)
    cuts = [a]
    for d in dips + a:
        if d - cuts[-1] >= min_len and b - d >= min_len:
            cuts.append(d)
    return list(zip(cuts, cuts[1:] + [b]))


def gate(voiced, db):
    """Drop 'voiced' frames in quiet stretches: mains hum (50/60 Hz) otherwise reads as low pitch."""
    return voiced & (db >= np.percentile(db, 99) - VOICE_GATE_DB)


def peak_track(y):
    """Frequency of the strongest spectral peak in FMIN-FMAX per frame; tonal frames count as voiced.

    Each frequency is first divided by its median level over the clip, so steady background tones
    (engines, insects, hum harmonics) drop out and only sounds that come and go stand out."""
    S = np.abs(librosa.stft(y, n_fft=FRAME_LEN, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=FRAME_LEN)
    band = (freqs >= FMIN) & (freqs <= FMAX)
    raw = S[band]
    B = raw / (np.median(raw, axis=1, keepdims=True) + 1e-12)
    k = np.argmax(B, axis=0)
    # parabolic interpolation between neighbouring bins for sub-bin accuracy
    lo, mid, hi = (np.log(B[np.clip(k + d, 0, len(B) - 1), np.arange(B.shape[1])] + 1e-12) for d in (-1, 0, 1))
    shift = np.clip(0.5 * (lo - hi) / (lo - 2 * mid + hi - 1e-12), -0.5, 0.5)
    f0 = freqs[band][0] + (k + shift) * (freqs[1] - freqs[0])
    prominence = 10 * np.log10(B.max(0) + 1e-12)
    voiced = (prominence >= PEAK_DB) & (k > 0) & (k < len(B) - 1)  # peaks on the band edge are leakage
    db = 10 * np.log10(raw.sum(0) + 1e-12)  # in-band loudness, so wind and traffic don't count
    return np.where(voiced, f0, np.nan), voiced, db


def track(path, cache):
    """pYIN pitch + loudness per 10 ms frame, cached as .npz."""
    cached = cache / (path.stem + ".npz")
    if cached.exists():
        z = np.load(cached)
        return z["f0"], z["voiced"], z["db"], float(z["dur"])
    y = load(path)
    if TRACKER == "peak":
        f0, voiced, db = peak_track(y)
    else:
        f0, voiced, _ = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=SR, frame_length=FRAME_LEN, hop_length=HOP)
        rms = librosa.feature.rms(y=y, frame_length=FRAME_LEN, hop_length=HOP)[0][: len(f0)]
        db = 20 * np.log10(rms + 1e-6)
    np.savez(cached, f0=f0, voiced=voiced, db=db, dur=len(y) / SR)
    return f0, voiced, db, len(y) / SR


def analyse(path, cache, median_hz=None):
    f0, voiced, db, duration = track(path, cache)
    voiced = gate(voiced, db)
    if voiced.sum() < 20 or voiced.mean() < MIN_VOICED:
        return None
    median_hz = median_hz or float(np.nanmedian(f0[voiced]))
    st = np.where(voiced, 12 * np.log2(f0 / median_hz), np.nan)  # semitones re. speaker median, NaN when unvoiced

    units = []
    runs = voiced_runs(voiced)
    pieces = []  # (start, end, gap after)
    for k, (a, b) in enumerate(runs):
        sy = syllables(a, b, db)
        gap = (runs[k + 1][0] - b) * FRAME_S if k + 1 < len(runs) else None
        pieces += [(s, e, 0.0) for s, e in sy[:-1]] + [(sy[-1][0], sy[-1][1], gap)]
    for a, b, gap in pieces:
        seg = st[a:b]
        seg = np.interp(np.arange(len(seg)), np.flatnonzero(~np.isnan(seg)), seg[~np.isnan(seg)])
        seg = np.convolve(np.pad(seg, 2, mode="edge"), np.ones(5) / 5, mode="valid")  # light smoothing
        contour = np.interp(np.linspace(0, len(seg) - 1, CONTOUR_PTS), np.arange(len(seg)), seg)
        dur = (b - a) * FRAME_S
        units.append({
            "t": round(a * FRAME_S, 3),
            "dur": round(dur, 3),
            "gap": None if gap is None else round(gap, 3),
            "level": round(float(np.mean(seg)), 2),
            "start": round(float(contour[0]), 2),
            "end": round(float(contour[-1]), 2),
            "slope": round(float((contour[-1] - contour[0]) / dur), 2),  # st per second
            "loud": round(float(np.mean(db[a:b]) - np.max(db)), 1),  # dB below clip peak
            "contour": [round(float(c), 2) for c in contour],
        })
    return {"file": path.name, "source": source_of(path), "median_hz": round(median_hz, 1),
            "duration": round(duration, 2), "units": units, "_st": st, "_db": db}


def draw(clip, out):
    """Pitch strokes on black: x = time, y = semitones, width/alpha = loudness."""
    st, db = clip["_st"], clip["_db"]
    t = np.arange(len(st)) * FRAME_S
    fig, ax = plt.subplots(figsize=(14, 4), dpi=110)
    fig.patch.set_facecolor("#0b0b10")
    ax.set_facecolor("#0b0b10")
    loud = np.clip((db - db.max() + 40) / 40, 0, 1)
    ax.scatter(t, st, s=2 + 14 * loud, c=st, cmap="plasma", vmin=-SPAN * 2 / 3, vmax=SPAN * 2 / 3, alpha=0.35 + 0.6 * loud, linewidths=0)
    for u in clip["units"]:
        ax.plot([u["t"], u["t"] + u["dur"]], [-SPAN + 1, -SPAN + 1], color="#ffffff30", lw=3, solid_capstyle="butt")
    ax.set_ylim(-SPAN, SPAN)
    ax.set_xlim(0, t[-1])
    ax.axis("off")
    title = clip.get("title") or f'{clip["file"]}   median {clip["median_hz"]} Hz'
    ax.set_title(title, color="#9999aa", fontsize=9, loc="left")
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def track_only(path, cache):
    try:
        track(path, cache)
    except Exception as e:  # corrupt/odd files shouldn't kill the batch
        print("skip", path.name, e)


def source_medians(files, cache):
    """Median pitch over all voiced frames of each source, for per-poet normalisation."""
    f0s = defaultdict(list)
    for f in files:
        cached = cache / (f.stem + ".npz")
        if cached.exists():
            z = np.load(cached)
            f0s[source_of(f)].append(z["f0"][gate(z["voiced"], z["db"])])
    return {s: float(np.nanmedian(np.concatenate(v))) for s, v in f0s.items()}


def work(path, cache, pics, median_hz):
    if not (cache / (path.stem + ".npz")).exists():
        return None
    clip = analyse(path, cache, median_hz)
    if clip is None:
        return None
    draw(clip, pics / (path.stem + ".png"))
    del clip["_st"], clip["_db"]
    return clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="ganjoor")
    ap.add_argument("--profile", choices=PROFILES, default="speech")
    ap.add_argument("--per-source", action="store_true", help="one pitch baseline per source instead of per clip")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    configure(a.profile)
    audio, cache, pics, units_file = paths(a.dataset)
    pics.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in audio.iterdir() if f.suffix in AUDIO_EXT)[: a.limit]
    # compile librosa's numba kernels once here; parallel first-compiles can corrupt its disk cache
    librosa.pyin(np.random.default_rng(0).standard_normal(SR).astype(np.float32) * 0.1,
                 fmin=FMIN, fmax=FMAX, sr=SR, frame_length=FRAME_LEN, hop_length=HOP)
    with ProcessPoolExecutor(initializer=configure, initargs=(a.profile,)) as pool:
        list(pool.map(track_only, files, repeat(cache)))
        medians = source_medians(files, cache) if a.per_source else {}
        baselines = [medians.get(source_of(f)) for f in files]
        clips = [c for c in pool.map(work, files, repeat(cache), repeat(pics), baselines) if c]
    clips.sort(key=lambda c: c["file"])
    units_file.write_text(json.dumps(clips))
    n_units = sum(len(c["units"]) for c in clips)
    print(f"{len(clips)}/{len(files)} clips kept, {n_units} units -> {units_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
