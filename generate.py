"""Generate new 'recitations' with a Markov chain, played on a single oscillator.

The chain walks the symbolic states from analyze.py (order k, backing off to shorter contexts
when a context was never seen). For each sounding state it borrows the contour, duration and
loudness of a random real unit in that state, so the micro-shape stays speech-like while the
sequence is new. Everything is rendered by one oscillator: pitch = base * 2^(st/12).

Usage: python generate.py [--dataset ganjoor] [--source NAME] [--order 2] [--seconds 30] [--base 160]
                          [--wave sine] [--range 1.0] [--tempo 1.0] [--seed N] [--out out/name.wav]
--source trains the chain on one source only (e.g. one poet).
--sing-like DATASET:SOURCE sets base, tempo, pause tempo and range to match another voice, e.g. a poet sung by a
blackbird: --dataset pennsound --source Ashbery --sing-like birds:Turdus_merula
--words "TEXT" (or --words-file, or --random-words N) puts the melody on spoken words, see words.py:
  --voice vocoder (the oscillator talks), speech (a voice re-intoned) or both; --tts-voice picks the
  macOS voice. The text sets the rhythm, so --tempo is ignored.
"""
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from analyze import load_clips, tokenize
import extract

ROOT = Path(__file__).parent
SR = 44100
PAUSE_S = {"|": (0.12, 0.45), "||": (0.45, 1.1)}
TIGHT_GAP = (0.0, 0.08)  # gap after a unit when no pause token follows


class Chain:
    def __init__(self, seqs, order):
        self.order = order
        self.table = defaultdict(Counter)  # context tuple -> next-token counts, all orders 0..k
        for s in seqs:
            toks = ["||"] * order + s
            for i in range(order, len(toks)):
                for k in range(order + 1):
                    self.table[tuple(toks[i - k:i])][toks[i]] += 1

    def next(self, history, rng):
        for k in range(self.order, -1, -1):
            ctx = tuple(history[-k:]) if k else ()
            if ctx in self.table:
                c = self.table[ctx]
                return rng.choices(list(c), weights=c.values())[0]


def build_model(clips, order):
    """Markov chain over state tokens, plus the real units behind each sounding state."""
    seqs, pools = [], defaultdict(list)
    for c in clips:
        toks = tokenize(c)
        seqs.append([t for t, _ in toks])
        for t, u in toks:
            if u is not None:
                pools[t].append(u)
    return Chain(seqs, order), pools


def compose(clips, order, seconds, tempo, pause_tempo, rng):
    """Return a list of events: ('note', unit, gap_after) with real-unit shapes."""
    chain, pools = build_model(clips, order)
    history, events, t = ["||"] * order, [], 0.0
    while t < seconds:
        tok = chain.next(history, rng)
        history.append(tok)
        if tok in PAUSE_S:
            if events:
                events[-1]["gap"] = rng.uniform(*PAUSE_S[tok]) / pause_tempo
        else:
            u = rng.choice(pools[tok])
            events.append({"state": tok, "contour": u["contour"], "dur": u["dur"] / tempo,
                           "loud": u["loud"], "gap": rng.uniform(*TIGHT_GAP) / tempo})
        t = sum(e["dur"] + e["gap"] for e in events)
    return events


def polyblep(t, dt):
    """Band-limiting correction around the waveform jump at phase 0, so high notes don't alias."""
    out = np.zeros_like(t)
    a = t < dt
    x = t[a] / dt[a]
    out[a] = 2 * x - x * x - 1
    b = t > 1 - dt
    x = (t[b] - 1) / dt[b]
    out[b] = x * x + 2 * x + 1
    return out


def oscillate(st, base, wave):
    """The single oscillator: semitone curve -> waveform at SR, phase-continuous."""
    freq = base * 2 ** (st / 12)
    dt = freq / SR
    phase = np.cumsum(dt) % 1.0
    if wave == "sine":
        return np.sin(2 * np.pi * phase)
    if wave == "triangle":
        return 4 * np.abs(phase - 0.5) - 1
    if wave == "saw":
        return 2 * phase - 1 - polyblep(phase, dt)
    return np.where(phase < 0.5, 1.0, -1.0) + polyblep(phase, dt) - polyblep((phase + 0.5) % 1.0, dt)


def render(events, base, wave, pitch_range):
    """One oscillator: continuous phase, pitch glides across tight gaps, silence in pauses."""
    total = sum(e["dur"] + e["gap"] for e in events) + 0.5
    n = int(total * SR)
    st = np.zeros(n)
    amp = np.zeros(n)
    i, last = 0, 0.0
    for e in events:
        m = int(e["dur"] * SR)
        g = int(e["gap"] * SR)
        cont = np.array(e["contour"]) * pitch_range
        st[i:i + m] = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, len(cont)), cont)
        env = np.ones(m)
        a, r = min(m // 4, int(0.012 * SR)), min(m // 3, int(0.04 * SR))
        env[:a] = np.linspace(0, 1, a)
        env[m - r:] = np.linspace(1, 0, r)
        amp[i:i + m] = env * 10 ** (max(e["loud"], -30) / 20)
        # glide toward the next unit through the gap so legato passages slur like speech
        st[i + m:i + m + g] = cont[-1]
        last = cont[-1]
        i += m + g
    st[i:] = last
    # short portamento smoothing so pitch never steps
    k = int(0.015 * SR)
    st = np.convolve(np.pad(st, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")

    y = oscillate(st, base, wave) * amp
    y /= np.max(np.abs(y)) + 1e-9
    return (0.9 * y).astype(np.float32), st, amp


def picture(st, amp, out):
    """Draw the generated line with the same renderer used for the real recitations."""
    hop = int(extract.FRAME_S * SR)
    st_f, amp_f = st[::hop].copy(), amp[::hop]
    st_f[amp_f < 0.05 * amp_f.max()] = np.nan  # skip attack/release glides
    extract.SPAN = 24 if np.nanmax(np.abs(st_f)) > 12 else 12  # birds move far more than speech
    extract.draw({"title": f"{out.stem}   (generated)", "units": [],
                  "_st": st_f, "_db": 20 * np.log10(amp_f + 1e-6)}, out)


def voice(clips):
    """Centre pitch (Hz), median unit duration (s), within-clip pitch spread (st) and median pause (s)."""
    spread = np.concatenate([np.array([u["level"] for u in c["units"]]) - np.mean([u["level"] for u in c["units"]])
                             for c in clips if c["units"]])
    gaps = np.array([u["gap"] for c in clips for u in c["units"] if u["gap"] is not None])
    return (float(np.median([c["median_hz"] for c in clips])),
            float(np.median([u["dur"] for c in clips for u in c["units"]])), float(spread.std()),
            float(np.median(gaps[gaps >= PAUSE_S["|"][0]])))


def select(clips, source):
    if not source:
        return clips
    clips = [c for c in clips if c["source"].lower() == source.lower()]
    if not clips:
        raise SystemExit(f"no clips with source {source!r}")
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ganjoor")
    ap.add_argument("--source", default=None, help="only learn from this source (case-insensitive)")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--base", type=float, default=160, help="oscillator centre pitch in Hz")
    ap.add_argument("--wave", choices=["sine", "triangle", "saw", "square"], default="sine")
    ap.add_argument("--range", type=float, default=1.0, help="scale pitch movement (2 = twice as wide)")
    ap.add_argument("--tempo", type=float, default=1.0, help="speed factor")
    ap.add_argument("--pause-tempo", type=float, default=None, help="speed factor for pauses (default: --tempo)")
    ap.add_argument("--sing-like", default=None, metavar="DATASET:SOURCE",
                    help="match base, tempo and range to another voice (overrides those flags)")
    ap.add_argument("--words", default=None, help="speak this text with the generated intonation")
    ap.add_argument("--words-file", type=Path, default=None, help="text file to speak")
    ap.add_argument("--random-words", type=int, default=None, metavar="N", help="speak N random words")
    ap.add_argument("--word-kind", choices=["real", "nonsense"], default="real",
                    help="random dictionary words or made-up pseudo-words")
    ap.add_argument("--voice", choices=["vocoder", "speech", "both"], default="both")
    ap.add_argument("--tts-voice", default="Daniel", help="macOS voice for the words (say -v '?')")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    clips = select(load_clips(a.dataset), a.source)
    if a.sing_like:
        ds, _, src = a.sing_like.partition(":")
        (_, dur, spread, pause), (hz, t_dur, t_spread, t_pause) = voice(clips), voice(select(load_clips(ds), src))
        # pauses scale separately: birds and poets pause for similar lengths even though their notes differ ~3x
        a.base, a.tempo, a.range, a.pause_tempo = hz, dur / t_dur, t_spread / spread, pause / t_pause
        print(f"singing like {a.sing_like}: base {a.base:.0f} Hz, tempo x{a.tempo:.2f}, "
              f"pauses x{a.pause_tempo:.2f}, range x{a.range:.2f}")
    tag = f"{a.source}_" if a.source else ""
    if a.sing_like:
        tag += f"as_{a.sing_like.partition(':')[2] or a.sing_like}_"
    seed_tag = a.seed if a.seed is not None else "rand"

    if a.words or a.words_file or a.random_words:
        import words  # imports this module, so only on demand

        if a.random_words:
            text = words.random_text(a.random_words, a.word_kind, rng)
            tag += f"random-{a.word_kind}_"
            print("text:", text)
        else:
            text = a.words_file.read_text().strip() if a.words_file else a.words
        wave = "saw" if a.wave == "sine" and a.voice != "speech" else a.wave  # a sine has no harmonics to shape
        out = a.out or ROOT / "out" / "words" / f"words_{a.dataset}_{tag}{seed_tag}.wav"
        words.speak(text, clips, a.order, a.base, a.range, wave, a.voice, a.tts_voice, rng, out)
        return

    events = compose(clips, a.order, a.seconds, a.tempo, a.pause_tempo or a.tempo, rng)
    y, st, amp = render(events, a.base, a.wave, a.range)
    name = f"gen_{tag}o{a.order}_{a.wave}_{seed_tag}.wav"
    out = a.out or ROOT / "out" / a.dataset / name
    out.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(out, SR, y)
    picture(st, amp, out.with_suffix(".png"))
    print(f"{len(events)} units, {len(y) / SR:.1f}s -> {out}")
    print("states:", " ".join(e["state"] for e in events[:24]), "...")


if __name__ == "__main__":
    main()
