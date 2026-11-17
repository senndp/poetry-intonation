"""Put words on the generated intonation.

The text is spoken by a macOS voice (`say`); its syllables set the rhythm. For every syllable the Markov
chain picks the next intonation unit, so the melody comes from the chosen poet or bird. Pauses in the
text are fed into the chain as pause states, so phrase context carries over. Two voices:

  vocoder  the oscillator is the sound; the spoken text only shapes its spectrum (a talkbox), with a
           little noise for consonants. Needs a harmonic-rich wave (saw/square).
  speech   the spoken text is kept and its pitch is replaced with the melody (Praat PSOLA resynthesis).

Used from generate.py:  python generate.py --dataset pennsound --source Ashbery --words "..." --voice both
"""
import json
import re
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

import extract
from analyze import PHRASE_BREAK, SHORT_PAUSE
from generate import SR, build_model, oscillate

DEFAULT_TEXT = "I celebrate myself, and sing myself, and what I assume you shall assume."  # Whitman, public domain
DICTIONARY = Path("/usr/share/dict/words")
ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "sh", "th", "br", "pl", "st"]
NUCLEI = ["a", "e", "i", "o", "u", "ai", "ou", "ee", "oo"]
DIP_STEPS_DB = [3.0, 2.5, 2.0, 1.5, 1.0, 0.7]  # syllable-dip thresholds tried for the TTS voice
N_BANDS = 40  # vocoder bands, log-spaced
NOISE_LEVEL = 0.3  # consonant noise relative to the oscillator


def random_text(n, kind, rng):
    """n random words with commas every 3-6 words and a full stop, so the chain gets pauses.

    kind 'real' draws from the system dictionary; 'nonsense' builds 1-3 syllable pseudo-words, so the
    melody can be heard without any meaning getting in the way."""
    if kind == "real":
        pool = [w for w in DICTIONARY.read_text().split() if w.isalpha() and w.islower() and 3 <= len(w) <= 7]
        ws = rng.sample(pool, n)
    else:
        ws = ["".join(rng.choice(ONSETS) + rng.choice(NUCLEI) for _ in range(rng.randint(1, 3))) for _ in range(n)]
    out, until_comma = [], rng.randint(3, 6)
    for i, w in enumerate(ws):
        until_comma -= 1
        if until_comma == 0 and i < n - 1:
            w += ","
            until_comma = rng.randint(3, 6)
        out.append(w)
    return " ".join(out).capitalize() + "."


def tts(text, voice):
    """Speak text with macOS `say`; returns the audio at SR."""
    with tempfile.TemporaryDirectory() as d:
        aiff = Path(d) / "tts.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", str(aiff), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                             capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def count_syllables(text):
    """Rough syllable count: vowel groups per word, minus a silent final e."""
    n = 0
    for w in re.findall(r"[a-z]+", text.lower()):
        k = len(re.findall(r"[aeiouy]+", w))
        if w.endswith("e") and not w.endswith(("le", "ee")) and k > 1:
            k -= 1
        n += max(1, k)
    return n


def syllables_of(y, text):
    """Syllable (start, end, gap after) in 10 ms frames, found the same way as in the recitations.

    A smooth TTS voice dips less between syllables than a reader does, so the dip threshold is lowered
    step by step and the split whose count is closest to the text's syllable count is kept."""
    y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
    f0, voiced, _ = librosa.pyin(y16, fmin=60, fmax=500, sr=16000, frame_length=1024, hop_length=160)
    db = 20 * np.log10(librosa.feature.rms(y=y16, frame_length=1024, hop_length=160)[0][: len(f0)] + 1e-6)
    voiced = extract.gate(voiced, db)
    runs = extract.voiced_runs(voiced)
    target, best = count_syllables(text), None
    for dip_db in DIP_STEPS_DB:
        pieces = []
        for k, (a, b) in enumerate(runs):
            sy = extract.syllables(a, b, db, dip_db)
            gap = (runs[k + 1][0] - b) * extract.FRAME_S if k + 1 < len(runs) else None
            pieces += [(s, e, 0.0) for s, e in sy[:-1]] + [(sy[-1][0], sy[-1][1], gap)]
        if best is None or abs(len(pieces) - target) < abs(len(best) - target):
            best = pieces
    return best, voiced, db


def melody(pieces, n_frames, clips, order, pitch_range, rng):
    """Semitone curve per 10 ms frame: one chain unit per syllable, held across the gaps."""
    chain, pools = build_model(clips, order)
    history = ["||"] * order
    st = np.full(n_frames, np.nan)
    for a, b, gap in pieces:
        tok = chain.next(history, rng)
        while tok in ("|", "||"):  # the text decides where pauses go
            tok = chain.next(history + [tok], rng)
        history.append(tok)
        cont = np.array(rng.choice(pools[tok])["contour"]) * pitch_range
        st[a:b] = np.interp(np.linspace(0, 1, b - a), np.linspace(0, 1, len(cont)), cont)
        if gap is None or gap > PHRASE_BREAK:
            history.append("||")
        elif gap > SHORT_PAUSE:
            history.append("|")
    sung = ~np.isnan(st)
    # hold the last pitch through gaps so the oscillator never jumps mid-consonant
    idx = np.where(sung, np.arange(n_frames), 0)
    np.maximum.accumulate(idx, out=idx)
    held = st[idx]
    held[: np.argmax(sung)] = st[np.argmax(sung)]
    return held, sung


def vocoder(modulator, carrier, voiced_frames):
    """Channel vocoder: the carrier takes the modulator's band envelope; noise covers consonants."""
    n_fft, hop = 1024, 256
    frame_t = np.arange(1 + len(modulator) // hop) * hop / SR
    unvoiced = 1 - np.interp(frame_t, np.arange(len(voiced_frames)) * extract.FRAME_S, voiced_frames.astype(float))
    noise = np.random.default_rng(0).standard_normal(len(carrier)) * np.std(carrier) * NOISE_LEVEL
    M = librosa.stft(modulator, n_fft=n_fft, hop_length=hop)
    C = librosa.stft(carrier, n_fft=n_fft, hop_length=hop)
    Nz = librosa.stft(noise, n_fft=n_fft, hop_length=hop)
    C = C + Nz * unvoiced[: C.shape[1]]
    freqs = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
    edges = np.geomspace(100, 12000, N_BANDS + 1)
    band = np.clip(np.searchsorted(edges, freqs) - 1, 0, N_BANDS - 1)
    gain = np.zeros(C.shape)
    for k in range(N_BANDS):
        rows = band == k
        if not rows.any():  # lowest bands can be narrower than one FFT bin
            continue
        e_m = np.mean(np.abs(M[rows]) ** 2, axis=0)
        e_c = np.mean(np.abs(C[rows]) ** 2, axis=0)
        gain[rows] = np.sqrt(e_m / (e_c + 1e-10))
    y = librosa.istft(C * gain, hop_length=hop, length=len(modulator))
    return y / (np.max(np.abs(y)) + 1e-9) * 0.9


def reintone(y, st, sung, base):
    """Replace the spoken pitch with the melody (Praat overlap-add resynthesis)."""
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=SR)
    manip = call(snd, "To Manipulation", 0.01, 60, 600)
    tier = call("Create PitchTier", "melody", 0, snd.duration)
    for i in np.flatnonzero(sung)[::2]:  # a point every 20 ms is plenty
        call(tier, "Add point", i * extract.FRAME_S, base * 2 ** (st[i] / 12))
    call([tier, manip], "Replace pitch tier")
    out = call(manip, "Get resynthesis (overlap-add)").values[0]
    return out / (np.max(np.abs(out)) + 1e-9) * 0.9


def speak(text, clips, order, base, pitch_range, wave, voice, tts_voice, rng, out):
    """Render the text with the chosen intonation; writes one wav per voice and a pitch picture."""
    from scipy.io import wavfile

    y = tts(text, tts_voice)
    pieces, voiced, db = syllables_of(y, text)
    n_frames = len(voiced)
    st, sung = melody(pieces, n_frames, clips, order, pitch_range, rng)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = []
    if voice in ("vocoder", "both"):
        st_audio = np.interp(np.arange(len(y)) / SR, np.arange(n_frames) * extract.FRAME_S, st)
        carrier = oscillate(st_audio, base, wave)
        path = out.with_name(out.stem + "_vocoder.wav")
        wavfile.write(path, SR, vocoder(y, carrier, voiced).astype(np.float32))
        written.append(path)
    if voice in ("speech", "both"):
        if base > 600:
            print(f"note: {base:.0f} Hz is far above any voice; the resynthesised speech will sound strained")
        path = out.with_name(out.stem + "_speech.wav")
        wavfile.write(path, SR, reintone(y, st, sung, base).astype(np.float32))
        written.append(path)
    drawn = np.where(sung, st, np.nan)
    # the melody itself, e.g. to check the renders or drive an external oscillator
    out.with_suffix(".json").write_text(json.dumps({
        "text": text, "base_hz": base, "frame_s": extract.FRAME_S,
        "hz": [None if np.isnan(v) else round(float(base * 2 ** (v / 12)), 2) for v in drawn]}))
    extract.SPAN = 24 if np.nanmax(np.abs(drawn)) > 12 else 12
    extract.draw({"title": f"{out.stem}   “{text[:60]}”", "units": [], "_st": drawn, "_db": db}, out.with_suffix(".png"))
    print(f"{len(pieces)} syllables, {len(y) / SR:.1f}s ->", ", ".join(str(p) for p in written))
