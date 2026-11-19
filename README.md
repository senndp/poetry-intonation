# Poetry intonation → birdsong oscillator

A poet's intonation, learned as a Markov chain, sung by a bird on a single oscillator.
Inspired by Lucio Arese's *Visualizing Bird Songs*.

```
poet recording  ──► units ──► labels ──► Markov chain ──► new label sequence ──► poet's real syllable shapes
                                                                                          │
bird recordings ──► units ──► 4 measurements ──────────────────────────────► scale pitch, speed, range
                                                                                          │
                                                                                 one oscillator ──► sound
```

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # also needs ffmpeg
```

## Run

```bash
# poet recordings: one of
.venv/bin/python download.py 220                     # Ganjoor Persian recitations -> data/ganjoor/audio/
.venv/bin/python pennsound.py                        # your MP3s in data/pennsound/raw/<Poet>/ -> 30 s chunks

# bird recordings (CC0 iNaturalist): blackbird, song thrush, nightingale, chaffinch
.venv/bin/python birds.py

# units (+ one pitch picture per clip in data/<dataset>/pictures/)
.venv/bin/python extract.py pennsound --per-source
.venv/bin/python extract.py ganjoor
.venv/bin/python extract.py birds --profile bird --per-source

# sound
.venv/bin/python generate.py --dataset pennsound --source Ashbery --bird Turdus_merula --seed 1
.venv/bin/python generate.py --dataset ganjoor --bird Fringilla_coelebs --seconds 20
```

Output goes to `out/` as a `.wav` with a matching pitch picture. Options: `--order` (Markov order,
default 2), `--seconds`, `--wave sine|triangle|saw|square`, `--seed`, `--out`.

## How it works

1. **Units** (`extract.py`): pitch every 10 ms, in semitones from the voice's own centre (`--per-source`
   gives one centre per poet). Voiced stretches are split into syllable-sized units at loudness dips. Each
   unit has an 8-point contour, a length, a loudness and the gap after it. Speech uses pYIN (60–500 Hz).
   Birds use the strongest spectral peak (1–10 kHz) after dividing out steady background noise. A loudness
   gate keeps mains hum from counting as voice.
2. **Labels** (`labels.py`): each unit becomes `LEVEL-SHAPE-LENGTH` (`L/M/H`, `fall/flat/rise`, `s/l`).
   Gaps become `|` (120–450 ms) or `||` (over 450 ms).
3. **Markov chain** (`generate.py`): it counts which label follows each pair of the poet's labels, backing
   off to one label, then to overall frequencies. It walks those counts to make a new label sequence. Each
   sounding label borrows a random real unit of the poet with that label.
4. **Bird measurements**: centre pitch, median note length, median pause and pitch spread within a clip.
   Compared with the poet's, they set the oscillator's centre, the speed of notes and of pauses (separately,
   because birds and poets pause for similar lengths), and how wide the pitch moves.
5. **One oscillator**: phase-continuous, band-limited for saw/square, with short glides, a fade in and out
   per note, and silence in pauses.

The recordings are not ours to redistribute (PennSound readings belong to the poets), so `data/` and `out/`
stay out of git.
