# Poetry intonation → oscillator

Persian poetry recitations ([Ganjoor audio dataset](https://huggingface.co/datasets/Reza2kn/ganjoor-chizzled-poetry-review-audio))
are transcribed into pitch strokes, as in Lucio Arese's *Visualizing Bird Songs*. The strokes are cut into
syllable-sized intonation units, and a Markov chain over those units drives a single oscillator.

## Pipeline

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python download.py 220        # random sample, spread over all dataset shards -> data/audio/
.venv/bin/python extract.py             # pitch tracks -> data/units.json + data/pictures/*.png
.venv/bin/python analyze.py             # relations -> analysis/summary.txt, overview.png, transitions.png
.venv/bin/python generate.py --seed 1   # new line on one oscillator -> out/*.wav + matching picture
```

Each dataset lives in `data/<dataset>/`, with results in `analysis/<dataset>/` and `out/<dataset>/`.
The default dataset is `ganjoor`.

### PennSound (comparing poets)

```bash
# put a few MP3 readings per poet in data/pennsound/raw/<Poet>/  (or: python pennsound.py download <Poet>)
.venv/bin/python pennsound.py chunk --minutes 15        # 30 s chunks per poet, skipping introductions
.venv/bin/python extract.py pennsound --per-source      # one pitch baseline per poet
.venv/bin/python analyze.py pennsound                   # adds a per-poet table + analysis/pennsound/sources.png
.venv/bin/python generate.py --dataset pennsound --source Ginsberg --seed 1
```

PennSound recordings are for noncommercial, educational use and the rights stay with the poets, so keep
`data/pennsound/` private. PennSound refuses scripted requests, so `download` may fail; downloading the
MP3s in a browser works.

### Birds (comparing poetry with birdsong)

```bash
.venv/bin/python birds.py                                # 4 songbirds, CC0 iNaturalist audio -> data/birds/
.venv/bin/python extract.py birds --profile bird --per-source
.venv/bin/python analyze.py compare ganjoor pennsound birds   # -> analysis/compare/sources.png
.venv/bin/python generate.py --dataset birds --source Turdus_merula --base 3280 --seconds 20
# a bird sings the poetry: poet's patterns, the bird's centre pitch, syllable speed and pitch spread
.venv/bin/python generate.py --dataset pennsound --source Ashbery --sing-like birds:Turdus_merula --seconds 20
# and the reverse: a poet speaks the blackbird's patterns
.venv/bin/python generate.py --dataset birds --source Turdus_merula --sing-like pennsound:Ashbery
```

`--sing-like` scales pauses by the ratio of median pause lengths, not by the syllable speed. Birds and
poets pause for similar lengths (240–320 ms) even though bird notes are about 3× shorter.

The bird profile follows the strongest frequency between 1 and 10 kHz, after dividing out each frequency's
steady background level. pYIN, used for speech, locks onto its range limits on noisy field recordings.
Because everything is measured in semitones from each source's own centre, birds and poets share one
scale.

### Words (`words.py`)

```bash
T="I celebrate myself, and sing myself, and what I assume you shall assume."
.venv/bin/python generate.py --dataset pennsound --source Ashbery --base 104 --words "$T" --seed 1
.venv/bin/python generate.py --dataset birds --source Turdus_merula --sing-like pennsound:Ashbery --words "$T"
.venv/bin/python generate.py --dataset birds --source Turdus_merula --base 3280 --voice vocoder --words "$T"
```

The text is spoken by a macOS voice (`--tts-voice`, default Daniel), and its syllables set the rhythm. For
each syllable the Markov chain picks the next intonation unit, and pauses in the text go back into the
chain as pause states. `--words-file` reads the text from a file instead.

`--random-words N` makes up a text: `--word-kind real` takes dictionary words and `nonsense` makes
pseudo-words, so you hear the melody without the meaning. Commas every 3–6 words give the chain pauses.
The same `--seed` gives the same text, so you can compare melodies on identical words:

```bash
.venv/bin/python generate.py --dataset ganjoor --base 155 --random-words 14 --word-kind nonsense --seed 2
```

A smooth TTS voice dips less between syllables than a reader does. So the syllable split lowers its dip
threshold until the count is close to the text's estimated syllable count.

Output goes to `out/words/`:

- `_vocoder.wav`: the oscillator is the sound, and the spoken text only shapes its spectrum (a talkbox),
  with a little noise for consonants. A sine has no harmonics to shape, so the vocoder switches to saw.
  At bird pitch the words blur, because the harmonics are too far apart to carry vowels.
- `_speech.wav`: the spoken text keeps its own voice, and its pitch is replaced by the melody using Praat's
  overlap-add resynthesis.
- `.png`: the melody per syllable.
- `.json`: the text and the melody in Hz per 10 ms frame (null in gaps). Use it to check a render, or to
  drive an external oscillator.

`--tempo` is ignored for words, because the text decides the timing. Both voices follow the target melody
to within about 0.1 semitone, measured by re-tracking the output.

### 1. Translate (`extract.py`)
pYIN pitch at 10 ms frames, in semitones relative to each reader's median, so different voices are comparable.
Voiced stretches are split into units at loudness dips (≈ syllables). Each unit stores its level, start and end
pitch, slope, duration, loudness, the gap that follows it and an 8-point contour. Raw tracks are cached in
`data/f0/`, so changing the segmentation re-runs in seconds.

### 2. Relations (`analyze.py`)
Each unit becomes a state `LEVEL-SHAPE-LENGTH` (L/M/H × fall/flat/rise × s/l). `|` marks a short pause and
`||` a phrase break. The script reports the transition matrix, pitch jumps between neighbouring units,
declination across phrases, phrase-final behaviour, rhythm and frequent 3-state patterns.

### 3. Generate (`generate.py`)
An order-k Markov chain (backs off to shorter contexts it has not seen) walks the states. Each sounding state
borrows the contour, duration and loudness of a random real unit in that state. Then one phase-continuous
oscillator plays `base · 2^(st/12)`, with short portamento and silence in pauses.

| flag | default | meaning |
|---|---|---|
| `--order` | 2 | Markov order: 1 = loose, 3+ = closer to real phrases |
| `--seconds` | 30 | length |
| `--base` | 160 | centre pitch in Hz |
| `--wave` | sine | sine / triangle / saw / square |
| `--range` | 1.0 | scale pitch movement |
| `--tempo` | 1.0 | speed factor |
| `--seed` | random | reproducible output |
