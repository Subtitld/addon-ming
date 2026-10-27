# Ming-3.0-tts add-on for Subtitld

Text-to-speech for [Subtitld](https://subtitld.org) dubbing with inclusionAI's
[Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) (0.5B): Chinese,
English, Japanese, Korean, German, French and Spanish at 44.1 kHz, in the
speaker's own voice cloned from the video, in one of the model's built-in
voices, or in a voice made from a description, with emotion and speed
controls. MIT, the model weights included.

## Voices

- **Speaker's own voice (cloned from the video).** Subtitld cuts a clip of the
  speaker from the video and sends it with the subtitle text spoken in it.
  Ming-3.0-tts then continues from the clip, which comes closest to the
  speaker. Turn on *Clone the timbre only* in Configure to clone the voice
  without the clip's delivery: try it when a dub into another language keeps
  the speaker's accent, or when the subtitles paraphrase the speech.
- **Built-in voices**: Emma (American), George and Charlotte (British), Ling
  Xiaorou and Ling Kai (Chinese). Speakers the model was trained on, the same
  on every line; each speaks all seven languages.
- **Described voices**: Warm woman, Gentle older woman, Calm man, Energetic
  young man, Narrator, and *Voice you describe*, which takes your own words
  from Configure ("An elderly woman, soft and warm voice, slow pace").
  Ming-3.0-tts invents a new voice every time it reads a description, so the
  add-on makes each speaker's voice once, as a fixed sentence in the line's
  language, and every line of that speaker clones it. Two speakers given the
  same described voice still sound like two people.

The voice is chosen per speaker in Subtitld's dubbing panel.

**Emotion** (happy, sad, angry, fearful, disgusted, surprised, neutral) and
**speed, pitch and volume** are settings in Configure, for the cloned and
described voices. They are the model's own controls, which take the
speaker's timbre but not their way of speaking; one at a time, so an emotion
overrides speed, pitch and volume. Built-in voices ignore them.

The model also reads inline sounds in Chinese or English text: `[Laughter]`,
`[Sigh]`, `[Breathing]`, `[Cough]`, `[Uhm]` and a few more (see upstream's
cookbook).

## Languages

Chinese, English, Japanese, Korean, German, French and Spanish. A line in any
other language is read as English.

## What it needs

- **Disk:** about 3.4 GB. The model, 2.99 GB, downloads from Hugging Face on
  the first synthesis, at a pinned revision, each file checked by sha256,
  into Subtitld's model cache. An interrupted download resumes. `HF_ENDPOINT`
  points it at a mirror; `MING_MODEL_DIR` at a folder that already holds the
  files.
- **Memory:** about 5 GB free. With less the add-on still runs, but far
  slower.
- **Time:** on a CPU each second of speech takes about four to five seconds to
  make (Intel Core i7-13700T, 8 performance cores). The add-on runs on the
  CPU; the Apple GPU can be chosen but has not been tried, and this build does
  not use NVIDIA GPUs.

After the download nothing leaves the machine.

## Settings

| Setting | Default | |
|---|---|---|
| Inference device | Automatic | the CPU; the Apple GPU (MPS) untested |
| CPU threads | 0 | the performance cores on a CPU that has efficiency cores too, else one per core |
| Emotion | As the voice speaks | see Voices |
| Speed, Pitch, Volume | Normal | see Voices |
| Voice description | | for *Voice you describe* |
| Fallback reference clip / transcript | | for the cloned voice when Subtitld has no clip to send |
| Clone the timbre only | off | see Voices |

## Building

Python 3.10 to 3.12.

```bash
# Linux and Windows: the CPU build of PyTorch
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchaudio==2.11.0
pip install -c constraints.txt -e '.[build,dev]'
pytest
pyinstaller pyinstaller.spec --noconfirm
dist/ming-addon/ming-addon --self-test
```

The release workflow builds, proves and publishes every platform on a tag
`v<version>` matching `manifest.json`. To run its Linux build locally, as
GitHub runs it, use [act](https://github.com/nektos/act):

```bash
act workflow_dispatch -W .github/workflows/release.yml --input targets=linux-x86_64 \
  -P ubuntu-22.04=catthehacker/ubuntu:act-22.04 --artifact-server-path /tmp/artifacts
```

A manual run on GitHub can build only some platforms (`targets`).
`tools/collect_licences.py` fails the build when a native file in the bundle has
no notice. The Linux archive carries numpy's libquadmath (LGPL-2.1), so the
release attaches its complete source, built by `tools/build_libquadmath_source.py`.

## How it runs the model

Upstream has no package to install; it runs from a checkout. The add-on
carries the inference code for the 0.5B model in `ming_addon/ming/`, with
the changes listed in `ming_addon/ming/README.md`. Two matter beyond
packaging:

- Upstream passes the language model a one-column attention mask on every
  decoding step. The transformers version it pins ignored it; current
  transformers pads it to the cache length and hides the whole prompt, which
  turns the speech into noise. The add-on passes none.
- The checkpoint is bfloat16 and upstream runs it under CUDA's autocast. On a
  CPU without native bfloat16 the audio decoder is then some 100 times
  slower, so the flow head and the decoder are loaded as float32, while the
  language model and the audio encoder, which lose little, stay bfloat16:
  4.5 GB in all. The model is built on PyTorch's `meta` device and takes each
  tensor as it is cast, so no second copy is made. See `ming_addon/engine.py`.

Upstream's FST text normaliser needs pynini, which has no Windows or macOS
wheels. The add-on spells out English numbers with upstream's own
inflect-based rules instead; Chinese digits the model reads as they are.

## Licence

The add-on is Apache-2.0. Ming-3.0-tts's code and the Ming-3.0-tts-0.5B
weights are MIT (inclusionAI, Ant Group). Each archive carries
`THIRD_PARTY_LICENSES/` with the licence of everything frozen into it.

Voice cloning makes speech that sounds like a real person. Clone voices you
have the right to use, and say that a dub is synthetic where it matters.
