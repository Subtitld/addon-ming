# Ming-3.0-tts inference code

`bin/_internal/ming_addon/ming/` (compiled into the executable's archive) is a
modified copy of the inference code of Ming-3.0-tts,
https://github.com/inclusionAI/Ming-3.0-tts at commit
9332fb9443325d4726ce02cefd4ec1c211df4079, under the MIT licence in `LICENSE`.
`ming_addon/ming/README.md` in the add-on's repository lists the changes.

Two small pieces of other projects are copied into it:

- `fm/rotary.py`: the rotary-embedding helpers of x-transformers 2.9.2
  (MIT, Phil Wang), `x-transformers-LICENSE`.
- `audio_tokenizer/modeling_audio_vae.py`, `_sample_posterior`: what
  diffusers' `OobleckDiagonalGaussianDistribution.sample` computes
  (Apache-2.0, The HuggingFace Team), `diffusers-LICENSE`.

The Ming-3.0-tts-0.5B weights are not in the archive. They download on first
use from https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B and are MIT.
