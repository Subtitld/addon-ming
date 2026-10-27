# Vendored Ming-3.0-tts inference code

The inference path of [Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts)
for its 0.5B dense checkpoint, copied at commit
`9332fb9443325d4726ce02cefd4ec1c211df4079` (2026-10-27). MIT, see `LICENSE`.
Upstream has no installable package; its README says to run from a checkout.

Copied: `ming_tts.py`, `tts_tasks.py`, `prompt_contract.py`,
`configuration_bailingmm.py`, `modeling_bailingmm.py`, `spkemb_extractor.py`,
`fm/` (`CFM`, `dit`, `flowloss`, `modules`), `audio_tokenizer/`
(`configuration_audio_vae`, `modeling_audio_vae`, `vae_modules`, `istft`) and
`frontend/` (`toolkit`, `text_segment_cut`, `number_en`). Not copied: the Tiny
MoE model (`ming_tiny/`), the FST text normaliser (`frontend/talker_tn`,
needs pynini), the CLI, the Whisper semantic module (the 0.5B checkpoint has
none), training code.

## Changes

Each is marked with a comment where it is made.

- Imports made relative to this package.
- `fm/rotary.py` replaces x-transformers (two helpers, MIT); the VAE's
  posterior sample replaces diffusers' `OobleckDiagonalGaussianDistribution`
  (Apache-2.0); loguru becomes `logging`; tqdm becomes a no-op that
  `engine.py` swaps for its step hook.
- **The decoding loop passes no attention mask.** Upstream passes a
  one-column mask on every cached step. transformers 4.46 ignored it;
  transformers 5 pads it with False to the cache length, which hides the whole
  prompt from every step after the first and turns the speech into noise.
- The decoding loop calls the transformer without its LM head (`model.model`)
  and reads `last_hidden_state`, the same state as upstream's
  `hidden_states[-1]`, without computing 152k-token logits each step.
- dtypes: autocast is off by default (`autocast = False`; upstream runs under
  CUDA's bfloat16 autocast, slow on a CPU), and values are cast where they
  cross between parts, so the language model and audio encoder can stay
  bfloat16 while the flow head and decoder run in float32 (see `engine.py`).
  The ISTFT head computes in float32: CPUs have no bfloat16 complex kernels.
- `flow_steps` exposes the flow head's step count (upstream fixes 10).
- `MingTTS` takes a `dtype` and reads and writes WAV with `ming_addon.audio`
  instead of torchaudio's I/O, which now needs torchcodec and FFmpeg.
