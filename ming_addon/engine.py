"""Ming-3.0-tts-0.5B for the add-on: fetch the weights, load them lean, synthesize.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0

Weights
-------
Ming-3.0-tts-0.5B is not in the archive: 2.99 GB, fetched on the first
synthesis from Hugging Face at a pinned revision, each file checked against
its sha256, into the host's model cache
($SUBTITLD_MODELS_DIR/ming/Ming-3.0-tts-0.5B). An interrupted download
resumes. $HF_ENDPOINT points it at a mirror; $MING_MODEL_DIR at a folder that
already holds the files, which is then used as it is.

Precision
---------
The checkpoint is bfloat16, and upstream runs it so under CUDA's autocast.
On a CPU without native bfloat16 that is the slow way for the parts that do
most of the work: the audio decoder takes some 100 times longer (35 s against
0.35 s for 4 s of speech on a 13th-gen Intel), the flow head is slower too,
and the vocoder's complex arithmetic has no bfloat16 kernels at all. Those
parts are loaded as float32. The language model and the audio encoder lose
little in bfloat16 and stay so (LEAN_PREFIXES): about 4.5 GB in all, against
6.9 GB in float32 throughout, for 10% more time.

Loading
-------
The model is built on PyTorch's `meta` device, without memory, and takes the
checkpoint's tensors as they are cast, so no second copy is ever made. What
the checkpoint does not hold is rebuilt: the language model's output head,
tied to its embeddings, and the rotary tables, which are computed.
"""

from __future__ import annotations

import hashlib
import logging
import os
import ssl
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ming_addon import __version__

log = logging.getLogger('ming_addon.engine')

MODEL_REPO = 'inclusionAI/Ming-3.0-tts-0.5B'
MODEL_REVISION = '7247f4c94a5b2e987404ed28a5cfec41b1a4ca61'
MODEL_DIRNAME = 'Ming-3.0-tts-0.5B'

# What the inference path reads, at MODEL_REVISION: (file, bytes, sha256).
# The tokenizer is Qwen2's, loaded from tokenizer.json; the repo holds no code.
MODEL_FILES = (
    ('config.json', 4754, '2fb0767361b3cd3f41dc9392709fc23fe5b40a39f19d70a38d04acabe991822c'),
    ('tokenizer.json', 11431389, '87d54ca3b44ea2f822ba39a0a45cb1b61644bd6606d9d3536dea75367bffb06d'),
    ('tokenizer_config.json', 16260, '12d6162c815edef24f8606e65b60d43fe0d92b0a4e150a0db8fa01c523aaf36a'),
    ('special_tokens_map.json', 2726, '2c2ec51b9da5db654ac20de8310350d0c9c1a65918ae0181b80dacd2919b4fc3'),
    ('campplus.onnx', 28303423, 'a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73'),
    ('model.safetensors', 2959055504, '45c4d78d8ad6b1d942e3da86ba035fece02de569f7b2ca7e9d121dfaf18d674c'),
)
MODEL_BYTES = sum(size for _name, size, _sha in MODEL_FILES)

_CHUNK = 1 << 20
_VERIFIED = '.verified'


class Cancelled(Exception):
    """The request was cancelled."""


class DownloadError(Exception):
    """The weights could not be fetched or did not check out."""


# ---------------------------------------------------------------------------
# Where the weights live
# ---------------------------------------------------------------------------
def cache_root() -> Path:
    """$SUBTITLD_MODELS_DIR/ming: the weights and the described voices' anchors."""
    base = (os.environ.get('SUBTITLD_MODELS_DIR') or '').strip()
    root = Path(os.path.expanduser(base)) if base else Path.home() / '.cache' / 'subtitld' / 'models'
    return root / 'ming'


def model_dir() -> Path:
    override = (os.environ.get('MING_MODEL_DIR') or '').strip()
    if override:
        return Path(os.path.expanduser(override))
    return cache_root() / MODEL_DIRNAME


def weights_present(folder: Path) -> bool:
    """True when every file is there at its expected size, checked once by sha256."""
    if (os.environ.get('MING_MODEL_DIR') or '').strip():
        return all((folder / name).is_file() for name, _size, _sha in MODEL_FILES)
    marker = folder / _VERIFIED
    try:
        if marker.read_text(encoding='utf-8').strip() != MODEL_REVISION:
            return False
    except OSError:
        return False
    return all((folder / name).is_file() and (folder / name).stat().st_size == size
               for name, size, _sha in MODEL_FILES)


def missing_bytes(folder: Path) -> int:
    """Bytes still to download, counting partial files as progress."""
    total = 0
    for name, size, _sha in MODEL_FILES:
        path = folder / name
        if path.is_file() and path.stat().st_size == size:
            continue
        part = folder / (name + '.part')
        have = part.stat().st_size if part.is_file() else 0
        total += max(0, size - have)
    return total


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # frozen macOS builds have no system CA bundle Python can read
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def _sha256(path: Path, cancel: Callable[[], bool]) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(_CHUNK * 8):
            if cancel():
                raise Cancelled()
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, dest: Path, size: int, sha256: str,
           on_bytes: Callable[[int], None], cancel: Callable[[], bool]) -> None:
    part = dest.with_name(dest.name + '.part')
    have = part.stat().st_size if part.is_file() else 0
    if have > size:
        part.unlink()
        have = 0
    if have < size:
        headers = {'User-Agent': f'subtitld-ming-addon/{__version__}'}
        if have:
            headers['Range'] = f'bytes={have}-'
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60, context=_ssl_context()) as response:
                if have and response.status != 206:
                    have = 0  # the server ignored the range: start over
                on_bytes(have)
                with part.open('ab' if have else 'wb') as out:
                    while chunk := response.read(_CHUNK):
                        if cancel():
                            raise Cancelled()
                        out.write(chunk)
                        on_bytes(len(chunk))
        except urllib.error.HTTPError as exc:
            raise DownloadError(f'{dest.name}: HTTP {exc.code} from {url}') from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise DownloadError(f'{dest.name}: {getattr(exc, "reason", exc)}') from exc
    got = part.stat().st_size
    if got != size:
        raise DownloadError(f'{dest.name}: got {got:,} bytes, expected {size:,}. Try again to resume.')
    if _sha256(part, cancel) != sha256:
        part.unlink()
        raise DownloadError(f'{dest.name} does not match its pinned sha256; the download was discarded.')
    os.replace(part, dest)


def ensure_weights(folder: Path, progress: Callable[[int, int], None],
                   cancel: Callable[[], bool]) -> None:
    """Download whatever is missing. `progress(done, total)` is in bytes."""
    if weights_present(folder):
        return
    if (os.environ.get('MING_MODEL_DIR') or '').strip():
        missing = [name for name, _s, _h in MODEL_FILES if not (folder / name).is_file()]
        raise DownloadError(f'MING_MODEL_DIR={folder} lacks {", ".join(missing)}')
    folder.mkdir(parents=True, exist_ok=True)
    endpoint = (os.environ.get('HF_ENDPOINT') or 'https://huggingface.co').rstrip('/')
    total = missing_bytes(folder)
    done = 0

    def on_bytes(count: int) -> None:
        nonlocal done
        done += count
        progress(min(done, total), total)

    for name, size, sha256 in MODEL_FILES:
        dest = folder / name
        if dest.is_file() and dest.stat().st_size == size and _sha256(dest, cancel) == sha256:
            continue
        part = folder / (name + '.part')
        before = part.stat().st_size if part.is_file() else 0
        done -= before  # _fetch reports the bytes already on disk again
        url = f'{endpoint}/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}'
        log.info('fetching %s (%s bytes)', url, f'{size:,}')
        _fetch(url, dest, size, sha256, on_bytes, cancel)
    (folder / _VERIFIED).write_text(MODEL_REVISION + '\n', encoding='utf-8')


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
# The decoding loop runs as `for step in tqdm(range(...))`. The module's tqdm
# is swapped for this, which reports each step to the thread's callback:
# progress, and a cancel that stops a line between two steps. Every
# generation refills the KV cache from its prompt, so stopping one leaves
# nothing behind for the next.
_step = threading.local()

# Decoding steps per second of audio: 50 latent frames a second, 4 per step.
STEPS_PER_SECOND = 12.5


def _stepping(iterable):
    callback = getattr(_step, 'callback', None)
    for item in iterable:
        if callback is not None:
            callback()
        yield item


@dataclass
class Loaded:
    tts: object            # ming_addon.ming.ming_tts.MingTTS
    device: str
    sample_rate: int


def _cpu_list(text: str) -> list[int]:
    """'0-3,8,10-11' -> [0, 1, 2, 3, 8, 10, 11]"""
    cpus: list[int] = []
    for part in text.strip().split(','):
        if '-' in part:
            low, high = part.split('-', 1)
            cpus.extend(range(int(low), int(high) + 1))
        elif part:
            cpus.append(int(part))
    return cpus


def performance_cores() -> int:
    """Physical performance cores of a hybrid CPU; 0 when not hybrid or not known.

    PyTorch's default, one thread per physical core, counts efficiency cores
    too, and the slower cores then hold every step back.
    """
    try:
        if sys.platform.startswith('linux'):
            cpus = _cpu_list(Path('/sys/devices/cpu_core/cpus').read_text())
            siblings = Path(f'/sys/devices/system/cpu/cpu{cpus[0]}/topology/thread_siblings_list')
            return max(1, len(cpus) // max(1, len(_cpu_list(siblings.read_text()))))
        if sys.platform == 'darwin':
            out = subprocess.run(['sysctl', '-n', 'hw.perflevel0.physicalcpu', 'hw.perflevel1.physicalcpu'],
                                 capture_output=True, text=True, timeout=5)
            counts = [int(line) for line in out.stdout.split()]
            return counts[0] if len(counts) == 2 and counts[1] > 0 else 0
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    return 0


def resolve_device(choice: str) -> str:
    import torch
    choice = (choice or 'auto').strip().lower()
    if choice == 'auto':
        # Not the Apple GPU: Ming-3.0-tts has not been tried on it, and the
        # vocoder's complex arithmetic may not be there. It is a choice.
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return choice


# The parts kept in bfloat16 by the lean layout: the language model and the
# audio encoder, which reads the reference clip. Both lose little speed in
# bfloat16 on a CPU, unlike the flow head and the audio decoder.
LEAN_PREFIXES = ('model.', 'audio.encoder.')


def build_model(folder: Path, device: str, lean: bool = True):
    """The model on `device`, from the checkpoint in `folder`.

    float32 throughout, or with `lean` the parts in LEAN_PREFIXES in
    bfloat16, which saves about 1.8 GB.
    """
    import torch
    from safetensors import safe_open
    from ming_addon.ming.configuration_bailingmm import BailingMMConfig
    from ming_addon.ming.modeling_bailingmm import BailingMMNativeForConditionalGeneration

    config = BailingMMConfig.from_pretrained(str(folder))
    with torch.device('meta'):
        model = BailingMMNativeForConditionalGeneration(config)
    state = {}
    with safe_open(str(folder / 'model.safetensors'), 'pt') as checkpoint:
        for key in checkpoint.keys():
            tensor = checkpoint.get_tensor(key)
            if tensor.is_floating_point():
                dtype = torch.bfloat16 if lean and key.startswith(LEAN_PREFIXES) else torch.float32
                tensor = tensor.to(device=device, dtype=dtype)
            state[key] = tensor.to(device)
            del tensor
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    del state
    if unexpected:
        raise RuntimeError(f'checkpoint has tensors the model does not: {unexpected[:5]}')
    if set(missing) - {'model.lm_head.weight'}:
        raise RuntimeError(f'checkpoint lacks {sorted(set(missing))[:5]}')

    # Tied, as upstream's loader ties it.
    model.model.lm_head.weight = model.model.get_input_embeddings().weight
    # The rotary tables are computed, not stored: rebuild them on the device.
    for module in list(model.modules()):
        rotary = getattr(module, 'rotary_emb', None)
        if rotary is not None and any(buffer.is_meta for buffer in rotary.buffers()):
            with torch.device(device):
                module.rotary_emb = type(rotary)(config=module.config)
    on_meta = [name for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
               if tensor.is_meta]
    if on_meta:
        raise RuntimeError(f'not loaded: {", ".join(on_meta[:5])}')
    return model.eval()


def load(folder: Path, device: str, threads: int = 0, lean: bool = True) -> Loaded:
    import torch
    from transformers import Qwen2TokenizerFast
    from ming_addon.ming import modeling_bailingmm
    from ming_addon.ming.ming_tts import MingTTS
    from ming_addon.ming.spkemb_extractor import SpkembExtractor

    modeling_bailingmm.tqdm = _stepping
    threads = threads if threads > 0 else performance_cores()
    if threads > 0:
        torch.set_num_threads(threads)
    model = build_model(folder, device, lean)
    # Upstream's __init__ loads the model its own way (bfloat16, Hub download).
    tts = MingTTS.__new__(MingTTS)
    tts.device = device
    tts.local_path = str(folder)
    tts.model = model
    model.tokenizer = Qwen2TokenizerFast.from_pretrained(str(folder))  # validated by the setter
    tts.sample_rate = model.config.audio_tokenizer_config.sample_rate
    tts.patch_size = model.patch_size
    tts.spkemb_extractor = SpkembExtractor(str(folder / 'campplus.onnx'))
    # Upstream's FST normaliser needs pynini, which has no Windows or macOS
    # wheels; numbers in English text are spelled out by text.py instead.
    tts.text_normalization = False
    tts._normalizer = None
    # Prompts are cached by path; the host's clips are temporary files whose
    # names can come back with other audio in them.
    tts.cache_prompts = False
    tts._prompt_cache = {}
    return Loaded(tts=tts, device=device, sample_rate=int(tts.sample_rate))


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
def synthesize(loaded: Loaded, *, seed: int | None = None,
               on_step: Callable[[], None] | None = None, **request):
    """A float32 waveform at `loaded.sample_rate`.

    `request` is what upstream's MingTTS.synthesize takes (task, text,
    language_id, controls, prompt_wav_path, prompt_text, use_spk_emb,
    use_zero_spk_emb). `on_step()` runs before each decoding step; raising
    from it stops the line.
    """
    import torch
    if seed is not None:
        torch.manual_seed(int(seed) & 0xFFFFFFFF)
    _step.callback = on_step
    try:
        with torch.inference_mode():
            wav = loaded.tts.synthesize(**request)
        return wav.reshape(-1).float().numpy()
    finally:
        _step.callback = None
