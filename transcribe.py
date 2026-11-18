#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["soundcard>=0.4.3", "numpy", "requests", "faster-whisper", "sherpa-onnx"]
# ///
"""Live transcription of system audio + microphone to plain text.

Both sources are mixed, cut at natural pauses, and each piece is transcribed:
by Whisper running in this process by default (faster-whisper, on the GPU when
there is one), or by an OpenAI-compatible /audio/transcriptions endpoint
(Ollama on this machine, or a hosted API).

Setup
    pip install soundcard numpy requests faster-whisper sherpa-onnx
    pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"   # NVIDIA GPU only
    ollama pull gemma4:e4b            # only for --backend ollama

Usage
    python transcribe.py                          # local Whisper large-v3, English
    python transcribe.py --language auto          # let Whisper detect the language
    python transcribe.py --backend ollama         # local, through Ollama
    python transcribe.py --backend openai         # hosted, needs OPENAI_API_KEY
    python transcribe.py --base-url URL --model NAME --api-key KEY   # anything compatible
    python transcribe.py --list-devices

Ctrl+C stops recording; audio still queued is transcribed before exit.
The transcript goes to stdout and a text file. Status messages go to stderr,
so `python transcribe.py > notes.txt` gives you only the text.

System-audio capture works on Windows (WASAPI loopback) and Linux
(PulseAudio/PipeWire monitor). macOS needs a virtual device such as BlackHole,
selected with --mic.
"""
from __future__ import annotations

import argparse
import io
import os
import queue
import sys
import threading
import time
import wave
import warnings
from collections import deque
from datetime import datetime, timedelta
from urllib.parse import urlparse

import numpy as np

SR = 16000            # sample rate sent to the model
FPS = 10              # analysis frames per second
FRAME = SR // FPS     # 100 ms
LEAD_IN = 3           # frames of silence kept around speech (0.3 s)
MIN_VOICED = 3        # a chunk needs at least this many loud frames to be sent

BACKENDS = {
    "whisper": {"base_url": None, "model": "large-v3"},
    "ollama": {"base_url": "http://localhost:11434/v1", "model": "gemma4:e4b"},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-transcribe"},
}


def log(*parts):
    print(*parts, file=sys.stderr, flush=True)


def describe(exc):
    """An exception as text; some (a bare assert) carry no message of their own."""
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


# ---------------------------------------------------------------- capture ---

class Source:
    """One capture device, read on its own thread into a sample buffer."""

    def __init__(self, label, device):
        self.label, self.device = label, device
        self.error = None
        self.reported = False
        self._chunks = deque()
        self._count = 0
        self._lock = threading.Lock()
        self.thread = None

    def start(self, stop):
        self.thread = threading.Thread(target=self._run, args=(stop,), daemon=True)
        self.thread.start()

    def _run(self, stop):
        try:
            if sys.platform == "win32":  # WASAPI is COM: make sure this thread has joined it
                import ctypes
                ctypes.windll.ole32.CoInitializeEx(None, 0)
            # Record the device's own channel layout (the OS resamples to 16 kHz
            # for us) and fold it to mono here. Dividing by at most 2 keeps
            # stereo content at full level on surround outputs.
            with self.device.recorder(samplerate=SR) as recorder:
                while not stop.is_set():
                    block = recorder.record(numframes=FRAME)
                    mono = (block.sum(axis=1) / min(block.shape[1], 2)).astype(np.float32)
                    with self._lock:
                        self._chunks.append(mono)
                        self._count += len(mono)
        except Exception as exc:  # reported by the main loop
            self.error = exc

    def available(self):
        with self._lock:
            return self._count

    def take(self, n):
        """Remove and return exactly n samples, zero-padded if fewer are buffered."""
        out = np.zeros(n, np.float32)
        filled = 0
        with self._lock:
            while filled < n and self._chunks:
                chunk = self._chunks[0]
                k = min(len(chunk), n - filled)
                out[filled:filled + k] = chunk[:k]
                if k == len(chunk):
                    self._chunks.popleft()
                else:
                    self._chunks[0] = chunk[k:]
                filled += k
                self._count -= k
        return out


def pull_mix(sources, everything=False):
    """Take the samples all live sources have in common.

    Returns an (n, 1 + len(sources)) array: column 0 is the clipped sum, then
    one column per source (silent once that source has failed), so later
    stages can still tell who was talking into which device.
    """
    live = [s for s in sources if s.error is None]
    if not live:
        return None
    counts = [s.available() for s in live]
    n = max(counts) if everything else min(counts)
    if not everything and max(counts) - n > SR:
        n = max(counts) - SR // 2      # one source stalled: don't let it hold the other back
    if n < (1 if everything else FRAME):
        return None
    out = np.zeros((n, 1 + len(sources)), np.float32)
    for i, source in enumerate(sources, start=1):
        if source.error is None:
            out[:, i] = source.take(n)
    out[:, 0] = np.clip(out[:, 1:].sum(axis=1), -1.0, 1.0)
    return out


# --------------------------------------------------------------- chunking ---

class Chunker:
    """Cuts a continuous stream into speech chunks at pauses.

    The loudness threshold follows the background noise level, so it adapts to
    quiet and noisy microphones alike. Stretches with no speech are dropped.
    """

    def __init__(self, min_s=3.0, max_s=28.0, pause_s=0.7):
        self.max_f = max(int(max_s * FPS), 2 * FPS)
        self.min_f = min(int(min_s * FPS), self.max_f)
        self.pause_f = max(1, round(pause_s * FPS))
        self.frames, self.levels, self.loud = [], [], []
        self.started = False
        self.quiet_run = 0
        self.history = deque(maxlen=30 * FPS)
        self.rest = None
        self.seen = 0

    def feed(self, samples):
        """Add (n, channels) samples, column 0 being the mix; return the chunks
        completed so far as (start_seconds, audio of the same channels)."""
        done = []
        buf = samples if self.rest is None or not len(self.rest) else np.concatenate([self.rest, samples])
        whole = len(buf) // FRAME
        for i in range(whole):
            chunk = self._push(buf[i * FRAME:(i + 1) * FRAME])
            if chunk is not None:
                done.append(chunk)
        self.rest = buf[whole * FRAME:]
        return done

    def flush(self):
        return self._cut(len(self.frames)) if self.frames else None

    def _threshold(self):
        if len(self.history) < FPS // 2:    # first half second: still measuring the room
            return float("inf")
        floor = float(np.percentile(self.history, 10))
        return min(max(3.0 * floor, 0.004), 0.03)

    def _push(self, frame):
        level = float(np.sqrt(np.mean(np.square(frame[:, 0], dtype=np.float64))))
        self.history.append(level)
        self.seen += 1
        is_loud = level >= self._threshold()
        self.frames.append(frame)
        self.levels.append(level)
        self.loud.append(is_loud)

        if is_loud:
            self.started, self.quiet_run = True, 0
        elif not self.started:          # still waiting for speech: keep a short lead-in only
            if len(self.frames) > LEAD_IN:
                del self.frames[0], self.levels[0], self.loud[0]
            return None
        else:
            self.quiet_run += 1

        n = len(self.frames)
        if n >= self.min_f and self.quiet_run >= self.pause_f:
            return self._cut(n)
        if n >= self.max_f:             # no pause in sight: cut at the quietest recent moment
            tail = min(5 * FPS, n // 2)
            return self._cut(n - tail + int(np.argmin(self.levels[-tail:])) + 1)
        return None

    def _cut(self, k):
        """Emit the first k buffered frames as a chunk and carry the rest over."""
        head, head_loud = self.frames[:k], self.loud[:k]
        del self.frames[:k], self.levels[:k], self.loud[:k]
        start_s = (self.seen - k - len(self.frames)) / FPS

        self.started = any(self.loud)
        if self.started:
            self.quiet_run = next(i for i, loud in enumerate(reversed(self.loud)) if loud)
        else:
            self.quiet_run = 0
            del self.frames[:-LEAD_IN], self.levels[:-LEAD_IN], self.loud[:-LEAD_IN]

        if sum(head_loud) < MIN_VOICED:
            return None
        last_loud = max(i for i, loud in enumerate(head_loud) if loud)
        return start_s, np.concatenate(head[:last_loud + 1 + LEAD_IN])


# ---------------------------------------------------------- transcription ---

def to_pcm16(audio):
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def wav_bytes(audio):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(to_pcm16(audio))
    return buf.getvalue()


class Rejected(Exception):
    """The server refused the request (4xx); sending the same thing again won't help."""


class Transcriber:
    """Client for an OpenAI-compatible /audio/transcriptions endpoint."""

    def __init__(self, base_url, model, api_key=None, language=None, prompt=None):
        import requests
        self._requests = requests
        self.url = base_url.rstrip("/") + "/audio/transcriptions"
        self.model, self.language, self.prompt = model, language, prompt
        self.http = requests.Session()
        if api_key:
            self.http.headers["Authorization"] = f"Bearer {api_key}"

    def __call__(self, audio, attempts=3):
        fields = [("model", self.model)]
        if self.language:
            # OpenAI's gpt-transcribe takes a `languages` list; other servers take `language`.
            key = "languages[]" if self.model.startswith("gpt-transcribe") else "language"
            fields.append((key, self.language))
        if self.prompt:
            fields.append(("prompt", self.prompt))
        files = {"file": ("audio.wav", wav_bytes(audio), "audio/wav")}

        error = None
        for attempt in range(attempts):
            try:
                r = self.http.post(self.url, data=fields, files=files, timeout=(10, 300))
            except self._requests.RequestException as exc:
                error = exc
            else:
                if r.ok:    # no word timing from a generic API: the whole chunk is one segment
                    text = self._text(r)
                    return [(0.0, len(audio) / SR, text)] if text else []
                detail = f"HTTP {r.status_code}: {r.text.strip()[:300]}"
                if r.status_code < 500 and r.status_code not in (408, 429):
                    raise Rejected(detail)
                error = RuntimeError(detail)
            if attempt + 1 < attempts:
                time.sleep(2 * (attempt + 1))
        raise error

    @staticmethod
    def _text(response):
        try:
            body = response.json()
        except ValueError:
            return response.text.strip()
        if isinstance(body, dict):
            return str(body.get("text") or "").strip()
        return str(body).strip()


def _add_nvidia_dlls():
    """On Windows, let CTranslate2 find the cuBLAS/cuDNN DLLs from the nvidia-* pip wheels."""
    if sys.platform != "win32":
        return
    import site
    roots = site.getsitepackages() + [site.getusersitepackages()]
    for root in roots:
        nvidia = os.path.join(root, "nvidia")
        if not os.path.isdir(nvidia):
            continue
        for lib in os.listdir(nvidia):
            bin_dir = os.path.join(nvidia, lib, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


class LocalWhisper:
    """Whisper in this process via faster-whisper. Same call shape as Transcriber.

    Each chunk is decoded with the end of the previous text as its prompt, so
    sentences cut at a pause keep their context (names, spelling, punctuation).
    """

    CONTEXT_CHARS = 400

    def __init__(self, model, language=None, prompt=None, device="auto"):
        _add_nvidia_dlls()
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            sys.exit("Missing dependency. Run: pip install faster-whisper")
        self.model_name, self.language, self.prompt = model, language, prompt
        self.model = None
        if device in ("auto", "cuda"):
            try:
                self.model = WhisperModel(model, device="cuda", compute_type="float16")
                self.device = "cuda"
            except Exception as exc:
                if device == "cuda":
                    raise
                log(f"GPU unavailable ({exc}); running Whisper on the CPU, which is much slower.")
        if self.model is None:
            self.model = WhisperModel(model, device="cpu", compute_type="int8")
            self.device = "cpu"
        self.context = ""

    supports_preview = True     # cheap enough to decode unfinished speech for a provisional text

    def __call__(self, audio, attempts=1, preview=False):
        """Return [(start_s, end_s, text), ...] relative to the start of audio.

        A preview is a quick look at speech that is still going on: decoded
        greedily, and left out of the running context, because the same audio
        comes back for its real decoding once the chunk is complete.
        """
        initial = " ".join(p for p in (self.prompt, self.context) if p) or None
        segments, _ = self.model.transcribe(
            audio.astype(np.float32),
            language=self.language,
            task="transcribe",
            beam_size=1 if preview else 5,
            initial_prompt=initial,
            condition_on_previous_text=False,
            vad_filter=True,                    # second guard against "Thank you." on noise
            vad_parameters={"min_silence_duration_ms": 500},
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        result = [(s.start, s.end, s.text.strip()) for s in segments
                  if s.no_speech_prob < 0.8 and s.text.strip()]
        text = " ".join(t for _, _, t in result)
        if text and not preview:
            self.context = (self.context + " " + text)[-self.CONTEXT_CHARS:]
        return result


# --------------------------------------------------------------- speakers ---

SPEAKER_MODEL = "nemo_en_titanet_small.onnx"
SPEAKER_MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                     "speaker-recongition-models/" + SPEAKER_MODEL)


def model_dir():
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache")
    path = os.path.join(base, "silent-ear")
    os.makedirs(path, exist_ok=True)
    return path


def fetch_speaker_model():
    path = os.path.join(model_dir(), SPEAKER_MODEL)
    if not os.path.exists(path):
        import requests
        log(f"Downloading speaker model {SPEAKER_MODEL} (40 MB) ...")
        r = requests.get(SPEAKER_MODEL_URL, timeout=(10, 300))
        r.raise_for_status()
        with open(path + ".part", "wb") as f:
            f.write(r.content)
        os.replace(path + ".part", path)
    return path


class Voices:
    """The voices heard so far on one channel, numbered in order of appearance."""

    def __init__(self, prefix):
        self.prefix = prefix
        self.centroids, self.counts = [], []
        self.last = None

    def name(self, index):
        return f"{self.prefix} {index + 1}"


class Speakers:
    """Names who is talking in each transcribed segment.

    With both a microphone and system audio, the devices themselves tell the
    two sides apart: speech on the microphone while the system audio is quiet
    is you; anything heard through the system audio is someone else. Remote
    voices (or every voice, with a single device) are then told apart by a
    speaker-embedding model and clustered on the fly: each new voice that
    matches no known one becomes the next "Speaker N".
    """

    MIN_NEW_S = 1.5     # a voice must speak this long before it can become a new speaker
    MIN_ID_S = 0.8      # shorter than this: too little audio to judge, keep the last speaker

    KNOWN_MARGIN = 0.05     # a saved voice must match a little better than a voice of this recording:
                            # a wrong name is worse than an unnamed "Speaker 2"

    def __init__(self, labels, threshold=0.6, split_mic=False, known=None):
        import sherpa_onnx
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=fetch_speaker_model(), num_threads=2)
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.mic_col = 1 + labels.index("Mic") if "Mic" in labels else None
        self.sys_col = 1 + labels.index("System") if "System" in labels else None
        self.threshold = threshold
        self.remote = Voices("Speaker")
        # Several people sharing the microphone: cluster its voices too ("Mic 1", "Mic 2", ...)
        # instead of calling everything heard there "You".
        self.local = Voices("Mic") if split_mic else None
        # Voices saved from earlier recordings, as (name, unit vector). When a voice of this
        # recording turns out to be one of them, `recognised` maps its label to that name.
        self.known = [(name, np.asarray(vector, np.float32)) for name, vector in (known or [])]
        self.recognised = {}

    def embed(self, voice):
        """The voice print of a stretch of speech: a unit vector, close to others of the same voice."""
        return self._embed(voice)

    def label(self, audio, start_s, end_s):
        """Who spoke audio[start_s:end_s] (audio as produced by pull_mix)."""
        seg = audio[max(0, int(start_s * SR)):max(int(end_s * SR), int(start_s * SR) + 1)]
        if self.mic_col is not None and self.sys_col is not None:
            if self._active(seg[:, self.sys_col]) < 0.25:
                if self.local is None:
                    return "You"
                return self._identify(seg[:, self.mic_col], self.local)
            return self._identify(seg[:, self.sys_col], self.remote)
        return self._identify(seg[:, 0], self.remote)

    @staticmethod
    def _active(signal):
        """Share of 100 ms frames carrying sound. Loopback audio is digitally
        clean, so a fixed floor works; the mic's echo of the call can't fool it."""
        n = len(signal) // FRAME
        if n == 0:
            return 0.0
        frames = signal[:n * FRAME].reshape(n, FRAME)
        return float(np.mean(np.sqrt(np.mean(frames ** 2, axis=1)) > 0.003))

    def _embed(self, voice):
        stream = self.extractor.create_stream()
        stream.accept_waveform(SR, np.ascontiguousarray(voice, dtype=np.float32))
        stream.input_finished()
        e = np.asarray(self.extractor.compute(stream), np.float32)
        return e / (np.linalg.norm(e) + 1e-9)

    def _identify(self, voice, voices):
        seconds = len(voice) / SR
        if seconds < self.MIN_ID_S and voices.last is not None:
            return voices.name(voices.last)
        e = self._embed(voice)
        best, score = None, -1.0
        if voices.centroids:
            scores = np.array(voices.centroids) @ e
            best = int(np.argmax(scores))
            score = float(scores[best])
        if best is None or (score < self.threshold and seconds >= self.MIN_NEW_S):
            voices.centroids.append(e)
            voices.counts.append(1)
            best = len(voices.centroids) - 1
        elif score >= self.threshold:   # confident match: let the voice profile learn
            n = min(voices.counts[best], 20)
            c = voices.centroids[best] * n + e
            voices.centroids[best] = c / np.linalg.norm(c)
            voices.counts[best] += 1
        voices.last = best
        label = voices.name(best)
        settled = voices.counts[best] >= 2 or seconds >= 2 * self.MIN_NEW_S    # not on a first short scrap
        if self.known and settled and label not in self.recognised:
            centroid = voices.centroids[best]
            scores = [float(vector @ centroid) if len(vector) == len(centroid) else -1.0
                      for _, vector in self.known]
            top = int(np.argmax(scores))
            if scores[top] >= self.threshold + self.KNOWN_MARGIN:
                self.recognised[label] = self.known[top][0]
        return label


class Output:
    """Writes each transcribed chunk to stdout and, optionally, a file.

    When a speaker is given and differs from the previous line's, a blank line
    and "Name:" mark the change of turn.
    """

    def __init__(self, path, timestamps, started_at):
        self.file = open(path, "a", encoding="utf-8") if path else None
        self.timestamps, self.started_at = timestamps, started_at
        self.speaker = None
        self.lines = 0

    def write(self, start_s, text, speaker=None):
        lines = []
        if speaker is not None and speaker != self.speaker:
            if self.lines:
                lines.append("")
            text = f"{speaker}: {text}"
            self.speaker = speaker
        if self.timestamps:
            clock = self.started_at + timedelta(seconds=start_s)
            text = f"[{clock:%H:%M:%S}] {text}"
        lines.append(text)
        self.lines += 1
        block = "\n".join(lines)
        print(block, flush=True)
        if self.file:
            self.file.write(block + "\n")
            self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


def speaker_turns(segments, audio, speakers, with_end=False):
    """Label each segment and merge neighbours with the same speaker:
    [(start_s, speaker or None, text), ...], with end_s as a fourth item on request."""
    turns = []
    for start, end, text in segments:
        who = None
        if speakers is not None:
            try:
                who = speakers.label(audio, start, end)
            except Exception as exc:
                log(f"[speaker detection failed: {exc}]")
        if turns and turns[-1][1] == who:
            turns[-1] = (turns[-1][0], who, turns[-1][2] + " " + text, end)
        else:
            turns.append((start, who, text, end))
    return turns if with_end else [turn[:3] for turn in turns]


def transcribe_worker(jobs, transcribe, output, state, stop, speakers=None):
    while True:
        job = jobs.get()
        if job is None:
            return
        start_s, audio = job
        seconds = len(audio) / SR
        try:
            segments = transcribe(audio[:, 0])
        except Rejected as exc:
            if state["ok"] == 0:        # nothing has ever worked: a setup problem, so stop
                state["fatal"] = exc
                stop.set()
                return
            log(f"[skipped {seconds:.0f}s of audio: {exc}]")
        except Exception as exc:
            log(f"[skipped {seconds:.0f}s of audio: {exc}]")
        else:
            state["ok"] += 1
            for offset, who, text in speaker_turns(segments, audio, speakers):
                output.write(start_s + offset, text, who)


# ---------------------------------------------------------------- devices ---

def _accept_plain_mix_formats():
    """Let soundcard open devices whose native format is not WAVE_FORMAT_EXTENSIBLE.

    soundcard builds its stream format by editing the device's own "mix format"
    and asserts that this is an extensible float one. Some devices (16-bit mono
    USB headset microphones, for instance) report a plain format instead, and
    opening them dies with a bare AssertionError. When that happens we hand
    WASAPI a float32 format of our own: in shared mode, with the auto-convert
    flags soundcard already sets, Windows converts from whatever the device uses.
    """
    import struct
    import uuid
    mf = sys.modules.get("soundcard.mediafoundation")
    client = getattr(mf, "_AudioClient", None)
    if client is None or getattr(client, "_plain_formats", False):
        return
    original = client.__init__
    ieee_float = uuid.UUID("00000003-0000-0010-8000-00aa00389b71").bytes_le

    def init(self, ptr, samplerate, channels, blocksize, isloopback, exclusive_mode=False):
        try:
            return original(self, ptr, samplerate, channels, blocksize, isloopback, exclusive_mode)
        except AssertionError:
            if exclusive_mode:
                raise
        self._ptr = ptr                 # nothing was initialised yet: the assert comes first
        self.channelmap = list(range(channels)) if isinstance(channels, int) else list(channels)
        count = len(set(self.channelmap))
        if blocksize is None:
            blocksize = self.deviceperiod[0] * samplerate
        mask = {1: 0x4, 2: 0x3}.get(count, (1 << count) - 1)       # front centre / left + right
        # WAVEFORMATEXTENSIBLE, packed as Windows lays it out (40 bytes).
        wave_format = mf._ffi.new("char[]", struct.pack(
            "<HHIIHHHHI", 0xFFFE, count, int(samplerate), int(samplerate) * count * 4,
            count * 4, 32, 22, 32, mask) + ieee_float)
        flags = 0x00100000 | 0x80000000 | 0x08000000 | 0x00080000   # as soundcard: convert, resample
        if isloopback:
            flags |= 0x00020000
        duration = int(blocksize / samplerate * 10000000)
        hr = ptr[0][0].lpVtbl.Initialize(
            ptr[0], mf._ole32.AUDCLNT_SHAREMODE_SHARED, flags, duration, 0,
            mf._ffi.cast("WAVEFORMATEXTENSIBLE*", wave_format), mf._ffi.NULL)
        mf._com.check_error(hr)
        self.samplerate = samplerate
        self._idle_start_time = None

    client.__init__ = init
    client._plain_formats = True


def load_soundcard():
    try:
        import soundcard
    except ImportError:
        sys.exit("Missing dependency. Run: pip install soundcard numpy requests")
    except Exception as exc:  # e.g. no audio server on Linux
        sys.exit(f"Could not initialise audio: {exc}")
    warnings.filterwarnings("ignore", category=soundcard.SoundcardRuntimeWarning)
    if sys.platform == "win32":
        try:
            _accept_plain_mix_formats()
        except Exception as exc:    # soundcard changed inside: its own behaviour stays
            log(f"Could not extend audio device support: {exc}")
    return soundcard


def loopback_of(sc, speaker):
    """The recordable 'what you hear' twin of an output device."""
    loopbacks = [m for m in sc.all_microphones(include_loopback=True) if m.isloopback]
    for mic in loopbacks:
        if mic.id in (speaker.id, f"{speaker.id}.monitor"):   # Windows / PulseAudio naming
            return mic
    for mic in loopbacks:
        if speaker.name in mic.name:
            return mic
    raise LookupError(f"no loopback device found for '{speaker.name}'")


def open_sources(sc, args):
    sources = []
    if not args.no_mic:
        try:
            mic = sc.get_microphone(args.mic) if args.mic else sc.default_microphone()
            sources.append(Source("Mic", mic))
        except Exception as exc:
            log(f"Microphone unavailable, continuing without it ({exc})")
    if not args.no_system:
        if sys.platform == "darwin":
            log("System audio isn't capturable on macOS directly; route it through a "
                "virtual device (e.g. BlackHole) and pick that with --mic.")
        else:
            try:
                speaker = sc.get_speaker(args.speaker) if args.speaker else sc.default_speaker()
                sources.append(Source("System", loopback_of(sc, speaker)))
            except Exception as exc:
                log(f"System audio unavailable, continuing without it ({exc})")
    return sources


def list_devices(sc):
    default_mic = default_speaker = None
    try:
        default_mic = sc.default_microphone().id
    except Exception:
        pass
    try:
        default_speaker = sc.default_speaker().id
    except Exception:
        pass
    print("Microphones (--mic):")
    for mic in sc.all_microphones():
        print(f"  {'*' if mic.id == default_mic else ' '} {mic.name}")
    print("Outputs whose audio can be captured (--speaker):")
    for speaker in sc.all_speakers():
        print(f"  {'*' if speaker.id == default_speaker else ' '} {speaker.name}")
    print("* = default")


# ------------------------------------------------------------------- main ---

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Transcribe system audio + microphone to plain text.",
        epilog="Prefer the OPENAI_API_KEY / TRANSCRIBE_API_KEY environment variables over "
               "--api-key, which ends up in your shell history.")
    p.add_argument("--backend", choices=BACKENDS, default="whisper",
                   help="whisper = local faster-whisper (default); ollama = local via Ollama; "
                        "openai = hosted API")
    p.add_argument("--base-url", help="any OpenAI-compatible API root (switches to that API)")
    p.add_argument("--model", help="default: large-v3 (whisper), gemma4:e4b (ollama), "
                                   "gpt-transcribe (openai)")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                   help="where local Whisper runs (default: GPU if available)")
    p.add_argument("--api-key", help="default: $TRANSCRIBE_API_KEY, or $OPENAI_API_KEY for openai")
    p.add_argument("--language", default="en",
                   help="spoken language as an ISO code, e.g. en or de; 'auto' to detect (default: en)")
    p.add_argument("--prompt", help="context hint such as names or jargon (support varies by model)")
    p.add_argument("-o", "--output",
                   help="transcript file (default: transcripts/transcript-<time>.txt; '-' for none)")
    p.add_argument("--timestamps", action="store_true", help="prefix each line with the clock time")
    p.add_argument("--save-audio", metavar="FILE.wav", help="also keep the mixed recording")
    p.add_argument("--mic", help="microphone name (substring is enough)")
    p.add_argument("--speaker", help="output device to capture (default: the system default)")
    p.add_argument("--no-mic", action="store_true", help="system audio only")
    p.add_argument("--no-system", action="store_true", help="microphone only")
    p.add_argument("--no-speakers", action="store_true",
                   help="don't detect who is speaking")
    p.add_argument("--mic-speakers", action="store_true",
                   help="several people share the microphone: tell them apart as Mic 1, Mic 2, ... "
                        "instead of labelling everything said there as You")
    p.add_argument("--speaker-threshold", type=float, default=0.6,
                   help="voice similarity (0-1) needed to count as a known speaker; raise it if "
                        "different people get merged, lower it if one person gets split (default 0.6)")
    p.add_argument("--pause", type=float, default=0.7, help="silence that ends a chunk, in seconds")
    p.add_argument("--min-chunk", type=float, default=5.0,
                   help="shortest chunk, in seconds; longer chunks give the model more context")
    p.add_argument("--max-chunk", type=float, default=28.0, help="longest chunk, in seconds")
    p.add_argument("--list-devices", action="store_true", help="show audio devices and exit")
    return p.parse_args(argv)


def is_local(url):
    return urlparse(url).hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0")


def warm_up(transcribe, base_url, model):
    """Load a local model before listening, and surface setup errors early."""
    log(f"Loading {model} ...")
    try:
        transcribe(np.zeros(SR, np.float32), attempts=1)
    except Rejected as exc:
        sys.exit(f"The server at {base_url} rejected the request: {exc}\n"
                 f"Check the model name. With Ollama: `ollama pull {model}` "
                 f"(it must be a model with audio input, on a recent Ollama).")
    except Exception as exc:
        sys.exit(f"Could not reach {base_url}: {exc}\nIs the server running? (Ollama: `ollama serve`)")


def main(argv=None):
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    sc = load_soundcard()
    if args.list_devices:
        list_devices(sc)
        return

    language = None if args.language.lower() == "auto" else args.language
    backend = "ollama" if args.backend == "whisper" and args.base_url else args.backend
    preset = BACKENDS[backend]
    base_url = args.base_url or preset["base_url"]
    model = args.model or preset["model"]

    if backend == "whisper":
        log(f"Loading Whisper {model} (downloaded on first use) ...")
        transcribe = LocalWhisper(model, language, args.prompt, args.device)
        where = f"local Whisper on {transcribe.device}"
    else:
        api_key = args.api_key or os.environ.get("TRANSCRIBE_API_KEY")
        if backend == "openai":
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                sys.exit("Set OPENAI_API_KEY (or pass --api-key) to use the OpenAI backend.")
        transcribe = Transcriber(base_url, model, api_key, language, args.prompt)
        where = base_url

    sources = open_sources(sc, args)
    if not sources:
        sys.exit("No audio source available. Try --list-devices.")
    if backend != "whisper" and is_local(base_url):
        warm_up(transcribe, base_url, model)

    speakers = None
    if not args.no_speakers:
        try:
            speakers = Speakers([s.label for s in sources], args.speaker_threshold, args.mic_speakers)
        except ImportError:
            log("Speaker detection off: run `pip install sherpa-onnx` to enable it.")
        except Exception as exc:
            log(f"Speaker detection off: {exc}")

    started_at = datetime.now()
    out_path = None if args.output == "-" else args.output
    if args.output is None:             # next to the sessions the UI keeps
        os.makedirs("transcripts", exist_ok=True)
        out_path = os.path.join("transcripts", f"transcript-{started_at:%Y%m%d-%H%M%S}.txt")
    output = Output(out_path, args.timestamps, started_at)
    recording = None
    if args.save_audio:
        recording = wave.open(args.save_audio, "wb")
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(SR)

    stop = threading.Event()
    jobs = queue.Queue()
    state = {"ok": 0, "fatal": None}
    worker = threading.Thread(target=transcribe_worker, daemon=True,
                              args=(jobs, transcribe, output, state, stop, speakers))
    worker.start()
    chunker = Chunker(min_s=args.min_chunk, max_s=args.max_chunk, pause_s=args.pause)

    def process(mix):
        if mix is None:
            return
        if recording:
            recording.writeframes(to_pcm16(mix[:, 0]))
        for job in chunker.feed(mix):
            jobs.put(job)

    for source in sources:
        log(f"{source.label + ':':<8}{source.device.name}")
        source.start(stop)
    log(f"Model:  {model} @ {where}")
    log("Listening. Ctrl+C to stop." + (f" Saving to {out_path}" if out_path else ""))

    try:
        while not stop.is_set():
            time.sleep(0.1)
            for source in sources:
                if source.error and not source.reported:
                    source.reported = True
                    log(f"{source.label} capture stopped: {describe(source.error)}")
            if all(source.error for source in sources):
                state["fatal"] = state["fatal"] or RuntimeError("all audio sources failed")
                break
            process(pull_mix(sources))
    except KeyboardInterrupt:
        pass
    stop.set()

    if state["fatal"] is None:
        for source in sources:
            source.thread.join(timeout=2)
        process(pull_mix(sources, everything=True))
        last = chunker.flush()
        if last:
            jobs.put(last)
        jobs.put(None)
        if jobs.qsize() > 1:
            log(f"Stopped. Finishing {jobs.qsize() - 1} queued chunk(s); Ctrl+C again to skip.")
        try:
            while worker.is_alive():    # short joins keep Ctrl+C responsive on Windows
                worker.join(0.2)
        except KeyboardInterrupt:
            log("Skipped the remaining audio.")

    if recording:
        recording.close()
    output.close()
    if state["fatal"] is not None:
        sys.exit(f"Stopped: {state['fatal']}")
    if out_path:
        log(f"Transcript saved to {out_path}")


if __name__ == "__main__":
    main()
