# Silent Ear

Live transcription of your computer's audio and your microphone, with a local web UI.
It listens to a call or meeting, writes down who said what as it happens, keeps the
recording, and can produce minutes afterwards.

Speech recognition and speaker detection run on your own machine. Nothing is sent
anywhere unless you ask for a summary or choose a hosted transcription backend
(see [Privacy](#privacy)).

## Features

- **Live transcript** of system audio and microphone together, cut at natural pauses.
  With a GPU, grey provisional text appears while someone is still speaking and is
  replaced by the final line when the sentence ends.
- **Speaker detection.** The microphone and the call are told apart by device; voices
  within each are told apart by a speaker model. Rename "Speaker 1" once and every past
  and future line follows.
- **Remembered voices.** Naming a speaker saves their voice, and later recordings label
  that person from the start.
- **Editing.** Fix a line, join lines, reassign a line to another speaker, join or
  remove whole transcripts. Removed transcripts go to a trash folder.
- **Dictionary** of preferred spellings per language (company names, jargon). Terms are
  given to the model as a hint, and listed mishearings are always replaced.
- **Search** across all transcripts, summaries and titles, ignoring case and accents.
- **Device picker with level meters.** Choose the microphone and the output to record,
  and see at a glance whether each one delivers sound.
- **Language switching** during a recording.
- **Audio kept and playable.** Click a line's time to play from there.
- **Clean up** a transcript from its recording: a second pass in long pieces replaces the
  live lines, with better punctuation and fewer sentences cut in two. On demand, or
  automatically when a recording stops.
- **Re-transcribe** a saved recording with different settings, into a new transcript.
- **Import** audio and video files (mp3, m4a, wav, mp4, mkv, webm, ...).
- **Summaries** with decisions and action items, written by Claude (optional, needs an
  API key).
- **Export** to Markdown, Word (.docx) or subtitles (.srt).
- **Plain files.** Each session is a `.txt`, a self-describing `.json` and a `.wav`.

## Requirements

- Python 3.9 or newer.
- **Windows** or **Linux** for capturing system audio (WASAPI loopback, or a
  PulseAudio/PipeWire monitor). On macOS only the microphone works out of the box; route
  system audio through a virtual device such as BlackHole and select it with `--mic`.
- An NVIDIA GPU is strongly recommended. Whisper `large-v3` also runs on a CPU, but
  slower than real time on most machines; pick a smaller model there (`--model small`).

## Install

```bash
pip install -r requirements.txt
```

For an NVIDIA GPU, also install the CUDA libraries Whisper needs:

```bash
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```

The Whisper model (about 3 GB for `large-v3`) and a small speaker model are downloaded
on first use.

## Use

```bash
python ui.py
```

This opens `http://127.0.0.1:8765`. Wait for the model to load, press **Record**, and
press **Stop** when you are done. The page is served to your own machine only.

Common options:

```bash
python ui.py --language pl                  # start in Polish ("auto" detects per chunk)
python ui.py --model small --device cpu     # a lighter model, no GPU
python ui.py --min-chunk 2 --max-chunk 12   # shorter chunks: faster updates, less context
python ui.py --dir D:\meetings              # keep the data somewhere else
python ui.py --prompt "Terms: Kubernetes, PostgreSQL"
python ui.py --mic "Jabra" --speaker "Realtek"
```

`python ui.py --help` and `python transcribe.py --help` list everything.
`python transcribe.py --list-devices` shows the audio devices.

### Summaries

Summaries use the Anthropic API. Create a key in the Claude Console, set it as the
`ANTHROPIC_API_KEY` environment variable, and start `ui.py` from a terminal that has it.
The default model is `claude-opus-5-5`; change it with `--summary-model`. Each summary is
billed to your key, typically a few cents to a few tens of cents for an hour-long
meeting, and the UI shows an estimate for each one.

### Command line only

`transcribe.py` works on its own, without the UI, and prints the transcript as it goes:

```bash
python transcribe.py
python transcribe.py --backend ollama        # a local model through Ollama
python transcribe.py --backend openai        # hosted, needs OPENAI_API_KEY
```

## Where things are kept

Everything the tool produces goes to the `transcripts` folder (or `--dir`):

| File | What it is |
|---|---|
| `transcript-<time>-<title>.txt` | The transcript as plain text, rewritten after every change. |
| `transcript-<time>-<title>.json` | The source of truth: entries with speaker, time and language, the speaker list, the summary, and an `about` text explaining the fields. Readable on its own by a person or a language model. |
| `transcript-<time>-<title>.wav` | The recording: 16 kHz, 16-bit, one channel per audio source (about 230 MB per hour with two sources). |
| `dictionary.json` | Your word lists, per language. |
| `voices.json` | Voice prints of the speakers you named. They identify people: treat the file as personal data. |
| `settings.json` | The UI switches. |
| `trash/` | Transcripts you removed or joined away. |

The folder is ignored by git.

## Privacy

- **Local by default.** Audio capture, Whisper and speaker detection run on your machine.
  The models are downloaded once from their public sources.
- **Summaries leave the machine.** Pressing Summarise sends the transcript text, never
  the audio, to the Anthropic API. The UI asks before the first one.
- **Hosted backends leave the machine.** `--backend openai` or a remote `--base-url`
  sends the audio to that service.
- **The web page is local.** The server listens on `127.0.0.1` only and refuses requests
  from other sites and host names.
- **Recording other people.** Many places require the consent of everyone being recorded.
  Check what applies to you and tell the people on the call.

## Limits

- Speaker detection is a guess by voice. One person can be split in two and two similar
  voices can be merged; renaming both labels to the same name fixes a split.
- Speaker numbers restart with every recording. "Speaker 1" in two recordings need not
  be the same person, unless you named them: named voices are recognised again.
- Overlapping speech is attributed to one speaker, the remote one when both sides talk.
- An imported file has a single mixed track, so everyone in it is a "Speaker N".
- Final text appears when a chunk ends. The provisional text before that is a quick
  guess, refreshed every couple of seconds, and can be wrong; it is off by default
  without a GPU.
- Only one recording, import or re-transcription runs at a time.

## Troubleshooting

- **"No speech was recognised"**: the message names a device if one failed to open.
  Otherwise open **Devices**, pick the right microphone, and watch its meter while you
  speak: a bar that stays empty means the device delivers no sound.
- **The GPU is not used**: install the two `nvidia-*` packages above. The top bar shows
  where the model runs ("on cuda" or "on cpu").
- **Summaries fail with "No Anthropic API key was found"**: the variable must be set in
  the terminal that starts `ui.py`.

## Licence

MIT. See [LICENSE](LICENSE).

The licence covers this code only. The libraries it uses and the models it downloads
(OpenAI's Whisper weights, and NVIDIA's TitaNet speaker model from the sherpa-onnx
releases) come under their own licences.
