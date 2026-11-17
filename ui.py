#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["soundcard>=0.4.3", "numpy", "requests", "faster-whisper", "sherpa-onnx"]
# ///
"""Browser UI for transcribe.py: live transcript, speaker names, editing, dictionary.

Usage
    python ui.py                       # opens http://127.0.0.1:8765
    python ui.py --port 9000 --dir D:\\meetings
    python ui.py --language pl --min-chunk 2 --max-chunk 12   # transcribe.py options work too

Every recording is a session, kept as two files in --dir:
    transcript-<time>[-<title>].json   the source of truth, written to be readable
                                       by a person or a language model on its own
    transcript-<time>[-<title>].txt    the plain text, rewritten after every change
Existing transcript-*.txt files show up as sessions. Removed sessions are moved
to the "trash" subfolder, never deleted. dictionary.json holds the word lists
(one per language plus a shared one), settings.json the UI switches.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
import unicodedata
import wave
import webbrowser
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

import numpy as np

import transcribe as core

HERE = os.path.dirname(os.path.abspath(__file__))
ID_RE = re.compile(r"transcript-[A-Za-z0-9_-]+")
STAMPED_ID_RE = re.compile(r"transcript-\d{8}-\d{6}")
STAMP_RE = re.compile(r"\[(\d\d):(\d\d):(\d\d)\] (.*)")
SPEAKER_RE = re.compile(r"([^:]{1,40}): (.+)")
GENERIC_RE = re.compile(r"(Speaker|Mic) \d+")                       # labels the voice clustering hands out
AUTO_RE = re.compile(r"(Speaker|Mic) \d+( \(part \d+\))?|You")      # any name the user did not choose
LANGUAGES = [
    ("auto", "Auto-detect"), ("en", "English"), ("pl", "Polish"), ("de", "German"),
    ("fr", "French"), ("es", "Spanish"), ("it", "Italian"), ("pt", "Portuguese"),
    ("nl", "Dutch"), ("cs", "Czech"), ("sk", "Slovak"), ("uk", "Ukrainian"),
    ("sv", "Swedish"), ("ja", "Japanese"), ("zh", "Chinese"),
]
PROMPT_TERMS_CHARS = 500    # Whisper keeps ~220 prompt tokens; leave room for the running context
ABOUT = (
    "Transcript of a recorded conversation, produced by automatic speech recognition. "
    "'entries' are in the order they were spoken. In each entry 'speaker' is who said it, "
    "'time' is the wall-clock time, 't' the seconds since 'started', and 'language' the "
    "language the recogniser was set to (null if it was detecting it). "
    "'speakers' lists everyone who talks. Where named_by_user is false the name is an automatic "
    "voice label: one person may appear under two labels and two people may share one; "
    "the label 'You' is the person who made the recording. "
    "'channel' says whether the voice came through the local microphone (people in the room "
    "with the person recording) or through system audio (remote participants of a call). "
    "The text can contain recognition errors, especially in names and jargon, and sentences "
    "may be split across entries. 'glossary' lists the correct spellings of terms that matter here. "
    "'audio' names the recording files next to this one (16 kHz WAV, one channel per audio source); "
    "'offset' is where each file starts, in seconds since 'started', so an entry's 't' minus that "
    "offset is its position in the file. "
    "'source_file', when set, is the audio or video file this transcript was imported from; "
    "'started' is then an estimate (the file's date minus its length) and every voice is on the "
    "single channel 'uploaded file'."
)


class Busy(Exception):
    """The request does not fit the current state (409)."""


def write_text(path, text):
    with open(path + ".part", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(path + ".part", path)


def write_json(path, data):
    write_text(path, json.dumps(data, ensure_ascii=False, indent=1))


def read_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        core.log(f"Ignoring unreadable {path}: {exc}")
        return default


def com_init():
    if sys.platform == "win32":     # soundcard talks to WASAPI through COM
        import ctypes
        ctypes.windll.ole32.CoInitializeEx(None, 0)


class Unreadable(ValueError):
    """An uploaded file that cannot be turned into audio; the message is for the user."""


def convert_to_wav(source, target, progress=None):
    """Decode the first audio track of any audio or video file to 16 kHz mono
    WAV, a piece at a time. Uses PyAV, which faster-whisper already brings along
    (it bundles FFmpeg, so nothing else needs installing). Returns the seconds written."""
    try:
        return _decode_to_wav(source, target, progress)
    except Unreadable:
        raise
    except Exception as exc:        # FFmpeg's own messages name temp files and error numbers
        core.log(f"Decoding failed: {exc!r}")
        raise Unreadable("it could not be read as an audio or video file") from exc


def _decode_to_wav(source, target, progress):
    import av
    with av.open(source) as container:
        if not container.streams.audio:
            raise Unreadable("the file has no audio track")
        stream = container.streams.audio[0]
        if stream.duration and stream.time_base:
            total = float(stream.duration * stream.time_base)
        else:
            total = (container.duration or 0) / 1_000_000
        resampler = av.AudioResampler(format="s16", layout="mono", rate=core.SR)
        samples, tick = 0, 0.0
        with wave.open(target, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(core.SR)
            for frame in container.decode(stream):
                for out in resampler.resample(frame) or []:
                    data = out.to_ndarray().tobytes()
                    w.writeframesraw(data)
                    samples += len(data) // 2
                if progress and total and time.monotonic() - tick >= 0.5:
                    tick = time.monotonic()
                    progress(min(samples / core.SR / total, 1.0))
            for out in resampler.resample(None) or []:      # what the resampler still holds
                data = out.to_ndarray().tobytes()
                w.writeframesraw(data)
                samples += len(data) // 2
    return samples / core.SR


def wav_peak(path):
    """The loudest sample of a 16-bit WAV, 0..32767."""
    peak = 0
    with wave.open(path) as w:
        while True:
            raw = w.readframes(1 << 18)
            if not raw:
                return peak
            peak = max(peak, int(np.abs(np.frombuffer(raw, "<i2").astype(np.int32)).max()))


def slug(title):
    """A title as a file-name fragment: ASCII letters and digits joined by dashes."""
    text = unicodedata.normalize("NFKD", title.replace("ł", "l").replace("Ł", "L"))
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()[:60].strip("-")


# ------------------------------------------------------------- dictionary ---

class Dictionary:
    """Preferred spellings, per language.

    The file maps a language code to a list of {"term": "Kubernetes", "heard": ["cooper netties"]};
    the list under "*" applies whatever the language. The terms are handed to
    the model as a prompt, which nudges it to spell them that way; the "heard"
    variants are then replaced in the text, which always works but is blind to
    context.
    """

    SHARED = "*"

    def __init__(self, path):
        self.path = path
        self.lists = self._clean_all(read_json(path, {}))
        self._rules = {}

    @staticmethod
    def _clean(entries):
        clean = []
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            term = str(entry.get("term", "")).strip()
            heard = [str(h).strip() for h in entry.get("heard", []) if str(h).strip()]
            if term:
                clean.append({"term": term, "heard": heard})
        return clean

    def _clean_all(self, data):
        if isinstance(data, list):          # the first version kept a single list
            data = {self.SHARED: data}
        lists = {}
        for language, entries in (data if isinstance(data, dict) else {}).items():
            language, entries = str(language).strip().lower(), self._clean(entries)
            if entries and (language == self.SHARED or re.fullmatch(r"[a-z]{2,3}", language)):
                lists[language] = entries
        return lists

    def set(self, data):
        self.lists = self._clean_all(data)
        write_json(self.path, self.lists)
        self._rules = {}

    def entries(self, language=None):
        """The entries in force for a language: its own, then the shared ones."""
        own = self.lists.get(language, []) if language else []
        return own + self.lists.get(self.SHARED, [])

    def _rules_for(self, language):
        if language not in self._rules:
            rules = []
            for entry in self.entries(language):
                for heard in entry["heard"]:
                    body = r"\s+".join(re.escape(word) for word in heard.split())
                    pattern = re.compile(r"(?<!\w)" + body + r"(?!\w)", re.IGNORECASE)
                    rules.append((len(heard), pattern, entry["term"]))
            rules.sort(key=lambda rule: -rule[0])   # longest first: "wide and" before "wide"
            self._rules[language] = [(pattern, term) for _, pattern, term in rules]
        return self._rules[language]

    def apply(self, text, language=None):
        for pattern, term in self._rules_for(language):
            text = pattern.sub(lambda match, term=term: term, text)
        return text

    def prompt(self, language=None):
        terms, size = [], 0
        for entry in self.entries(language):
            size += len(entry["term"]) + 2
            if size > PROMPT_TERMS_CHARS:
                break
            terms.append(entry["term"])
        return ", ".join(terms) + "." if terms else None

    def glossary(self, languages):
        terms = [e["term"] for language in languages for e in self.lists.get(language, [])]
        terms += [e["term"] for e in self.lists.get(self.SHARED, [])]
        return list(dict.fromkeys(terms))


# ---------------------------------------------------------------- storage ---

def new_session(sid, started, title=None, model=None, sources=None, source_file=None):
    return {"id": sid, "title": title or started.strftime("%Y-%m-%d %H:%M"),
            "started": started.isoformat(timespec="seconds"),
            "model": model, "sources": sources, "source_file": source_file, "audio": [],
            "names": {}, "entries": [], "next": 1}


def add_entry(session, t, speaker, text, language=None):
    entry = {"id": session["next"], "t": t, "speaker": speaker, "language": language, "text": text}
    session["next"] += 1
    session["entries"].append(entry)
    if speaker:
        session["names"].setdefault(speaker, speaker)
    return entry


def display_name(session, entry):
    key = entry["speaker"]
    return session["names"].get(key, key) if key else None


def channel_of(key, sources):
    """Where a voice label came from, as far as the label and the devices tell."""
    if sources == ["File"]:
        return "uploaded file"
    if key == "You" or re.fullmatch(r"Mic \d+", key):
        return "microphone"
    if sources and len(sources) == 1:
        return "microphone" if sources[0] == "Mic" else "system audio"
    if sources and re.fullmatch(r"Speaker \d+", key):
        return "system audio"
    return None


def render(session, timestamps=False):
    """The session as plain text, in the format transcribe.py writes."""
    started = datetime.fromisoformat(session["started"])
    lines, last = [], None
    for entry in session["entries"]:
        name, text = display_name(session, entry), entry["text"]
        if name is not None and name != last:
            if lines:
                lines.append("")
            text = f"{name}: {text}"
            last = name
        if timestamps and entry["t"] is not None:
            text = f"[{started + timedelta(seconds=entry['t']):%H:%M:%S}] {text}"
        lines.append(text)
    return "\n".join(lines) + "\n" if lines else ""


def to_file(session, dictionary=None):
    """The .json as written to disk: the session plus everything a reader who
    has only this file needs (resolved names, clock times, who is who)."""
    started = datetime.fromisoformat(session["started"])
    entries, speakers = [], []
    for e in session["entries"]:
        clock = None if e["t"] is None else f"{started + timedelta(seconds=e['t']):%H:%M:%S}"
        entries.append({"id": e["id"], "time": clock, "t": e["t"],
                        "speaker": display_name(session, e), "speaker_id": e["speaker"],
                        "language": e.get("language"), "text": e["text"]})
    for key, name in session["names"].items():
        own = [e for e in session["entries"] if e["speaker"] == key]
        if own:
            speakers.append({"id": key, "name": name, "named_by_user": not AUTO_RE.fullmatch(name),
                             "channel": channel_of(key, session.get("sources")),
                             "entries": len(own), "words": sum(len(e["text"].split()) for e in own)})
    times = [e["t"] for e in session["entries"] if e["t"] is not None]
    languages = sorted({e["language"] for e in entries if e["language"]})
    return {
        "format": "silent-ear/2",
        "about": ABOUT,
        "id": session["id"],
        "title": session["title"],
        "started": session["started"],
        "weekday": started.strftime("%A"),
        "duration_minutes": round(max(times) / 60, 1) if times else None,
        "languages": languages,
        "recognition_model": session.get("model"),
        "audio_sources": session.get("sources"),
        "source_file": session.get("source_file"),
        "audio": session.get("audio", []),
        "glossary": dictionary.glossary(languages) if dictionary else [],
        "speakers": speakers,
        "entries": entries,
        "next": session["next"],
    }


def from_file(data):
    """Back to the working form: entries keyed by speaker id, plus an id -> name map."""
    if "names" in data:                     # silent-ear/1 was the working form itself
        session = new_session(data["id"], datetime.fromisoformat(data["started"]), data.get("title"))
        session["names"], session["next"] = data["names"], data["next"]
        session["entries"] = [{"id": e["id"], "t": e["t"], "speaker": e["speaker"],
                               "language": e.get("language"), "text": e["text"]}
                              for e in data["entries"]]
        return session
    session = new_session(data["id"], datetime.fromisoformat(data["started"]), data.get("title"),
                          data.get("recognition_model"), data.get("audio_sources"),
                          data.get("source_file"))
    session["names"] = {s["id"]: s["name"] for s in data.get("speakers", [])}
    for item in data.get("audio") or []:
        name = item.get("file") if isinstance(item, dict) else None
        if isinstance(name, str) and name == os.path.basename(name) and name.endswith(".wav"):
            session["audio"].append({"file": name, "offset": item.get("offset") or 0,
                                     "channels": item.get("channels")})
    for e in data.get("entries", []):
        session["entries"].append({"id": e["id"], "t": e.get("t"), "speaker": e.get("speaker_id"),
                                   "language": e.get("language"), "text": e["text"]})
        if e.get("speaker_id"):
            session["names"].setdefault(e["speaker_id"], e.get("speaker") or e["speaker_id"])
    session["next"] = max([data.get("next", 1)] + [e["id"] + 1 for e in session["entries"]])
    return session


class Store:
    """Sessions on disk. Every change rewrites the .json and the .txt."""

    def __init__(self, folder, timestamps=False, dictionary=None):
        self.dir, self.timestamps, self.dictionary = folder, timestamps, dictionary
        self.lock = threading.RLock()
        os.makedirs(folder, exist_ok=True)

    def _path(self, sid, ext):
        return os.path.join(self.dir, sid + ext)

    def ids(self):
        found = set()
        for name in os.listdir(self.dir):
            stem, ext = os.path.splitext(name)
            if ext in (".json", ".txt") and ID_RE.fullmatch(stem):
                found.add(stem)
        return sorted(found, reverse=True)

    def load(self, sid):
        if not ID_RE.fullmatch(sid):
            raise KeyError(sid)
        with self.lock:
            if os.path.exists(self._path(sid, ".json")):
                with open(self._path(sid, ".json"), encoding="utf-8") as f:
                    session = from_file(json.load(f))
                session["id"] = sid         # the file name wins if someone renamed the file
                return session
            if os.path.exists(self._path(sid, ".txt")):
                return self._import(sid, self._path(sid, ".txt"))
        raise KeyError(sid)

    def _import(self, sid, path):
        """Read a transcript written by transcribe.py (no .json beside it yet)."""
        try:
            started = datetime.strptime(sid[:26], "transcript-%Y%m%d-%H%M%S")
        except ValueError:
            started = datetime.fromtimestamp(os.path.getmtime(path)).replace(microsecond=0)
        session = new_session(sid, started)
        speaker, fresh = None, True
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line, t = raw.strip(), None
                if not line:
                    fresh = True        # a blank line precedes every change of speaker
                    continue
                stamp = STAMP_RE.fullmatch(line)
                if stamp:
                    h, m, s = (int(stamp.group(i)) for i in (1, 2, 3))
                    clock = started.replace(hour=h, minute=m, second=s)
                    t = (clock - started).total_seconds() % 86400
                    line = stamp.group(4)
                if fresh:
                    named = SPEAKER_RE.fullmatch(line)
                    if named:
                        speaker, line = named.group(1), named.group(2)
                    fresh = False
                add_entry(session, t, speaker, line)
        return session

    def save(self, session):
        with self.lock:
            write_json(self._path(session["id"], ".json"), to_file(session, self.dictionary))
            write_text(self._path(session["id"], ".txt"), render(session, self.timestamps))

    def summaries(self):
        out = []
        for sid in self.ids():
            try:
                s = self.load(sid)
            except (KeyError, OSError, ValueError):
                continue
            names = {display_name(s, e) for e in s["entries"]} - {None}
            out.append({"id": sid, "title": s["title"], "started": s["started"],
                        "count": len(s["entries"]), "speakers": sorted(names)})
        out.sort(key=lambda s: s["started"], reverse=True)
        return out

    def _unique(self, sid):
        candidate, n = sid, 1
        while any(os.path.exists(self._path(candidate, ext)) for ext in (".json", ".txt", ".wav")):
            n += 1
            candidate = f"{sid}-{n}"
        return candidate

    def _align(self, session):
        """Save under a file name that carries the title: transcript-<time>-<title>."""
        old = session["id"]
        stamp = STAMPED_ID_RE.match(old)
        if stamp:
            title = session["title"]
            default = f"{datetime.fromisoformat(session['started']):%Y-%m-%d %H:%M}"
            if title.startswith(default):       # the date is in the name already
                title = title[len(default):]
            tail = slug(title)
            want = f"{stamp.group(0)}-{tail}" if tail else stamp.group(0)
            if old != want and not re.fullmatch(re.escape(want) + r"-\d+", old):
                session["id"] = self._unique(want)
        new = session["id"]
        if new != old:
            try:                        # the recordings first: they are the ones that can be busy
                for item in session["audio"]:
                    source = os.path.join(self.dir, item["file"])
                    if item["file"].startswith(old) and os.path.exists(source):
                        renamed = new + item["file"][len(old):]
                        os.replace(source, os.path.join(self.dir, renamed))
                        item["file"] = renamed
            except OSError:
                session["id"] = old
                self.save(session)
                raise Busy("The title is saved, but the files keep their old names: "
                           "the recording is in use. Pause playback and rename again.")
        self.save(session)
        if new != old:
            for ext in (".json", ".txt"):
                if os.path.exists(self._path(old, ext)):
                    os.remove(self._path(old, ext))

    def align(self, sid):
        with self.lock:
            session = self.load(sid)
            self._align(session)
            return session["id"]

    def create(self, started, model=None, sources=None, title=None, source_file=None):
        with self.lock:
            sid = self._unique(f"transcript-{started:%Y%m%d-%H%M%S}")
            session = new_session(sid, started, title, model, sources, source_file)
            if title:
                self._align(session)        # the file name carries the title from the start
            else:
                self.save(session)
            return session

    def copy_audio(self, source, sid):
        """Give session sid its own files for another session's recordings."""
        with self.lock:
            audio = []
            for item in source["audio"]:
                origin = os.path.join(self.dir, item["file"])
                if not os.path.isfile(origin):
                    continue
                n = len(audio) + 1
                name = sid + (".wav" if n == 1 else f".part{n}.wav")
                try:                        # a hard link: a second name, no second copy on disk
                    os.link(origin, os.path.join(self.dir, name))
                except OSError:             # not supported on this drive
                    shutil.copyfile(origin, os.path.join(self.dir, name))
                audio.append({"file": name, "offset": item["offset"], "channels": item.get("channels")})
            self.set_audio(sid, audio)
            return audio

    def set_audio(self, sid, audio):
        with self.lock:
            session = self.load(sid)
            session["audio"] = audio
            self.save(session)

    def audio_path(self, sid, index):
        """The file behind the index-th recording of a session."""
        audio = self.load(sid)["audio"]
        if not 0 <= index < len(audio):
            raise KeyError(index)
        path = os.path.join(self.dir, audio[index]["file"])
        if not os.path.isfile(path):
            raise KeyError(path)
        return path

    def append(self, sid, t, speaker, text, language=None):
        with self.lock:
            session = self.load(sid)
            entry = add_entry(session, t, speaker, text, language)
            self.save(session)
            return entry, session["names"]

    def update(self, sid, title=None, names=None, rename_files=True):
        with self.lock:
            session = self.load(sid)
            title = str(title).strip() if title is not None else ""
            retitled = bool(title) and title != session["title"]
            if retitled:
                session["title"] = title
            for key, name in (names or {}).items():
                if key in session["names"] and str(name).strip():
                    session["names"][key] = str(name).strip()
            if retitled and rename_files:
                self._align(session)
            else:
                self.save(session)
            return session

    def _entry(self, session, eid):
        for index, entry in enumerate(session["entries"]):
            if entry["id"] == eid:
                return index, entry
        raise KeyError(eid)

    def edit_entry(self, sid, eid, text=None, speaker=None):
        with self.lock:
            session = self.load(sid)
            _, entry = self._entry(session, eid)
            if text is not None and str(text).strip():
                entry["text"] = " ".join(str(text).split())
            if speaker is not None:
                key = str(speaker).strip()
                entry["speaker"] = key or None
                if key:
                    session["names"].setdefault(key, key)
            self.save(session)
            return session

    def delete_entry(self, sid, eid):
        with self.lock:
            session = self.load(sid)
            index, _ = self._entry(session, eid)
            del session["entries"][index]
            self.save(session)
            return session

    def join_entry(self, sid, eid):
        """Fold an entry into the one before it."""
        with self.lock:
            session = self.load(sid)
            index, entry = self._entry(session, eid)
            if index == 0:
                raise Busy("The first line has nothing before it to join.")
            session["entries"][index - 1]["text"] += " " + entry["text"]
            del session["entries"][index]
            self.save(session)
            return session

    def apply_dictionary(self, sid, dictionary, fallback_language=None):
        """Fix the lines already written; each with the list of its own language."""
        with self.lock:
            session = self.load(sid)
            changed = 0
            for entry in session["entries"]:
                fixed = dictionary.apply(entry["text"], entry.get("language") or fallback_language)
                if fixed != entry["text"]:
                    entry["text"] = fixed
                    changed += 1
            if changed:
                self.save(session)
            return changed

    def trash(self, sid):
        """Move a session's files to the trash subfolder."""
        with self.lock:
            session = self.load(sid)                         # 404 if it does not exist
            bin_dir = os.path.join(self.dir, "trash")
            os.makedirs(bin_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            for name in [sid + ".json", sid + ".txt"] + [a["file"] for a in session["audio"]]:
                source = os.path.join(self.dir, name)
                if os.path.exists(source):
                    target = os.path.join(bin_dir, name)
                    if os.path.exists(target):
                        stem, ext = os.path.splitext(name)
                        target = os.path.join(bin_dir, f"{stem}.{stamp}{ext}")
                    os.replace(source, target)

    def discard_if_empty(self, sid):
        """Drop a recording in which nothing was said; True if it is gone."""
        with self.lock:
            try:
                session = self.load(sid)
            except KeyError:
                return True
            if session["entries"]:
                return False
            for name in [sid + ".json", sid + ".txt"] + [a["file"] for a in session["audio"]]:
                if os.path.exists(os.path.join(self.dir, name)):
                    os.remove(os.path.join(self.dir, name))
            return True

    def merge(self, sids, title=None):
        """Join sessions in time order into a new one; the originals go to the trash."""
        with self.lock:
            parts = sorted((self.load(sid) for sid in dict.fromkeys(sids)), key=lambda s: s["started"])
            if len(parts) < 2:
                raise Busy("Pick at least two transcripts to join.")
            first = datetime.fromisoformat(parts[0]["started"])
            stamp = STAMPED_ID_RE.match(parts[0]["id"])
            base = stamp.group(0) if stamp else parts[0]["id"]
            merged = new_session(self._unique(base + "-joined"), first,
                                 title or parts[0]["title"] + " (joined)",
                                 parts[0].get("model"))
            for number, part in enumerate(parts, start=1):
                shift = (datetime.fromisoformat(part["started"]) - first).total_seconds()
                for entry in part["entries"]:
                    name = display_name(part, entry)
                    # "Speaker 2" of another recording is a different voice: keep them apart.
                    if name and number > 1 and GENERIC_RE.fullmatch(name):
                        name = f"{name} (part {number})"
                    t = None if entry["t"] is None else entry["t"] + shift
                    add_entry(merged, t, name, entry["text"], entry.get("language"))
                for item in part["audio"]:      # the recordings move into the joined session
                    source = os.path.join(self.dir, item["file"])
                    n = len(merged["audio"]) + 1
                    moved = merged["id"] + (".wav" if n == 1 else f".part{n}.wav")
                    try:
                        os.replace(source, os.path.join(self.dir, moved))
                    except OSError:             # missing or in use: it stays with the original
                        continue
                    merged["audio"].append({"file": moved, "offset": round(shift + item["offset"], 1),
                                            "channels": item.get("channels")})
                    item["file"] = moved        # so the trash below leaves it alone
            self.save(merged)
            for part in parts:
                self.trash(part["id"])
            return merged


# ----------------------------------------------------------------- events ---

class Hub:
    """Fans events out to every open browser tab (server-sent events)."""

    def __init__(self):
        self.clients = set()
        self.lock = threading.Lock()

    def open(self):
        q = queue.Queue(maxsize=1000)
        with self.lock:
            self.clients.add(q)
        return q

    def close(self, q):
        with self.lock:
            self.clients.discard(q)

    def send(self, **event):
        data = json.dumps(event, ensure_ascii=False)
        with self.lock:
            for q in self.clients:
                try:
                    q.put_nowait(data)
                except queue.Full:
                    pass


# -------------------------------------------------------------- recording ---

class Engine:
    """Owns the model and the one recording that can run at a time."""

    def __init__(self, args, sc, store, dictionary, hub):
        self.args, self.sc = args, sc
        self.store, self.dictionary, self.hub = store, dictionary, hub
        self.language = None if args.language.lower() == "auto" else args.language.lower()
        self.settings_path = os.path.join(store.dir, "settings.json")
        saved = read_json(self.settings_path, {})
        saved = saved if isinstance(saved, dict) else {}
        self.mic_speakers = bool(args.mic_speakers or saved.get("mic_speakers"))
        self.save_audio = bool(saved.get("save_audio", True))
        self.transcribe = None
        self.model = ""
        self.phase = "loading"          # loading, idle, starting, recording, finishing, error
        self.error = None
        self.session = None
        self.replaying = False          # the running "recording" is fed from a saved file
        self.stop_event = None
        self.worker = None
        self.lock = threading.Lock()

    def status(self):
        return {"phase": self.phase, "error": self.error, "model": self.model,
                "language": self.language or "auto", "session": self.session,
                "replaying": self.replaying,
                "mic_speakers": self.mic_speakers, "save_audio": self.save_audio}

    def _changed(self):
        self.hub.send(type="changed")

    def load(self):
        args = self.args
        backend = "ollama" if args.backend == "whisper" and args.base_url else args.backend
        preset = core.BACKENDS[backend]
        base_url = args.base_url or preset["base_url"]
        model = args.model or preset["model"]
        try:
            if backend == "whisper":
                core.log(f"Loading Whisper {model} (downloaded on first use) ...")
                transcribe = core.LocalWhisper(model, self.language, args.prompt, args.device)
                self.model = f"Whisper {model} on {transcribe.device}"
            else:
                api_key = args.api_key or os.environ.get("TRANSCRIBE_API_KEY")
                if backend == "openai":
                    api_key = api_key or os.environ.get("OPENAI_API_KEY")
                    if not api_key:
                        raise RuntimeError("Set OPENAI_API_KEY to use the OpenAI backend.")
                transcribe = core.Transcriber(base_url, model, api_key, self.language, args.prompt)
                self.model = f"{model} @ {base_url}"
                if core.is_local(base_url):
                    core.warm_up(transcribe, base_url, model)
        except (Exception, SystemExit) as exc:      # the core reports setup problems by exiting
            self.error, self.phase = str(exc), "error"
            core.log(f"Model failed to load: {exc}")
        else:
            self.transcribe = transcribe
            self.transcribe.language = self.language    # may have been changed while loading
            self.refresh_prompt()
            self.phase = "idle"
            core.log(f"Model ready: {self.model}")
        self._changed()

    def refresh_prompt(self):
        """Hand the model the word list of the language now in force."""
        if self.transcribe is not None:
            parts = [p for p in (self.args.prompt, self.dictionary.prompt(self.language)) if p]
            self.transcribe.prompt = " ".join(parts) or None

    def set_language(self, code):
        code = str(code).strip().lower()
        if code != "auto" and not re.fullmatch(r"[a-z]{2,3}", code):
            raise ValueError("Language must be an ISO code such as en or pl, or auto.")
        self.language = None if code == "auto" else code
        if self.transcribe is not None:
            self.transcribe.language = self.language
            if hasattr(self.transcribe, "context"):
                # Text in the old language as a prompt pulls the model back to it.
                self.transcribe.context = ""
        self.refresh_prompt()
        self._changed()

    def set_settings(self, body):
        if "mic_speakers" in body:
            self.mic_speakers = bool(body["mic_speakers"])
        if "save_audio" in body:
            self.save_audio = bool(body["save_audio"])
        write_json(self.settings_path, {"mic_speakers": self.mic_speakers,
                                        "save_audio": self.save_audio})
        self._changed()

    def start(self):
        with self.lock:
            if self.phase != "idle":
                raise Busy({"loading": "The model is still loading.",
                            "error": self.error or "The model failed to load."}
                           .get(self.phase, "A recording is already running."))
            self.phase = "starting"
        try:
            com_init()
            sources = core.open_sources(self.sc, self.args)
            if not sources:
                raise RuntimeError("No audio source available.")
            labels = [s.label for s in sources]
            speakers = None
            if not self.args.no_speakers:
                try:
                    speakers = core.Speakers(labels, self.args.speaker_threshold, self.mic_speakers)
                except Exception as exc:
                    core.log(f"Speaker detection off: {exc}")
            session = self.store.create(datetime.now().replace(microsecond=0), self.model, labels)
            recording = None
            if self.save_audio:
                # One channel per source, so a saved recording can be fed back through the
                # whole pipeline later, speaker detection included. The header is kept
                # current after every write: the file is valid even if we crash.
                name = session["id"] + ".wav"
                recording = wave.open(os.path.join(self.store.dir, name), "wb")
                recording.setnchannels(len(sources))
                recording.setsampwidth(2)
                recording.setframerate(core.SR)
                self.store.set_audio(session["id"], [{"file": name, "offset": 0, "channels": labels}])
        except Exception:
            self.phase = "idle"
            raise
        if hasattr(self.transcribe, "context"):
            self.transcribe.context = ""
        stop, jobs = threading.Event(), queue.Queue()
        chunker = core.Chunker(min_s=self.args.min_chunk, max_s=self.args.max_chunk,
                               pause_s=self.args.pause)
        self.stop_event, self.session, self.phase = stop, session["id"], "recording"
        self.worker = threading.Thread(target=self._work, daemon=True,
                                       args=(jobs, speakers, stop, session["id"]))
        self.worker.start()
        threading.Thread(target=self._capture, daemon=True,
                         args=(sources, chunker, jobs, stop, recording)).start()
        self._changed()
        return session

    def replay(self, sid):
        """Transcribe a session's saved recording again, into a new session, with
        the language, dictionary and speaker settings in force now."""
        with self.lock:
            if self.phase != "idle":
                raise Busy({"loading": "The model is still loading.",
                            "error": self.error or "The model failed to load."}
                           .get(self.phase, "Stop the running recording first."))
            self.phase = "starting"
        try:
            original = self.store.load(sid)
            if not any(os.path.isfile(os.path.join(self.store.dir, a["file"])) for a in original["audio"]):
                raise Busy("This transcript has no saved recording to transcribe again.")
            labels = original["audio"][0].get("channels") or original.get("sources") or ["Mic", "System"]
            session = self.store.create(datetime.fromisoformat(original["started"]), self.model,
                                        labels, original["title"] + " (re-run)",
                                        original.get("source_file"))
            audio = self.store.copy_audio(original, session["id"])
        except Exception:
            self.phase = "idle"
            raise
        self._begin_replay(session, labels, audio)
        return session

    def _begin_replay(self, session, labels, audio):
        """Start transcribing a session's saved recordings (phase is "starting" or "importing")."""
        speakers = None
        if not self.args.no_speakers:
            try:
                speakers = core.Speakers(labels, self.args.speaker_threshold, self.mic_speakers)
            except Exception as exc:
                core.log(f"Speaker detection off: {exc}")
        if hasattr(self.transcribe, "context"):
            self.transcribe.context = ""
        stop, jobs = threading.Event(), queue.Queue(maxsize=4)     # bounded: the file outruns the model
        self.stop_event, self.session, self.replaying = stop, session["id"], True
        self.phase = "recording"
        self.worker = threading.Thread(target=self._work, daemon=True,
                                       args=(jobs, speakers, stop, session["id"]))
        self.worker.start()
        parts = [(os.path.join(self.store.dir, a["file"]), a["offset"]) for a in audio]
        threading.Thread(target=self._replay, daemon=True,
                         args=(parts, len(labels), jobs, stop)).start()
        self._changed()

    def reserve_import(self):
        """Claim the engine for an upload that is about to arrive."""
        with self.lock:
            if self.phase != "idle":
                raise Busy({"loading": "The model is still loading.",
                            "error": self.error or "The model failed to load."}
                           .get(self.phase, "Wait for the running recording or import to finish."))
            self.phase = "importing"
        self._changed()

    def cancel_import(self):
        with self.lock:
            if self.phase == "importing":
                self.phase = "idle"
        self._changed()

    def import_file(self, upload, name, modified_ms=None):
        """Turn an uploaded audio or video file into a session and transcribe it.
        Runs in the background; `upload` is deleted when done."""
        threading.Thread(target=self._import, daemon=True, args=(upload, name, modified_ms)).start()

    def _import(self, upload, name, modified_ms):
        wav = upload + ".wav"
        try:
            seconds = convert_to_wav(
                upload, wav, lambda share: self.hub.send(type="progress", convert=share, name=name))
            if seconds < 0.5:
                raise Unreadable("the audio track is empty")
            if wav_peak(wav) < 33:          # below -60 dB: nothing a microphone ever recorded
                raise Unreadable(f"its {seconds:.0f} seconds of audio are pure silence, "
                                 "so there is nothing to transcribe")
            # A recorder stamps the file when it finishes, so the start is that minus the length.
            now = datetime.now()
            ended = now
            if modified_ms:
                try:
                    ended = min(datetime.fromtimestamp(float(modified_ms) / 1000), now)
                except (ValueError, OSError, OverflowError):
                    pass
            started = (ended - timedelta(seconds=seconds)).replace(microsecond=0)
            title = os.path.splitext(name)[0].strip() or "Imported recording"
            session = self.store.create(started, self.model, ["File"], title, name)
            final = session["id"] + ".wav"
            os.replace(wav, os.path.join(self.store.dir, final))
            audio = [{"file": final, "offset": 0, "channels": ["File"]}]
            self.store.set_audio(session["id"], audio)
            self._begin_replay(session, ["File"], audio)
        except Exception as exc:
            core.log(f"Import of {name} failed: {exc!r}")
            self.hub.send(type="notice", text=f"Could not import {name}: {exc}")
            self.cancel_import()
        finally:
            for leftover in (upload, wav):
                try:
                    os.remove(leftover)
                except OSError:
                    pass

    def _replay(self, parts, columns, jobs, stop):
        """Feed saved recordings through the same chunker the live capture uses."""
        def put(job):
            while not stop.is_set():
                try:
                    return jobs.put(job, timeout=0.2)
                except queue.Full:
                    pass

        try:
            total = 0
            for path, _ in parts:
                with wave.open(path) as w:
                    total += w.getnframes()
            done, tick = 0, 0.0
            for path, offset in parts:
                chunker = core.Chunker(min_s=self.args.min_chunk, max_s=self.args.max_chunk,
                                       pause_s=self.args.pause)
                with wave.open(path) as w:
                    if w.getframerate() != core.SR or w.getsampwidth() != 2:
                        raise RuntimeError(f"{os.path.basename(path)} is not 16 kHz 16-bit audio.")
                    channels = w.getnchannels()
                    while not stop.is_set():
                        raw = w.readframes(core.SR)             # a second at a time
                        if not raw:
                            break
                        block = np.frombuffer(raw, "<i2").reshape(-1, channels).astype(np.float32) / 32767.0
                        mix = np.zeros((len(block), 1 + columns), np.float32)   # the shape pull_mix gives
                        shared = min(channels, columns)
                        mix[:, 1:1 + shared] = block[:, :shared]
                        mix[:, 0] = np.clip(block.sum(axis=1), -1.0, 1.0)
                        for start_s, audio in chunker.feed(mix):
                            put((start_s + offset, audio))
                        done += len(block)
                        if time.monotonic() - tick >= 0.5:
                            tick = time.monotonic()
                            self.hub.send(type="progress", buffered=0, queued=jobs.qsize(),
                                          replay=done / max(total, 1))
                last = None if stop.is_set() else chunker.flush()
                if last:
                    put((last[0] + offset, last[1]))
        except Exception as exc:
            self.hub.send(type="notice", text=f"Re-transcription stopped: {exc}")
        with self.lock:
            self.phase = "finishing"
        self._changed()
        jobs.put(None)

    def stop(self):
        with self.lock:
            if self.phase != "recording":
                return
            self.phase = "finishing"
        self.stop_event.set()
        self._changed()

    def wait(self, timeout):
        if self.worker is not None:
            self.worker.join(timeout)

    def _capture(self, sources, chunker, jobs, stop, recording=None):
        def feed(mix):
            if mix is None:
                return
            if recording is not None:
                try:
                    recording.writeframes(core.to_pcm16(mix[:, 1:]))
                except OSError as exc:      # e.g. disk full: keep transcribing
                    if not feed.failed:
                        feed.failed = True
                        self.hub.send(type="notice", text=f"Audio is no longer being saved: {exc}")
            for job in chunker.feed(mix):
                jobs.put(job)
        feed.failed = False

        com_init()
        for source in sources:
            core.log(f"{source.label + ':':<8}{source.device.name}")
            source.start(stop)
        tick = 0.0
        while not stop.is_set():
            time.sleep(0.1)
            for source in sources:
                if source.error and not source.reported:
                    source.reported = True
                    self.hub.send(type="notice", text=f"{source.label} capture stopped: {source.error}")
            if all(source.error for source in sources):
                self.hub.send(type="notice", text="Recording stopped: all audio sources failed.")
                stop.set()
                break
            feed(core.pull_mix(sources))
            if time.monotonic() - tick >= 1.0:
                tick = time.monotonic()
                buffered = len(chunker.frames) / core.FPS if chunker.started else 0.0
                self.hub.send(type="progress", buffered=buffered, queued=jobs.qsize())
        with self.lock:
            self.phase = "finishing"
        self._changed()
        for source in sources:
            if source.thread is not None:
                source.thread.join(timeout=2)
        feed(core.pull_mix(sources, everything=True))
        if recording is not None:
            try:
                recording.close()
            except OSError:
                pass
        last = chunker.flush()
        if last:
            jobs.put(last)
        jobs.put(None)

    def _work(self, jobs, speakers, stop, sid):
        ok, dead = 0, False
        while True:
            job = jobs.get()
            if job is None:
                break
            if dead:
                continue
            start_s, audio = job
            seconds = len(audio) / core.SR
            language = self.language
            try:
                segments = self.transcribe(audio[:, 0])
            except core.Rejected as exc:
                if ok == 0:             # nothing has ever worked: a setup problem, so stop
                    self.hub.send(type="notice", text=f"Recording stopped: {exc}")
                    dead = True
                    stop.set()
                else:
                    self.hub.send(type="notice", text=f"Skipped {seconds:.0f}s of audio: {exc}")
            except Exception as exc:
                self.hub.send(type="notice", text=f"Skipped {seconds:.0f}s of audio: {exc}")
            else:
                ok += 1
                fix = self.dictionary.apply
                segments = [(a, b, fix(text, language)) for a, b, text in segments]
                if getattr(self.transcribe, "context", None):   # don't feed misspellings back in
                    self.transcribe.context = fix(self.transcribe.context, language)
                for offset, who, text in core.speaker_turns(segments, audio, speakers):
                    entry, names = self.store.append(sid, round(start_s + offset, 1), who, text, language)
                    self.hub.send(type="entry", session=sid, entry=entry, names=names)
        if not self.store.discard_if_empty(sid):
            self.store.align(sid)       # a title given while recording now reaches the file name
        elif not dead:                  # say why the transcript that was on screen is gone
            what = "the saved recording" if self.replaying else "the recording"
            self.hub.send(type="notice", text=f"No speech was recognised in {what}, "
                                              "so no transcript was kept.")
        with self.lock:
            self.phase, self.session, self.replaying = "idle", None, False
        self._changed()


# ------------------------------------------------------------------- http ---

class App:
    def __init__(self, engine, store, dictionary, hub):
        self.engine, self.store, self.dictionary, self.hub = engine, store, dictionary, hub

    def state(self):
        return {"status": self.engine.status(), "sessions": self.store.summaries(),
                "dictionary": self.dictionary.lists,
                "languages": [{"code": c, "name": n} for c, n in LANGUAGES]}

    def _not_recording(self, sids):
        if self.engine.session in sids:
            raise Busy("Stop the recording before removing or joining its transcript.")

    def handle(self, method, parts, body):
        """Route one /api request; returns the JSON-ready result."""
        store, engine = self.store, self.engine
        if parts == ["state"] and method == "GET":
            return self.state()
        if parts == ["record", "start"] and method == "POST":
            return {"session": engine.start()["id"]}
        if parts == ["record", "stop"] and method == "POST":
            engine.stop()
            return {}
        if parts == ["language"] and method == "POST":
            engine.set_language(body.get("language", ""))
            return {}
        if parts == ["settings"] and method == "POST":
            engine.set_settings(body)
            return {}
        if parts == ["dictionary"] and method == "PUT":
            self.dictionary.set(body.get("dictionary", {}))
            engine.refresh_prompt()
            self.hub.send(type="changed")
            return {"dictionary": self.dictionary.lists}
        if parts[:1] != ["sessions"]:
            raise KeyError("/".join(parts))

        result = None
        if parts == ["sessions", "merge"] and method == "POST":
            sids = [str(s) for s in body.get("ids", [])]
            self._not_recording(sids)
            result = store.merge(sids, body.get("title"))
        elif len(parts) == 2:
            sid = parts[1]
            if method == "GET":
                return store.load(sid)
            if method == "PATCH":       # the file keeps its name while it is being recorded into
                result = store.update(sid, body.get("title"), body.get("names"),
                                      rename_files=sid != engine.session)
            elif method == "DELETE":
                self._not_recording([sid])
                store.trash(sid)
                result = {}
        elif len(parts) == 3 and parts[2] == "replay" and method == "POST":
            return {"session": engine.replay(parts[1])["id"]}
        elif len(parts) == 3 and parts[2] == "apply-dictionary" and method == "POST":
            result = {"changed": store.apply_dictionary(parts[1], self.dictionary, engine.language)}
        elif len(parts) >= 4 and parts[2] == "entries" and parts[3].isdigit():
            sid, eid = parts[1], int(parts[3])
            if len(parts) == 4 and method == "PATCH":
                result = store.edit_entry(sid, eid, body.get("text"), body.get("speaker"))
            elif len(parts) == 4 and method == "DELETE":
                result = store.delete_entry(sid, eid)
            elif parts[4:] == ["join"] and method == "POST":
                result = store.join_entry(sid, eid)
        if result is None:
            raise KeyError("/".join(parts))
        self.hub.send(type="changed")
        return result


class Handler(BaseHTTPRequestHandler):
    server_version = "silent-ear"

    def log_message(self, *args):
        pass

    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code=200):
        self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _trusted(self):
        """Only this machine's own page may talk to the server: refuse other
        host names (DNS rebinding) and writes coming from other web sites."""
        host = urlparse("//" + (self.headers.get("Host") or "")).hostname
        if host not in ("127.0.0.1", "localhost", "::1"):
            return False
        origin = self.headers.get("Origin")
        if self.command != "GET" and origin is not None:
            return urlparse(origin).netloc == self.headers.get("Host")
        return True

    def _events(self):
        hub = self.server.app.hub
        q = hub.open()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b"retry: 2000\n\n")
            while True:
                try:
                    chunk = f"data: {q.get(timeout=15)}\n\n"
                except queue.Empty:
                    chunk = ": ping\n\n"
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
        except OSError:
            pass
        finally:
            hub.close(q)

    def _audio(self, sid, index):
        """Stream a recording, honouring Range so the browser can seek in it."""
        try:
            path = self.server.app.store.audio_path(sid, index)
            size = os.path.getsize(path)
        except (KeyError, OSError, ValueError):
            return self._json({"error": "Not found."}, 404)
        start, end, code = 0, size - 1, 200
        wanted = re.fullmatch(r"bytes=(\d*)-(\d*)", (self.headers.get("Range") or "").strip())
        if wanted and (wanted.group(1) or wanted.group(2)):
            code = 206
            if wanted.group(1):
                start = int(wanted.group(1))
                if wanted.group(2):
                    end = min(int(wanted.group(2)), size - 1)
            else:
                start = max(0, size - int(wanted.group(2)))
            if start >= size or start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        self.send_response(code)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        try:
            position = start
            while position <= end:
                # Open briefly for each block: a file held open could not be renamed on Windows.
                with open(path, "rb") as f:
                    f.seek(position)
                    block = f.read(min(262144, end - position + 1))
                if not block:
                    break
                self.wfile.write(block)
                position += len(block)
        except OSError:
            pass

    def _import(self):
        """Receive an audio or video file as the raw request body and hand it to the engine."""
        app = self.server.app
        name = os.path.basename(unquote(self.headers.get("X-File-Name") or "upload").replace("\\", "/"))
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._json({"error": "No file was sent."}, 400)
        try:
            app.engine.reserve_import()
        except Busy as exc:
            return self._json({"error": str(exc)}, 409)
        upload = os.path.join(app.store.dir, f".upload-{time.time_ns()}.part")
        try:
            with open(upload, "wb") as f:
                left = length
                while left > 0:
                    block = self.rfile.read(min(1 << 20, left))
                    if not block:
                        raise OSError("the upload was cut short")
                    f.write(block)
                    left -= len(block)
        except OSError as exc:
            if os.path.exists(upload):
                os.remove(upload)
            app.engine.cancel_import()
            return self._json({"error": f"Upload failed: {exc}"}, 400)
        app.engine.import_file(upload, name or "upload", self.headers.get("X-File-Modified"))
        self._json({})

    def _route(self):
        if not self._trusted():
            return self._json({"error": "Request refused: not from this machine's page."}, 403)
        path = urlparse(self.path).path
        if self.command == "GET" and path == "/":
            with open(os.path.join(HERE, "ui.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        parts = [p for p in path.split("/") if p]
        if parts[:1] != ["api"]:
            return self._json({"error": "Not found."}, 404)
        if self.command == "GET" and parts == ["api", "events"]:
            return self._events()
        if (self.command == "GET" and len(parts) == 5 and parts[:2] == ["api", "sessions"]
                and parts[3] == "audio" and parts[4].isdigit()):
            return self._audio(parts[2], int(parts[4]))
        if self.command == "POST" and parts == ["api", "import"]:
            return self._import()
        try:
            body = {}
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("The request body must be a JSON object.")
            self._json(self.server.app.handle(self.command, parts[1:], body))
        except KeyError:
            self._json({"error": "Not found."}, 404)
        except Busy as exc:
            self._json({"error": str(exc)}, 409)
        except ValueError as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:
            core.log(f"{self.command} {path} failed: {exc!r}")
            self._json({"error": str(exc)}, 500)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _route


class Server(ThreadingHTTPServer):
    daemon_threads = True


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Browser UI for live transcription.",
        epilog="Options of transcribe.py (--language, --backend, --model, --mic, --no-speakers, "
               "--mic-speakers, --min-chunk, --max-chunk, --pause, --timestamps, ...) are accepted too.")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--dir", default=".", help="folder holding the transcripts (default: current)")
    p.add_argument("--no-browser", action="store_true", help="don't open the page automatically")
    own, rest = p.parse_known_args(argv)
    args = core.parse_args(rest)

    sc = core.load_soundcard()
    hub = Hub()
    folder = os.path.abspath(own.dir)
    os.makedirs(folder, exist_ok=True)
    dictionary = Dictionary(os.path.join(folder, "dictionary.json"))
    store = Store(folder, args.timestamps, dictionary)
    for name in os.listdir(folder):         # uploads a previous run did not get to finish
        if name.startswith(".upload-"):
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                pass
    engine = Engine(args, sc, store, dictionary, hub)
    server = Server(("127.0.0.1", own.port), Handler)
    server.app = App(engine, store, dictionary, hub)
    threading.Thread(target=engine.load, daemon=True).start()

    url = f"http://127.0.0.1:{own.port}"
    core.log(f"Silent Ear UI at {url}  (transcripts in {store.dir}; Ctrl+C to quit)")
    if not own.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    if engine.phase in ("recording", "finishing"):
        core.log("Finishing the recording ...")
        engine.stop()
        try:
            engine.wait(120)
        except KeyboardInterrupt:
            core.log("Skipped the remaining audio.")


if __name__ == "__main__":
    main()
