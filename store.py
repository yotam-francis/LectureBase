"""Persist the raw transcript once, so `index` and `summarize` can both reuse it
instead of running Whisper twice (transcription is the slow part)."""
import hashlib
import json
import os

from .transcribe import Segment


def _slug(audio_path: str) -> str:
    return hashlib.sha1(os.path.abspath(audio_path).encode()).hexdigest()[:12]


def transcript_path(persist_dir: str, audio_path: str) -> str:
    return os.path.join(persist_dir, f"transcript_{_slug(audio_path)}.json")


def save_transcript(persist_dir: str, audio_path: str, segments) -> None:
    os.makedirs(persist_dir, exist_ok=True)
    data = {"source": audio_path,
            "segments": [{"start": s.start, "end": s.end, "text": s.text}
                         for s in segments]}
    with open(transcript_path(persist_dir, audio_path), "w") as f:
        json.dump(data, f)


def load_transcript(persist_dir: str, audio_path: str) -> list[Segment]:
    with open(transcript_path(persist_dir, audio_path)) as f:
        raw = json.load(f)["segments"]
    return [Segment(**s) for s in raw]