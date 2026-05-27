"""Transcribe audio into timestamped segments using faster-whisper.

We keep timestamps so that answers can cite *where* in the audio a fact came
from — that citation step is what turns a generic RAG demo into an audio one.
"""
from dataclasses import dataclass


@dataclass
class Segment:
    start: float  # seconds
    end: float
    text: str


def transcribe(audio_path: str, model_size: str = "base",
               device: str = "cpu", compute_type: str = "int8", language: str | None = None) -> list[Segment]:
    # Lazy import so the rest of the package imports without the heavy dep.
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    # vad_filter drops silence — fewer empty/garbage segments to embed.
    segments, _info = model.transcribe(audio_path, vad_filter=True,language = language)
    return [Segment(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]
