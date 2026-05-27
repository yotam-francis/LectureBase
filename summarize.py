"""Summarize a recording and pull out action items / decisions.

Reuses the transcript saved by `index` (no re-transcription). Uses map-reduce so
it works on long recordings instead of overflowing the model's context:
  map    -> summarize each chunk
  reduce -> combine partial summaries into one summary + action items.
"""
from .chunk import chunk_segments
from .llm import complete
from .store import load_transcript

_MAP = """Summarize the key points of this part of a recording in 2-4 bullets.
Be factual and concise; skip filler.

Transcript:
{text}"""

_REDUCE = """Below are ordered partial summaries of one recording.
Produce exactly these sections:

SUMMARY: 3-5 sentences covering the whole recording.
KEY TOPICS: bullet points.
ACTION ITEMS: concrete decisions or to-dos as bullets; write "None" if there are none.

Partial summaries:
{summaries}"""


def summarize(audio_path: str, persist_dir: str = ".audiorag",
              target_chars: int = 2000) -> str:
    segments = load_transcript(persist_dir, audio_path)
    chunks = chunk_segments(segments, target_chars=target_chars)
    partials = [complete(_MAP.format(text=c.text)) for c in chunks]
    return complete(_REDUCE.format(summaries="\n\n".join(partials)))
