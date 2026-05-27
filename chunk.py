"""Group short transcript segments into overlapping, retrieval-sized chunks.

Whisper emits short segments (a sentence or two). Embeddings retrieve better on
slightly larger windows, so we merge consecutive segments up to a character
budget. We carry the start/end timestamps forward so an answer can point back to
the exact moment in the audio — that timestamp is the whole reason this is an
*audio* RAG tool and not a text one.
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    start: float
    end: float


def chunk_segments(segments, target_chars: int = 500, overlap_segments: int = 1):
    """segments: any objects exposing .start, .end, .text (e.g. Segment)."""
    chunks: list[Chunk] = []
    i, n = 0, len(segments)
    while i < n:
        buf, total = [], 0
        start = segments[i].start
        end = segments[i].end
        j = i
        while j < n and total < target_chars:
            buf.append(segments[j].text)
            end = segments[j].end
            total += len(segments[j].text)
            j += 1
        chunks.append(Chunk(text=" ".join(buf), start=start, end=end))
        if j >= n:
            break
        i = max(j - overlap_segments, i + 1)  # overlap so context isn't cut mid-idea
    return chunks
