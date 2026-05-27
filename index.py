"""Build the searchable index for an audio file.

One pass: transcribe -> save transcript -> chunk -> embed -> upsert to Chroma.
Embeddings + the vector store are loaded lazily so the package imports fast and
stays unit-testable without the heavy deps installed.
"""
from .chunk import chunk_segments
from .store import save_transcript
from .transcribe import transcribe

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # small, fast, CPU-friendly, good enough for this


def build_index(audio_path: str, persist_dir: str = ".audiorag",
                collection: str = "lectures", model_size: str = "base") -> int:
    from sentence_transformers import SentenceTransformer
    import chromadb

    segments = transcribe(audio_path, model_size=model_size)
    save_transcript(persist_dir, audio_path, segments)
    chunks = chunk_segments(segments)

    embedder = SentenceTransformer(EMBED_MODEL)
    vectors = embedder.encode([c.text for c in chunks]).tolist()

    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_or_create_collection(collection)
    col.upsert(
        ids=[f"{audio_path}::{k}" for k in range(len(chunks))],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[{"start": c.start, "end": c.end, "source": audio_path}
                   for c in chunks],
    )
    return len(chunks)
