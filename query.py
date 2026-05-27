"""Answer a question over indexed lectures, citing audio timestamps.

`ask` searches the whole collection (all lectures you've indexed), so you can ask
across a semester, not just one recording.
"""
from .llm import complete

EMBED_MODEL = "all-MiniLM-L6-v2"

_PROMPT = """You are helping a student review lecture recordings.
Answer the question using ONLY the transcript excerpts below.
If the answer isn't in them, say so plainly. Cite the timestamp(s) you used
in [mm:ss] form.

Excerpts:
{context}

Question: {question}
Answer:"""


def fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def build_context(hits) -> str:
    """hits: list of (document_text, metadata_dict)."""
    return "\n".join(f"[{fmt_ts(meta['start'])}] {doc}" for doc, meta in hits)


def retrieve(question: str, persist_dir: str = ".audiorag",
             collection: str = "lectures", k: int = 4):
    from sentence_transformers import SentenceTransformer
    import chromadb

    embedder = SentenceTransformer(EMBED_MODEL)
    qvec = embedder.encode([question]).tolist()
    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_collection(collection)
    res = col.query(query_embeddings=qvec, n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0]))


def ask(question: str, persist_dir: str = ".audiorag",
        collection: str = "lectures", k: int = 4) -> str:
    hits = retrieve(question, persist_dir, collection, k)
    if not hits:
        return "Nothing indexed yet — run `audiorag index <file>` first."
    prompt = _PROMPT.format(context=build_context(hits), question=question)
    return complete(prompt)
