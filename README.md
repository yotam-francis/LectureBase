# LectureBase

Ask questions over your lecture recordings and get answers with timestamps,
or generate a summary and action items for any recording. A local, private
RAG pipeline built on audio.

```
audio ──> Whisper ──> segments ──> chunk ──> embed ──> Chroma
                                                         │
question ──> embed ──> similarity search ────────────────┘
                          │
                          └─> top-k chunks ──> LLM ──> answer [mm:ss]
```

Everything runs locally. No data leaves your machine.

---

## Install

```bash
pip install -r requirements.txt
cp .env.example .env        # configure your LLM backend
ollama pull llama3.1        # recommended: 8B, reliable tool use
```

Requires [Ollama](https://ollama.com) for the LLM and [ffmpeg](https://ffmpeg.org) for audio processing.

---

## Usage

```bash
# transcribe + index a recording (or a whole folder)
python -m lecturebase index lectures/lecture1.mp3
python -m lecturebase index lectures/*.mp3 --language he
python -m lecturebase index lectures/lecture1.mp3 --model-size small  # better quality, slower

# ask a question across all indexed lectures (timestamped answer)
python -m lecturebase ask "how does a MUX work as a function creator?"
python -m lecturebase ask "explain two's complement" -k 8

# agentic Q&A — model decides what to search and can search multiple times
python -m lecturebase agent "compare sign magnitude vs two's complement with an example"

# summarise one recording (summary + key topics + action items)
python -m lecturebase summarize lectures/lecture1.mp3

# diagnose audio quality before indexing
python -m lecturebase diagnose lectures/lecture1.mp3 --offset 1200 --duration 300
```

---

## Configuration

Copy `.env.example` to `.env`:

```env
# LLM backend (default: Ollama, fully local and free)
AUDIORAG_LLM=ollama
OLLAMA_MODEL=llama3.1

# Alternatively, use Anthropic (requires API key)
# AUDIORAG_LLM=anthropic
# ANTHROPIC_API_KEY=sk-ant-...
# AUDIORAG_MODEL=claude-haiku-4-5-20251001
```

---

## Limitations

**Transcription quality depends on Whisper model size.**
The default `base` model (74M params) hallucinates on long recordings,
especially when audio starts with silence or when the speaker switches
languages mid-lecture. Use `--model-size small` (244M) for reliable
transcription. Expect ~3× slower indexing on CPU.

**Local LLMs are unreliable for tool use.**
The `agent` command requires the model to make structured tool calls.
Models under ~8B parameters frequently generate fake search results
instead of calling the tool. `llama3.1` (8B) is the minimum practical
size. For guaranteed reliable tool use, configure the Anthropic backend.

**Answer quality is bounded by retrieval quality.**
If a lecture section was hallucinated during transcription (common with
the `base` Whisper model), that content is not in the index and cannot
be retrieved. Re-index with `--model-size small` if answers are missing
known content.

**Everything runs on CPU by default.**
Indexing a 1-hour lecture takes roughly 10–20 minutes on CPU with the
`small` model. A GPU is not required but significantly speeds up both
transcription and embedding.
My private laptop has no fancy GPU ¯\_(ツ)_/¯

---

## Future work

**Meeting transcription agent.**
The same pipeline generalises directly to meeting recordings: transcribe
→ index → ask questions, or summarise with action items per speaker.
Adding `pyannote.audio` for speaker diarisation would turn the
`summarize` command into a proper meeting assistant — each action item
attributed to the person who said it.

**Evaluation harness.**
No current measurement of retrieval hit-rate or answer faithfulness.
A small set of question/answer pairs per lecture would let us tune `k`,
chunk size, and the embedding model against actual numbers rather than
vibes.

**Multilingual support.**
`all-MiniLM-L6-v2` is English-biased. Switching to
`paraphrase-multilingual-MiniLM-L12-v2` and passing `--language he`
at index time would give proper Hebrew support — relevant for
Israeli university lectures.

**Reranking.**
Retrieve top-20 cheaply, then a cross-encoder reranks to the best 4.
Largest quality-per-effort improvement for larger lecture collections.
