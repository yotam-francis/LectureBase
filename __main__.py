"""CLI: index a recording, ask questions over it, or summarize it.

    python -m audiorag index lecture1.mp3
    python -m audiorag ask "what did they say about backpropagation?"
    python -m audiorag summarize lecture1.mp3
    python -m audiorag agent "explain the difference between RNN and transformer"
"""
import argparse
from dotenv import load_dotenv


def main():
    load_dotenv()  # reads .env into the environment before anything else runs
    p = argparse.ArgumentParser(prog="audiorag")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="transcribe + index audio file(s)")
    pi.add_argument("audio", nargs="+", help="one or more files or glob patterns e.g. lectures\\*.mp3")
    pi.add_argument("--model-size", default="base",
                    help="whisper size: tiny/base/small/medium/large-v3")
    pi.add_argument("--language", default=None,
                    help="audio language e.g. 'he' for Hebrew, 'en' for English. "
                         "Auto-detects if omitted.")

    pa = sub.add_parser("ask", help="ask a question across indexed lectures")
    pa.add_argument("question")
    pa.add_argument("-k", type=int, default=4, help="how many chunks to retrieve")

    ps = sub.add_parser("summarize", help="summary + action items for one file")
    ps.add_argument("audio")

    pag = sub.add_parser("agent", help="agentic Q&A — model decides when and what to search")
    pag.add_argument("question")
    pag.add_argument("--max-steps", type=int, default=6,
                     help="max search rounds before giving up (default 6)")

    pd = sub.add_parser("diagnose", help="analyse audio signal quality before indexing")
    pd.add_argument("audio", nargs="+", help="one or more audio files to analyse")
    pd.add_argument("--offset", type=float, default=1200.0,
                    help="start analysis at this many seconds into the file (default 1200 = 20 min)")
    pd.add_argument("--duration", type=float, default=300.0,
                    help="how many seconds to analyse (default 300 = 5 min)")

    args = p.parse_args()
    if args.cmd == "index":
        import glob as _glob
        from .index import build_index
        files = []
        for pattern in args.audio:
            expanded = _glob.glob(pattern)
            files.extend(expanded if expanded else [pattern])
        for f in files:
            print(f"Indexing {f} ...")
            n = build_index(f, model_size=args.model_size, language=args.language)
            print(f"  → {n} chunks indexed.")
    elif args.cmd == "ask":
        from .query import ask
        print(ask(args.question, k=args.k))
    elif args.cmd == "summarize":
        from .summarize import summarize
        print(summarize(args.audio))
    elif args.cmd == "agent":
        from .agent import run_agent
        print(run_agent(args.question, max_steps=args.max_steps))
    elif args.cmd == "diagnose":
        from .diagnose import diagnose
        for f in args.audio:
            diagnose(f, offset=args.offset, duration=args.duration)


if __name__ == "__main__":
    main()
