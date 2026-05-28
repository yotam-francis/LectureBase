"""Retrieval evaluation harness.

Measures hit-rate: for each question, does the correct chunk appear in top-k?
This is the single most important metric for a RAG system — if retrieval fails,
the LLM can't answer correctly no matter how good it is.

Usage:
    python -m lecturebase eval eval_set.json
    python -m lecturebase eval eval_set.json --k 8 --persist-dir .lecturebase

eval_set.json format:
    [
      {
        "question": "what is two's complement?",
        "must_contain": "flip the bits and add one",
        "note": "optional human note"
      },
      ...
    ]

must_contain: a substring (case-insensitive) that should appear in at
least one of the top-k retrieved chunks if retrieval is working correctly.
Keep it short and distinctive — a unique phrase from the lecture, not a
generic word.
"""
import json


def run_eval(eval_path: str, k: int = 4, persist_dir: str = ".lecturebase") -> dict:
    from lecturebase.query import retrieve

    with open(eval_path) as f:
        cases = json.load(f)

    hits, misses = [], []

    for case in cases:
        question = case["question"]
        needle = case["must_contain"].lower()
        results = retrieve(question, persist_dir=persist_dir, k=k)
        found = any(needle in doc.lower() for doc, _ in results)

        if found:
            hits.append(question)
        else:
            misses.append({
                "question": question,
                "expected": case["must_contain"],
                "retrieved": [doc[:120] for doc, _ in results],
            })

    total = len(cases)
    hit_rate = len(hits) / total if total else 0

    return {
        "total": total,
        "hits": len(hits),
        "misses": len(misses),
        "hit_rate": hit_rate,
        "failed": misses,
    }


def print_report(results: dict, k: int) -> None:
    print(f"\n{'─'*50}")
    print(f"  Retrieval eval  (k={k})")
    print(f"{'─'*50}")
    print(f"  Questions : {results['total']}")
    print(f"  Hits      : {results['hits']}")
    print(f"  Misses    : {results['misses']}")
    print(f"  Hit-rate  : {results['hit_rate']*100:.1f}%")

    if results["failed"]:
        print(f"\n── Failures ──")
        for f in results["failed"]:
            print(f"\n  Q: {f['question']}")
            print(f"  Expected to contain: \"{f['expected']}\"")
            print(f"  Top retrieved chunks:")
            for i, chunk in enumerate(f["retrieved"], 1):
                print(f"    [{i}] {chunk}...")
    print()
