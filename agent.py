"""RAG agent: the model decides when to search, what to search for, and
whether to search again before answering.

Uses Ollama's OpenAI-compatible endpoint so tool-use works properly.
The retrieval logic (retrieve, build_context) is identical to the pipeline —
the only new thing is the loop that lets the model drive it.
"""
import json
import os

from .query import build_context, retrieve

# Tool definition — plain English description the model reads to decide
# when and how to call the search. The parameters block is a JSON Schema.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_lectures",
            "description": (
                "Search indexed lecture transcripts by meaning. "
                "Returns relevant excerpts with [mm:ss] timestamps. "
                "Call this whenever you need information from the lectures."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A natural-language search query.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

SYSTEM = (
    "You are a helpful study assistant. You have access to a search tool "
    "that retrieves excerpts from indexed lecture recordings with timestamps. "
    "Always search before answering factual questions about the lectures. "
    "Cite [mm:ss] timestamps in your answers."
)


def run_agent(question: str, max_steps: int = 6) -> str:
    from openai import OpenAI  # lazy import — only needed for the agent command

    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # Ollama requires the field but ignores the value
    )

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
        )
        msg = response.choices[0].message

        # --- build the assistant turn to add to history ---
        assistant_turn = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_turn["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_turn)

        # --- no tool calls → model is done ---
        if not msg.tool_calls:
            return msg.content or "(no response)"

        # --- execute each tool call and feed results back ---
        for tc in msg.tool_calls:
            if tc.function.name == "search_lectures":
                args = json.loads(tc.function.arguments)
                query = args.get("query", "")
                # small models (e.g. llama3.2:3b) sometimes return the schema
                # dict instead of a plain string: {"type": "string", "value": "..."}
                if isinstance(query, dict):
                    query = query.get("value") or query.get("query") or str(query)
                hits = retrieve(query)
                result = build_context(hits) if hits else "No results found."
            else:
                result = f"Unknown tool: {tc.function.name}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

        print(f"  [step {step + 1}] searched: {query!r}")

    return "Stopped: reached maximum search steps."