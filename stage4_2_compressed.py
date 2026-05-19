"""
Stage 4.2: Same agent as 3.5 + structured output, but with working-memory compression.
When the conversation exceeds a token threshold, old turns are summarized.
"""

from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import requests
import io
import contextlib
import traceback

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "openai/gpt-4o-mini"


# ---------- Calculator ----------

def calculator(expression: str) -> str:
    expression = expression.replace("^", "**")
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ---------- Lookup ----------

KNOWLEDGE_BASE = {
    "population of tokyo": "13960000",
    "population of delhi": "32941000",
    "population of new york city": "8336000",
    "population of london": "8982000",
    "population of paris": "2161000",
    "population of mumbai": "20411000",
    "year python was created": "1991",
    "year javascript was created": "1995",
    "year java was created": "1995",
    "year c++ was created": "1985",
    "height of mount everest in meters": "8849",
    "height of k2 in meters": "8611",
    "height of kangchenjunga in meters": "8586",
    "speed of light in meters per second": "299792458",
    "speed of sound in meters per second": "343",
    "distance from earth to moon in km": "384400",
    "distance from earth to sun in km": "149600000",
}

STOPWORDS = {"of", "the", "in", "a", "an", "is", "was", "are", "to", "for"}


def lookup(query: str) -> str:
    normalized = query.lower().strip()
    if normalized in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[normalized]
    query_words = {w for w in normalized.split() if w not in STOPWORDS and len(w) > 1}
    scored = []
    for key in KNOWLEDGE_BASE:
        key_words = {w for w in key.split() if w not in STOPWORDS}
        overlap = len(query_words & key_words)
        if overlap > 0:
            scored.append((overlap, key))
    scored.sort(reverse=True)
    if scored:
        top = [k for _, k in scored[:3]]
        return f"Not found. Closest entries in the knowledge base: {top}"
    categories = sorted({key.split()[0] + "..." for key in KNOWLEDGE_BASE})
    return f"Not found. No similar entries. This knowledge base only contains entries starting with: {categories}"


# ---------- Wikipedia search ----------

def wikipedia_search(query: str, num_results: int = 5) -> str:
    """Searches Wikipedia for article titles matching the query."""
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": num_results,
                "format": "json",
            },
            headers={"User-Agent": "AgenticAILearningCourse/1.0"},
            timeout=10,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return "Error: Wikipedia search timed out."
    except requests.exceptions.RequestException as e:
        return f"Error: Wikipedia search failed: {e}"

    data = response.json()
    hits = data.get("query", {}).get("search", [])

    if not hits:
        return f"No Wikipedia articles found for '{query}'. Try a more specific or differently-phrased query."

    results = []
    for hit in hits:
        title = hit["title"]
        # Snippet is HTML-formatted; strip the tags crudely for cleanliness.
        snippet = hit["snippet"].replace("<span class=\"searchmatch\">", "").replace("</span>", "")
        results.append(f"Title: {title}\nSnippet: {snippet}")

    return "\n\n".join(results)


# ---------- Wikipedia article fetch ----------

def wikipedia_get_article(title: str, max_chars: int = 5000) -> str:
    """Fetches the plain-text content of a Wikipedia article by exact title."""
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": True,  # return plain text, not HTML
                "exintro": False,     # full article, not just intro
                "titles": title,
                "format": "json",
                "redirects": 1,       # follow redirects (e.g. "JS" -> "JavaScript")
            },
            headers={"User-Agent": "AgenticAILearningCourse/1.0"},
            timeout=10,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return "Error: Wikipedia fetch timed out."
    except requests.exceptions.RequestException as e:
        return f"Error: Wikipedia fetch failed: {e}"

    data = response.json()
    pages = data.get("query", {}).get("pages", {})

    if not pages:
        return f"No article data returned for '{title}'."

    # The pages dict is keyed by page ID; just take the first (and usually only) page.
    page = next(iter(pages.values()))

    if "missing" in page:
        return f"No Wikipedia article exists with the exact title '{title}'. Try wikipedia_search first to find the correct title."

    extract = page.get("extract", "")
    if not extract.strip():
        return f"Article '{title}' exists but has no extractable text."

    if len(extract) > max_chars:
        extract = extract[:max_chars] + f"\n\n[Content truncated. Original article is {len(page['extract'])} chars.]"

    return f"Article: {page['title']}\n\n{extract}"


# ---------- Python execution ----------

def python_exec(code: str) -> str:
    """Executes Python code and returns printed output. Trusted environment — full Python is available."""
    stdout_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, {})
        output = stdout_buffer.getvalue()
        if not output.strip():
            output = "(no output — code ran successfully but printed nothing)"
        return output
    except Exception:
        return f"Error executing code:\n{traceback.format_exc()}"
    

# ---------- Tokenizer and Summarize - compression ----------
# NEW: rough token counter. Real systems use tiktoken or the model's tokenizer.
def estimate_tokens(messages: list) -> int:
    """Rough token estimate: ~4 chars per token. Good enough for compression triggers."""
    total_chars = 0
    for msg in messages:
        # Each message has overhead (role labels, formatting) — add ~10 tokens per message.
        total_chars += 40
        if msg.get("content"):
            total_chars += len(msg["content"])
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                total_chars += len(tc["function"]["name"])
                total_chars += len(tc["function"]["arguments"])
    return total_chars // 4


# NEW: summarize a chunk of conversation using the LLM itself.
# NEW: summarize a chunk of conversation using the LLM itself.
def summarize_conversation(messages_to_summarize: list, user_goal: str) -> str:
    """Asks the LLM to summarize a slice of conversation. Returns a summary string."""
    # Format the messages into a readable transcript for the summarizer.
    transcript_parts = []
    for msg in messages_to_summarize:
        role = msg["role"]
        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                transcript_parts.append(f"ASSISTANT called {tc['function']['name']}({tc['function']['arguments']})")
        elif role == "tool":
            transcript_parts.append(f"TOOL RESULT: {msg['content']}")
        elif role == "assistant":
            transcript_parts.append(f"ASSISTANT: {msg['content']}")
        elif role == "user":
            transcript_parts.append(f"USER: {msg['content']}")
    
    transcript = "\n\n".join(transcript_parts)
    
    summary_prompt = f"""You are summarizing the early steps of an agent solving a task.

ORIGINAL USER GOAL:
{user_goal}

TRANSCRIPT TO SUMMARIZE:
{transcript}

Produce a CONCISE summary that captures:
1. Key facts the agent established (with specific numbers/values)
2. What tools were tried and what worked vs. didn't
3. Any constraints, dead ends, or decisions made
4. Current state of progress toward the goal

Do NOT include verbose intermediate reasoning. Do NOT repeat the tool call syntax.
Be terse but specific — preserve exact numbers and findings."""
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.0,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


# NEW: compress the conversation if it's too long.
def compress_if_needed(messages: list, user_goal: str, token_threshold: int = 2000, keep_last_n_turns: int = 2) -> list:
    """
    If messages exceed token_threshold, summarize the older portion.
    
    Strategy: keep the system prompt and the most recent `keep_last_n_turns` complete 
    turns full-fidelity. Summarize everything in between.
    """
    if estimate_tokens(messages) < token_threshold:
        return messages  # No compression needed
    
    # Index 0 is the system prompt — always keep.
    # Index 1 is the user goal — always keep.
    # We want to keep the last N "turn groups" intact.
    # A turn group starts with an assistant message (possibly with tool_calls) 
    # and includes all subsequent tool messages until the next assistant message.
    
    # Find the indices where each assistant message starts.
    assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    
    if len(assistant_indices) <= keep_last_n_turns:
        return messages  # Not enough turns to bother compressing
    
    # The first assistant message we want to KEEP is the (N from the end)th one.
    cutoff_index = assistant_indices[-keep_last_n_turns]
    
    # Messages to summarize: everything from index 2 (just after system + user goal) up to (but not including) cutoff.
    messages_to_summarize = messages[2:cutoff_index]
    if not messages_to_summarize:
        return messages
    
    print(f"\n[Compressing {len(messages_to_summarize)} old messages — context was ~{estimate_tokens(messages)} tokens]")
    summary = summarize_conversation(messages_to_summarize, user_goal)
    print(f"[Summary length: ~{len(summary) // 4} tokens]\n")
    
    # Build the new compressed message list.
    compressed = (
        messages[:2]  # system + original user goal
        + [{"role": "user", "content": f"[SUMMARY OF EARLIER STEPS]\n{summary}\n[END SUMMARY — continuing from here]"}]
        + messages[cutoff_index:]  # the kept turns
    )
    
    return compressed



# ---------- Tool registry and declarations ----------

TOOLS = {
    "calculator": calculator,
    "lookup": lookup,
    "wikipedia_search": wikipedia_search,
    "wikipedia_get_article": wikipedia_get_article,
    "python_exec": python_exec,
}

TOOL_DECLARATIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluates a Python math expression. Use ** for exponentiation. Good for simple arithmetic. For anything more involved, use python_exec.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "A Python math expression, e.g. '23 * 47 + 100'."}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": (
                "Looks up a fact in a small in-memory knowledge base. "
                "Available facts: populations of Tokyo, Delhi, NYC, London, Paris, Mumbai; "
                "creation years of Python, JavaScript, Java, C++; "
                "heights of Everest, K2, Kangchenjunga (in meters); "
                "speed of light/sound, distance Earth-Moon/Sun. "
                "For anything else, use wikipedia_search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Lowercase phrase, e.g. 'population of tokyo'."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": (
                "Searches Wikipedia for the top article titles matching a query. Returns title + short snippet for each. "
                "Use to find which Wikipedia article(s) might have the info you need. "
                "Once you've identified a good title, use wikipedia_get_article to fetch the full text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, e.g. 'Burj Khalifa height' or 'Nobel Prize in Physics 2024'."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_get_article",
            "description": (
                "Fetches the plain text of a Wikipedia article by exact title (case-sensitive, though common redirects work). "
                "Content truncated to ~5000 chars. Use after wikipedia_search to read the most promising result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Exact article title, e.g. 'Burj Khalifa' or 'Diocletian'."}
                },
                "required": ["title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": (
                "Executes Python code. Full Python is available — import anything you need (math, statistics, datetime, json, re, etc.). "
                "IMPORTANT: only PRINTED output is returned. Use print() to see results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code, e.g. 'import datetime\\nprint((datetime.date.today() - datetime.date(1776,7,4)).days)'."}
                },
                "required": ["code"]
            }
        }
    },
]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The direct answer to the user's question, in plain text."
        },
        "key_facts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The specific facts you used to arrive at the answer. One fact per entry. Include source where relevant."
        },
        "sources_used": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["calculator", "lookup", "wikipedia_search", "wikipedia_get_article", "python_exec", "prior_knowledge"]
            },
            "description": "Which tools (or prior knowledge) you used to answer."
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "high = verified by tools or trivially known; medium = inferred or partially verified; low = uncertain or guessing."
        },
        "limitations": {
            "type": "string",
            "description": "Anything you couldn't determine, caveats about your answer, or assumptions you made. Empty string if none."
        }
    },
    "required": ["answer", "key_facts", "sources_used", "confidence", "limitations"],
    "additionalProperties": False
}


# ---------- System prompt ----------

SYSTEM_PROMPT = """You are a research agent that solves problems by calling tools.

You have access to: calculator, lookup (small fact KB), wikipedia_search, wikipedia_get_article, and python_exec.

Strategy:
- For factual questions: try lookup first, fall back to wikipedia_search if needed.
- For Wikipedia: search first to find article titles, then fetch the article(s) you want.
- For computation: calculator for simple arithmetic, python_exec for anything more involved.
- Don't loop. If a tool fails twice, try a different approach.

When you have your final answer, respond with a JSON object containing:
- answer: the direct answer
- key_facts: list of specific facts you used (with sources)
- sources_used: which tools you used (or "prior_knowledge" if you relied on what you already knew)
- confidence: high/medium/low based on how well-verified your answer is
- limitations: caveats, assumptions, or things you couldn't determine

Be honest about confidence. If you guessed at a number, say so. If you couldn't verify something, say so."""



# ---------- Agent loop with loop detection ----------

def run_agent(user_goal: str):
    print(f"USER GOAL: {user_goal}\n")
    print("=" * 60)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_goal},
    ]

    max_iterations = 20  # Higher since we can now run longer
    recent_calls = []

    for step in range(max_iterations):
        print(f"\n--- Step {step + 1} (context: ~{estimate_tokens(messages)} tokens) ---")

        # NEW: compress before making the LLM call.
        messages = compress_if_needed(messages, user_goal)

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_DECLARATIONS,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=1500,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "research_result",
                    "strict": True,
                    "schema": RESPONSE_SCHEMA
                }
            },
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    }
                    for tc in message.tool_calls
                ]
            })

            for tc in message.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    result = "Error: tool arguments were not valid JSON."
                    print(f"Tool call: {name}(<malformed args>)")
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    continue

                args_preview = {k: (v[:100] + "...") if isinstance(v, str) and len(v) > 100 else v for k, v in args.items()}
                print(f"Tool call: {name}({args_preview})")

                if name in TOOLS:
                    result = TOOLS[name](**args)
                else:
                    result = f"Error: unknown tool '{name}'. Valid: {list(TOOLS)}"

                # Loop detection (unchanged from 3.3)
                call_signature = (name, json.dumps(args, sort_keys=True)[:200])
                recent_calls.append(call_signature)
                if len(recent_calls) > 6:
                    recent_calls.pop(0)
                if recent_calls.count(call_signature) >= 3:
                    result = f"{result}\n\n[SYSTEM NOTE: You've made this exact call 3+ times. Try a different approach.]"
                elif len(recent_calls) >= 4:
                    last_4_tools = [c[0] for c in recent_calls[-4:]]
                    if last_4_tools.count(name) >= 4:
                        result = f"{result}\n\n[SYSTEM NOTE: '{name}' tried 4 times with no success. Use a different tool or give a partial answer.]"

                result_preview = result[:300] + ("..." if len(result) > 300 else "")
                print(f"Result: {result_preview}")

                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

        else:
            # The final answer should be JSON conforming to RESPONSE_SCHEMA.
            print("\n--- FINAL STRUCTURED ANSWER ---")
            try:
                parsed = json.loads(message.content)
                print(json.dumps(parsed, indent=2))
                print("=" * 60)
                return parsed
            except json.JSONDecodeError:
                print("ERROR: model did not return valid JSON.")
                print("Raw content:", message.content)
                print("=" * 60)
                return None

    print("Max iterations reached without final answer.")
    return None


if __name__ == "__main__":
    run_agent(
        "Pick three different Wikipedia articles about different famous scientists "
        "(Albert Einstein, Marie Curie, and Isaac Newton). For each, find and report: "
        "(1) the year they were born, (2) their most famous discovery or contribution, "
        "and (3) what field they worked in. Then tell me which one was born first."
    )