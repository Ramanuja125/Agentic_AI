"""
Stage 3.5 similar to 3.3 real tools but has response_schema Agent with Wikipedia search, Wikipedia fetch, Python exec,
calculator, and lookup. Includes loop detection.
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

MODEL = "google/gemini-2.5-flash"


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

# --- reflection
def critique_answer(client, model: str, user_goal: str, proposed_answer: dict) -> dict:
    """
    Have a separate LLM call critique a proposed structured answer.
    Returns {'verdict': 'APPROVED' | 'REVISE', 'critique': str}.
    """
    critic_prompt = f"""You are a critical reviewer. Your job is to find problems with proposed answers BEFORE they're delivered to the user.

USER'S ORIGINAL QUESTION:
{user_goal}

PROPOSED ANSWER:
{json.dumps(proposed_answer, indent=2)}

Find issues. Look specifically for:
- Does the answer actually address what was asked? (Wrong direction, missing part of question, etc.)
- Are numeric claims internally consistent? (Did the agent quote 829.8 but use 828 in calculations? Did units convert correctly?)
- Are factual claims appropriately sourced or hedged?
- Is the `confidence` level honest given the evidence?
- Is the `limitations` field empty when it should mention real caveats (stale data, assumptions, ambiguities, etc.)?
- Does the answer contradict the key_facts?

Default to finding issues. Most answers have at least one. Don't just say "looks good" unless you've genuinely tried to find problems and can't.

Respond with EXACTLY this JSON format:
{{
  "verdict": "APPROVED" or "REVISE",
  "critique": "Specific issues found, or empty if APPROVED. Be concrete: name the exact field and what's wrong."
}}

If even one real issue exists, verdict should be REVISE. APPROVED means you genuinely cannot find a problem."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": critic_prompt}],
        temperature=0.0,
        max_tokens=500,
        response_format={"type": "json_object"},
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        # If critic produces malformed JSON, treat as approval (fail open).
        return {"verdict": "APPROVED", "critique": "Critic output unparseable."}
    

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

    max_iterations = 15
    recent_calls = []

    for step in range(max_iterations):
        print(f"\n--- Step {step + 1} ---")

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

                # Loop detection
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
            # Final answer reached — run reflection before delivering.
            print("\n--- PROPOSED ANSWER (before reflection) ---")
            try:
                proposed = json.loads(message.content)
                print(json.dumps(proposed, indent=2))
            except json.JSONDecodeError:
                print("ERROR: model did not return valid JSON.")
                print("Raw content:", message.content)
                return None

            # CRITIC PASS
            print("\n[Running critic...]")
            critique = critique_answer(client, MODEL, user_goal, proposed)
            print(f"[Critic verdict: {critique['verdict']}]")

            if critique["verdict"] == "APPROVED":
                print(f"\n--- FINAL ANSWER (approved on first try) ---")
                print(json.dumps(proposed, indent=2))
                print("=" * 60)
                return proposed

            # REVISE: send critique back to the agent for one revision round.
            print(f"[Critique]: {critique['critique']}")
            print("\n[Asking agent to revise...]")

            messages.append({
                "role": "assistant",
                "content": message.content,
            })
            messages.append({
                "role": "user",
                "content": (
                    f"Your answer has been reviewed and needs revision. "
                    f"Critique:\n{critique['critique']}\n\n"
                    f"Please produce a revised answer that addresses these issues. "
                    f"Use tools if needed to verify facts."
                )
            })

            # Revision sub-loop — allow up to 5 more steps for the agent to verify and revise.
            for revise_step in range(5):
                print(f"\n--- Revision step {revise_step + 1} ---")

                revise_response = client.chat.completions.create(
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

                revise_message = revise_response.choices[0].message

                if revise_message.tool_calls:
                    # Agent wants to verify something — handle tool calls.
                    messages.append({
                        "role": "assistant",
                        "content": revise_message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                            }
                            for tc in revise_message.tool_calls
                        ]
                    })

                    for tc in revise_message.tool_calls:
                        name = tc.function.name
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            result = "Error: tool arguments were not valid JSON."
                            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                            continue

                        print(f"Revision tool call: {name}({args})")
                        if name in TOOLS:
                            result = TOOLS[name](**args)
                        else:
                            result = f"Error: unknown tool '{name}'."
                        result_preview = result[:200] + ("..." if len(result) > 200 else "")
                        print(f"Result: {result_preview}")
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
                else:
                    # Agent produced a revised final answer.
                    try:
                        revised = json.loads(revise_message.content)
                        print("\n--- FINAL ANSWER (after revision) ---")
                        print(json.dumps(revised, indent=2))
                        print("=" * 60)
                        return revised
                    except json.JSONDecodeError:
                        print("Revised answer was not valid JSON. Returning original proposed answer.")
                        return proposed

            # Revision loop ran out of steps.
            print("Revision didn't complete in 5 steps. Returning original proposed answer.")
            return proposed

    print("Max iterations reached without final answer.")
    return None


if __name__ == "__main__":
    run_agent(
        "What's the population of Berlin? Find it on Wikipedia, then tell me what it is "
        "per square kilometer if Berlin's area is 891.7 square kilometers."
    )