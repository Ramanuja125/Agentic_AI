"""
Stage 5: Multi-agent system with Researcher + Writer pipeline.
"""

from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import requests

load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
MODEL = "openai/gpt-4o-mini"


# ---------- Tools (only researcher uses these) ----------

def wikipedia_search(query: str, num_results: int = 5) -> str:
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search",
                "srsearch": query, "srlimit": num_results, "format": "json",
            },
            headers={"User-Agent": "AgenticAILearningCourse/1.0"},
            timeout=10,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Error: search failed: {e}"

    hits = response.json().get("query", {}).get("search", [])
    if not hits:
        return f"No articles found for '{query}'."
    return "\n\n".join(
        f"Title: {h['title']}\nSnippet: {h['snippet'].replace(chr(60)+'span class=\"searchmatch\">', '').replace('</span>', '')}"
        for h in hits
    )


def wikipedia_get_article(title: str, max_chars: int = 5000) -> str:
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "prop": "extracts", "explaintext": True,
                "titles": title, "format": "json", "redirects": 1,
            },
            headers={"User-Agent": "AgenticAILearningCourse/1.0"},
            timeout=10,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Error: fetch failed: {e}"

    pages = response.json().get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    if "missing" in page:
        return f"No article with title '{title}'."
    extract = page.get("extract", "")
    if len(extract) > max_chars:
        extract = extract[:max_chars] + f"\n\n[Truncated. Original {len(page['extract'])} chars.]"
    return f"Article: {page.get('title')}\n\n{extract}"


RESEARCHER_TOOLS = {
    "wikipedia_search": wikipedia_search,
    "wikipedia_get_article": wikipedia_get_article,
}

RESEARCHER_TOOL_DECLARATIONS = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Search Wikipedia for article titles matching a query. Returns top 5 results with snippets.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_get_article",
            "description": "Fetch the plain text of a Wikipedia article by title. Truncated to ~5000 chars.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string", "description": "Article title"}},
                "required": ["title"]
            }
        }
    },
]


# ---------- Researcher Agent ----------

RESEARCHER_PROMPT = """You are a researcher agent. Your job is to gather factual information from Wikipedia about a topic.

You have access to wikipedia_search and wikipedia_get_article.

Your output should be a structured JSON object with these fields:
- topic: the subject you researched
- key_facts: a list of specific factual statements you discovered (with categories like "early_life", "scientific_work", "significance")
- sources: list of Wikipedia article titles you consulted

Do NOT write prose summaries or analysis — just facts. Your output will be passed to a writer agent who will compose the final response.

Be thorough but concise. Aim for 8-15 key facts covering different aspects of the topic."""


RESEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "key_facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "fact": {"type": "string"}
                },
                "required": ["category", "fact"],
                "additionalProperties": False
            }
        },
        "sources": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["topic", "key_facts", "sources"],
    "additionalProperties": False
}


def run_researcher(research_brief: str, max_iterations: int = 10) -> dict:
    """Run the researcher agent. Returns structured research output."""
    print("\n" + "=" * 60)
    print("RESEARCHER AGENT")
    print("=" * 60)

    messages = [
        {"role": "system", "content": RESEARCHER_PROMPT},
        {"role": "user", "content": research_brief},
    ]

    for step in range(max_iterations):
        print(f"\n[Researcher Step {step + 1}]")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=RESEARCHER_TOOL_DECLARATIONS,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=1500,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "research_output", "strict": True, "schema": RESEARCH_SCHEMA}
            },
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ]
            })

            for tc in message.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                print(f"  Tool: {name}({args})")
                result = RESEARCHER_TOOLS[name](**args)
                preview = result[:200] + ("..." if len(result) > 200 else "")
                print(f"  Result: {preview}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
        else:
            # Researcher produced final output
            findings = json.loads(message.content)
            print(f"\n[Researcher complete: {len(findings['key_facts'])} facts across {len(findings['sources'])} sources]")
            return findings

    print("[Researcher hit max iterations]")
    return {"topic": "unknown", "key_facts": [], "sources": []}


# ---------- Writer Agent ----------

WRITER_PROMPT = """You are a writer agent. Your job is to take structured research findings and produce a polished, engaging written response to the user's original request.

You have NO tools — you only write. The research has already been done for you.

Guidelines:
- Write in flowing prose, not bullet points (unless the user specifically asked for a list).
- Organize naturally — early life, work, significance, etc.
- Use only the facts provided. Do NOT invent details.
- If facts conflict or are unclear, acknowledge briefly rather than picking one.
- Length should match the user's request — a "short summary" should be 2-4 paragraphs."""


def run_writer(user_goal: str, research: dict) -> str:
    """Run the writer agent. Takes research findings, returns written prose."""
    print("\n" + "=" * 60)
    print("WRITER AGENT")
    print("=" * 60)

    facts_text = "\n".join(f"- [{f['category']}] {f['fact']}" for f in research["key_facts"])
    sources_text = ", ".join(research["sources"])

    writer_input = f"""USER'S ORIGINAL REQUEST:
{user_goal}

RESEARCH TOPIC: {research['topic']}

KEY FACTS:
{facts_text}

SOURCES CONSULTED: {sources_text}

Write the response now."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": WRITER_PROMPT},
            {"role": "user", "content": writer_input},
        ],
        temperature=0.3,  # slightly higher for prose variation
        max_tokens=1500,
    )

    return response.choices[0].message.content


# ---------- Orchestrator: the pipeline ----------

def run_pipeline(user_goal: str):
    print(f"\nUSER GOAL: {user_goal}")

    # Step 1: brief the researcher
    research_brief = f"Research the topic of the following user request, gathering facts that would help compose a response:\n\n{user_goal}"

    # Step 2: run the researcher
    findings = run_researcher(research_brief)

    # Step 3: pass findings to the writer
    final_output = run_writer(user_goal, findings)

    print("\n" + "=" * 60)
    print("FINAL OUTPUT (from writer)")
    print("=" * 60)
    print(final_output)
    print("=" * 60)

    return {"research": findings, "final_output": final_output}


if __name__ == "__main__":
    run_pipeline(
        "Write a short biographical summary of Marie Curie covering her early life, "
        "scientific work, and historical significance."
    )