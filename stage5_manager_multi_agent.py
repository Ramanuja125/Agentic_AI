"""
Stage 5b: Manager/Worker multi-agent system.
A trip-planning manager delegates to specialized worker agents.
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


# ============================================================
# SHARED TOOLS (used by some workers)
# ============================================================

def wikipedia_search(query: str) -> str:
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query, "srlimit": 5, "format": "json"},
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


def wikipedia_get_article(title: str, max_chars: int = 3000) -> str:
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "prop": "extracts", "explaintext": True, "titles": title, "format": "json", "redirects": 1},
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
        extract = extract[:max_chars] + "..."
    return f"Article: {page.get('title')}\n\n{extract}"


WIKIPEDIA_TOOLS = {
    "wikipedia_search": wikipedia_search,
    "wikipedia_get_article": wikipedia_get_article,
}

WIKIPEDIA_TOOL_DECLARATIONS = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Search Wikipedia for article titles.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_get_article",
            "description": "Fetch a Wikipedia article by title.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"]
            }
        }
    },
]


# ============================================================
# WORKER AGENTS — each runs a small focused agent
# ============================================================

def _run_wikipedia_worker(system_prompt: str, user_input: str, label: str, max_iterations: int = 6) -> str:
    """Helper: run a small ReAct loop with Wikipedia tools, return the final text."""
    print(f"\n  [Worker: {label}] received: {user_input[:100]}...")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    
    for step in range(max_iterations):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=WIKIPEDIA_TOOL_DECLARATIONS,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=1000,
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
                args = json.loads(tc.function.arguments)
                print(f"    [{label}] calling {tc.function.name}({list(args.values())[0][:50]}...)")
                result = WIKIPEDIA_TOOLS[tc.function.name](**args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
        else:
            print(f"  [Worker: {label}] returning response ({len(message.content)} chars)")
            return message.content
    
    return "Worker hit max iterations without producing output."


def destination_researcher(destination: str, travel_style: str = "general tourism") -> str:
    """Worker: research a destination's key practical info."""
    system_prompt = """You are a destination researcher. Given a destination and travel style, gather KEY PRACTICAL INFO from Wikipedia:
- Best time to visit (climate, seasons)
- Getting there and around (transportation)
- Major neighborhoods or regions
- Practical considerations (currency, language, safety basics)

Use wikipedia_search to find the right article, then wikipedia_get_article to read it. Be efficient — 2-3 tool calls is usually enough.

Return a concise summary of practical info. Plain text, not JSON. Aim for 150-300 words."""
    
    user_input = f"Destination: {destination}\nTravel style: {travel_style}"
    return _run_wikipedia_worker(system_prompt, user_input, "DestinationResearcher")


def activities_researcher(destination: str, interests: str = "general sightseeing") -> str:
    """Worker: research things to do at a destination."""
    system_prompt = """You are an activities researcher. Given a destination and interests, find NOTABLE THINGS TO DO from Wikipedia:
- Major attractions and landmarks
- Cultural sites (museums, historic places)
- Unique experiences specific to that location
- Activities that match the user's interests

Use wikipedia_search and wikipedia_get_article. Be efficient — 2-3 tool calls is usually enough.

Return a list of 5-8 specific recommendations with brief explanations. Plain text. Aim for 200-400 words."""
    
    user_input = f"Destination: {destination}\nUser interests: {interests}"
    return _run_wikipedia_worker(system_prompt, user_input, "ActivitiesResearcher")


def budget_analyzer(destination: str, num_days: int, budget_usd: int, traveler_count: int = 1) -> str:
    """Worker: analyze whether a budget is realistic. No tools — just LLM reasoning."""
    print(f"\n  [Worker: BudgetAnalyzer] analyzing ${budget_usd} for {num_days} days in {destination}")
    
    system_prompt = """You are a budget analyst for travel planning. Given a destination, duration, and budget, provide:
1. A realistic assessment: is this budget tight, comfortable, or generous for this destination?
2. Rough breakdown of how the budget should be allocated (accommodation, food, activities, transport, misc)
3. Specific tips for staying within budget OR for using a generous budget well

Use your general knowledge of travel costs. Be honest — if a budget is unrealistic, say so. Plain text response, 150-250 words."""
    
    user_input = f"Destination: {destination}\nDuration: {num_days} days\nBudget: ${budget_usd} USD\nTravelers: {traveler_count}"
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    result = response.choices[0].message.content
    print(f"  [Worker: BudgetAnalyzer] returning analysis ({len(result)} chars)")
    return result


# ============================================================
# MANAGER — exposes workers AS TOOLS, decides which to call
# ============================================================

MANAGER_WORKERS = {
    "destination_researcher": destination_researcher,
    "activities_researcher": activities_researcher,
    "budget_analyzer": budget_analyzer,
}

MANAGER_WORKER_DECLARATIONS = [
    {
        "type": "function",
        "function": {
            "name": "destination_researcher",
            "description": "Worker agent: researches practical info about a destination (climate, transport, regions, currency). Returns a text summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "City or region name, e.g. 'Kyoto'"},
                    "travel_style": {"type": "string", "description": "e.g. 'cultural tourism', 'adventure', 'relaxation'"}
                },
                "required": ["destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "activities_researcher",
            "description": "Worker agent: researches notable things to do at a destination. Returns a list of recommendations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string"},
                    "interests": {"type": "string", "description": "User's stated interests, e.g. 'history and food'"}
                },
                "required": ["destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "budget_analyzer",
            "description": "Worker agent: analyzes whether a travel budget is realistic and how to allocate it. No tools used — uses general travel cost knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string"},
                    "num_days": {"type": "integer"},
                    "budget_usd": {"type": "integer", "description": "Total budget in US dollars"},
                    "traveler_count": {"type": "integer", "description": "Number of travelers (default 1)"}
                },
                "required": ["destination", "num_days", "budget_usd"]
            }
        }
    },
]


MANAGER_PROMPT = """You are a trip planning manager. You coordinate a team of specialized workers to plan trips.

Your workers (each is a specialized agent — call them like tools):
- destination_researcher: gathers practical info about a place
- activities_researcher: finds things to do
- budget_analyzer: assesses whether a budget is realistic

Your job:
1. Read the user's trip request carefully. Extract destination, duration, budget, interests, etc.
2. Decide which workers to invoke and in what order. You don't have to use all of them — only use what's relevant.
3. Workers run independently — they don't know about each other. Give each worker enough context in its inputs.
4. Synthesize the workers' outputs into a single, well-organized trip plan for the user.

Important coordination notes:
- Workers may take 10-20 seconds each. Don't call workers you don't need.
- If the user didn't specify a budget, skip the budget_analyzer.
- If the user gave specific interests, pass them to activities_researcher.
- When you have enough info, produce the final trip plan in your response (no more tool calls).

Final output should be a friendly, well-structured trip plan that the user can actually use."""


def run_manager(user_request: str, max_iterations: int = 10):
    print(f"\nUSER REQUEST: {user_request}")
    print("\n" + "=" * 60)
    print("MANAGER AGENT")
    print("=" * 60)
    
    messages = [
        {"role": "system", "content": MANAGER_PROMPT},
        {"role": "user", "content": user_request},
    ]
    
    for step in range(max_iterations):
        print(f"\n[Manager Step {step + 1}]")
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=MANAGER_WORKER_DECLARATIONS,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=2000,
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
                worker_name = tc.function.name
                args = json.loads(tc.function.arguments)
                print(f"  Manager delegates to: {worker_name}({args})")
                
                if worker_name in MANAGER_WORKERS:
                    result = MANAGER_WORKERS[worker_name](**args)
                else:
                    result = f"Error: unknown worker '{worker_name}'."
                
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            print("\n" + "=" * 60)
            print("FINAL TRIP PLAN (from manager)")
            print("=" * 60)
            print(message.content)
            print("=" * 60)
            return message.content
    
    print("Manager hit max iterations without producing final plan.")
    return None


if __name__ == "__main__":
    run_manager(
        "What's the best time to visit Reykjavik, Iceland? I just want a quick weather and season overview.")