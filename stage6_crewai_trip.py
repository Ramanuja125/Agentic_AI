"""
CrewAI implementation of the trip planner from Module 5b.
Compare to stage5_manager_multi_agent.py — same task, very different code shape.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import requests

load_dotenv()

# CrewAI uses LiteLLM under the hood. Model strings prefix with the provider routing.
llm = LLM(
    model="openrouter/openai/gpt-4o-mini",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


# ============================================================
# TOOLS — same Wikipedia tools as before, but wrapped with @tool
# ============================================================

@tool("Wikipedia Search")
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for article titles matching a query. Returns top 5 results with snippets."""
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
        f"Title: {h['title']}\nSnippet: {h['snippet']}"
        for h in hits
    )


@tool("Wikipedia Get Article")
def wikipedia_get_article(title: str) -> str:
    """Fetch the plain text of a Wikipedia article by title. Truncated to ~3000 chars."""
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
    if len(extract) > 3000:
        extract = extract[:3000] + "..."
    return f"Article: {page.get('title')}\n\n{extract}"


# ============================================================
# AGENTS — role / goal / backstory becomes the system prompt
# ============================================================

destination_researcher = Agent(
    role="Destination Researcher",
    goal="Find key practical information about a travel destination — climate, best season, transportation, currency, and safety basics.",
    backstory=(
        "You're an experienced travel researcher who specializes in distilling the practical "
        "essentials about a destination. You know Wikipedia is your best source. You're efficient — "
        "you don't fetch articles you don't need."
    ),
    tools=[wikipedia_search, wikipedia_get_article],
    llm=llm,
    verbose=True,
)

activities_researcher = Agent(
    role="Activities Researcher",
    goal="Find notable things to do at a destination, matched to the traveler's stated interests.",
    backstory=(
        "You're a travel curator who finds the most interesting things to do at any destination. "
        "You take the traveler's interests seriously — a history lover gets different recommendations "
        "than a foodie. You use Wikipedia to discover specific landmarks, museums, and unique experiences."
    ),
    tools=[wikipedia_search, wikipedia_get_article],
    llm=llm,
    verbose=True,
)

budget_analyzer = Agent(
    role="Travel Budget Analyst",
    goal="Assess whether a travel budget is realistic for a destination and duration, and suggest how to allocate it well.",
    backstory=(
        "You're a frugal-yet-realistic travel cost analyst. You know typical travel costs for major "
        "destinations and you give honest assessments. If a budget is too tight, you say so. If it's "
        "generous, you suggest how to use it well."
    ),
    llm=llm,
    verbose=True,
    # No tools — this agent uses pure LLM reasoning.
)


# ============================================================
# TASKS — what each agent should do
# ============================================================

USER_REQUEST = (
    "I want to plan a 5-day trip to Kyoto, Japan. I'm into history and food, "
    "and my total budget is $2000 USD. Help me figure out what to do and "
    "whether my budget is realistic."
)

destination_task = Task(
    description=(
        f"Research practical information about the destination in this trip request:\n\n{USER_REQUEST}\n\n"
        "Focus on: climate and best time to visit, transportation, and practical considerations "
        "like currency, language, and safety. Be efficient — 2-3 Wikipedia tool calls is usually enough."
    ),
    expected_output="A concise practical-info summary, 150-300 words, in plain text.",
    agent=destination_researcher,
)

activities_task = Task(
    description=(
        f"Find specific, notable things to do at the destination in this trip request:\n\n{USER_REQUEST}\n\n"
        "Pay attention to the traveler's interests and recommend accordingly. "
        "Be efficient with Wikipedia — 2-3 tool calls is usually enough."
    ),
    expected_output="A list of 5-8 specific recommendations with brief explanations, 200-400 words.",
    agent=activities_researcher,
)

budget_task = Task(
    description=(
        f"Analyze whether the budget in this trip request is realistic, and suggest an allocation:\n\n{USER_REQUEST}\n\n"
        "Provide an honest assessment (tight / comfortable / generous), a rough breakdown of how the "
        "budget should be allocated, and 2-3 specific budget tips."
    ),
    expected_output="A budget analysis with assessment, breakdown, and tips. 150-250 words.",
    agent=budget_analyzer,
)


# ============================================================
# CREW — the orchestration
# ============================================================

trip_crew = Crew(
    agents=[destination_researcher, activities_researcher, budget_analyzer],
    tasks=[destination_task, activities_task, budget_task],
    process=Process.sequential,  # Tasks run one after another in order.
    verbose=True,
)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("KICKING OFF CREW")
    print("=" * 60)
    
    result = trip_crew.kickoff()
    
    print("\n" + "=" * 60)
    print("FINAL CREW OUTPUT")
    print("=" * 60)
    print(result)