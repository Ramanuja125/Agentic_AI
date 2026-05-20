"""
Episodic memory for the agent.
Stores lessons learned from past runs, retrieves relevant ones for new tasks.
"""

import json
import os
from pathlib import Path
import numpy as np
from openai import OpenAI

MEMORY_FILE = Path("agent_memory.json")
EMBEDDING_MODEL = "openai/text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.35  # Lessons below this aren't retrieved
TOP_K = 3  # Max lessons to inject per run


def _get_client():
    """Lazy client init so this module is importable without an API key."""
    from dotenv import load_dotenv
    load_dotenv()
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


def embed(text: str) -> list:
    """Get an embedding vector for a single piece of text."""
    client = _get_client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def load_memory() -> list:
    """Load all stored lessons. Returns a list of {lesson, embedding, metadata} dicts."""
    if not MEMORY_FILE.exists():
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def save_memory(lessons: list):
    """Persist the full memory list to disk."""
    with open(MEMORY_FILE, "w") as f:
        json.dump(lessons, f, indent=2)


def cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two embedding vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def retrieve_relevant_lessons(user_goal: str, top_k: int = TOP_K) -> list:
    """Find the most relevant lessons for a given user goal."""
    lessons = load_memory()
    if not lessons:
        return []

    goal_embedding = embed(user_goal)

    scored = []
    for lesson in lessons:
        sim = cosine_similarity(goal_embedding, lesson["embedding"])
        if sim >= SIMILARITY_THRESHOLD:
            scored.append((sim, lesson))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [{"lesson": l["lesson"], "similarity": round(s, 3)} for s, l in scored[:top_k]]


def add_lesson(lesson_text: str, task_description: str):
    """Store a new lesson with its embedding and originating task."""
    lessons = load_memory()
    lesson_embedding = embed(lesson_text)
    lessons.append({
        "lesson": lesson_text,
        "embedding": lesson_embedding,
        "task": task_description,
    })
    save_memory(lessons)
    print(f"[Memory: saved lesson — total stored: {len(lessons)}]")


def extract_lesson_from_run(client, model: str, user_goal: str, conversation_summary: str) -> str:
    """Ask the LLM to distill a one-sentence lesson — only if something notable happened."""
    extraction_prompt = f"""You are reviewing a completed agent run to extract a lesson — but ONLY if there's a real lesson worth saving.

TASK:
{user_goal}

WHAT HAPPENED:
{conversation_summary}

Decide: did anything NOTABLE happen in this run that future runs could learn from?

NOTABLE means at least one of:
- A tool failed in a specific way, and the agent had to work around it
- The agent discovered a non-obvious pattern in how a tool behaves
- The agent encountered a data format issue that took effort to resolve
- The agent made a mistake and had to correct itself
- A specific phrasing or approach worked unusually well after others failed

NOT NOTABLE:
- The run was straightforward and everything just worked
- The lesson would be a generic platitude ("be clear", "use the right tool", "verify outputs")
- The lesson restates the obvious ("tasks involving X require X-related tools")

If notable: write ONE specific, actionable sentence under 30 words. Reference the SPECIFIC tool, error, or pattern. A good lesson example: "wikipedia_get_article rate-limits with HTTP 429 if 3+ requests fire within 1 second; space them out or fetch sequentially."

If NOT notable: respond with exactly NO_LESSON.

Default to NO_LESSON. Only save a lesson if you're confident it's specific and useful.

Output ONLY the lesson sentence OR the text NO_LESSON, nothing else."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": extraction_prompt}],
        temperature=0.0,
        max_tokens=100,
    )
    lesson = response.choices[0].message.content.strip()
    return lesson if lesson != "NO_LESSON" else None