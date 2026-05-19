"""
Stage 3.5: The same two-tool agent, rebuilt with native tool calling.
Same calculator + lookup tools. Same ReAct loop. Different transport.
"""

from dotenv import load_dotenv
from openai import OpenAI
import os
import json

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "google/gemini-2.5-flash"


# ---------- Tools (unchanged from Stage 3) ----------

def calculator(expression: str) -> str:
    """Evaluates a math expression."""
    expression = expression.replace("^", "**")
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Store values as raw strings without comma formatting.
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

# Words too common to be useful for fuzzy matching.
STOPWORDS = {"of", "the", "in", "a", "an", "is", "was", "are", "to", "for"}


def lookup(query: str) -> str:
    """Looks up a fact in the knowledge base."""
    normalized = query.lower().strip()

    # Exact match — return the value.
    if normalized in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[normalized]

    # Fuzzy match: find entries that share meaningful (non-stopword) words.
    query_words = {w for w in normalized.split() if w not in STOPWORDS and len(w) > 1}
    scored = []
    for key in KNOWLEDGE_BASE:
        key_words = {w for w in key.split() if w not in STOPWORDS}
        overlap = len(query_words & key_words)
        if overlap > 0:
            scored.append((overlap, key))
    scored.sort(reverse=True)  # Best matches first

    if scored:
        top = [k for _, k in scored[:3]]
        return f"Not found. Closest entries in the knowledge base: {top}"

    # No useful matches — give the agent context about what IS available.
    categories = sorted({key.split()[0] + "..." for key in KNOWLEDGE_BASE})
    return (
        f"Not found. No similar entries. "
        f"This knowledge base only contains entries starting with: {categories}"
    )

# ---------- Tool registry: function lookup ----------

TOOLS = {
    "calculator": calculator,
    "lookup": lookup,
}


# ---------- Tool declarations: the JSON Schema for each tool ----------
# This replaces the "here are your tools" section of the old system prompt.

TOOL_DECLARATIONS = [
        {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": (
                "Looks up a fact in a small in-memory knowledge base. "
                "Available facts: populations of Tokyo, Delhi, New York City, London, Paris, Mumbai; "
                "creation years of Python, JavaScript, Java, C++; "
                "heights of Mount Everest, K2, Kangchenjunga (in meters); "
                "speed of light, speed of sound, distance from Earth to Moon and Sun. "
                "Returns the raw numeric value or a 'not found' message with similar entries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Lowercase phrase matching the stored entries. "
                            "Examples: 'population of tokyo', 'year python was created', "
                            "'height of mount everest in meters'."
                        )
                    }
                },
                "required": ["query"]
            }
        }
},
]


# ---------- A much smaller system prompt ----------
# The tools are declared structurally now, so we don't describe them here.

SYSTEM_PROMPT = """You are a helpful agent that solves problems by calling tools.
When a problem requires computation or factual lookup, use the available tools.
Think step by step. When you have the final answer, just respond with it directly — no tool call needed."""


# ---------- The agent loop ----------

def run_agent(user_goal: str):
    print(f"USER GOAL: {user_goal}\n")
    print("=" * 60)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_goal},
    ]

    max_iterations = 10

    for step in range(max_iterations):
        print(f"\n--- Step {step + 1} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_DECLARATIONS,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=500,
        )

        message = response.choices[0].message

        # CASE 1: the LLM wants to call one or more tools.
        if message.tool_calls:
            # Append the assistant's message (with the tool_calls) to history.
            # Convert the SDK object to a plain dict so we can re-send it.
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            # Process each tool call. (Models can request multiple in parallel.)
            for tc in message.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                print(f"Tool call: {name}({args})")

                if name in TOOLS:
                    result = TOOLS[name](**args)
                else:
                    result = f"Error: unknown tool '{name}'. Valid: {list(TOOLS)}"

                print(f"Result: {result}")

                # Append the tool result with the corresponding tool_call_id.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })

        # CASE 2: the LLM returned a final text answer, no tool calls.
        else:
            print(f"FINAL ANSWER: {message.content}")
            print("=" * 60)
            return message.content

    print("Max iterations reached without final answer.")
    return None


if __name__ == "__main__":
    run_agent(
        "What's the population of Berlin, multiplied by 2?" 
    )