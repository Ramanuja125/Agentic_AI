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


KNOWLEDGE_BASE = {
    "population of tokyo": "13,960,000",
    "population of delhi": "32,941,000",
    "population of new york city": "8,336,000",
    "population of london": "8,982,000",
    "population of paris": "2,161,000",
    "population of mumbai": "20,411,000",
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


def lookup(query: str) -> str:
    """Looks up a fact in the knowledge base."""
    normalized = query.lower().strip()
    if normalized in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[normalized]
    suggestions = [k for k in KNOWLEDGE_BASE if any(word in k for word in normalized.split())]
    if suggestions:
        return f"Not found. Did you mean one of: {suggestions[:3]}?"
    return f"Not found. No similar entries in the knowledge base."


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
            "name": "calculator",
            "description": "Evaluates a Python math expression and returns the numeric result. Use ** for exponentiation (not ^).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A Python math expression, e.g. '23 * 47 + 100'."
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Looks up a fact in an in-memory knowledge base of populations, language creation years, mountain heights, and a few physics constants. Use simple lowercase phrases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Lowercase query, e.g. 'population of tokyo' or 'year python was created'."
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
        "What is the population of Mumbai plus the population of Delhi?"
    )