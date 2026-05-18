"""
Stage 3: A ReAct agent with two tools.
Now the agent must DECIDE which tool to use AND COMPOSE them.
"""

from dotenv import load_dotenv
from openai import OpenAI
import os
import time


load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
MODEL = "google/gemini-2.5-flash"


# ---------- Tools ----------

def calculator(expression: str) -> str:
    """Evaluates a math expression like '23 * 47' and returns the result."""
    expression = expression.replace("^", "**")  # forgiveness from Stage 2
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# A small in-memory knowledge base. The "encyclopedia."
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
    """Looks up a fact in the knowledge base. Returns the fact or an error."""
    normalized = query.lower().strip()
    if normalized in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[normalized]
    # Helpful: suggest similar keys if no exact match.
    suggestions = [k for k in KNOWLEDGE_BASE if any(word in k for word in normalized.split())]
    if suggestions:
        return f"Not found. Did you mean one of: {suggestions[:3]}?"
    return f"Not found. No similar entries in the knowledge base."


# ---------- System prompt: now with two tools ----------

SYSTEM_PROMPT = """You are a ReAct agent that solves problems step by step.

You have access to TWO tools:
- calculator(expression): evaluates a Python math expression, e.g. calculator("23 * 47"). Use ** for exponentiation.
- lookup(query): looks up a fact in a knowledge base, e.g. lookup("population of tokyo"). Use lowercase queries.

On every turn, you MUST respond in this exact format:
Thought: <your reasoning about what to do next>
Action: <tool_name>("<argument>")

The only valid tool names are: calculator, lookup, finish

When you have the final answer, use:
Action: finish("<the answer>")

Rules:
- Output EXACTLY one Thought line and EXACTLY one Action line. Nothing else.
- Do not output multiple actions in one turn.
- Do not wrap your output in code blocks or markdown.
- The argument must be in double quotes.
- For lookup queries, try simple lowercase phrases like "population of tokyo" or "year python was created"."""


# ---------- Parser (unchanged) ----------

def parse_action(text: str):
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Action:"):
            action_part = line[len("Action:"):].strip()
            try:
                paren_index = action_part.index("(")
            except ValueError:
                return None, None
            tool_name = action_part[:paren_index]
            argument = action_part[paren_index + 1:-1].strip().strip('"').strip("'")
            return tool_name, argument
    return None, None


# ---------- LLM call (unchanged structurally) ----------

def call_llm(conversation_history: list) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i, entry in enumerate(conversation_history):
        if i == 0:
            messages.append({"role": "user", "content": entry})
        elif entry.startswith("Observation:"):
            messages.append({"role": "user", "content": entry})
        else:
            messages.append({"role": "assistant", "content": entry})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


# ---------- The agent loop (now with a tool registry) ----------

# A dict mapping tool name to the actual function. Cleaner than a chain of ifs.
TOOLS = {
    "calculator": calculator,
    "lookup": lookup,
}


def run_agent(user_goal: str):
    print(f"USER GOAL: {user_goal}\n")
    print("=" * 60)

    conversation = [user_goal]
    max_iterations = 10

    for step in range(max_iterations):
        print(f"\n--- Step {step + 1} ---")

        agent_output = call_llm(conversation)
        print(agent_output)
        conversation.append(agent_output)

        tool_name, argument = parse_action(agent_output)

        if tool_name is None:
            print("No valid action found. Stopping.")
            break

        if tool_name == "finish":
            print(f"\n{'=' * 60}")
            print(f"FINAL ANSWER: {argument}")
            return argument

        # Dispatch via the tool registry.
        if tool_name in TOOLS:
            observation = TOOLS[tool_name](argument)
        else:
            valid = ", ".join(list(TOOLS.keys()) + ["finish"])
            observation = f"Error: unknown tool '{tool_name}'. Valid tools: {valid}"

        print(f"Observation: {observation}")
        conversation.append(f"Observation: {observation}")

    print("Max iterations reached without finishing.")
    return None


if __name__ == "__main__":
    # A task that requires BOTH tools: lookup a fact, then compute with it.
    run_agent(
        "What is the population of Mumbai plus the population of Delhi?"
    )