"""
Stage 2: A real ReAct agent powered by Gemini.
Same loop as Stage 1 — but the 'agent output' now comes from an LLM
that decides for itself what to think and do next.
"""

from dotenv import load_dotenv
from openai import OpenAI
#from google.genai import types
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
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ---------- The system prompt: this is the LLM's "job description" ----------

SYSTEM_PROMPT = """You are a ReAct agent that solves problems step by step.

You have access to ONE tool:
- calculator(expression): evaluates a math expression, e.g. calculator("23 * 47")

On every turn, you MUST respond in this exact format:
Thought: <your reasoning about what to do next>
Action: <tool_name>("<argument>")

The only valid tool names are: calculator, finish

When you have the final answer, use:
Action: finish("<the answer>")

Rules:
- Output EXACTLY one Thought line and EXACTLY one Action line. Nothing else.
- Do not output multiple actions in one turn.
- Do not wrap your output in code blocks or markdown.
- The argument must be in double quotes."""


# ---------- Parser (unchanged from Stage 1) ----------

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


# ---------- The LLM-powered agent ----------

def call_llm(conversation_history: list) -> str:
    """Send the conversation history to Gemini and return the next agent output."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=conversation_history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,  # We want deterministic, careful reasoning, not creativity.
        ),
    )
    return response.text.strip()
def call_llm(conversation_history: list) -> str:
    """Send the conversation to the LLM and return the next agent output."""
    # OpenRouter (OpenAI-style) wants messages as a list of {role, content} dicts.
    # We rebuild the message list every call: system prompt first, then alternating
    # user/assistant turns reflecting the conversation so far.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # The first item in `conversation` is the user goal. Everything after alternates:
    # agent_output (assistant), observation (user), agent_output (assistant), ...
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
        max_tokens=500,  # Cap output length — we only need 2 lines per turn.
    )
    return response.choices[0].message.content.strip()


def run_agent(user_goal: str):
    print(f"USER GOAL: {user_goal}\n")
    print("=" * 60)

    # The conversation history. This is the agent's MEMORY.
    # We start it with the user's goal.
    conversation = [user_goal]

    max_iterations = 10
    for step in range(max_iterations):
        print(f"\n--- Step {step + 1} ---")

        # Ask the LLM what to do next, given everything that's happened so far.
        agent_output = call_llm(conversation)
        print(agent_output)

        # Append the agent's output to the conversation so it sees it next time.
        conversation.append(agent_output)

        # Parse the action.
        tool_name, argument = parse_action(agent_output)

        if tool_name is None:
            print("No valid action found. Stopping.")
            break

        if tool_name == "finish":
            print(f"\n{'=' * 60}")
            print(f"FINAL ANSWER: {argument}")
            return argument

        # Run the tool.
        if tool_name == "calculator":
            observation = calculator(argument)
        else:
            observation = f"Error: unknown tool '{tool_name}'. Valid tools: calculator, finish"

        print(f"Observation: {observation}")

        # Feed the observation back to the agent for the next turn.
        conversation.append(f"Observation: {observation}")

        # Polite pause to stay under Gemini's free-tier rate limit (10-15 req/min).
        time.sleep(4)

    print("Max iterations reached without finishing.")
    return None


if __name__ == "__main__":
    run_agent("Solve x^2 + 5x + 6 = 0")