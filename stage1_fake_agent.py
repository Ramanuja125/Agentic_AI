"""
Stage 1: A fake ReAct agent.
No LLM. The 'agent' is just a list of pre-scripted Thought/Action turns.
The point is to see the LOOP clearly before adding the LLM.
"""

# Our tiny "tool": a calculator.
def calculator(expression: str) -> str:
    """Evaluates a math expression like '23 * 47' and returns the result as a string."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# Pre-scripted agent responses. In Stage 2, the LLM will generate these.
# Each entry is what the "agent" would emit on that turn.
fake_agent_responses = [
    """Thought: The user wants to know what 23 * 47 + 100 is. I should use the calculator.
Action: calculator("23 * 47")""",

    """Thought: 23 * 47 is 1081. Now I need to add 100 to it.
Action: calculator("1081 + 100")""",

    """Thought: The final answer is 1181.
Action: finish("1181")""",
]


def parse_action(text: str):
    """
    Extract the Action line from the agent's output.
    Returns (tool_name, argument) or (None, None) if no action found.
    """
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Action:"):
            # Example: 'Action: calculator("23 * 47")'
            action_part = line[len("Action:"):].strip()
            # Split into tool name and argument
            paren_index = action_part.index("(")
            tool_name = action_part[:paren_index]
            # The argument is everything inside the parens, stripped of quotes
            argument = action_part[paren_index + 1:-1].strip().strip('"').strip("'")
            return tool_name, argument
    return None, None


def run_agent(user_goal: str):
    """The ReAct loop. This is the heart of every agent we'll build."""
    print(f"USER GOAL: {user_goal}\n")
    print("=" * 60)

    max_iterations = 10
    for step in range(max_iterations):
        print(f"\n--- Step {step + 1} ---")

        # In Stage 1, "agent output" is hardcoded. In Stage 2, it'll come from the LLM.
        agent_output = fake_agent_responses[step]
        print(agent_output)

        # Parse what the agent decided to do.
        tool_name, argument = parse_action(agent_output)

        if tool_name is None:
            print("No action found. Stopping.")
            break

        if tool_name == "finish":
            print(f"\n{'=' * 60}")
            print(f"FINAL ANSWER: {argument}")
            return argument

        # Run the tool.
        if tool_name == "calculator":
            observation = calculator(argument)
        else:
            observation = f"Error: unknown tool '{tool_name}'"

        print(f"Observation: {observation}")

    print("Max iterations reached.")
    return None


if __name__ == "__main__":
    run_agent("What is 23 times 47, plus 100?")