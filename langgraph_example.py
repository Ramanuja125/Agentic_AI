"""
A minimal LangGraph agent with two tools (calculator and lookup).
The same agent you've built in plain Python, now expressed as a graph.
"""

import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.runnables.graph import MermaidDrawMethod
import operator

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

# --- 1. The model ---

# Through OpenRouter — same setup you've been using
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)


# --- 2. The tools (using the @tool decorator) ---

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Use ** for exponentiation. Example: '23 * 47 + 100'."""
    try:
        return str(eval(expression.replace("^", "**"), {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


KB = {
    "population of tokyo": "13960000",
    "population of paris": "2161000",
    "height of mount everest in meters": "8849",
}


@tool
def lookup(query: str) -> str:
    """Look up a fact in a small knowledge base. Contains populations of Tokyo, Paris, and height of Everest. Use lowercase."""
    return KB.get(query.lower().strip(), f"Not found. Available keys: {list(KB)}")


tools = [calculator, lookup]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)


# --- 3. The state ---

class State(TypedDict):
    messages: Annotated[list, operator.add]


# --- 4. The nodes ---

def agent_node(state: State):
    """Calls the LLM. The LLM decides whether to call a tool or respond directly."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def tool_node(state: State):
    """Executes each tool call in the most recent AIMessage."""
    last_message = state["messages"][-1]
    results = []
    for tc in last_message.tool_calls:
        result = tools_by_name[tc["name"]].invoke(tc["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}


# --- 5. The routing logic ---

def should_continue(state: State):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


# --- 6. The graph ---

graph = StateGraph(State)
graph.add_node("agent_node", agent_node)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "agent_node")
graph.add_conditional_edges("agent_node", should_continue, ["tool_node", END])
graph.add_edge("tool_node", "agent_node")

agent = graph.compile()


# --- 7. Run it ---

if __name__ == "__main__":
    result = agent.invoke({
        "messages": [
            SystemMessage(content="You are a helpful assistant. Use tools when needed."),
            HumanMessage(content="What's the population of Tokyo divided by the height of Mount Everest in meters?"),
        ]
    })
    
    print("=" * 60)
    print("FINAL TRACE:")
    print("=" * 60)
    for msg in result["messages"]:
        msg.pretty_print()
    print(agent.get_graph().draw_mermaid())