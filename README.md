# Agentic AI — Learning Journey

A self-paced course on Agentic AI, built from scratch in plain Python (no frameworks). This README tracks the concepts, code, and experiments completed so far.

---

## Table of Contents

- [Module 1: Foundations](#module-1-foundations)
- [Module 2: Building a ReAct Agent](#module-2-building-a-react-agent)
- [Module 2 (extended theory)](#module-2-extended-theory)
- [Module 3: Expanding the Action Space](#module-3-expanding-the-action-space)
- [Cheatsheets](#cheatsheets)
- [Code Artifacts](#code-artifacts)
- [Key Lessons Learned](#key-lessons-learned)

---

## Module 1: Foundations

### What "agentic" means

Agentic AI isn't a binary — it's a spectrum of autonomy.

| Level | Description | Example |
|-------|-------------|---------|
| 0 | Pure function | A sentiment classifier |
| 1 | Chatbot (state, no goals) | Basic conversational LLM |
| 2 | Tool-augmented LLM (single turn) | LLM that can call a calculator |
| 3 | ReAct agent (own loop) | LLM that reasons, acts, observes, repeats |
| 4 | Planning agent | Decomposes goals into plans, replans on failure |
| 5 | Multi-agent system | Manager delegates to specialized workers |
| 6 | Self-improving agent | Modifies its own prompts/tools/strategies |

Most production systems mix levels.

### The five ingredients of an agent

1. **Goal specification** — what does "done" look like? The fuzzier the goal, the harder the agent's job.
2. **World model / state** — what the agent knows about its current situation.
3. **Action space** — what the agent can do. Closed (enumerated) vs. open (LLM generates any tool call).
4. **Policy** — given state and goal, which action next? In LLM agents, this is a prompt.
5. **Feedback signal** — how the agent knows it's making progress.

### Why LLMs changed agentic AI

Pre-LLM agents had to have **enumerable action spaces**. LLMs broke this — the action space became *anything you can describe in natural language*. Source of both their power and every problem in modern agentic AI.

### The ReAct pattern

```
Thought: <reasoning about what to do next>
Action: <tool_call>
Observation: <result from tool>
... repeat ...
Action: finish(<answer>)
```

The LLM emits text. The runtime parses, executes, observes, loops.

**Key insight:** the agent IS the loop. The LLM just fills in the Thoughts and Actions.

### Why "Thought" before "Action" matters

Forcing the model to verbalize reasoning before committing to an action measurably improves decision quality. It's not decorative — it's a performance technique.

### LLM API statelessness

LLM APIs are **stateless** — every call is independent. Conversation "memory" is an illusion created by the *client*, which stores the conversation locally and re-sends the entire history.

---

## Module 2: Building a ReAct Agent

### The four phases of any LLM call

1. **Load credentials** (via `.env` and `python-dotenv`)
2. **Create a client** (authenticated, reusable connection)
3. **Make a call** (specifying model + prompt)
4. **Extract the result** (pull text from response object)

### Stage 1 — The fake agent (no LLM)

Built a ReAct loop with hardcoded "agent outputs" to isolate the **loop mechanics** from LLM behavior.

**Learned:** the loop, parser, and tool registry are the agent's skeleton. The LLM is just what fills in the decision-making blanks.

### Stage 2 — The real agent (LLM-driven)

Replaced hardcoded outputs with LLM calls (via OpenRouter, OpenAI-compatible SDK).

#### Key concepts

- **System prompt** = LLM's job description. Defines role, tools, output format, termination signal.
- **Conversation history** = manual state management because APIs are stateless.
- **Role-based message format** = `{"role": "system" | "user" | "assistant", "content": "..."}`. Observations are sent as "user" messages.
- **Temperature** = `0.0` for reasoning tasks. Determinism, not creativity.
- **Max tokens** = always cap output. Prevents runaway generations.

#### Observed LLM behaviors

- LLMs sometimes skip the `Thought` line on trivially easy problems despite instructions.
- LLMs treat tools as **advisors**, not oracles — when output contradicts their prior belief, they retry differently.
- Different models have different "personalities": fine-grained vs. coarse-grained planning.

### Stage 3 — Two tools and composition

Added a `lookup` tool alongside calculator. Tasks now require *composing* tools.

#### Key concepts

- **Tool registry pattern** — `TOOLS = {name: function}` dict replaces if/else dispatches.
- **Forgiving tools** — normalize inputs because LLMs phrase things inconsistently.
- **Helpful failure messages** — return suggestions, not just "not found." Agents recover faster.
- **Tool composition as planning** — given multiple tools, agent figures out the order of operations from natural-language goal alone.

#### Observed behaviors

- **Adaptive give-up thresholds** emerge from LLM reasoning, not from code.
- **Creative tool use** — agents may guess a fact and use the tool to verify, working around tool limitations.
- **Silent format fixes** — agents correct format mismatches (comma-stripped numbers) without being told.
- **Plan compaction** — some models bundle multi-step calculations into one tool call.

---

## Module 2 (extended theory)

### Native tool calling vs. text-based ReAct

Two transport layers for the same ReAct pattern:

**Text-based** (Stage 1-3): LLM emits text like `Action: calculator("23 * 47")`. We parse the text ourselves with regex.

**Native tool calling** (Stage 3.5+): LLM emits structured JSON (`{name: "calculator", arguments: {expression: "23 * 47"}}`). API parses for us.

| Aspect | Text-based | Native |
|--------|-----------|--------|
| Parser needed | Yes | No |
| Format failures | Possible | Effectively zero |
| Parallel tool calls | No | Yes |
| Reasoning visibility | Explicit | Hidden |
| System prompt size | Large (must teach format) | Small |
| Final answer style | Constrained (`finish()`) | Natural prose |

**When to use which:**
- Production with mainstream models → Native
- Open-source models without tool training → Text-based
- Learning / debugging → Text-based (for visibility)
- Need parallel tool calls → Native

### ReAct family of patterns

| Pattern | One-line summary |
|---------|------------------|
| **ReAct** | Reason and act one step at a time. Maximum reactivity. |
| **Plan-and-Execute** | Plan all steps upfront, then execute. Replan on failure. |
| **Reflexion** | ReAct + cross-episode memory: agent reflects on past tasks and stores lessons. |
| **Tree of Thoughts** | Generate multiple candidate next steps, score, pick best. Expensive. |
| **ReWOO** | Plan all tool calls upfront, execute in parallel, synthesize. Only works when steps are independent. |

All of these answer the same question: *how much should the agent plan vs. react?*

### When to use a framework

| Situation | Recommendation |
|-----------|----------------|
| Learning | Build from scratch |
| Prototyping a one-off | Build from scratch |
| Production with simple needs | Build from scratch + small utilities (Pydantic, tenacity) |
| Production with complex orchestration | **LangGraph** |
| Role-based delegation | **CrewAI** |
| RAG-heavy research | **LlamaIndex** for retrieval, your own loop for the agent |

The conceptual work transfers everywhere. Frameworks are dialects; concepts are the language.

### Production failure modes

1. **Context window overflow** — conversation grows linearly, hits limits, gets expensive, gets confused ("lost in the middle").
2. **Cost runaways** — stuck loops on premium models burn thousands in hours.
3. **Infinite loops** — `max_iterations` cap protects against; subtle versions (near-identical queries) require smarter detection.
4. **Prompt injection** — external content can contain instructions that override the system prompt. The biggest unaddressed security issue in agentic AI.
5. **Hallucinated tool calls** — LLM invents tools that don't exist. Mitigated by native tool calling.
6. **Tool misuse** — right tool, wrong arguments. Validate inputs, fail loudly.
7. **Silent over-confidence** — plausible-looking wrong answers. The hardest failure mode to detect.
8. **Rate limits / transient failures** — retry with backoff, distinguish retryable from non-retryable errors.

---

## Module 3: Expanding the Action Space

### 3.1 — Native tool calling refactor

Rebuilt Stage 3's two-tool agent using structured tool calls. Same behavior, different transport.

#### What changed

- Tool declarations moved from system prompt into structured `TOOL_DECLARATIONS` (JSON Schema).
- System prompt shrank from ~25 lines to ~3 lines.
- Parser deleted. API hands us structured `tool_calls` objects.
- New conversation role: `"tool"`, linked to its tool call via `tool_call_id`.
- No more `finish()` action — termination is "LLM returns no tool calls."

#### What got better

- **Parallel tool calls**: agent can fire off independent lookups simultaneously.
- **Cleaner final answers**: natural prose instead of `finish("answer")`.
- **No format violations**: API guarantees structure.

#### What got worse (tradeoffs)

- **Reasoning hidden**: no more `Thought:` lines visible to us.
- **Less recovery from tool errors**: parallel calls miss the chance to learn from each other within a turn.

### 3.2 — Tool design principles

**The framing:** tools aren't APIs for code, they're interfaces between two intelligences (the LLM and your code). Design them like documentation for a smart but unfamiliar collaborator with 5 seconds to read.

#### The six principles (detail in [cheatsheet below](#tool-design-cheatsheet))

1. The description is more important than the code
2. Parameter names and descriptions teach format
3. Errors are teaching opportunities
4. Be forgiving about input, strict about output
5. Tools should have non-overlapping purposes
6. Tool count has cognitive cost

#### Lookup tool refactor: results

Same agent, same model, same problems. Only the tool changed (better description, smarter suggestion logic via stopword filtering, raw numeric outputs, category fallback for completely missing entries).

| Task | Before | After |
|------|--------|-------|
| Everest vs K2 | 6 steps then 2-step fail (Stage 3 → 3.5) | 2 steps success |
| Berlin × 2 | 5 steps, vague reasoning | 2 steps, specific reasoning |
| Mumbai + Delhi | Worked, with silent comma-stripping | Worked, no silent reformatting needed |

**The lesson:** the agent didn't get smarter. The tool did. *Tool design is leverage.*

---

## Cheatsheets

### Tool Design Cheatsheet

> Anything the agent has to discover at runtime is a place you should consider documenting at design time.

#### The six principles

**1. The description is more important than the code.**
The LLM never reads your function body. It only reads: name, description, parameter names, parameter descriptions. That's the entire API surface from the LLM's perspective.

- ❌ `"description": "A search tool."`
- ✅ `"description": "Looks up populations of Tokyo, Delhi, NYC, London, Paris, Mumbai; creation years of Python, JS, Java, C++; heights of Everest, K2, Kangchenjunga in meters."`

**2. Parameter names and descriptions teach format.**
Always include a concrete example. Models latch onto examples.

- ❌ `"description": "The expression"`
- ✅ `"description": "A Python math expression, e.g. '23 * 47 + 100'. Use ** for exponentiation."`

**3. Errors are teaching opportunities.**
Error messages are the agent's only signal about what went wrong. Make them actionable.

- ❌ `"Error: not found"`
- ✅ `"Not found. Closest entries: ['population of tokyo', 'population of paris']. This KB only contains: populations, language years, mountain heights, physics constants."`

Helpful means relevant. Junk suggestions are worse than no suggestions.

**4. Be forgiving about input, strict about output.**

Forgiving input: normalize case, strip whitespace, handle plural/singular, handle common rephrasings.

Strict output: consistent format, no presentation cruft, raw values when downstream tools will consume them.

- ❌ Output: `"Result: 13,960,000"` (commas, prefix)
- ✅ Output: `"13960000"` (raw)

**5. Tools should have non-overlapping purposes.**
If two tools could plausibly handle the same request, the LLM has to guess. Sharpen distinctions:

- ❌ `search_facts`, `find_data`, `lookup` (all the same thing in different words)
- ✅ `lookup_population`, `lookup_constant`, `lookup_year` (purposes distinct)

Or: one well-named tool with clear domain.

**6. Tool count has cognitive cost.**
More tools = more tokens spent on tool docs every turn, harder choice for the LLM, slightly-wrong tool selections become more common.

Rule of thumb: past 10-15 tools, be deliberate about each addition. Consider merging, routing, or hiding behind composite tools.

#### The iteration loop in production

When an agent misbehaves, ask first: **is this a tool problem?**

- Agent wastes turns trying multiple phrasings → tool description is missing an example
- Agent picks the wrong tool → descriptions overlap or are vague
- Agent gives up too early → error messages aren't helpful
- Agent silently produces wrong answers → tool output format mismatches downstream consumption

Most of the time, the fix is in the tool, not the agent.

### Cost Control Cheatsheet

| Safeguard | Why |
|-----------|-----|
| `max_iterations` cap on every loop | Defense against infinite loops |
| `max_tokens` cap on every API call | Defense against runaway generations |
| Default to cheap models | Gemini Flash, Haiku 4.5, GPT-4o-mini are 10-30× cheaper than premium |
| Set a hard monthly cap on your API provider account | Last line of defense |
| Check usage dashboard weekly | Catch anomalies before they're catastrophic |

For this course's typical agents (5-15 steps, small context): **$0.001-$0.005 per run** on cheap models. $5 budget covers thousands of runs.

---

## Code Artifacts

| File | Purpose |
|------|---------|
| `test_setup.py` | Verifies LLM API connection (single hello-world call) |
| `stage1_fake_agent.py` | ReAct loop with hardcoded outputs. Isolates the loop mechanism. |
| `stage2_real_agent.py` | LLM-driven ReAct agent with one tool (calculator), text-based. |
| `stage3_two_tools.py` | Two-tool agent (calculator + lookup), text-based, with tool registry. |
| `stage3_5_native_tools.py` | Same two-tool agent, native tool calling. |
| `stage3_5_native_tools_upgraded_tools.py` | Above + refactored `lookup` with better description, smarter suggestions, raw outputs. |

### Setup

```bash
pip install openai python-dotenv requests beautifulsoup4
```

`.env` file:
```
OPENROUTER_API_KEY=sk-or-v1-...
```

---

## Key Lessons Learned

### Conceptual

- **An agent is a loop, not a model.** Intelligence is in the LLM; agency lives in the loop.
- **State management is half the job.** LLMs are amnesiac; you maintain history manually.
- **Policy is a prompt.** Want different behavior? Rewrite the system prompt.
- **Feedback signal quality determines agent quality.** Clean signals → strong agents.
- **Tool design is leverage.** Most agent improvements come from tool refinement, not agent refinement.
- **The agent is only as smart as its dumbest tool.**

### Practical

- Always cap `max_iterations` and `max_tokens`.
- Default to cheap models during development.
- Errors should become Observations, not crashes.
- Tool error messages are teaching messages — design them to guide the agent.
- System prompt and tool registry must stay synchronized.
- Anything the agent discovers at runtime is a candidate for documenting at design time.

### Surprising things observed

- Same loop + same prompt + different LLM = qualitatively different agent behavior.
- LLMs distrust their tools when output contradicts priors — sometimes rightly, sometimes not.
- An agent reverse-engineered a broken calculator (`^` as XOR) and worked around it.
- An agent worked around a forward-only lookup by guessing a fact and verifying.
- Agents calibrate their own persistence based on the informational content of feedback.
- Tool description refactors produced larger agent-behavior improvements than model upgrades would have.
- Native tool calling enables parallel tool calls, sometimes for free, sometimes at the cost of in-turn recovery.

---

## Status

- ✅ Module 1: Foundations
- ✅ Module 2: Building a ReAct Agent (Stages 1–3)
- ✅ Module 2: Extended theory
- ✅ Module 3.1: Native tool calling refactor
- ✅ Module 3.2: Tool design principles
- ⏳ Module 3.3: Real-world tools (web search, fetch, Python exec, file I/O)
- ⏳ Module 3.4: Tool safety and sandboxing
- ⏳ Module 3.5: Structured outputs
- ⏳ Module 3.6: Capstone task
- ⏳ Module 4: Memory, planning, reflection
- ⏳ Module 5: Multi-agent systems
- ⏳ Module 6: Frameworks (LangGraph, CrewAI)