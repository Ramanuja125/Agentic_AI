## this is a agentic AI course work.
### module 1 has the setup - connection to Google Gen AI using LLM api key


# Agentic AI — Learning Journey

A self-paced course on Agentic AI, built from scratch in plain Python (no frameworks). This README tracks the concepts, code, and experiments completed so far.

---

## Table of Contents

- [Module 1: Foundations](#module-1-foundations)
- [Module 2: Building a ReAct Agent](#module-2-building-a-react-agent)
- [Code Artifacts](#code-artifacts)
- [Key Lessons Learned](#key-lessons-learned)

---

## Module 1: Foundations

### What "agentic" means

Agentic AI isn't a binary — it's a spectrum of autonomy. Roughly:

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

Every agent — narrow or general, LLM-based or not — needs these five things:

1. **Goal specification** — what does "done" look like? The fuzzier the goal, the harder the agent's job.
2. **World model / state** — what the agent knows about its current situation. Compression matters: keep the signal, drop the noise.
3. **Action space** — what the agent can do. Closed (enumerated) vs. open (LLM generates any tool call).
4. **Policy** — given state and goal, which action next? In classical agents this is an algorithm; in LLM agents it's a prompt.
5. **Feedback signal** — how the agent knows it's making progress. Clean signals (test results, ROC-AUC scores) → strong agents. Fuzzy signals (subjective quality) → weak agents.

### Why LLMs changed agentic AI

Pre-LLM agents had to have **enumerable action spaces** (a fixed set of moves). LLMs broke this — the action space became *anything you can describe in natural language*. This is the source of both their power and every problem in modern agentic AI: hallucinated actions, infinite loops, plans citing nonexistent functions, etc.

### The ReAct pattern

ReAct ("Reasoning and Acting", Yao et al. 2022) is the foundational pattern behind most modern agents:

```
Thought: <reasoning about what to do next>
Action: <tool_call>
Observation: <result from tool>
Thought: ...
Action: ...
...
Action: finish(<answer>)
```

The LLM emits text in this format. The runtime parses the `Action`, executes the tool, appends the `Observation`, and asks the LLM to continue.

**Key insight:** the agent IS the loop. The LLM just fills in the Thoughts and Actions. The loop, parser, and tool registry sit *outside* the LLM.

### Why "Thought" before "Action" matters

Forcing the model to verbalize reasoning before committing to an action measurably improves decision quality. It's not decorative — it's a performance technique. LLMs are better at acting when they're forced to think out loud first.

### LLM API statelessness

LLM APIs are **stateless** — every call is independent. Conversation "memory" is an illusion created by the *client*, which stores the conversation locally and re-sends the entire history on every new request. This means:

- Agents must maintain their own conversation history
- Context grows linearly with steps
- Eventually context window limits, cost, and "lost-in-the-middle" effects start hurting

---

## Module 2: Building a ReAct Agent

### The four phases of any LLM call

1. **Load credentials** (via `.env` and `python-dotenv`)
2. **Create a client** (authenticated, reusable connection)
3. **Make a call** (specifying model + prompt)
4. **Extract the result** (pull text from response object)

### Stage 1 — The fake agent (no LLM)

Built a ReAct loop with hardcoded "agent outputs" — pre-scripted Thought/Action strings — to isolate the **loop mechanics** from LLM behavior.

**Learned:** the loop, parser, and tool registry are the agent's skeleton. The LLM is just what fills in the decision-making blanks.

### Stage 2 — The real agent (LLM-driven)

Replaced the hardcoded outputs with calls to an LLM (via OpenRouter, OpenAI-compatible SDK). Only the *source* of the agent's output changed — the loop, parser, and tools stayed the same.

#### Key new concepts

- **The system prompt** — the LLM's "job description." It defines the role, lists available tools, enforces the output format, and specifies the termination signal. Whatever you want the agent to do, you bake into the system prompt. *Policy is now a prompt, not an algorithm.*
- **Conversation history (manual state)** — since LLM APIs are stateless, we maintain a list of `(user_goal, agent_output, observation, ...)` and re-send it on every turn. This is the agent's working memory.
- **The role-based message format** — OpenAI-style APIs use `{"role": "system" | "user" | "assistant", "content": "..."}`. Observations are sent as "user" messages because, from the LLM's perspective, observations are external inputs (same category as user messages).
- **Temperature** — `temperature=0.0` for reasoning tasks. We want determinism, not creativity. Lower temperature → better tool-use accuracy.
- **Max tokens** — always cap output. Prevents runaway generations and caps cost per call.

#### Observed LLM behaviors

- LLMs sometimes skip the `Thought` line on trivially easy problems despite system-prompt instructions saying "MUST output Thought."
- LLMs treat tools as **advisors**, not oracles — when a tool's answer disagrees with the LLM's prior belief, the LLM may retry with a different formulation rather than accept the tool's result.
- Different models have different "personalities": some decompose problems fine-grained (cautious), some bundle work coarsely (efficient). Neither is better — it depends on the task.

#### Robustness patterns

- Always have `max_iterations` capped — defense against infinite loops.
- Wrap parser logic in `try/except` — LLMs will emit malformed actions occasionally.
- Return errors as Observations (don't crash) — gives the agent a chance to self-correct.
- Use the Observation channel as a **teaching channel** — when a tool fails, the error message can guide the agent toward the correct usage.

### Stage 3 — Two tools and composition

Added a `lookup` tool (in-memory knowledge base) alongside the calculator. Tasks now require *composing* tools — e.g., "look up Tokyo's population, then calculate."

#### Key new concepts

- **Tool registry pattern** — a `TOOLS = {name: function}` dict replaces a chain of `if/else` dispatches. Adding a tool now requires only (1) writing the function, (2) adding it to the registry, (3) updating the system prompt. The loop code never changes.
- **Forgiving tools** — normalize inputs (lowercase, strip whitespace) because LLMs will phrase things inconsistently.
- **Helpful failure messages** — when a tool fails, return *suggestions* for what to try. Agents recover dramatically faster when Observations contain useful guidance.
- **Tool composition as planning** — given two tools, the agent must figure out the order of operations from a natural-language goal. This is real planning, learned for free from the LLM's general reasoning ability.

#### Observed agent behaviors

- **Adaptive give-up thresholds** — agents stop trying after gathering enough evidence of repeated failure. This is *emergent* from LLM reasoning, not coded. The agent's persistence depends on whether feedback varies (keep trying) or stays the same (give up).
- **Creative tool use** — when one tool can't answer a question directly (e.g., reverse lookup), the agent may use its prior knowledge to *guess* and then use the tool to *verify*. Powerful but risky — works when the guess is right, produces confident hallucinations when wrong.
- **Silent format fixes** — agents often correct mismatches between tool output formats (e.g., comma-formatted numbers from lookup → clean integers for the calculator) without being told. Convenient when it works, brittle when it doesn't.
- **Plan compaction** — some models bundle multi-step calculations into one tool call (e.g., `13960000 * 0.5 / 1000` instead of two separate calculator calls). Efficient but slightly riskier than fine-grained decomposition.

---

## Code Artifacts

| File | Purpose |
|------|---------|
| `test_setup.py` | Verifies the LLM API connection works (single hello-world call). |
| `stage1_fake_agent.py` | ReAct loop with hardcoded agent outputs. Isolates the loop mechanism. |
| `stage2_real_agent.py` | LLM-driven ReAct agent with one tool (calculator). |
| `stage3_two_tools.py` | Two-tool agent (calculator + lookup) with tool registry and tool composition tasks. |

### Setup

```bash
pip install openai python-dotenv
```

`.env` file:
```
OPENROUTER_API_KEY=sk-or-v1-...
```

---

## Key Lessons Learned

### Conceptual

- **An agent is a loop, not a model.** The intelligence is in the LLM, but the *agency* lives in the surrounding loop, parser, and tool registry.
- **State management is half the job.** LLMs are amnesiac; you maintain history manually.
- **Policy is a prompt.** Want different behavior? Rewrite the system prompt. No retraining needed.
- **Feedback signal quality determines agent quality.** Clean numeric signals → strong agents. Fuzzy subjective signals → weak agents. This predicts which domains agents will work well in.
- **LLM agents combine tool use with prior knowledge.** This produces flashes of brilliance and pockets of confident hallucination — sometimes in the same task.

### Practical

- Always cap `max_iterations` and `max_tokens`. Always.
- Use cheap models (Gemini Flash, Haiku, GPT-4o-mini) for development. Save premium models for one-off comparison runs.
- Errors should become Observations, not crashes. The agent can recover from feedback; it can't recover from an exception.
- Tool error messages are teaching messages — design them to guide the agent toward success.
- The system prompt and the tool registry must stay synchronized — every registered tool must be documented in the prompt.

### Surprising things observed

- Same loop + same prompt + different LLM = qualitatively different agent behavior.
- LLMs distrust their tools when tool output contradicts their priors — sometimes they're right to, sometimes not.
- An agent reverse-engineered a broken calculator (`^` parsed as XOR, not exponentiation) and worked around the bug without being told to.
- An agent worked around a forward-only lookup tool by guessing a fact and using the tool to verify it.
- Agents calibrate their own persistence based on the informational content of feedback. No "give up after N tries" rule needed.

### Surprising things observed (continued — from Module 3.3)

- Wikipedia's official API is dramatically more reliable than scraping search engines. Choosing the right *source* matters more than building clever scrapers.
- An agent computed dates by hand (counting leap years, summing months) when its Python tool was broken — getting close to the right answer (90,328 vs ~91,200) but not exactly. Demonstrates both LLM resilience and the unreliability of relying on it.
- Agents silently reformat data from tools when it doesn't fit their preferred output shape: 829.8 became 828; "10.69 times shorter" became "10.69 times taller." The result *looks* confident and is genuinely wrong.
- An agent inverted a question ("Burj Khalifa relative to Everest") to match the answer shape it could produce cleanly. LLMs gravitate toward easy-to-produce framings, even when the user asked for the harder one.
- Loop detection logic, properly designed, stays silent when tools work and only fires when something is wrong. Good safeguards are defense in depth, not constant intervention.
- Sandboxing Python at the language level (custom `__builtins__`, blocked imports) is much harder than it looks and creates more bugs than it prevents. Production sandboxing uses *environmental* isolation (containers, microVMs, hosted execution services like e2b), not language-level restrictions.

---

## Module 3.3 — Real-World Tools

Built an agent with five tools spanning fact lookup, search, content retrieval, and computation:

| Tool | Purpose |
|------|---------|
| `calculator` | Simple arithmetic |
| `lookup` | Small in-memory knowledge base |
| `wikipedia_search` | Find article titles via Wikipedia's official API |
| `wikipedia_get_article` | Fetch full plain-text article content |
| `python_exec` | Execute arbitrary Python (trusted local environment) |

### Key concepts

- **Real-world tools fail constantly.** Network timeouts, scraper blocks, format changes, missing data. Tool design must assume failure as the baseline, not the exception.
- **Timeouts on every external call.** Non-negotiable. Hanging calls take down agents.
- **Output truncation matters.** Real web content is huge; raw HTML is mostly junk. Cleaning + truncating to ~5000 chars protects context windows and costs.
- **Structured APIs beat scraping.** Wikipedia's API returned exactly what we needed, in JSON, with no key, with no anti-bot defenses. The right *source* shortcuts most tool-design work.
- **Trust boundaries beat sandboxes.** For learning/local use, full Python access is fine because *you* run the agent. For production, isolate environmentally (container, microVM) — don't try to neuter the language.
- **Loop detection via observation channel.** Inject `[SYSTEM NOTE: ...]` into tool results when the agent has called the same tool 3+ times with the same arguments. The agent reads it as part of the tool output and (usually) responds to it. Cheaper and more flexible than modifying the system prompt mid-run.

### Failures encountered (and what they taught)

1. **DuckDuckGo scraper blocked** — search engines actively defend against scrapers. Lesson: prefer real APIs over scraping when possible.
2. **`__builtins__` sandbox broken on `datetime.date.today()`** — Python's name resolution needs `__import__` in builtins even when the code doesn't import. Lesson: language-level sandboxing has dragons; use environmental isolation instead.
3. **Subtle infinite loop on partial success** — agent kept retrying the Burj Khalifa search because lookup *was* succeeding on Everest, so it kept getting "progress" reinforcement. Lesson: mixed-success feedback is more dangerous than total failure for triggering give-up logic.
4. **Silent number transcription error** — agent read 829.8 from a Wikipedia article and used 828 in the calculator. Final answer was wrong but confident-sounding. Lesson: LLMs reformat tool outputs silently; downstream consumers can't tell. This is "silent over-confidence" — the hardest failure mode to detect.
5. **Question inversion** — agent asked to compute "X relative to Y" silently switched to "Y relative to X" because the latter produced a cleaner number. Lesson: LLMs gravitate toward easy-to-produce output shapes, even when they conflict with the user's actual question.

---

## Cheatsheets (additions)

### Real-World Tool Design Cheatsheet

- **Always set timeouts** on every external call (10s is a reasonable default).
- **Validate inputs** — never trust the LLM to format URLs, queries, or IDs correctly.
- **Catch network errors distinctly** from generic exceptions — timeouts may be retryable, 404s aren't.
- **Truncate large outputs** before feeding back to the agent. Web pages can be 50KB+; that wrecks context windows.
- **Strip junk before truncation** — for HTML, remove `<script>`, `<style>`, `<nav>`, `<footer>` before extracting text.
- **Tell the agent when content was truncated** so it knows to fetch more if needed.
- **Prefer structured APIs over scraping** when available. JSON beats HTML.
- **Include a User-Agent header** that identifies your agent honestly.
- **Test failure modes explicitly** — what happens when the API is down? When the response is empty? When the URL is malformed? Each path should return a useful error message.

### Loop Detection Cheatsheet

The right pattern: track recent tool calls, intervene through the *observation* channel.

```python
# Track last N tool calls as (name, args_signature) tuples
recent_calls.append((name, json.dumps(args, sort_keys=True)[:200]))

# Intervene if same exact call repeated 3+ times
if recent_calls.count(call_signature) >= 3:
    result += "\n\n[SYSTEM NOTE: You've made this exact call 3+ times. Try a different approach.]"

# Or: intervene if same tool called 4+ times in a row regardless of args
if last_4_tools.count(name) >= 4:
    result += "\n\n[SYSTEM NOTE: This tool isn't yielding results. Try a different tool.]"
```

Why this is better than modifying the system prompt: it activates only when needed, gives the agent specific actionable advice in context, and doesn't bloat the prompt for normal cases.

---

## Status

- ✅ Module 1: Foundations
- ✅ Module 2: Building a ReAct Agent (Stages 1–3)
- ✅ Module 2: Extended theory
- ✅ Module 3.1: Native tool calling refactor
- ✅ Module 3.2: Tool design principles
- ✅ Module 3.3: Real-world tools (Wikipedia, Python exec, loop detection)
- ⏳ Module 3.4: Tool safety and sandboxing
- ⏳ Module 3.5: Structured outputs
- ⏳ Module 3.6: Capstone task
- ⏳ Module 4: Memory, planning, reflection
- ⏳ Module 5: Multi-agent systems
- ⏳ Module 6: Frameworks (LangGraph, CrewAI)