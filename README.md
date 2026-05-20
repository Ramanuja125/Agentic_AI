## This is an entire repo of various Agentic AI projects, along with clear explanation of what I have built and learnt
### Each Module is built for specific use case and has examples as mentioned.
### This contains simplest agents with access to only one tool (Calculator) upto complex multi tool agent access


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
---

## Module 3.4 — Tool Safety and Sandboxing

### The real threat model

Three threats, in descending likelihood:

1. **Agent mistakes** — LLMs hallucinate arguments, pick wrong tools, misread observations. Most "safety incidents" are this.
2. **Prompt injection** — external content (web pages, emails, documents) contains instructions designed to manipulate the agent.
3. **Excessive authority** — the agent has access to tools whose effects exceed what the task actually needed.

The agent is the *vector*, not the attacker. The attacker is whoever wrote the malicious input, or whoever designed the system without the right guardrails.

### Four principles of safe tool design

1. **Distinguish reversible from irreversible actions.** Reversible actions (reads, searches, computations) can be automated freely. Irreversible actions (sends, deletes, posts, charges) must never be fully automated.

2. **Trust boundaries beat technical sandboxes.** Don't try to neuter a dangerous thing — run it in a place where its consequences are bounded. Docker, microVMs, dedicated directories, allowlisted proxies, least-privilege DB users.

3. **Confirmation gates for irreversible actions.** Agent *proposes* the action via one tool; a human *approves* via a separate step; only then does a third tool actually execute.

4. **Treat external content as data, not instructions.** Wrap untrusted content in clear delimiters. Tell the agent in the system prompt that nothing inside those delimiters should be followed as instructions.

### Reversibility is contextual, not absolute

The same tool can be reversible or catastrophic depending on what it's pointed at.

- `write_file("scratch.txt", "hello")` → reversible
- `write_file("/etc/hosts", "...")` → catastrophic
- `write_file("~/.bashrc", "...")` → catastrophic with delayed effect

You can't classify a *tool* — you have to classify the action it would take in context. The tool's job is to refuse contexts that make it dangerous.

### The lethal trifecta

A single agent with all three of these is a data-exfiltration pipeline:

1. Access to private/sensitive data
2. Ability to communicate externally (network, email, file writes outside the sandbox)
3. Exposure to untrusted content (web pages, user documents, third-party API responses)

Prompt injection in (3) can exfiltrate (1) through (2). Split agents to break the trifecta — give each agent at most two of these three capabilities.

### What production systems actually use

| Need | Production solution |
|------|---------------------|
| Code execution | Docker container, Firecracker microVM, e2b.dev, Pyodide (WASM) |
| File operations | Dedicated working directory with path-traversal protection |
| Network operations | Outbound proxy with domain allowlist |
| Database access | Least-privilege DB user with restricted schema access |
| Irreversible APIs (email, payment) | Confirmation gate (prepare → review → confirm) |

### Confirmation gate pattern

```python
pending_actions = {}

def prepare_email(to, subject, body):
    action_id = generate_id()
    pending_actions[action_id] = {"type": "email", "to": to, ...}
    return f"Email prepared (NOT sent). Action ID: {action_id}\n  To: {to}\n  ..."

def confirm_action(action_id, user_approved):
    # Called by the loop in response to user approval, NOT exposed to the agent.
    if not user_approved:
        del pending_actions[action_id]
        return "Cancelled."
    # Only here does the irreversible thing actually happen.
    actually_send(...)
```

The agent only has access to `prepare_email`. The execution is gated outside the agent's tool surface.

### Cheatsheet: questions to ask for every new tool

1. **Reversible or irreversible?** (And in what contexts could it become irreversible?)
2. **What's the worst the agent could do by mistake?**
3. **What's the worst an attacker could do via prompt injection?**
4. **What's the trust boundary?** (Container? Directory? Proxy? DB permissions?)
5. **Does this tool combined with another tool create the lethal trifecta?**
6. **Does this need a confirmation gate?**
7. **What's the blast radius if it goes wrong?**

If you can't answer #4 confidently, you don't have a safe tool — you have a hopeful tool.

---

## Module 3.5 — Structured Outputs

Added structured JSON output to the agent's final answer using OpenAI-compatible `response_format` with JSON Schema enforcement.

### Why structured outputs

- Real agents almost never produce final outputs for human consumption only — they feed dashboards, databases, downstream services.
- Free-form prose requires parsing; parsing is unreliable; agents word things inconsistently.
- Structured outputs guarantee field names, types, and constraints (enums, required fields).

### The three approaches

1. **Ask in the prompt.** ~95% reliable. Models sometimes wrap in markdown, add preamble, get fields wrong.
2. **JSON mode.** Format guaranteed; content (fields, types) still free-form.
3. **Structured outputs with schema.** Both format and content guaranteed. The production answer.

We used #3, passing a JSON Schema via `response_format`.

### The schema we used

```json
{
  "answer": "string",
  "key_facts": ["string"],
  "sources_used": ["calculator" | "lookup" | "wikipedia_search" | "wikipedia_get_article" | "python_exec" | "prior_knowledge"],
  "confidence": "high" | "medium" | "low",
  "limitations": "string"
}
```

Enums on `sources_used` and `confidence` are critical — without them, models invent vocabulary ("pretty sure", "the calculator tool").

### Surprising things observed

- **Structured outputs guarantee format, not meaning.** Agents fill in fields per the schema, but their *interpretation* of what each field is *for* often differs from the designer's. The `limitations` field was treated as "external/situational caveats" rather than "internal/epistemic uncertainty" most of the time.
- **Agents propagate uncertainty when it's articulated; they don't originate it.** Berlin's stale 2019 figure didn't get flagged. Burj Khalifa's two different heights didn't get flagged. But Atlantis being mythical and "sound through vacuum" being impossible *were* flagged — because both were structurally obvious in the input.
- **`confidence` is almost always "high".** This is real model miscalibration. To get useful confidence signals you need either (a) explicit prompting that defines what each level *means*, (b) a verifier LLM that re-scores confidence, or (c) evals that measure calibration across many runs.
- **Prior knowledge gets used liberally.** Models will skip the lookup tool on facts they're confident about (capital of Mongolia), even with prior_knowledge being structurally undesirable. To force verification, the system prompt has to explicitly say "you MUST verify all factual claims via tools."

---

## Module 3.6 — Capstone

A single multi-step task exercising the whole stack: lookup, search, fetch, computation, conceptual reasoning, and structured output.

**Task:** Compare light vs sound travel time from Moon to Earth, compute the ratio, and explain why "lightning before thunder" matters.

### Behaviors observed

- **Aggressive parallel lookups.** Three independent lookups fired simultaneously in step 1. When all failed (close but not exact matches), the agent re-queried using the suggestion engine's hints, also in parallel. Total cost: two API turns, six tool calls. This is what well-tooled native-tool-calling agents look like in production.
- **Unit-aware variable naming.** The agent named variables `distance_km` and `distance_m` in its Python code, and converted explicitly between them. This is real-engineer-quality discipline, picked up from training on code.
- **Limitations field filled correctly.** First time we saw this. Worked because the question explicitly contained an assumption ("which it can't, but assume it can"). The agent propagated the caveat. Same agent doesn't originate caveats on its own.
- **Loop detection fired incorrectly.** SYSTEM NOTE warned about 4 lookups in a row, but those lookups were successfully recovering from misnamed queries — actually progressing, not stuck. Agent correctly ignored the false alarm and continued. Lesson: loop detection should distinguish "many calls" from "many *failed* calls".

### The architecture's ceiling (so far)

What this 100-line agent + 5-tool registry CAN do reliably:
- Multi-step research with 3-8 steps
- Numerical computation with unit awareness
- Conceptual reasoning + factual lookup combined
- Producing structured output for downstream consumption

What it CANNOT do (deferred to later modules):
- Learn between runs (Module 4: memory/reflection)
- Plan 20+ step tasks coherently (Module 4: planning)
- Delegate to specialists (Module 5: multi-agent)
- Production-grade observability and orchestration (Module 6: frameworks)

---

---

## Module 4.1 — What "Memory" Actually Means in Agentic AI

The word "memory" gets used for at least four different things in the field. They have different mechanisms, costs, and purposes.

### The four types of memory

| Type | What it stores | Where it lives | Lifetime |
|------|---------------|---------------|----------|
| **Working** | Current conversation context | `messages` list, in memory | Dies when run ends |
| **Episodic** | Past runs and their outcomes | Database / file | Persistent across runs |
| **Semantic** | Structured general knowledge | KB / vector DB / docs | Persistent, usually read-only |
| **Procedural** | How-to skills and patterns | System prompt / fine-tuning | Persistent, hard to update dynamically |

### What each type solves

- **Working memory** — lets the agent reason over what's happened in the current task.
- **Episodic memory** — lets the agent learn over time without retraining. ("Have I solved this before? What worked?")
- **Semantic memory** — gives the agent access to specific knowledge the LLM doesn't have or shouldn't memorize (private data, current data, niche expertise). RAG patterns live here.
- **Procedural memory** — encodes how the agent should behave. The system prompt is the simplest form; production systems sometimes dynamically update this.

### What we already have vs. what's missing

- ✅ Working memory (the `messages` list)
- ⚠️ Partial semantic memory (the `KNOWLEDGE_BASE` in `lookup`, but tiny)
- ❌ Episodic memory (no persistence across runs — agent re-derives everything)
- ✅ Static procedural memory (the system prompt)

### Why the distinction matters

- "Give the agent a memory" is ambiguous. RAG, Reflexion, context summarization, and skill libraries are all "memory" but solve different problems.
- Each type has different cost/complexity tradeoffs. Working memory is cheap and fragile; episodic memory is moderately expensive but transformative; semantic memory needs real retrieval infrastructure.

---

## Module 4.2 — Working Memory and Context Compression

The problem: the `messages` list grows on every turn. For long-running agents this becomes a context-window issue, a cost issue, and a "lost in the middle" reasoning issue.

### The solution: periodically summarize old turns

Strategy: keep the system prompt and original user goal verbatim. Keep the last N "turn groups" verbatim. Replace everything in between with a single LLM-generated summary.

### The three structural rules

1. **Always keep the system prompt** (agent identity).
2. **Always keep the original user goal** (task being solved).
3. **Always cut on a turn boundary** — never split an assistant's tool_calls from their results. The API enforces tool_call_id linking, so breaking turn groups produces 400 errors.

A "turn group" = one assistant message + all tool result messages before the next assistant message.

### Implementation

- Token-based trigger: if conversation exceeds threshold (e.g., 8000 tokens), compress.
- Cutoff = index of the Nth-from-last assistant message.
- Summarize indices [2 : cutoff], keep [cutoff : end].
- Summary inserted as a `"user"` role message wrapped in `[SUMMARY OF EARLIER STEPS]` markers.
- Compression uses an LLM call itself — typically the same model, but cheaper models work too.

### What good summaries preserve

- Original user goal anchor
- Specific facts (with exact numbers/values)
- What tools were tried and what worked/didn't
- Decisions and constraints discovered

### What summaries drop

- Verbose intermediate reasoning
- Failed exploration paths in full detail
- Exact tool_call syntax and IDs

### Surprising things observed

- **Compression makes long agents possible but doesn't make them better.** It solves cost and context-overflow problems, not reasoning problems.
- **Compression is lossy by definition.** The summary captures less than the original — the skill is summarizing the *right* things.
- **Compression can cause redundant work.** When the agent's actual findings (e.g., article text it just fetched) get summarized away, the agent may re-fetch them because the summary doesn't fully replace the original content in its mental model.
- **`temperature=0.0` is not actually deterministic.** Same task, same inputs, same model occasionally produces different paths — floating-point non-determinism on GPUs, batching effects, backend variance. Production agents must assume non-determinism even at temperature 0.
- **The Gemini-via-OpenRouter combo doesn't support `tools` + `response_format` (JSON schema) simultaneously.** Switched to `openai/gpt-4o-mini`, which does. Worth knowing: feature compatibility varies by provider, even through unified APIs.
- **Aggressive compression settings cause jarring context drops.** With `keep_last_n_turns=2`, compression replaces 5-8 turn groups at once, dropping context by 75% in one step. Production settings would compress less aggressively to avoid disrupting the agent's flow.

### The deep architectural lesson

When facts live only inside the conversation, compression blurs them. A better architecture has facts living *outside* the conversation — in a scratchpad, structured intermediate output, or persistent memory. Compression then doesn't lose them.

This motivates Module 4.3.

---

## Cheatsheets (additions)

### Context Compression Cheatsheet

**Trigger:** token threshold (e.g., 8000 tokens). Adaptive — fires when needed, not at fixed intervals.

**Cutoff:** always at an assistant-message index. Slicing in the middle of a turn group breaks tool_call_id linking → 400 error from API.

**Slice pattern:**
```
compressed = (
    messages[:2]                          # system + original user goal
    + [summary_message]                   # one LLM-generated summary
    + messages[cutoff_index:]             # last N turn groups intact
)
```

**Summary prompt structure:**
- Include original user goal (anchors relevance)
- Format transcript human-readably
- Explicit instructions on what to preserve (exact numbers) and drop (verbose reasoning)
- Cap with `max_tokens` (e.g., 400) to keep summary terse

**Watch for:** the agent re-doing work that got summarized away. If you see redundant tool calls after compression, your summary is dropping things the agent needs.

---

## Code Artifacts (additions)

| File | Purpose |
|------|---------|
| `stage4_2_compressed.py` | Same agent as 3.5 + working-memory compression triggered by token threshold |

---

## Module 4.3 — Episodic Memory

Built a memory system that stores lessons learned from past runs and retrieves relevant ones for new tasks.

### Architecture

- **Storage:** JSON file (`agent_memory.json`) — each entry has `lesson`, `embedding`, `task`.
- **Embedding model:** `openai/text-embedding-3-small` (1536 dimensions, ~$0.02/M tokens).
- **Similarity:** cosine similarity via NumPy.
- **Retrieval:** embed the user goal, score against all stored lessons, return top-K above threshold.
- **Threshold:** 0.35 default; tuned empirically based on observed similarity scores.

### The agent loop with memory

```
BEFORE the loop:
  - Embed user goal
  - Retrieve top-K relevant lessons (similarity >= threshold)
  - Inject into system prompt with "MIGHT be relevant — use judgment" framing

DURING the loop:
  - Track tool calls and results for later lesson extraction

AFTER the loop:
  - Pass run summary to an LLM "lesson extractor"
  - Extractor returns either a specific actionable lesson OR `NO_LESSON`
  - If a lesson is returned, embed and store it
```

### Two persistent problems and how we addressed them

**Problem 1: Lesson extractors invent platitudes.**
Default prompts produce lessons like "ensure outputs are clear" — generic, unactionable, useless. Memory fills with noise.

*Solution:* explicit positive examples (a real specific lesson), explicit negative examples (the platitudes to avoid), and a strong default toward `NO_LESSON`. After this fix, multiple successful runs correctly produced no lessons.

**Problem 2: Relevant-looking lessons may not embed close enough to retrieve.**
A lesson about "researching historical figures" had only 0.273 similarity to a goal about a specific historical figure. Below threshold → not retrieved.

*Solution:* phrase lessons in vocabulary similar to expected future queries. Reworded the same lesson with overlapping terms ("Roman emperor", "Wikipedia") and similarity jumped to 0.539.

### Surprising things observed

- **Memory works mechanically but has modest impact per single run.** Embedding, storage, retrieval, threshold filtering all worked correctly. But measuring the effect of one lesson on one run is nearly impossible — non-determinism dominates.
- **The "use judgment" framing in lesson injection is doing real work.** Seeded a deliberately misleading lesson (search for nonexistent "Disambiguation Index"). It was retrieved. The agent ignored it completely and used its own judgment.
- **This implies real memory systems give statistical, not absolute, wins.** A bad lesson won't break behavior. A good lesson won't dictate behavior. Both are weighted inputs, not commands. Production gains are 5-15% on aggregate metrics, not dramatic per-run transformations.
- **Lesson phrasing has to match expected future task phrasing for retrieval to work.** This is the central craft of writing lessons for production memory systems: write them in the vocabulary the agent will use when describing related future tasks.
- **Successful, uneventful runs produce no learnable lessons.** Most of our well-architected tasks just succeed cleanly. Memory systems benefit most when paired with tasks that *do* fail in distinctive ways — which is most production work, but not our toy capstone tasks.

### The bootstrap problem

Cold-start memory is useless. Production systems often seed lessons manually based on engineer knowledge before deploying. The seed gets *added to* by organic learning over time. Without a bootstrap, memory takes hundreds of runs to become valuable.

We demonstrated this directly: a manually-seeded lesson about Wikipedia research patterns was correctly retrieved on related future tasks.

### Memory injection design principles

| Choice | Effect |
|--------|--------|
| Framing as "MIGHT be relevant — use judgment" | Agent treats lessons as suggestions; ignores bad ones |
| Framing as "MANDATORY rules from past runs" | Agent follows rigidly; one bad lesson breaks everything |
| Lessons phrased like future queries | Higher retrieval similarity, more often surfaced |
| Lessons phrased abstractly | Lower retrieval similarity; rarely surfaced |
| Include exact tool names and error patterns | Specific lessons retrieve sharply when conditions recur |
| Include generic advice | Diffuse retrieval, low signal |

### Cheatsheet: writing lessons that work

1. **Be specific.** Mention exact tool names, error patterns, numeric thresholds. "wikipedia_get_article returns HTTP 429 after 3+ rapid calls" beats "be careful with Wikipedia."
2. **Use task-vocabulary.** Write lessons in the words the agent would use to describe a related future task. "When researching a Roman emperor..." beats "for historical figure research..."
3. **Default to NO_LESSON.** Most runs aren't notable. Save lessons only when something *specific* happened.
4. **Lessons are weighted inputs, not commands.** Frame injection with "use judgment" to keep the agent's own reasoning in the loop.
5. **Bootstrap manually.** Don't wait for organic learning. Seed lessons from your own observations of past failures.

---

## Code Artifacts (additions)

| File | Purpose |
|------|---------|
| `memory.py` | Embedding, storage, retrieval, and lesson extraction |
| `stage4_3_memory.py` | Agent loop with memory retrieval before run and lesson saving after |
| `agent_memory.json` | Persistent storage of lessons + their embedding vectors |

---

## Module 4.4 – 4.6 — Planning, Reflection, Capstone (theory only)

Skipped implementation in favor of conceptual coverage. Key takeaways:

### Planning patterns

**Problem ReAct doesn't solve:** goal drift on long tasks — the agent's attention gets dominated by recent context and loses sight of the original goal.

**Plan-and-Execute:**
1. Planner LLM produces an explicit numbered plan
2. Executor runs each step (often as a small ReAct sub-agent)
3. Replan when reality contradicts the plan

**When it helps:** structured, predictable tasks (research, comparisons, multi-source synthesis).

**When it hurts:** highly exploratory tasks where the right next step depends on what you find.

**Why it matters:** plans are valuable because they're explicit. You (or another system) can inspect, critique, and modify them. ReAct's implicit plan lives in the LLM's head only.

### Reflection patterns

**Problem:** the agent produces confidently wrong answers with no native way to catch itself. Calibration failure.

**Self-critique:** after the agent produces an answer, a separate LLM call critiques it. If issues found, agent revises. Repeat until satisfied (capped).

**Why it sometimes works:** LLMs are often better at recognizing problems in existing text than at avoiding them while generating.

**Why it sometimes fails:** if the critic shares the generator's blind spots, neither catches the issue.

**Reflexion (named pattern):** reflection + episodic memory. After each run, generate a written reflection ("what went well/badly"), store as a lesson, retrieve for future related runs. This is what your `extract_lesson_from_run` is already doing.

**Verifier patterns:** when possible, use deterministic verification (run tests, re-derive math, search for confirmation) instead of open-ended critique. Tighter signal, drives retry loops better. This is why coding agents outperform research agents — they have verifiers built into their environment.

**Best place to deploy reflection:** at decision boundaries, especially before delivering output to the user. One extra LLM call, catches a meaningful fraction of bad outputs.

### How they compose

The three patterns are complementary layers of robustness:
- **Planning** = structural safety (won't wander)
- **Memory** = experiential safety (won't repeat known mistakes)
- **Reflection** = output safety (won't ship confidently wrong)

Each costs additional LLM calls. Each addresses a different failure mode.

### Production naming

Same patterns appear in literature under different names:
- "Agentic workflows" = planning
- "Reflexion" = memory + reflection
- "Self-refine" = iterative reflection
- "Constitutional AI" = reflection with a written rubric

The names matter less than the structure.

### Reflection: built and tested

Built a reflection layer on top of `stage3_5_structured.py` as `stage4_5_reflection.py`. After the agent produces a proposed structured answer, a separate LLM critic reviews it. Critic returns APPROVED or REVISE with specific critique. If REVISE, agent gets a capped (5-step) revision loop to address the critique.

#### Test 1: Burj Khalifa height vs. Mount Everest

- **Original answer:** "Burj Khalifa is approximately 0.0938 times the height of Mount Everest" (confusing phrasing — 0.0938 doesn't intuitively read as "fraction of height")
- **Critic caught:** the semantic mismatch between calculation direction and the question's phrasing
- **Revised answer:** "Mount Everest is approximately 10.66 times taller than the Burj Khalifa" + non-empty `limitations` field acknowledging the interpretive choice
- **Cost:** +2 LLM calls (1 critic + 1 revision step)
- **Verdict:** Clean win. Reflection caught a real issue the generator missed.

#### Test 2: Berlin population density

- **Original answer:** "Berlin's population is 4 million... density ~4485 per sq km"
- **Critic caught:** the 4 million figure is rounded; precise number is ~3.77 million
- **Revision loop:** Agent re-fetched articles, found Demographics of Berlin's precise 3,769,495 figure, computed correct density of 4227.3
- **Outcome:** Revision loop hit the 5-step cap *before* producing the final answer. System fell back to original (wrong) answer.
- **Verdict:** Mixed. Critic was right, revision work was correct, but budget exhausted just shy of completion.

### Lessons learned about reflection

1. **It does catch real mistakes.** The Burj Khalifa phrasing error would have shipped without reflection.
2. **It also catches debatable issues.** The 4-million Berlin figure was defensibly rough; the critic still flagged it.
3. **Cost roughly triples.** ~4-step task became ~11 calls with reflection + revision.
4. **Iteration caps interact badly with revision.** The agent might need real verification work; a 5-step cap can run out.
5. **Critic calibration is a craft.** Too-soft critic approves bad answers; too-strict critic causes expensive revisions for marginal improvements.
6. **Reflection works statistically, not per-run.** Single runs can win or lose; aggregate improvement is what matters.

### Where reflection helps most

- Tasks where output quality matters more than latency or cost
- Final user-facing outputs (reports, customer responses)
- Outputs that will feed downstream systems where errors propagate

### Where reflection helps least

- Exploratory work where partial answers are fine
- High-volume internal tasks where 3x cost is prohibitive
- Tasks where the critic's likely blind spots match the generator's (e.g., both miss the same factual error)
---

## Status

- ✅ Module 1: Foundations
- ✅ Module 2: Building a ReAct Agent
- ✅ Module 3: Expanding the Action Space
- ✅ Module 4: Memory, planning, reflection (4.4–4.6 covered as theory)
- ⏳ Module 5: Multi-agent systems
- ⏳ Module 6: Frameworks