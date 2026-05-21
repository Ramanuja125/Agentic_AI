"""
Agentic AI — Learning Journey
A Streamlit UI for navigating the coursework, modules, and code artifacts.

Run with:
    streamlit run app.py

Place this file next to your Python files (stage1_fake_agent.py, etc.)
so the "Open file" links resolve correctly.
"""

import os
from pathlib import Path
import streamlit as st

# ---------- Page config ----------
st.set_page_config(
    page_title="Agentic AI — Learning Journey",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .pill-done { background: #dcfce7; color: #166534; }
    .pill-module { background: #e0e7ff; color: #3730a3; }
    .pill-stage { background: #fef3c7; color: #92400e; }
    .card {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        background: #fafafa;
        margin-bottom: 0.8rem;
    }
    .card-title {
        font-weight: 700;
        margin-bottom: 0.3rem;
        font-size: 1.05rem;
    }
    .file-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
    .small-muted {
        color: #6b7280;
        font-size: 0.85rem;
    }
    code {
        background: #f3f4f6;
        padding: 1px 5px;
        border-radius: 4px;
        font-size: 0.85em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Data: code artifacts ----------
# Each entry: (filename, module, purpose)
CODE_ARTIFACTS = [
    ("test_setup.py", "Setup", "Verifies LLM API connection works (single hello-world call)."),
    ("stage1_fake_agent.py", "Module 2.1", "ReAct loop with hardcoded outputs (no LLM). Isolates loop mechanics."),
    ("stage2_real_agent.py", "Module 2.2", "First LLM-driven ReAct agent with one tool (calculator)."),
    ("stage3_two_tools.py", "Module 2.3", "Two-tool agent (calculator + lookup) with tool registry."),
    ("stage3_5_native_tools.py", "Module 3.1", "Same agent rebuilt with native tool calling."),
    ("stage3_5_native_tools_upgraded_tools.py", "Module 3.2", "Improved lookup tool with stopword filtering & category fallback."),
    ("stage3_3_real_tools.py", "Module 3.3", "Five tools: Wikipedia search/fetch, Python exec, calculator, lookup."),
    ("stage3_5_structured.py", "Module 3.5", "Structured JSON output with schema enforcement."),
    ("stage3_6_structured_complex_ip.py", "Module 3.6", "Capstone task variant — Moon/Earth physics problem."),
    ("stage4_2_compressed.py", "Module 4.2", "Working-memory compression via token-threshold summarization."),
    ("memory.py", "Module 4.3", "Embedding, storage, retrieval, lesson extraction."),
    ("stage4_3_memory.py", "Module 4.3", "Episodic memory: retrieve before run, save after."),
    ("agent_memory.json", "Module 4.3", "Persistent storage of lessons with embeddings."),
    ("stage4_5_reflection.py", "Module 4.5", "Critic LLM + revision loop before final answer."),
    ("stage5_multiagent.py", "Module 5a", "Pipeline pattern: researcher → writer."),
    ("stage5_manager_multi_agent.py", "Module 5b", "Manager/worker pattern: trip planner with 3 specialists."),
    ("langgraph_example.py", "Module 6", "Minimal LangGraph implementation of the core agent."),
]


# ---------- Helpers ----------
def file_link(filename: str):
    """Render a file row with an Open button if the file exists locally."""
    path = Path(filename)
    exists = path.exists()
    col1, col2, col3 = st.columns([4, 2, 2])
    with col1:
        st.markdown(f"**`{filename}`**")
    with col2:
        if exists:
            st.markdown('<span class="pill pill-done">Found</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="small-muted">Not in current directory</span>', unsafe_allow_html=True)
    with col3:
        if exists:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            st.download_button(
                "Download",
                content,
                file_name=filename,
                key=f"dl_{filename}",
                use_container_width=True,
            )


def view_file_inline(filename: str):
    """Show file content in an expander if it exists in the working directory."""
    path = Path(filename)
    if path.exists():
        with st.expander(f"📄 View `{filename}`"):
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                lang = "python" if filename.endswith(".py") else "json" if filename.endswith(".json") else "text"
                st.code(content, language=lang)
            except Exception as e:
                st.error(f"Couldn't read file: {e}")
    else:
        st.caption(f"`{filename}` is not in the current directory — drop it next to `app.py` to view it here.")


# ---------- Sidebar navigation ----------
st.sidebar.title("🤖 Agentic AI")
st.sidebar.caption("Learning Journey")

PAGES = [
    "🏠 Home",
    "📘 How to read this",
    "1️⃣ Module 1: Foundations",
    "2️⃣ Module 2: ReAct Agent",
    "3️⃣ Module 3: Real-World Tools",
    "4️⃣ Module 4: Memory, Planning, Reflection",
    "5️⃣ Module 5: Multi-Agent Systems",
    "6️⃣ Module 6: Frameworks",
    "📂 Code Artifacts",
    "💡 Key Lessons",
    "📋 Cheatsheets & References",
    "🎓 Final Reflection",
]

page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("### Status")
st.sidebar.markdown(
    """
    ✅ Module 1: Foundations
    ✅ Module 2: ReAct Agent
    ✅ Module 3: Real-World Tools
    ✅ Module 4: Memory, Planning, Reflection
    ✅ Module 5: Multi-agent
    ✅ Module 6: Frameworks
    """
)
st.sidebar.success("🎓 Course complete")


# ============================================================
# PAGE: HOME
# ============================================================
if page == "🏠 Home":
    st.markdown('<div class="main-title">🤖 Agentic AI — Learning Journey</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">A self-paced course on Agentic AI, built from scratch in plain Python (no frameworks).</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        This repository documents an end-to-end build of Agentic AI systems — starting from a hardcoded
        ReAct loop with zero LLMs, all the way to multi-agent orchestration and frameworks like LangGraph.

        Every concept here was implemented from scratch in plain Python **before** any framework was introduced.
        Module 6 (Frameworks) sits at the end precisely because frameworks make most sense once you've built
        the patterns by hand.
        """
    )

    st.markdown("### What's inside")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Modules completed", "6 / 6")
    with col2:
        st.metric("Code artifacts", f"{len(CODE_ARTIFACTS)}")
    with col3:
        st.metric("Tools built", "5")

    st.markdown("### Course arc at a glance")
    st.markdown(
        """
        | Module | Theme | Key build |
        |--------|-------|-----------|
        | **1** | Foundations | The five ingredients of an agent; ReAct pattern |
        | **2** | ReAct Agent | Fake agent → real LLM-driven agent → two tools |
        | **3** | Real-World Tools | 5-tool agent with Wikipedia, Python exec, structured outputs |
        | **4** | Memory, Planning, Reflection | Compression, episodic memory, critic/revision loops |
        | **5** | Multi-Agent Systems | Pipeline + Manager/Worker patterns |
        | **6** | Frameworks | LangGraph & CrewAI mapped to hand-built patterns |
        """
    )

    st.info(
        "👈 Use the sidebar to navigate. Start with **How to read this** if you're new, "
        "or jump straight to **Code Artifacts** to see what was built."
    )


# ============================================================
# PAGE: How to read this
# ============================================================
elif page == "📘 How to read this":
    st.title("📘 How to read this")
    st.markdown(
        """
        This course is organized **chronologically**. Module 1 covers foundational concepts;
        each later module builds on the earlier ones.

        ### Different paths through the material

        - **Want to learn agentic AI from scratch?**
          → Read modules in order, starting with **Module 1**.

        - **Want to skim the key insights?**
          → Read the **Key Lessons** page and the "Surprising things observed" subsections in each module.

        - **Want to reference specific patterns?**
          → Use the **Cheatsheets & References** page.

        - **Want to see actual code?**
          → Refer to the **Code Artifacts** page — each file builds on the previous.

        ### Philosophy

        The pattern throughout this course was: **build first, observe failures, extract principles.**
        Most agent tutorials reverse this — they teach principles first, build last.
        The build-first approach means every principle in this course was discovered by hitting it
        as a wall, then named.
        """
    )


# ============================================================
# PAGE: Module 1
# ============================================================
elif page == "1️⃣ Module 1: Foundations":
    st.title("1️⃣ Module 1: Foundations")
    st.caption("The conceptual scaffolding before any code is written.")

    st.markdown("## What 'agentic' means")
    st.markdown(
        "Agentic AI isn't binary — it's a **spectrum of autonomy**. Most production systems mix levels."
    )
    st.markdown(
        """
        | Level | Description | Example |
        |-------|-------------|---------|
        | 0 | Pure function | A sentiment classifier |
        | 1 | Chatbot (state, no goals) | Basic conversational LLM |
        | 2 | Tool-augmented LLM (single turn) | LLM that can call a calculator |
        | 3 | ReAct agent (own loop) | LLM that reasons, acts, observes, repeats |
        | 4 | Planning agent | Decomposes goals into plans, replans on failure |
        | 5 | Multi-agent system | Manager delegates to specialized workers |
        | 6 | Self-improving agent | Modifies its own prompts/tools/strategies |
        """
    )

    st.markdown("## The five ingredients of an agent")
    st.markdown(
        """
        Every agent — narrow or general, LLM-based or not — needs these five things:

        1. **Goal specification** — what does "done" look like? The fuzzier the goal, the harder the job.
        2. **World model / state** — what the agent knows about its current situation. Compression matters.
        3. **Action space** — what the agent can do. Closed (enumerated) vs. open (LLM generates any tool call).
        4. **Policy** — given state and goal, which action next? Classical agents: algorithm. LLM agents: a prompt.
        5. **Feedback signal** — how the agent knows it's progressing. Clean signals → strong agents.
        """
    )

    st.markdown("## Why LLMs changed agentic AI")
    st.info(
        "Pre-LLM agents needed **enumerable action spaces** (a fixed set of moves). "
        "LLMs broke this — the action space became *anything you can describe in natural language*. "
        "This is the source of both their power and every problem in modern agentic AI: "
        "hallucinated actions, infinite loops, plans citing nonexistent functions, etc."
    )

    st.markdown("## The ReAct pattern")
    st.code(
        """Thought: <reasoning about what to do next>
Action: <tool_call>
Observation: <result from tool>
Thought: ...
Action: ...
...
Action: finish(<answer>)""",
        language="text",
    )
    st.markdown(
        "**Key insight:** the agent IS the loop. The LLM just fills in the Thoughts and Actions. "
        "The loop, parser, and tool registry sit *outside* the LLM."
    )

    st.markdown("## Why 'Thought' before 'Action' matters")
    st.markdown(
        "Forcing the model to verbalize reasoning before committing to an action measurably improves "
        "decision quality. It's not decorative — it's a performance technique."
    )

    st.markdown("## LLM API statelessness")
    st.warning(
        "LLM APIs are **stateless** — every call is independent. Conversation 'memory' is an illusion "
        "created by the *client*, which stores the conversation locally and re-sends the entire history "
        "on every new request."
    )


# ============================================================
# PAGE: Module 2
# ============================================================
elif page == "2️⃣ Module 2: ReAct Agent":
    st.title("2️⃣ Module 2: Building a ReAct Agent")
    st.caption("From a hardcoded fake agent → an LLM-driven agent with two composable tools.")

    st.markdown("## The four phases of any LLM call")
    st.markdown(
        """
        1. **Load credentials** (via `.env` and `python-dotenv`)
        2. **Create a client** (authenticated, reusable connection)
        3. **Make a call** (specifying model + prompt)
        4. **Extract the result** (pull text from response object)
        """
    )

    st.markdown("## Stage 1 — The fake agent (no LLM)")
    st.markdown(
        "Built a ReAct loop with **hardcoded** 'agent outputs' — pre-scripted Thought/Action strings — "
        "to isolate the **loop mechanics** from LLM behavior."
    )
    st.success(
        "**Learned:** the loop, parser, and tool registry are the agent's skeleton. "
        "The LLM is just what fills in the decision-making blanks."
    )
    view_file_inline("stage1_fake_agent.py")

    st.markdown("## Stage 2 — The real agent (LLM-driven)")
    st.markdown(
        "Replaced hardcoded outputs with calls to an LLM via OpenRouter. "
        "Only the *source* of the agent's output changed — the loop, parser, and tools stayed the same."
    )

    st.markdown("### Key new concepts")
    st.markdown(
        """
        - **The system prompt** — the LLM's "job description". Defines role, lists tools, enforces output format.
          *Policy is now a prompt, not an algorithm.*
        - **Conversation history (manual state)** — LLM APIs are stateless; maintain `(user_goal, agent_output, observation, ...)`.
        - **Role-based message format** — `{"role": "system" | "user" | "assistant", "content": "..."}`. Observations are sent as "user" messages.
        - **Temperature** — `temperature=0.0` for reasoning tasks. Lower temperature → better tool-use accuracy.
        - **Max tokens** — always cap output. Prevents runaway generations.
        """
    )

    st.markdown("### Observed LLM behaviors")
    st.markdown(
        """
        - LLMs sometimes **skip the `Thought` line** on trivially easy problems despite system-prompt instructions.
        - LLMs treat tools as **advisors, not oracles** — when a tool disagrees with the LLM's prior belief, the LLM may retry with a different formulation.
        - Different models have different "personalities": some decompose fine-grained, some bundle coarsely.
        """
    )

    st.markdown("### Robustness patterns")
    st.markdown(
        """
        - Always cap `max_iterations`.
        - Wrap parser logic in `try/except`.
        - Return errors as **Observations**, not crashes — gives the agent a chance to self-correct.
        - Use the Observation channel as a **teaching channel**.
        """
    )
    view_file_inline("stage2_real_agent.py")

    st.markdown("## Stage 3 — Two tools and composition")
    st.markdown(
        "Added a `lookup` tool alongside the calculator. Tasks now require *composing* tools — "
        "e.g., 'look up Tokyo's population, then calculate.'"
    )

    st.markdown("### Key new concepts")
    st.markdown(
        """
        - **Tool registry pattern** — `TOOLS = {name: function}` replaces `if/else` dispatches.
        - **Forgiving tools** — normalize inputs (lowercase, strip whitespace).
        - **Helpful failure messages** — when a tool fails, return *suggestions*.
        - **Tool composition as planning** — agent figures out order of operations from natural-language goal.
        """
    )

    st.markdown("### Observed agent behaviors")
    st.markdown(
        """
        - **Adaptive give-up thresholds** — agents stop after evidence of repeated failure. *Emergent*, not coded.
        - **Creative tool use** — agent uses prior knowledge to *guess*, then uses the tool to *verify*.
        - **Silent format fixes** — agents reformat data between tools without being told.
        - **Plan compaction** — some models bundle multi-step calculations into one tool call.
        """
    )
    view_file_inline("stage3_two_tools.py")


# ============================================================
# PAGE: Module 3
# ============================================================
elif page == "3️⃣ Module 3: Real-World Tools":
    st.title("3️⃣ Module 3: Real-World Tools")
    st.caption("Native tool calling, five real tools, structured outputs, and a multi-step capstone.")

    st.markdown("## Module 3.1–3.2 — Native tool calling")
    st.markdown(
        "Rebuilt the agent using **native tool calling** (JSON Schema declared via API parameter) "
        "instead of text-based ReAct parsing."
    )
    st.markdown(
        """
        | Aspect | Text-based ReAct | Native tool calling |
        |--------|------------------|---------------------|
        | Tool declaration | English in prompt | JSON Schema |
        | Format failures | Possible | Effectively zero |
        | Parallel tool calls | No | Yes |
        | System prompt size | Large | Small |
        | Termination | `finish(answer)` | Response without `tool_calls` |
        """
    )
    view_file_inline("stage3_5_native_tools.py")
    view_file_inline("stage3_5_native_tools_upgraded_tools.py")

    st.markdown("## Module 3.3 — Real-world tools")
    st.markdown("Built an agent with **five tools** spanning fact lookup, search, content retrieval, and computation:")
    st.markdown(
        """
        | Tool | Purpose |
        |------|---------|
        | `calculator` | Simple arithmetic |
        | `lookup` | Small in-memory knowledge base |
        | `wikipedia_search` | Find article titles via Wikipedia's official API |
        | `wikipedia_get_article` | Fetch full plain-text article content |
        | `python_exec` | Execute arbitrary Python (trusted local environment) |
        """
    )

    st.markdown("### Key concepts")
    st.markdown(
        """
        - **Real-world tools fail constantly** — network timeouts, scraper blocks, format changes.
        - **Timeouts on every external call** — non-negotiable.
        - **Output truncation matters** — real web content is huge; truncate to ~5000 chars.
        - **Structured APIs beat scraping** — Wikipedia's API beats DuckDuckGo scraping.
        - **Trust boundaries beat sandboxes** — for learning/local use, full Python access is fine.
        - **Loop detection via observation channel** — inject `[SYSTEM NOTE: ...]` into tool results.
        """
    )

    st.markdown("### Failures encountered (and what they taught)")
    st.markdown(
        """
        1. **DuckDuckGo scraper blocked** → prefer real APIs over scraping.
        2. **`__builtins__` sandbox broken** → use environmental isolation, not language-level.
        3. **Subtle infinite loop on partial success** → mixed-success feedback is more dangerous than total failure.
        4. **Silent number transcription error** (829.8 → 828) → LLMs reformat tool outputs silently.
        5. **Question inversion** → LLMs gravitate toward easy-to-produce framings.
        """
    )
    view_file_inline("stage3_3_real_tools.py")

    st.markdown("## Module 3.5 — Structured outputs")
    st.markdown(
        "Added JSON schema enforcement to the agent's final answer. The schema:"
    )
    st.code(
        """{
  "answer": "string",
  "key_facts": ["string"],
  "sources_used": ["calculator" | "lookup" | "wikipedia_search" | "wikipedia_get_article" | "python_exec" | "prior_knowledge"],
  "confidence": "high" | "medium" | "low",
  "limitations": "string"
}""",
        language="json",
    )
    st.warning(
        "**Surprising:** Structured outputs guarantee **format, not meaning**. Agents fill fields per schema, "
        "but their *interpretation* of what each field is *for* often differs from the designer's. "
        "`confidence` is almost always 'high' — real model miscalibration."
    )
    view_file_inline("stage3_5_structured.py")

    st.markdown("## Module 3.6 — Capstone")
    st.markdown(
        "**Task:** Compare light vs sound travel time from Moon to Earth, compute the ratio, and explain why "
        "'lightning before thunder' matters."
    )
    st.markdown(
        """
        **Behaviors observed:**
        - **Aggressive parallel lookups** — three independent lookups fired simultaneously
        - **Unit-aware variable naming** in Python (`distance_km`, `distance_m`)
        - **`limitations` field filled correctly** for the first time (the question made the caveat explicit)
        - **Loop detection fired incorrectly** — system noted 4 lookups, but they were progressing

        **What this stack CAN do reliably:** multi-step research (3–8 steps), unit-aware numerical computation,
        conceptual + factual reasoning, structured output.

        **What it CANNOT do:** learn between runs, plan 20+ steps, delegate to specialists, production observability.
        """
    )
    view_file_inline("stage3_6_structured_complex_ip.py")


# ============================================================
# PAGE: Module 4
# ============================================================
elif page == "4️⃣ Module 4: Memory, Planning, Reflection":
    st.title("4️⃣ Module 4: Memory, Planning, Reflection")
    st.caption("Working memory compression, episodic memory with embeddings, and self-critique loops.")

    st.markdown("## Module 4.1 — The four types of memory")
    st.markdown(
        """
        | Type | What it stores | Where it lives | Lifetime |
        |------|----------------|----------------|----------|
        | **Working** | Current conversation context | `messages` list | Dies when run ends |
        | **Episodic** | Past runs and their outcomes | DB / file | Persistent across runs |
        | **Semantic** | Structured general knowledge | KB / vector DB | Persistent, usually read-only |
        | **Procedural** | How-to skills and patterns | System prompt / fine-tuning | Persistent, hard to update |
        """
    )
    st.markdown(
        """
        **What we already had vs. what was missing:**
        - ✅ Working memory (the `messages` list)
        - ⚠️ Partial semantic memory (tiny `KNOWLEDGE_BASE` in lookup)
        - ❌ Episodic memory (no persistence across runs)
        - ✅ Static procedural memory (the system prompt)
        """
    )

    st.markdown("## Module 4.2 — Working memory compression")
    st.markdown(
        "**Problem:** the `messages` list grows on every turn → context window issues, cost, "
        "and 'lost-in-the-middle' reasoning issues."
    )
    st.markdown("### Three structural rules")
    st.markdown(
        """
        1. **Always keep the system prompt** (agent identity).
        2. **Always keep the original user goal** (task being solved).
        3. **Always cut on a turn boundary** — never split tool_calls from results (API enforces tool_call_id linking).
        """
    )
    st.code(
        """compressed = (
    messages[:2]                          # system + original user goal
    + [summary_message]                   # one LLM-generated summary
    + messages[cutoff_index:]             # last N turn groups intact
)""",
        language="python",
    )
    st.info(
        "**Deep lesson:** When facts live only inside the conversation, compression blurs them. "
        "A better architecture has facts living *outside* the conversation — in a scratchpad or persistent memory. "
        "This motivates Module 4.3."
    )
    view_file_inline("stage4_2_compressed.py")

    st.markdown("## Module 4.3 — Episodic memory")
    st.markdown(
        """
        **Architecture:**
        - Storage: JSON file (`agent_memory.json`)
        - Embedding model: `openai/text-embedding-3-small` (1536 dimensions)
        - Similarity: cosine via NumPy
        - Retrieval: embed user goal, score all lessons, return top-K above threshold (0.35)
        """
    )
    st.code(
        """BEFORE the loop:
  - Embed user goal
  - Retrieve top-K relevant lessons (similarity >= threshold)
  - Inject into system prompt with "MIGHT be relevant — use judgment" framing

DURING the loop:
  - Track tool calls and results for later lesson extraction

AFTER the loop:
  - Pass run summary to an LLM "lesson extractor"
  - Extractor returns either a specific actionable lesson OR NO_LESSON
  - If a lesson is returned, embed and store it""",
        language="text",
    )

    st.markdown("### Two persistent problems")
    with st.expander("Problem 1: Lesson extractors invent platitudes"):
        st.markdown(
            "Default prompts produce generic lessons like 'ensure outputs are clear' — useless. "
            "**Solution:** explicit positive examples, negative examples, strong default toward `NO_LESSON`."
        )
    with st.expander("Problem 2: Relevant lessons may not embed close enough"):
        st.markdown(
            "A lesson about 'researching historical figures' had only **0.273** similarity to a goal about "
            "a specific historical figure — below threshold. "
            "**Solution:** rephrased with overlapping vocabulary ('Roman emperor', 'Wikipedia') → similarity jumped to **0.539**."
        )

    st.markdown("### Surprising observations")
    st.markdown(
        """
        - Memory works mechanically but has **modest impact per single run** — non-determinism dominates.
        - The **"use judgment" framing** is doing real work — a deliberately misleading lesson was retrieved and ignored.
        - Real memory systems give **statistical, not absolute, wins**. Production gains: 5–15% on aggregate metrics.
        - **Lesson phrasing must match expected future task phrasing** for retrieval to work.
        - Successful, uneventful runs produce **no learnable lessons** — that's the goal.
        """
    )
    view_file_inline("memory.py")
    view_file_inline("stage4_3_memory.py")
    view_file_inline("agent_memory.json")

    st.markdown("## Module 4.5 — Reflection")
    st.markdown(
        "After the agent produces an answer, a separate **critic** LLM reviews it. "
        "If issues found, the agent enters a capped revision loop."
    )

    st.markdown("### Two test cases")
    with st.expander("Test 1: Burj Khalifa height vs. Mount Everest — **Clean win**"):
        st.markdown(
            """
            - **Original:** "Burj Khalifa is approximately 0.0938 times the height of Mount Everest" (confusing)
            - **Critic caught:** semantic mismatch between calculation direction and question phrasing
            - **Revised:** "Mount Everest is approximately 10.66 times taller than the Burj Khalifa"
            - **Cost:** +2 LLM calls
            - **Verdict:** Reflection caught a real issue the generator missed.
            """
        )
    with st.expander("Test 2: Berlin population density — **Mixed**"):
        st.markdown(
            """
            - **Original:** "4 million... density ~4485 per sq km"
            - **Critic caught:** the 4 million figure is rounded; precise number is ~3.77 million
            - **Revision loop:** re-fetched articles, found precise 3,769,495 figure, computed 4227.3
            - **Outcome:** Hit the 5-step cap *before* producing the final answer — fell back to original (wrong) answer.
            - **Verdict:** Critic was right, but budget exhausted just shy of completion.
            """
        )

    st.markdown("### Composition: three layers of robustness")
    st.markdown(
        """
        - **Planning** = structural safety (won't wander)
        - **Memory** = experiential safety (won't repeat known mistakes)
        - **Reflection** = output safety (won't ship confidently wrong)

        Each costs additional LLM calls. Each addresses a different failure mode.
        """
    )
    view_file_inline("stage4_5_reflection.py")


# ============================================================
# PAGE: Module 5
# ============================================================
elif page == "5️⃣ Module 5: Multi-Agent Systems":
    st.title("5️⃣ Module 5: Multi-Agent Systems")
    st.caption("Pipeline (sequential) and Manager/Worker (orchestrated) patterns.")

    st.markdown("## Why multi-agent at all")
    st.markdown(
        """
        The motivation isn't capability — a single agent with the right tools can do almost anything
        multi-agent can. The motivation is:

        - **Quality** — specialized prompts focused on one task outperform monolithic prompts.
        - **Clarity** — failures localize to specific agents.
        - **Scale** — different agents can use different models, infrastructure, or be maintained by different teams.
        """
    )

    st.markdown("## The two foundational patterns")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Pipeline (sequential)")
        st.code("User goal → [Researcher] → findings → [Writer] → final output", language="text")
        st.markdown("Use when the task has **clear, predictable phases.**")
    with col2:
        st.markdown("### Manager/Worker (orchestrated)")
        st.code(
            """          [Manager]
         ↙    ↓    ↘
  [Worker A] [Worker B] [Worker C]""",
            language="text",
        )
        st.markdown("Use when the **path through workers depends on the input.**")

    st.info(
        "**Key conceptual unlock:** Workers are exposed to the manager as **tools** — native tool calling, "
        "exactly like `calculator` or `wikipedia_search`. The tool function just happens to internally run another "
        "LLM-based agent. Once you see this, manager/worker stops feeling like a different paradigm — it's recursive ReAct."
    )

    st.markdown("## Module 5a — Pipeline (Researcher + Writer)")
    st.markdown(
        "**Task:** *Write a short biographical summary of Marie Curie covering her early life, "
        "scientific work, and historical significance.*"
    )
    st.markdown(
        """
        - **Researcher:** ReAct agent with Wikipedia tools, outputs structured `{topic, key_facts, sources}`
        - **Writer:** single LLM call, no tools, produces prose
        - **Orchestrator:** hardcoded Python function (no LLM)

        **Comparison with single-agent baseline:**
        - Multi-agent prose was **visibly better-organized** (three paragraphs matching the three requested aspects)
        - Single-agent prose was a **flat info-dump** — the agent was still in "tool-using mode" when it produced output
        - Multi-agent used ~60% more LLM calls

        **Lesson:** Separating cognitive modes (research vs. composition) produces better outputs.
        """
    )
    view_file_inline("stage5_multiagent.py")

    st.markdown("## Module 5b — Manager/Worker (Trip Planner)")
    st.markdown(
        "**Task:** *Plan a 5-day trip to Kyoto with budget \\$2000 and interests in history and food.*"
    )
    st.markdown(
        """
        **Workers:**
        - `destination_researcher` — practical info via Wikipedia
        - `activities_researcher` — things to do via Wikipedia
        - `budget_analyzer` — no tools, pure LLM reasoning about costs

        **Behavior on Kyoto run:**
        - Manager called `activities_researcher` and `budget_analyzer` **in parallel** in step 1
        - Correctly skipped `destination_researcher`
        - Synthesized outputs into a structured trip plan

        **Behavior on Reykjavik run (simpler task):**
        - Manager called only `destination_researcher`
        - But the worker returned **more than asked for** — its prompt has its own goals
        """
    )

    st.warning(
        "**Real production problem:** Workers are independent agents with their own prompts. "
        "**Manager guidance is a soft suggestion, not a hard constraint.** "
        "Mitigations: parameter-responsive worker prompts, more granular workers, or post-processing in the manager."
    )

    st.markdown("### Cost reality")
    st.markdown(
        """
        - Single agent: ~3–5 LLM calls per task
        - Pipeline: ~5–7 calls
        - Manager/worker: ~8–12 calls (each worker is itself a small ReAct loop)
        """
    )
    view_file_inline("stage5_manager_multi_agent.py")

    st.markdown("## The industry trend")
    st.markdown(
        """
        Multi-agent was heavily hyped 2023–2024. The trend has moved back toward
        **one well-designed agent with many tools**, because:

        - Frontier models are good enough to handle complex tasks alone
        - Coordination loses information at every handoff
        - Multi-agent systems are harder to debug, monitor, and maintain
        - Added cost rarely justifies marginal quality gains

        **Honest recommendation:** Start with one well-designed agent. Add specialization only when you can
        clearly articulate (a) what cognitive mode separation it provides, (b) why a single agent can't do
        the same with the right prompt, and (c) what the handoff schema between agents looks like.
        """
    )


# ============================================================
# PAGE: Module 6
# ============================================================
elif page == "6️⃣ Module 6: Frameworks":
    st.title("6️⃣ Module 6: Frameworks")
    st.caption("LangGraph and CrewAI mapped onto the patterns built from scratch.")

    st.markdown("## The mapping (LangGraph)")
    st.markdown(
        """
        | Hand-built code | LangGraph equivalent |
        |-----------------|----------------------|
        | `messages` list | State object with reducer (`Annotated[list, operator.add]`) |
        | `while step in range(max_iterations):` | Graph runtime |
        | `if message.tool_calls:` check | Conditional edge function (`should_continue`) |
        | Tool execution loop | `tool_node` function |
        | LLM call | `agent_node` function |
        | Final answer return | Edge to `END` |
        """
    )
    st.success("**The graph is structurally identical to the hand-coded loop.** LangGraph gives names to the parts and a framework to wire them together.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### What LangGraph buys you")
        st.markdown(
            """
            - Visualization (mermaid diagrams)
            - Persistence & resumption (checkpointers)
            - Streaming of intermediate state
            - LangSmith integration
            - Built-in patterns (ReAct, plan-and-execute, supervisor)
            - Parallelism primitives
            """
        )
    with col2:
        st.markdown("### What it costs")
        st.markdown(
            """
            - More verbose for simple cases
            - Learning curve for state/reducer/edge abstractions
            - Custom patterns require graph nodes
            - Framework lock-in
            """
        )

    st.markdown("## CrewAI in brief")
    st.markdown(
        """
        Different philosophy: **"team of specialists with roles."**

        - `Agent` (role + goal + backstory → assembled into system prompt)
        - `Task` (input + expected output)
        - `Crew` with `Process.sequential` (pipeline) or `Process.hierarchical` (manager/worker)

        Maps cleanly to Module 5 patterns. Trades control for opinionated orchestration.
        """
    )

    st.markdown("## The decision framework")
    st.markdown(
        """
        | Situation | Recommendation |
        |-----------|----------------|
        | Learning, prototyping, simple agents | **Raw Python** |
        | Complex orchestration, production observability needs | **LangGraph** |
        | Team-of-specialists fit, non-engineer stakeholders | **CrewAI** |
        | Most agents in 2026 | Raw Python is still the right answer more often than people admit |
        """
    )

    st.info(
        "The bar for adopting a framework should be: *does this solve a real problem I have?* "
        "Not: *is this popular?*"
    )

    st.markdown("## Surprising observations")
    st.markdown(
        """
        - **The principles transfer.** Tool-design issues, error-message design, loop detection — they all matter
          equally in LangGraph as in raw Python.
        - **LangGraph is shorter for common patterns, longer for unusual ones.** Compression, custom memory —
          easier to write directly than to fit into graph abstractions.
        - **CrewAI hides the manager prompt** when using `Process.hierarchical`. Convenient until you need to customize.
        - **Framework choice is less consequential than people make it.** Most agents fail or succeed based on
          tool design, prompt quality, and feedback signal quality — not on which framework wraps the loop.
        """
    )
    view_file_inline("langgraph_example.py")


# ============================================================
# PAGE: Code Artifacts
# ============================================================
elif page == "📂 Code Artifacts":
    st.title("📂 Code Artifacts")
    st.markdown(
        "Consolidated list of all files built during the course, in order. "
        "Each stage builds on the previous — to understand any later stage fully, "
        "the earlier ones provide context."
    )

    # Filter
    modules_present = sorted(set(a[1] for a in CODE_ARTIFACTS))
    selected = st.multiselect(
        "Filter by module",
        options=modules_present,
        default=modules_present,
    )

    st.markdown("---")

    for filename, module, purpose in CODE_ARTIFACTS:
        if module not in selected:
            continue
        path = Path(filename)
        exists = path.exists()

        with st.container():
            cols = st.columns([3, 1.2, 4, 1.2])
            with cols[0]:
                st.markdown(f"**`{filename}`**")
            with cols[1]:
                st.markdown(f'<span class="pill pill-module">{module}</span>', unsafe_allow_html=True)
            with cols[2]:
                st.caption(purpose)
            with cols[3]:
                if exists:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    st.download_button(
                        "📥 Download",
                        content,
                        file_name=filename,
                        key=f"art_dl_{filename}",
                        use_container_width=True,
                    )
                else:
                    st.markdown('<span class="small-muted">— not in dir —</span>', unsafe_allow_html=True)

            if exists:
                with st.expander(f"View `{filename}`"):
                    content = path.read_text(encoding="utf-8", errors="replace")
                    lang = "python" if filename.endswith(".py") else "json" if filename.endswith(".json") else "text"
                    st.code(content, language=lang)

    st.markdown("---")
    st.info(
        "📌 **Tip:** Place this `app.py` in the same directory as your `.py` files "
        "(`stage1_fake_agent.py`, etc.) — they'll then show up as **Found** and become viewable/downloadable inline."
    )

    st.markdown("### Setup")
    st.code("pip install openai python-dotenv streamlit", language="bash")
    st.markdown("`.env` file:")
    st.code("OPENROUTER_API_KEY=sk-or-v1-...", language="text")


# ============================================================
# PAGE: Key Lessons
# ============================================================
elif page == "💡 Key Lessons":
    st.title("💡 Key Lessons Learned")

    st.markdown("## The five takeaways")
    st.markdown(
        """
        1. **The agent is the loop, not the model.**
        2. **Most agent problems are tool design problems.**
        3. **Structured outputs guarantee format, not meaning.**
        4. **Production agents are non-deterministic distributed systems — plan for failure modes.**
        5. **Frameworks are dialects. Concepts are the language.**
        """
    )

    st.markdown("## Conceptual lessons")
    st.markdown(
        """
        - **An agent is a loop, not a model.** Intelligence is in the LLM, but *agency* lives in the surrounding loop, parser, and tool registry.
        - **State management is half the job.** LLMs are amnesiac; you maintain history manually.
        - **Policy is a prompt.** Want different behavior? Rewrite the system prompt. No retraining needed.
        - **Feedback signal quality determines agent quality.** Clean numeric signals → strong agents. Fuzzy subjective signals → weak agents.
        - **LLM agents combine tool use with prior knowledge.** Flashes of brilliance and pockets of confident hallucination — sometimes in the same task.
        """
    )

    st.markdown("## Practical lessons")
    st.markdown(
        """
        - Always cap `max_iterations` and `max_tokens`. **Always.**
        - Use cheap models (Gemini Flash, Haiku, GPT-4o-mini) for development. Save premium models for one-off comparison runs.
        - Errors should become Observations, not crashes.
        - Tool error messages are **teaching messages** — design them to guide the agent toward success.
        - The system prompt and tool registry must stay synchronized — every registered tool must be documented in the prompt.
        """
    )

    st.markdown("## Surprising things observed")
    with st.expander("Module 2–3 surprises"):
        st.markdown(
            """
            - Same loop + same prompt + different LLM = qualitatively different agent behavior.
            - LLMs distrust their tools when tool output contradicts their priors — sometimes right to, sometimes not.
            - An agent reverse-engineered a broken calculator (`^` parsed as XOR) and worked around the bug without being told to.
            - An agent worked around a forward-only lookup tool by guessing a fact and using the tool to verify it.
            - Agents calibrate their own persistence based on the informational content of feedback. No "give up after N tries" rule needed.
            """
        )
    with st.expander("Module 3.3 surprises"):
        st.markdown(
            """
            - Wikipedia's official API is dramatically more reliable than scraping search engines.
            - An agent computed dates by hand (counting leap years, summing months) when its Python tool was broken.
            - Agents silently reformat data from tools when it doesn't fit their preferred output shape (829.8 → 828; "10.69 times shorter" → "10.69 times taller").
            - An agent **inverted a question** to match the answer shape it could produce cleanly.
            - Sandboxing Python at the language level is much harder than it looks. Use environmental isolation.
            """
        )
    with st.expander("Module 4 surprises"):
        st.markdown(
            """
            - **Compression makes long agents possible but doesn't make them better.** Solves cost & context-overflow, not reasoning.
            - **`temperature=0.0` is not actually deterministic.** Floating-point non-determinism, batching effects, backend variance.
            - **The Gemini-via-OpenRouter combo doesn't support `tools` + `response_format` simultaneously.** Feature compatibility varies by provider.
            - **Memory works mechanically but has modest impact per single run.** Production gains: 5–15% on aggregate metrics.
            - **The "use judgment" framing is doing real work** — a deliberately misleading lesson was retrieved and ignored.
            """
        )
    with st.expander("Module 5 surprises"):
        st.markdown(
            """
            - **Single agents in "tool-using mode" produce list-like prose.** Recent context dominates style.
            - **The writer agent, with no tools, produces visibly better prose.** Cognitive mode separation is real.
            - **Manager routing intelligence is genuine** — skipping `budget_analyzer` when no budget is mentioned, calling workers in parallel when independent.
            - **Workers don't always honor manager guidance.** Manager passes hints; worker prompts may override them.
            - **The handoff schema is everything.** Categorized facts → organized paragraphs; uncategorized facts → flat dump.
            """
        )


# ============================================================
# PAGE: Cheatsheets
# ============================================================
elif page == "📋 Cheatsheets & References":
    st.title("📋 Cheatsheets & References")

    st.markdown("## Text-based ReAct vs. Native tool calling")
    st.markdown(
        """
        | Aspect | Text-based ReAct | Native tool calling |
        |--------|------------------|---------------------|
        | Tool declaration | English in system prompt | JSON Schema via API parameter |
        | Tool dispatch | Parse `Action:` line with regex | Read `tool_calls` from response object |
        | Format failures | Possible (parser breaks) | Effectively zero (API enforces) |
        | Parallel tool calls | No | Yes |
        | Reasoning visibility | Explicit `Thought:` lines | Hidden (model internalizes) |
        | System prompt size | Large (must teach format) | Small (API handles format) |
        | Termination signal | `finish(answer)` action | Response without `tool_calls` |
        | Final answer style | Constrained text | Natural prose or structured |
        """
    )

    st.markdown("**When to use which:**")
    st.markdown(
        """
        - Production with mainstream models → **Native**
        - Open-source models without tool-call training → **Text-based**
        - Learning / debugging / inspecting reasoning → **Text-based**
        - Need parallel tool calls → **Native**
        - Need to enforce strict output format → **Native (with strict mode)**
        """
    )

    st.markdown("---")
    st.markdown("## ReAct Family of Patterns")
    st.markdown(
        """
        | Pattern | One-line summary |
        |---------|------------------|
        | **ReAct** | Reason and act one step at a time. Maximum reactivity. |
        | **Plan-and-Execute** | Plan all steps upfront, then execute. Replan on failure. |
        | **Reflexion** | ReAct + cross-episode memory: agent reflects on past tasks, stores lessons. |
        | **Tree of Thoughts** | Generate multiple candidate next steps, score, pick best. Expensive. |
        | **ReWOO** | Plan all tool calls upfront, execute in parallel, synthesize. Works only when steps are independent. |
        """
    )
    st.markdown("*All of these answer the same question: how much should the agent plan vs. react?*")

    st.markdown("---")
    st.markdown("## Production Failure Modes")
    st.markdown(
        """
        The eight major failure modes any production agent has to handle:

        1. **Context window overflow** — conversation grows linearly, hits limits, gets expensive, gets confused.
        2. **Cost runaways** — stuck loops on premium models burn money fast.
           *Mitigation:* hard caps on iterations & tokens, monthly budget caps.
        3. **Infinite loops** — `max_iterations` cap protects; subtle versions require smarter detection.
        4. **Prompt injection** — external content can contain instructions that override the system prompt.
           *Biggest unaddressed security issue in agentic AI.*
        5. **Hallucinated tool calls** — LLM invents tools that don't exist. Mitigated by native tool calling.
        6. **Tool misuse** — right tool, wrong arguments. *Mitigation:* validate inputs, fail loudly.
        7. **Silent over-confidence** — plausible-looking wrong answers with no signal of error. The hardest to detect.
        8. **Rate limits / transient failures** — retry with backoff, distinguish retryable from non-retryable.
        """
    )

    st.markdown("---")
    st.markdown("## Real-World Tool Design Cheatsheet")
    st.markdown(
        """
        - **Always set timeouts** on every external call (10s default).
        - **Validate inputs** — never trust the LLM to format URLs, queries, or IDs correctly.
        - **Catch network errors distinctly** from generic exceptions — timeouts may be retryable, 404s aren't.
        - **Truncate large outputs** before feeding back. Web pages can be 50KB+; that wrecks context windows.
        - **Strip junk before truncation** — for HTML, remove `<script>`, `<style>`, `<nav>`, `<footer>` first.
        - **Tell the agent when content was truncated** so it knows to fetch more if needed.
        - **Prefer structured APIs over scraping** when available. JSON beats HTML.
        """
    )

    st.markdown("---")
    st.markdown("## Context Compression Cheatsheet")
    st.markdown(
        """
        - **Trigger:** token threshold (e.g., 8000 tokens). Adaptive — fires when needed, not at fixed intervals.
        - **Cutoff:** always at an assistant-message index. Slicing mid-turn-group breaks `tool_call_id` linking → 400 error.
        - **Summary prompt structure:** include original user goal, format transcript human-readably, explicit instructions on what to preserve (exact numbers) and drop (verbose reasoning).
        - **Cap with `max_tokens`** (e.g., 400) to keep summary terse.
        - **Watch for:** the agent re-doing work that got summarized away. If you see redundant tool calls after compression, your summary is dropping things the agent needs.
        """
    )

    st.markdown("---")
    st.markdown("## Writing Lessons That Work (Episodic Memory)")
    st.markdown(
        """
        1. **Be specific.** Mention exact tool names, error patterns, numeric thresholds.
           *"wikipedia_get_article returns HTTP 429 after 3+ rapid calls"* beats *"be careful with Wikipedia."*
        2. **Use task-vocabulary.** Write lessons in the words the agent would use to describe a related future task.
           *"When researching a Roman emperor..."* beats *"for historical figure research..."*
        3. **Default to NO_LESSON.** Most runs aren't notable. Save lessons only when something *specific* happened.
        """
    )

    st.markdown("---")
    st.markdown("## When to Use Which Multi-Agent Pattern")
    st.markdown(
        """
        | Situation | Recommended pattern |
        |-----------|---------------------|
        | Task has one cognitive mode | Single agent |
        | Task has clear, fixed phases | Pipeline |
        | Task requires dynamic routing among specialists | Manager/Worker |
        | Workers need different tools, models, or rate limits | Multi-agent (either) |
        | Output quality matters more than cost | Multi-agent |
        | High-volume, latency-sensitive | Single agent |
        """
    )


# ============================================================
# PAGE: Final Reflection
# ============================================================
elif page == "🎓 Final Reflection":
    st.title("🎓 Final Reflection")

    st.markdown(
        """
        This course was built in dialogue — **Socratic, experimental, and iterative.**

        The pattern was: **build first, observe failures, extract principles.**

        Most agent tutorials reverse this — they teach principles first, build last. The build-first
        approach means every principle in this course was discovered by hitting it as a wall, then named.
        """
    )

    st.markdown("## What this course deliberately did NOT cover")
    st.markdown(
        """
        - **Fine-tuning models for agent behavior** — real production sometimes does this; out of scope here.
        - **Production observability stacks** (LangSmith, AgentOps, Helicone) — mentioned but not used.
        - **Specific evals frameworks** (Phoenix, Braintrust, OpenAI evals) — production necessity, but a separate skill.
        - **Cost optimization at scale** (caching, model routing, prompt compression for cost).
        - **Building agents that interact with real APIs** (Stripe, Salesforce, Gmail) — same patterns, different surface area.

        For each of these, the patterns in this course transfer. **The mental model is what matters.**
        """
    )

    st.markdown("## Status")
    cols = st.columns(3)
    statuses = [
        ("Module 1", "Foundations"),
        ("Module 2", "ReAct Agent"),
        ("Module 3", "Real-World Tools"),
        ("Module 4", "Memory, Planning, Reflection"),
        ("Module 5", "Multi-Agent Systems"),
        ("Module 6", "Frameworks"),
    ]
    for i, (m, name) in enumerate(statuses):
        with cols[i % 3]:
            st.success(f"✅ **{m}**: {name}")

    st.markdown("---")
    st.markdown("## 🎓 Course complete.")
    st.balloons()
