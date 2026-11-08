import logging
from typing import Optional
import re

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage
)
from langchain_ollama import ChatOllama

from src import config
from src.tools import ALL_TOOLS
from src import citations

logger = logging.getLogger(__name__)

_llm: Optional[ChatOllama] = None
_llm_with_tools = None


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model = config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            headers={"Authorization": f"Bearer {config.OLLAMA_API_KEY}"},
            temperature=0.0,
            num_predict=512
        )
    return _llm


def get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = get_llm().bind_tools(ALL_TOOLS)
    return _llm_with_tools


# Helpers
def _last_human_message(state: dict) -> str:
    for msg in reversed(state['messages']):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


# Node 1: Planning
def planning_node(state: dict) -> dict:
    """
    Read the student's question, classify its intent, and produce a short plan.
    Also resets loop counters and retrieved_context for a fresh turn.
    """
    logger.info("NODE: planning")
    question = _last_human_message(state)

    prompt = f"""You are a Python 101 tutor. A student asked:

{question}

Classify the intent as exactly one of:
- concept_explanation  (they want a definition/explanation)
- debugging            (they pasted an error or broken code)
- quiz_prep            (they want practice or a quiz)

Then write a 4-step plan (numbered 1-4) for answering at a beginner level.

Format your reply as:
INTENT: <intent>
PLAN:
1. ...
2. ...
3. ...
4. ...
    """

    try:
        response = get_llm().invoke(prompt)
        text = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as e:
        logger.exception("planning_node LLM call failed")
        text = "INTENT: concept_explanation\nPLAN:\n1. Ask for clarification.\n"

        # Parse intent
    intent = "concept_explanation"
    for line in text.splitlines():
        if line.upper().startswith("INTENT:"):
            intent = line.split(":", 1)[1].strip().lower()
            break

    logger.info(" intent=%s", intent)
    logger.info(" plan=%s", text.replace("\n", " , ")[:200])

    return {
        "intent": intent,
        "plan": text,
        "retrieved_context": "", # Reset for the new turn
        "draft": "",
        "reflection": "",
        "reflection_passed": False,
        "iterations": 0
    }

# Node 2: Retrieve context
def retrieve_node(state: dict) -> dict:
    """Deterministically call retrieve_textbook so context always exists."""
    logger.info("NODE: retrieve")
    question = _last_human_message(state)
    from src.tools import retrieve_textbook
    try:
        result = retrieve_textbook.invoke({"query": question, "top_k": 5})
        if str(result).startswith(("ERROR:", "NO_RESULTS:")):
            logger.warning("  retrieval returned: %s", str(result)[:100])
            context = ""
        else:
            context = str(result)
            logger.info("  retrieved_context set (%d chars)", len(context))
    except Exception:
        logger.exception("retrieve_node failed")
        context = ""
    return {"retrieved_context": context}


# Node 3: ReAct Agent
def agent_node(state: dict) -> dict:
    """
    The ReAct step: the LLM decides whether to call a tool.
    If it emits tool_calls, routing sends us to the tool's node.
    Otherwise, routing sends us to the draft node.
    """
    logger.info("NODE: agent")
    try:
        response = get_llm_with_tools().invoke(state["messages"])
    except Exception as e:
        logger.exception("agent_node LLM call failed")
        response = AIMessage(content="I ran into a technical issue. Please try again")

    if getattr(response, "tool_calls", None):
        for tc in response.tool_calls:
            logger.info(" agent wants tool: %s(%s)", tc["name"], tc["args"])
    else:
        logger.info(" agent produced a final answer (no tool calls)")

    return {"messages": [response]}


# Node 4: Tools
def tools_node(state: dict) -> dict:
    """
    Execute the tool calls emitted by the agent.
    IMPORTANT: if the tool is retrieve_textbook, also write its output into
    state['retrieved_context'] so the draft node can use it.
    """
    logger.info("NODE: tools")
    last = state["messages"][-1]
    tool_messages: list[ToolMessage] = []
    retrieved_context = state.get("retrieved_context", "")

    tool_map = {t.name: t for t in ALL_TOOLS}

    for call in getattr(last, "tool_calls", []) or []:
        name  = call["name"]
        args = call.get("args", {})
        call_id = call["id"]

        tool = tool_map.get(name)
        if tool is None:
            result = f"ERROR: Unknown tool '{name}'."
            logger.warning("  unknown tool: %s", name)
        else:
            try:
                result = tool.invoke(args)
                logger.info("  tool %s returned %d chars", name, len(str(result)))
            except Exception as e:
                logger.exception("  tool %s failed", name)
                result = f"ERROR: Tool '{name}' failed ({e})."

        if name == "retrieve_textbook" and not str(result).startswith(("ERROR:", "NO_RESULTS:")):
            retrieved_context = str(result)
            logger.info("  retrieved context set (%d chars)", len(retrieved_context))

        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

    logger.info("  tools_node returning keys: %s, retrieved_context len=%d",
                ["messages", "retrieved_context"], len(retrieved_context))
    return {
        "messages": tool_messages,
        "retrieved_context": retrieved_context,
    }


# Node 5: Draft
def draft_node(state: dict) -> dict:
    """
    Write the answer using only the retrieved textbook context and any tool output.
    Falls back to a clarifying question if nothing useful was retrieved.
    """
    logger.info("NODE: draft")
    question = _last_human_message(state)
    context = state.get("retrieved_context", "")
    logger.info("  context page markers: %s", citations.extract_available_pages(context))

    # 1. Abstain if retrieval produced nothing usable
    if len(context) < 100:
        logger.warning("  insufficient context (%d chars) —> abstaining", len(context))
        return {"draft": (
            "I couldn't find this in the course textbook, so I don't want to guess. "
            "I've flagged this question for a TA, they'll follow up within one working day."
        )}

    # 2. Derive citation metadata once
    available_pages = citations.extract_available_pages(context)
    source_map = citations.extract_id_page_map(context)
    logger.info("  available_pages=%s", available_pages)
    logger.info("  source_map=%s", source_map)

    # 3. Collect ToolMessages from the current turn for non-retrieval tools which produce their own content.
    tool_outputs = []
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage) and getattr(msg, "name", "") != "retrieve_textbook":
            tool_outputs.append(f"[Tool: {msg.name}]\n{msg.content}")
    tool_text = "\n\n".join(reversed(tool_outputs))

    # 4. Build the prompt
    tool_block = f"\nTOOL OUTPUT:\n{tool_text[:2000]}\n" if tool_text else ""
    prompt = f"""You are a friendly Python 101 tutor. Answer this student's question:

{question}

Use ONLY the textbook excerpts below. Keep your answer beginner-friendly.
- Plain language, no jargon.
- One short code example (2-5 lines).
- The excerpts are labelled [S1], [S2], etc. Cite the label at the end of each claim.
- Example of a correctly cited sentence: "A list is mutable. [S1]"
- If the excerpts do not answer the question, say so explicitly.
- Do not use outside knowledge.

TEXTBOOK CONTEXT:
{context[:3000]}
{tool_block}
Answer:
"""
    logger.info("  prompt length=%d chars", len(prompt))

    # 5. Call the LLM (narrow try: post-processing must not be swallowed)
    try:
        response = get_llm().invoke(prompt)
    except Exception as e:
        logger.exception("draft_node LLM call failed")
        return {"draft":"I hit a technical issue while drafting your answer. Please try again."}

    raw_draft = response.content if isinstance(response.content, str) else str(response.content)

    if not raw_draft.strip():
        logger.warning("  LLM returned empty content. Using fallback")
        return {"draft": "I couldn't generate an answer. Please rephrase your question."}

    # 6. Log raw citations before resolving
    raw_labels = re.findall(r"\[S\d+\]", raw_draft)
    logger.info("  raw draft citations: %s", raw_labels)
    if raw_labels and not source_map:
        logger.warning(
            "  model cited labels but source_map is empty — "
            "context may not contain '[S#, page N]' markers"
        )

    # 7. Resolve [S#] → [page N]
    draft = citations.resolve_source_ids(raw_draft, source_map)
    logger.info("  resolved citations: %s", citations.extract_cited_pages(draft))
    logger.info("  draft length=%d chars", len(draft))

    return {"draft": draft}


def parse_verdict(text: str) -> bool:
    """Extract PASS/FAIL from the model's verdict line, tolerating markdown."""
    match = re.search(r"VERDICT\s*:\s*\**\s*(PASS|FAIL)", text, re.IGNORECASE)
    return bool(match and match.group(1).upper() == "PASS")


# Node 6: Reflect
def reflect_node(state: dict) -> dict:
    """
    Evaluate the draft against a small rubric plus a deterministic citation check.
    Returns PASS/FAIL + feedback.
    """
    logger.info("NODE: reflect")
    logger.info("  reflect_node sees state keys: %s", list(state.keys()))
    logger.info("  reflect draft field: %r",
                state.get("draft", "MISSING")[:80] if state.get("draft") else "EMPTY/MISSING")
    draft = state.get("draft", "") or ""
    context = state.get("retrieved_context", "") or ""

    available_pages = citations.extract_available_pages(context)
    cited = citations.extract_cited_pages(draft)
    invalid = citations.find_invalid_citations(draft, available_pages)
    if invalid:
        draft = citations.strip_invalid_citations(draft, available_pages)

    citation_ok = bool(cited) and not invalid
    logger.info(
        "  citation check: cited=%s invalid=%s ok=%s",
        cited, invalid, citation_ok,
    )

    if citation_ok:
        citation_block = "CITATION STATUS: valid , all cited pages exist in the context."
    else:
        citation_block = (
            "CITATION STATUS: missing or invalid, the answer MUST be revised "
            "to include valid citations."
        )

    prompt = f"""You are a strict evaluator for a Python 101 tutoring system.

Score the DRAFT answer on four criteria (each 0-10):
1. Groundedness: is it supported by the textbook context?
2. Clarity: is it understandable to a beginner?
3. Correctness: is it technically accurate?
4. Level-appropriateness: no advanced shortcuts or jargon?

TEXTBOOK CONTEXT:
\"\"\"
{context[:2000]}
\"\"\"

DRAFT:
\"\"\"
{draft[:2000]}
\"\"\"

{citation_block}

Reply with EXACTLY this format:
SCORES: groundedness=X, clarity=X, correctness=X, level=X
VERDICT: PASS or FAIL
FEEDBACK: <one sentence of specific improvement advice if FAIL>
"""

    logger.info("  reflect draft len=%d, context len=%d", len(draft), len(context))
    logger.info("  reflect prompt preview: %r", prompt[:400])

    try:
        response = get_llm().invoke(prompt)
        text = response.content if isinstance(response.content, str) else str(response.content)
    except Exception:
        logger.exception("reflect_node LLM call failed")
        text = "SCORES: n/a\nVERDICT: PASS\nFEEDBACK: evaluation unavailable"

    llm_passed = parse_verdict(text)
    passed = citation_ok and llm_passed

    logger.info("  llm verdict: %s", "PASS" if llm_passed else "FAIL")
    logger.info("  combined verdict: %s", "PASS" if passed else "FAIL")
    logger.info("  reflection text: %s", text.replace("\n", " | ")[:300])

    return {"reflection": text, "reflection_passed": passed}


# Node 7: Revise
def revise_node(state: dict) -> dict:
    """
    Rewrite the draft using the reflection feedback.
    Increments the iterations counter to bound the loop.
    """
    logger.info("NODE: revise (iteration %d)", state.get("iterations", 0) +1)
    draft = state.get("draft", "")
    feedback = state.get("reflection", "")
    context = state.get("retrieved_context", "")

    prompt = f"""Revise this Python 101 tutoring answer based on the feedback.

TEXTBOOK CONTEXT:
{context[:2500]}

CURRENT DRAFT:
{draft}

FEEDBACK:
{feedback}

Rules:
- Keep the citation labels from the current draft. Do not remove them.
- If a claim in your revision comes from the textbook, cite it with the
  matching [S#] label.
- Beginner-friendly language, one short code example.
Return ONLY the revised answer.
"""

    try:
        response = get_llm().invoke(prompt)
        revised = response.content if isinstance(response.content, str) else str(response.content)
    except Exception:
        logger.exception("revise_node LLM call failed")
        revised = draft

    if not revised.strip():
        logger.warning("  revise returned empty, keeping previous draft")
        revised = draft

    source_map = citations.extract_id_page_map(context)
    before = citations.extract_cited_pages(revised)
    revised = citations.resolve_source_ids(revised, source_map)
    after = citations.extract_cited_pages(revised)
    logger.info("  revise citations: before=%s after=%s", before, after)

    original_citations = citations.extract_cited_pages(draft)
    if original_citations and not before:
        logger.warning(
            "  revision dropped all citations (%d → 0) — restoring original",
            len(original_citations),
        )
        revised = draft

    return {
        "draft": revised,
        "iterations": state.get("iterations", 0)+1,
    }


# Node 8: Finalize
def finalize_node(state: dict) -> dict:
    """
    Emit the final answer and append it to the message history.
    """
    logger.info("NODE: finalize")
    answer = state.get("draft") or "I'm sorry, I couldn't produce an answer."
    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }