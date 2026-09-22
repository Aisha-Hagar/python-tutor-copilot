import logging
from typing import Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage
)
from langchain_ollama import ChatOllama

from src import config
from src.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

_llm: Optional[ChatOllama] = None
_llm_with_tools = None


def get_llm() -> ChatOllama:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model = config.LLM_MODEL,
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


# Node 2: ReAct Agent
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


# Node 3: Tools
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

    return {
        "messages": tool_messages,
        "retrieved_context": retrieved_context,
    }


# Node 4: Draft
def draft_node(state: dict) -> dict:
    """
    Write the answer using the retrieved textbook context and any tool output.
    Falls back to a clarifying question if nothing useful was retrieved.
    """
    logger.info("NODE: draft")
    question = _last_human_message(state)
    context = state.get("retrieved_context", "")

    # Collect ToolMessages from the current turn for non-retrieval tools which produce their own content.
    tool_outputs = []
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            tool_outputs.append(f"[Tool: {msg.name}]\n{msg.content}")
        elif isinstance(msg, HumanMessage):
            break
    tool_text = "\n\n".join(reversed(tool_outputs))

    if not context and not tool_text:
        logger.info("  no context or tool output. Asking for clarification.")
        return {
            "draft": (
                "I don't have enough information to answer that yet. "
                "Could you give me a bit more detail, or point me to the "
                "specific topic you'd like me to explain?"
            )
        }

    prompt = f"""You are a friendly Python 101 tutor. Answer this student's question:

{question}

Use ONLY the following sources. Keep your answer beginner-friendly:
- Plain language, no jargon.
- One short code example (2-5 lines).
- Cite the textbook pages you used, in brackets, e.g. [page 42].

TEXTBOOK CONTEXT:
{context[:3000]}

TOOL OUTPUT:
{tool_text[:2000]}

Answer:
"""

    try:
        response = get_llm().invoke(prompt)
        draft = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as e:
        logger.exception("draft_node LLM call failed")
        draft = "I hit a technical issue while drafting your answer. Please try again."

    logger.info("  draft length=%d chars", len(draft))
    return {"draft": draft}


# Node 5: Reflect
def reflect_node(state: dict) -> dict:
    """
    Evaluate the draft against a small rubric. Returns PASS/FAIL + feedback.
    """
    logger.info("NODE: reflect")
    draft = state.get("draft", "")
    context = state.get("retrieved_context", "")

    prompt = f"""You are a strict evaluator for a Python 101 tutoring system.

Score the DRAFT answer on four criteria (each 0-10):
1. Groundedness: is it supported by the textbook context?
2. Clarity: is it understandable to a beginner?
3. Correctness: is it technically accurate?
4. Level-appropriateness: no advanced shortcuts or jargon?

TEXTBOOK CONTEXT:
{context[:2000]}

DRAFT:
{draft[:2000]}

Reply with EXACTLY this format:
SCORES: groundedness=X, clarity=X, correctness=X, level=X
VERDICT: PASS or FAIL
FEEDBACK: <one sentence of specific improvement advice if FAIL>
"""

    try:
        response = get_llm().invoke(prompt)
        text = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as e:
        logger.exception("reflect_node LLM call failed")
        text = "SCORES: n/a\nVERDICT: PASS\nFEEDBACK: evaluation unavailable"

    passed = "VERDICT: PASS" in text.upper() or "PASS" in text.upper().split("VERDICT:")[-1][:10]
    logger.info("  reflection verdict=%s", "PASS" if passed else "FAIL")
    logger.info("  reflection text=%s", text.replace("\n", " | ")[:200])

    return {"reflection": text, "reflection_passed": passed}


# Node 6: Revise
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
{context[:2000]}

CURRENT DRAFT:
{draft}

FEEDBACK:
{feedback}

Rules: beginner-friendly language, one short code example, cite textbook pages in [brackets].
Return ONLY the revised answer.
"""

    try:
        response = get_llm().invoke(prompt)
        revised = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as e:
        logger.exception("revise_node LLM call failed")
        revised = draft

    return {
        "draft": revised,
        "iterations": state.get("iterations", 0)+1,
    }


# Node 7: Finalize
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