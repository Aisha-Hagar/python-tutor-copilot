import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.nodes import (
    agent_node,
    draft_node,
    finalize_node,
    planning_node,
    reflect_node,
    revise_node,
    tools_node,
)
from src.state import TutorState

logger = logging.getLogger(__name__)


# Routing functions
def route_after_agent(state: TutorState) -> str:
    """
    If the LLM emitted tool calls -> go to the tools node.
    If it produced a final answer -> go to draft.
    """
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        logger.info("ROUTE: agent -> tools")
        return "tools"
    logger.info("ROUTE: agent -> draft")
    return "draft"

def route_after_reflect(state: TutorState) -> str:
    """
    Decide whether to finalize or revise.
    If reflection passed -> finalize.
    If already revised twice -> finalize (bounded loop)
    Otherwise -> revise.
    """
    if state.get("reflection_passed", False):
        logger.info("ROUTE: reflect -> finalize (PASS)")
        return "finalize"
    if state.get("iterations", 0) >=2:
        logger.info("ROUTE: reflect -> finalize (iteration limit reached)")
        return "finalize"
    logger.info("ROUTE: reflect -> revise (attempt %d)", state.get("iterations", 0)+1)
    return "revise"


# Graph assembly
def build_graph():
    """
    Build and compile the tutor graph.
    """
    builder = StateGraph(TutorState)

    # Nodes
    builder.add_node("planning", planning_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_node("draft", draft_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("revise", revise_node)
    builder.add_node("finalize", finalize_node)

    # Entry point
    builder.add_edge(START, "planning")
    builder.add_edge("planning", "agent")

    # Routing to tool
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "draft": "draft"},
    )

    # Back to agent
    builder.add_edge("tools", "agent")

    # Draft -> reflect
    builder.add_edge("draft", "reflect")

    # Routing to revise or finalize
    builder.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"revise": "revise", "finalize": "finalize"},
    )

    # Back to reflect
    builder.add_edge("revise", "reflect")

    # Finalize
    builder.add_edge("finalize", END)

    # Compile with in-memory checkpointer
    checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)

    logger.info("Graph compiled.")
    return app