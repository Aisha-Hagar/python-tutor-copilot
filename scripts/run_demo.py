import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage

from src.graph import build_graph
from src.observability import get_callback_handler, get_langfuse_client, score_trace, resolve_trace_id
from src.scoring import deterministic_scores
from src.sanitize import sanitize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tutor.demo")

DEMO_QUESTIONS = [
    "What is the difference between a list and a tuple?",
    "How do I write a for loop in Python?",
    "Why is my code giving an IndentationError?",
    "Can you create a practice quiz about functions?",
    "Can you make me a 2-week study plan for learning loops?",
]


def main():
    app = build_graph()
    handler = get_callback_handler()
    session_id = f"demo-{uuid.uuid4().hex[:8]}"

    config = {
        "configurable": {"thread_id": session_id},
        "callbacks": [handler],
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_tags": ["python-tutor", "demo"],
        },
    }

    print("Python 101 Tutor Copilot: Demo Run")
    print(f"Session: {session_id}")
    print(f"Questions: {len(DEMO_QUESTIONS)}")

    for i, question in enumerate(DEMO_QUESTIONS, start=1):
        print(f"[Turn {i}/{len(DEMO_QUESTIONS)}]")
        print(f"Student: {sanitize(question)}")

        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=sanitize(question))]},
                config=config,
            )
        except Exception:
            logger.exception("Turn %d failed", i)
            print("Tutor: [system error, see logs]\n")
            continue

        reflection = result.get("reflection", "")
        if reflection:
            passed = result.get("reflection_passed", False)
            print(f"[reflection: {'PASS' if passed else 'FAIL'}]")
            print(f"[detail: {sanitize(reflection)[:200]}]")

        draft = result.get("draft", "")
        if draft:
            print(f"[draft length: {len(draft)} chars]")

        citations = result.get("retrieved_context", "")
        if citations:
            print(f"[context retrieved: {len(citations)} chars]")

        messages = result.get("messages", [])
        print(f"[conversation length: {len(messages)} messages]")

        trace_id = resolve_trace_id(handler)
        scores = deterministic_scores(result)
        if trace_id:
            for name, value in scores.items():
                score_trace(trace_id, name, value)
        else:
            logger.warning("turn %d: no trace_id — scores not sent", i)
        print(f"[scores: {scores}]")

    get_langfuse_client().flush()
    print("Scores flushed to Langfuse.")
    print("Demo complete. Review traces in Langfuse.")


if __name__ == "__main__":
    main()