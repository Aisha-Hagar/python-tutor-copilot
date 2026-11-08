import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage

from src import citations
from src.graph import build_graph
from src.observability import get_callback_handler, get_langfuse_client, score_trace

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("tutor.eval")

EVAL_FILE = Path(__file__).resolve().parent.parent / "evaluation" / "eval_questions.txt"


def load_questions() -> list[str]:
    return [
        line.strip()
        for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def deterministic_scores(result: dict) -> dict[str, float]:
    """
    Compute scores that are deterministic.
    """
    draft = result.get("draft", "") or ""
    intent = result.get("intent", "")
    requires_context = intent != "quiz_prep"
    context = result.get("retrieved_context", "") or ""
    grounded = 1.0 if (not requires_context or len(context) > 200) else 0.0
    available = citations.extract_available_pages(context)
    cited = citations.extract_cited_pages(draft)
    invalid = [p for p in cited if p not in available]

    if cited:
        citation_validity = 1.0 if not invalid else 0.0
    else:
        citation_validity = 0.0

    return {
        "draft_length": float(len(draft)),
        "has_citations": 1.0 if cited else 0.0,
        "citation_validity": citation_validity,
        "grounded_context_present": grounded,
        "reflection_passed": 1.0 if result.get("reflection_passed") else 0.0,
    }


def main():
    app = build_graph()
    questions = load_questions()
    handler = get_callback_handler()
    langfuse = get_langfuse_client()

    print(f"\nRunning evaluation on {len(questions)} questions\n")

    summary = []

    for i, question in enumerate(questions, start=1):
        session_id = f"eval-{uuid.uuid4().hex[:8]}"
        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [handler],
            "metadata": {
                "langfuse_session_id": session_id,
                "langfuse_tags": ["python-tutor", "evaluation"],
            }
        }

        print(f"[{i}/{len(questions)}] {question}")
        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=question)]},
                config=config,
            )
        except Exception as e:
            logger.exception("eval run failed")
            print(f" ERROR: {e}\n")
            continue

        #trace_id = handler.last_trace_id if hasattr(handler, "last_trace_id") else None
        trace_id = None
        if hasattr(handler, "last_trace_id"):
            trace_id = handler.last_trace_id
        elif hasattr(handler, "get_trace_id"):
            trace_id = handler.get_trace_id

        scores = deterministic_scores(result)
        logger.info("Q%d: intent=%s, context_len=%d, cited=%s",
                    i,
                    result.get("intent", "?"),
                    len(result.get("retrieved_context", "") or ""),
                    citations.extract_cited_pages(result.get("draft", "") or ""))
        if trace_id:
            for name, value in scores.items():
                score_trace(trace_id, name, value)

        answer = result.get("final_answer", "")[:120]
        print(f"  answer: {answer}...")
        print(f"  scores: {scores}")
        print()

        summary.append({"question": question, **scores})

    # Print a summary table
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    if summary:
        keys = [k for k in summary[0] if k != "question"]
        for k in keys:
            avg = sum(row[k] for row in summary) / len(summary)
            print(f"  {k:24s}: {avg:.3f}")

    # Flush pending spans to Langfuse
    langfuse.flush()
    print("\nLangfuse traces flushed.")


if __name__ == "__main__":
    main()