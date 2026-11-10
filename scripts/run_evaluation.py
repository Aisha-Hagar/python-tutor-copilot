import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage

from src import citations
from src.graph import build_graph
from src.scoring import deterministic_scores
from src.observability import get_callback_handler, get_langfuse_client, resolve_trace_id, score_trace

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

        trace_id = resolve_trace_id(handler)
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