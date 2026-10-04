import logging
import sys
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph import build_graph

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    )

logger = logging.getLogger("tutor.main")

def run_cli():
    """Interactive CLI loop with persistent memory."""
    app = build_graph()
    thread_id = f"student-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\nPython 101 Tutor Copilot (session: {thread_id})")
    print("Type your question, or 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye")
            break

        if question.lower() in {"quit", "exit", "q"}:
            print("\nGoodbye")
            break
        if not question:
            print("(Empty question, please type your question.)")
            continue

        print()
        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=question)]},
                config=config,
            )
            answer = result.get("final_answer") or "(no answer produced)"
            print(f"Tutor: {answer}\n")
        except Exception:
            logger.exception("Graph invocation failed.")
            print("Tutor: I hit a technical issue. Please try again.\n")


if __name__ == "__main__":
    run_cli()