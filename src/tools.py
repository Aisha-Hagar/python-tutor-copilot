import logging
from typing import Optional

from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient

from src import config

logger = logging.getLogger(__name__)

# Shared Qdrant + embeddings clients (created once, reused across tool calls)
_qdrant_client: Optional[QdrantClient] = None
_embeddings: Optional[OllamaEmbeddings] = None

def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )
    return _qdrant_client

def _get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    return _embeddings


# Tools
# Tool 1: Retrieval from the textbook
@tool
def retrieve_textbook(query: str, top_k: int = 3) -> str:
    """
    Search the Python textbook for passages relevant to the student's
    question. Use this whenever you need a definition, explanation, or example
    that must match the question. Returns the top matching passages
    with page numbers and chunk IDs for citation.
    """
    logger.info("TOOL CALL: retrieve_textbook(query=%r, top_k=%r)", query, top_k)

    if not query or not query.strip():
        return "ERROR: Empty query. Please provide a topic to search for"

    try:
        client = _get_qdrant()
        embeddings = _get_embeddings()
        vector = embeddings.embed_query(query)

        hits = client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=vector,
            limit=top_k,
            with_payload=True,
        ).points

        if not hits:
            return "NO_RESULTS: The textbook doesn't appear to cover this topic."

        parts = []
        for h in hits:
            payload = h.payload or {}
            parts.append(
                f"[Source: {payload.get("source", "unknown")} | "
                f"page: {payload.get("page", "?")} | "
                f"chunk: {payload.get("chunk_id", "?")}]\n"
                f"{payload.get("text", "")}"
            )
        return "\n\n---\n\n".join(parts)

    except Exception as e:
        logger.exception("retrieve_textbook failed")
        return f"ERROR: Retrieval failed {e}"


# Tool 2: Practice quiz
@tool
def create_practice_quiz(topic: str, num_questions: int=3) ->str:
    """
    Create a short practice quiz on a given Python topic. Use this when the
    student asks for practice, a quiz, or a way to test their understanding.
    Returns a numbered list of questions.
    """
    logger.info("TOOL CALL: create_practice_quiz(topic=%r, num_questions=%d)", topic, num_questions)

    if not topic or not topic.strip():
        return "ERROR: NO topic provided."

    num_questions = max(1, min(num_questions, 5))

    source = retrieve_textbook.invoke({"query": topic, "top_k": 2})

    questions = [
        f"1. In your own words, what is {topic}?",
        f"2. Write a one-line Python example that demonstrates {topic}.",
        f"3. Name one common mistake beginners make with {topic}, and how to avoid it.",
        f"4. How does {topic} differ from a related concept you've learned?",
        f"5. Where in the textbook is {topic} explained? (Hint: check the cited pages.)",
    ]

    quiz = "\n".join(questions[:num_questions])
    return f"Practice quiz on '{topic}':\n\n{quiz}\n\nSuggested reading:\n{source[:600]}"


# Tool 3: Recommend exercises
@tool
def recommend_exercises(topic: str, difficulty: str = "beginner") -> str:
    """
    Recommend hands-on exercises from the textbook for a given topic.
    difficulty must be one of: 'beginner', 'intermediate'. Use this when the
    student wants to practise.
    """
    logger.info("TOOL CALL: recommend_exercises(topic=%r, difficulty=%r)", topic, difficulty)

    if not topic or not topic.strip():
        return "ERROR: No topic provided."

    difficulty = difficulty.lower()
    if difficulty not in {"beginner", "intermediate"}:
        difficulty = "beginner"

    source = retrieve_textbook.invoke({"query": topic, "top_k": 2})

    plan = (
        f"Exercises for '{topic}' ({difficulty}):\n"
        f"1. Read the textbook section cited below.\n"
        f"2. Type out every code example by hand (no copy-paste) and run it.\n"
        f"3. Modify each example: change a value, add a print, break it on purpose.\n"
        f"4. Write a 5-line program that uses {topic} to solve a small problem.\n"
        f"5. Explain {topic} to a friend in 30 seconds.\n\n"
        f"Relevant textbook passages:\n{source[:600]}"
    )
    return plan


# Tool 4: Escalate to TA
@tool
def escalate_to_ta(reason: str, student_question: str) -> str:
    """
    Escalate a student's question to a human TA when the Copilot cannot answer
    confidently from the textbook. Provide a short reason and the original
    question. This logs the escalation for follow-up.
    """
    logger.warning("TOOL CALL: escalate_to_ta(reason=%r)", reason)
    logger.warning("  student_question=%r", student_question)
    return (
        f"Escalated to a TA.\n"
        f"Reason: {reason}\n"
        f"Question: {student_question}\n"
        f"A TA will reply within one working day."
    )


# Tool 5: Study plan
@tool
def build_study_plan(topic: str, weeks: int=2) -> str:
    """
    Build a simple week-by-week study plan for a Python topic. Use this when
    the student asks 'how should I study X?' or 'make me a plan'.
    """
    logger.info("TOOL CALL: build_study_plan(topic=%r, weeks=%d)", topic, weeks)

    if not topic or not topic.strip():
        return "ERROR: No topic provided."

    weeks = max(1, min(weeks, 6))
    source = retrieve_textbook.invoke({"query": topic, "top_k": 2})

    plan_lines = [f"{weeks}-week study plan for '{topic}':\n"]
    for w in range(1, weeks + 1):
        plan_lines.append(
            f"Week {w}: read the textbook section, solve all the examples,"
            f"complete 3 exercises, then write a mini-program using {topic}."
        )
    plan_lines.append(f"\nStart here:\n{source[:600]}")
    return "\n".join(plan_lines)


# Tool registry
ALL_TOOLS = [
    retrieve_textbook,
    create_practice_quiz,
    recommend_exercises,
    escalate_to_ta,
    build_study_plan,
]