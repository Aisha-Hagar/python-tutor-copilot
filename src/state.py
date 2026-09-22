from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

class TutorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Intent classification: "concept_explanation", "debugging", "quiz_prep"
    intent: str
    plan: str
    retrieved_context: str
    reflection_passed: bool
    iterations: int
    final_answer: str
