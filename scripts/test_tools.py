import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools import retrieve_textbook, create_practice_quiz, escalate_to_ta, build_study_plan

print("=== retrieve_textbook ===")
print(retrieve_textbook.invoke({"query": "difference between list and tuple", "top_k": 2}))
print()

print("=== create_practice_quiz ===")
print(create_practice_quiz.invoke({"topic": "loops", "num_questions": 3}))
print()

print("=== build_study_plan ===")
print(build_study_plan.invoke({"topic": "functions", "weeks": 2}))
print()

print("=== escalate_to_ta ===")
print(escalate_to_ta.invoke({"reason": "out of scope", "student_question": "How do I install Python 4?"}))