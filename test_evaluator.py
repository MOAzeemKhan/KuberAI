# test_evaluator.py - Enhanced KuberAI QA Evaluator
import json
from difflib import SequenceMatcher
from app import qa_chain  # assumes app.py is in the same folder and loads chain

with open("test_set.json") as f:
    test_cases = json.load(f)

results = []
correct = 0

def fuzzy_match(expected, answer):
    return SequenceMatcher(None, expected.lower(), answer.lower()).ratio() > 0.6

for case in test_cases:
    question = case["question"]
    expected = case.get("expected", "")
    expected_keywords = case.get("expected_keywords", [])

    response = qa_chain.run(question)
    answer = response.strip().split("Answer (friendly and clear):")[-1].strip().lower()

    # Check if expected or all keywords are present
    if expected_keywords:
        is_correct = all(k.lower() in answer for k in expected_keywords)
    elif expected:
        is_correct = any(k.lower() in answer for k in expected.split())

    else:
        is_correct = False

    results.append({
        "question": question,
        "expected": expected or ", ".join(expected_keywords),
        "answer": answer,
        "correct": is_correct
    })

    if is_correct:
        correct += 1

# Final accuracy
accuracy = round((correct / len(results)) * 100, 2)
print(f"\nAccuracy: {accuracy}% ({correct}/{len(results)})\n")

# Output table
from tabulate import tabulate
print(tabulate([
    [r["question"], r["correct"], r["answer"][:80] + ("..." if len(r["answer"]) > 80 else "")]
    for r in results
], headers=["Question", "Correct", "Model Answer"], tablefmt="github"))
