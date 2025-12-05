# backend/evaluation/run_eval.py

import json
from pathlib import Path

from backend.agents.editorial_agent import agent_answer


def main():
    eval_path = Path(__file__).parent / "eval_questions.json"
    questions = json.loads(eval_path.read_text(encoding="utf-8"))

    print("=== EVALUACIÓN EDITORIALCOPILOT ===\n")

    for i, q in enumerate(questions, start=1):
        print(f"[{i}] Pregunta: {q}")
        answer = agent_answer(q, history=[])
        print(f"Respuesta:\n{answer}\n")
        print("-" * 80)

    print("Fin de la evaluación. Revisa las respuestas para seleccionar ejemplos de demo.")


if __name__ == "__main__":
    main()
