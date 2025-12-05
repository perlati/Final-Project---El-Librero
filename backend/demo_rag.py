# backend/demo_rag.py

"""
Quick RAG demo over the books collection.

Usage:
    python -m backend.demo_rag "¿Qué tienen los libros en común?"
or:
    python -m backend.demo_rag
    (then type your question when prompted)
"""

import sys

from backend.rag.chains import build_books_rag_chain


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Pregunta / Question: ").strip()

    if not question:
        print("No question provided.")
        return

    chain = build_books_rag_chain()
    print("\n[ANSWER]\n")
    answer = chain.invoke(question)
    print(answer)


if __name__ == "__main__":
    main()
