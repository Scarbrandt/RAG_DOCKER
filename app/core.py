# app/core.py
from typing import List

def generate_answer(query: str) -> str:
    # Ваша RAG-логика
    return f"Ответ на: {query}"

def batch_generate(queries: List[str]) -> List[str]:
    return [generate_answer(q) for q in queries]
