"""Grounded generation through a local Ollama runtime."""

from __future__ import annotations

from typing import Any, Protocol

from langchain_ollama import OllamaLLM

from .config import KnowledgeOSConfig


class LocalLLM(Protocol):
    """Minimal interface implemented by supported local inference adapters."""

    def invoke(self, prompt: str) -> Any: ...


def create_local_llm(config: KnowledgeOSConfig) -> OllamaLLM:
    """Create a deterministic local Ollama language-model interface."""

    return OllamaLLM(model=config.ollama_model_name, temperature=0)


def build_grounded_prompt(question: str, context: str) -> str:
    """Build a short prompt that requires evidence and traceable citations."""

    return f"""Answer only from CONTEXT. Cite claims with exact reference IDs such as [1].
Do not invent or alter citation fields.
If the answer is absent or evidence is weak, reply exactly:
"It is not available in the retrieved documents."

CONTEXT
{context}

QUESTION
{question}

ANSWER
"""


def generate_answer(llm: LocalLLM, question: str, context: str) -> str:
    """Generate a grounded answer using the configured local runtime."""

    if not context.strip():
        return "It is not available in the retrieved documents."
    return str(llm.invoke(build_grounded_prompt(question, context))).strip()
