"""Grounded generation through a local Ollama runtime."""

from __future__ import annotations

from typing import Any, Protocol

from httpx import HTTPError
from langchain_ollama import OllamaLLM
from ollama import ResponseError, list as list_ollama_models

from .config import KnowledgeOSConfig

PHASE1_NO_EVIDENCE_RESPONSE = "It is not available in the retrieved documents."


class GenerationFailedError(RuntimeError):
    """Report an exhausted Phase 4 generation attempt without losing its cause.

    Inputs are the original exception and total attempt count. The raised error
    exposes the original type/message for batch row metadata and chains the
    source exception for full logs. Earlier phases do not use this exception;
    it is raised only by the Phase 4 retry boundary.
    """

    def __init__(self, original: Exception, *, attempts: int) -> None:
        self.original_error_type = type(original).__name__
        self.original_error_message = str(original)
        self.attempts = attempts
        super().__init__(
            f"{self.original_error_type}: {self.original_error_message} "
            f"(generation attempts: {attempts})"
        )


class LocalLLM(Protocol):
    """Minimal interface implemented by supported local inference adapters."""

    def invoke(self, prompt: str) -> Any: ...


def create_local_llm(config: KnowledgeOSConfig) -> OllamaLLM:
    """Validate and create a deterministic local Ollama model interface."""

    try:
        available_models = {
            model.model
            for model in list_ollama_models().models
            if model.model is not None
        }
    except (HTTPError, OSError, ResponseError) as exc:
        raise RuntimeError(
            "The local Ollama service is unavailable. Start Ollama and confirm "
            f"that the configured model '{config.ollama_model_name}' is installed."
        ) from exc

    if config.ollama_model_name not in available_models:
        raise RuntimeError(
            f"Configured Ollama model '{config.ollama_model_name}' is not installed "
            "locally. Install or transfer that model, or change "
            "KnowledgeOSConfig.ollama_model_name. No model was downloaded."
        )

    return OllamaLLM(model=config.ollama_model_name, temperature=0)


def build_grounded_prompt(
    question: str,
    context: str,
    *,
    no_evidence_response: str = PHASE1_NO_EVIDENCE_RESPONSE,
) -> str:
    """Build a strict grounded prompt that requires direct evidence."""

    return f"""You are a strict grounded-answering system.

Answer the QUESTION using only the provided CONTEXT.

Rules:
1. Use only facts directly supported by the CONTEXT.
2. Do not use outside knowledge.
3. Do not guess, infer beyond the evidence, or generalize from related cybersecurity guidance.
4. If the CONTEXT is only loosely related, incomplete, ambiguous, or does not directly answer the QUESTION, reply exactly:
"{no_evidence_response}"
5. If the question asks for organization-specific facts, predictions, passwords, budgets, vendors, live status, or information not explicitly present in CONTEXT, reply exactly:
"{no_evidence_response}"
6. Cite supported claims inline using exact reference IDs such as [1].
7. Do not invent, alter, or renumber reference IDs.
8. Answer concisely.
9. Prefer 5–8 bullets unless the question requires a longer explanation.
10. Do not include long background explanations.
11. Do not restate the context.
12. Do not add filler, introductions, or conclusions.

CONTEXT
{context}

QUESTION
{question}

ANSWER
"""


def generate_answer(
    llm: LocalLLM,
    question: str,
    context: str,
    *,
    no_evidence_response: str = PHASE1_NO_EVIDENCE_RESPONSE,
) -> str:
    """Generate a grounded answer using the configured local runtime."""

    if not context.strip():
        return no_evidence_response
    prompt = build_grounded_prompt(
        question,
        context,
        no_evidence_response=no_evidence_response,
    )
    return str(llm.invoke(prompt)).strip()
