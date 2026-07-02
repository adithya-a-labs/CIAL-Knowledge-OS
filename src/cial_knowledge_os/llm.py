"""Grounded generation through a local Ollama runtime."""

from __future__ import annotations

from typing import Any, Protocol

from httpx import HTTPError
from langchain_ollama import OllamaLLM
from ollama import ResponseError, list as list_ollama_models

from .config import KnowledgeOSConfig

PHASE1_NO_EVIDENCE_RESPONSE = "It is not available in the retrieved documents."


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
    """Build a short prompt that requires evidence and traceable citations."""

    return f"""Answer only from CONTEXT. Cite claims with exact reference IDs such as [1].
Use reference IDs inline. Do not add a separate reference list.
The application resolves reference IDs locally.
Do not invent or alter reference IDs.
If the answer is absent or evidence is weak, reply exactly:
"{no_evidence_response}"

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
