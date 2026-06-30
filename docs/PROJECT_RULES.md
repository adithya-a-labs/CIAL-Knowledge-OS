# CIAL KnowledgeOS — Project Rules

The canonical project requirements are maintained in `PROJECT_REQUIREMENTS.md`. This document provides detailed development rules and must remain consistent with those requirements.

## 1. On-Premise First

- No cloud deployment is allowed.
- No AWS, Azure, GCP, Vercel, Railway, Render, or hosted inference APIs.
- The system must be designed to run fully inside CIAL-controlled infrastructure.
- Prefer local-first tools, local databases, and self-hosted services.

## 2. Open-Source Model First

- Use open-source LLMs wherever possible.
- Prefer models that can run through Ollama, llama.cpp, vLLM, or Hugging Face local inference.
- Do not design core features around OpenAI, Claude, Gemini, Groq, Cerebras, or other cloud APIs.
- Any cloud-model experiment must be clearly marked as temporary and non-production.

# Model Agnosticism & Local AI Deployment

### Core Principle

- Knowledge OS must remain **model-agnostic**.
- Never design the system around a single LLM vendor or model family.
- All AI functionality must communicate through a common abstraction layer.

### Local-First AI

The entire AI stack must run on infrastructure owned by the organization.

No cloud inference providers may be required for core functionality, including but not limited to:

- OpenAI
- Anthropic
- Google Gemini API
- AWS Bedrock
- Azure OpenAI
- Any other hosted LLM APIs

All prompts, retrieved documents, embeddings, intermediate reasoning, and generated responses must remain inside the organization's infrastructure.

### Supported Model Families

The platform should support local deployment of multiple open-weight model families, including:

- Meta (Llama)
- Google (Gemma)
- Microsoft (Phi)
- Mistral AI
- Qwen
- DeepSeek

The architecture must not favor any particular model family.

### Model Abstraction Layer

All LLM integrations must use a unified interface.

Changing models should require configuration changes only.

The implementation should support multiple local inference engines, including:

- Ollama (development)
- vLLM (production)
- Future local inference runtimes

### Development Policy

Notebook experiments must:

- benchmark multiple models
- avoid model-specific assumptions
- keep retrieval independent from the chosen LLM
- make switching models possible through configuration only

### Production Philosophy

Knowledge OS is an AI platform—not an application tied to a specific language model.

Organizations should be free to choose whichever local open-weight model best satisfies their requirements for:

- performance
- licensing
- multilingual capability
- hardware availability
- cost
- security

without requiring application-level code changes.

## 3. Token Efficiency

- Optimize every RAG experiment for low token usage.
- Do not pass entire documents into the LLM.
- Use chunking, retrieval, reranking, and context compression before generation.
- Keep prompt templates short and purposeful.
- Log estimated input/output token usage where possible.
- Compare retrieval quality before increasing context size.

## 4. Retrieval Before Generation

- The LLM should answer only from retrieved context.
- Retrieval quality must be improved before increasing model size.
- Every answer should be grounded in source chunks.
- If evidence is weak, the system should say that the available documents are insufficient.

## 5. Citations Are Mandatory

- Every generated answer must include document references.
- Track source file name, page number if available, chunk ID, and metadata.
- Never generate enterprise answers without traceability.

## 6. Metadata-Aware RAG

All documents should be indexed with metadata such as:

- department
- document type
- asset/system
- location
- date
- version
- access level
- source file
- page number
- owner/responsible team

Retrieval should use metadata filters wherever possible.

## 7. Security and Access Control

- Design with role-based access control in mind from the beginning.
- Users must only retrieve documents they are authorized to access.
- Personal workspace documents must remain isolated from the central repository.
- Do not mix private user uploads with organization-wide documents unless explicitly allowed.

## 8. Offline-Friendly Stack

Preferred stack during experimentation:

- Python
- Jupyter Notebook
- Ollama / llama.cpp / vLLM
- Qdrant / Chroma / FAISS
- Sentence Transformers
- BGE / E5 / Qwen embeddings
- BGE reranker / local cross-encoder reranker
- FastAPI later for backend
- Frontend dashboard later

Avoid tools that require cloud-only execution.

## 9. Notebook Experimentation Rules

Each notebook should:

- Start with a clear objective.
- List the technique being tested.
- Use small sample documents first.
- Print retrieved chunks before generation.
- Show similarity scores or rerank scores.
- Show source metadata.
- Compare outputs before and after improvements.
- End with observations, limitations, and next steps.

## 10. Evaluation First Mindset

Every RAG improvement must be tested against:

- answer correctness
- citation accuracy
- retrieval relevance
- hallucination risk
- latency
- token usage
- local hardware feasibility

Do not assume a technique is better just because it is more advanced.

## 11. Enterprise Reliability

The system should handle:

- vague questions
- multi-part questions
- missing information
- conflicting documents
- outdated documents
- duplicate documents
- scanned PDFs
- long manuals
- tables
- acronyms and airport-specific terminology

## 12. Minimal Agent Usage

- Do not use agents where simple deterministic code is enough.
- Agents should be used only for clear roles such as:
  - query planner
  - router
  - verifier
  - citation checker
  - hallucination critic
- Keep agent workflows inspectable and debuggable.

## 13. No Black Box Pipelines

- Every stage must be inspectable.
- Always expose:
  - original query
  - rewritten query
  - retrieved chunks
  - reranked chunks
  - final context
  - generated answer
  - verifier output

## 14. Hardware-Aware Development

- Experiments should be runnable on a laptop first.
- Optimize for 12GB VRAM testing where possible.
- Later scale to workstation/server GPUs.
- Avoid assuming unlimited GPU memory.
- Prefer quantized models for local testing.

## 15. Data Privacy

- Do not upload CIAL documents to external services.
- Do not use external APIs for document processing unless explicitly approved.
- Keep raw documents inside the local repository or controlled local storage.
- Add sensitive data handling notes before using real enterprise documents.

## 16. Modularity

Even though the current stage uses notebooks, code should be written so it can later be moved into modules:

- ingestion
- chunking
- embeddings
- vectorstore
- retrieval
- reranking
- generation
- verification
- evaluation

## 17. Hybrid Retrieval Preference

Enterprise RAG should not rely only on vector search.

Prefer combining:

- vector search
- keyword/BM25 search
- metadata filters
- reranking

## 18. Failure Handling

The system must support safe failure responses:

- “I could not find enough evidence in the uploaded documents.”
- “The retrieved documents disagree.”
- “This answer may be incomplete because the relevant SOP is missing.”
- “Please upload the latest version of the document.”

## 19. No Premature Productionization

During the notebook stage:

- Do not build a large backend too early.
- Do not over-engineer UI.
- Do not add complex deployment scripts before the RAG pipeline works.
- Prioritize correctness, observability, and learning.

## 20. Final Goal

The final system should become:

A secure, on-premise, enterprise-grade Knowledge OS for CIAL that allows employees to search, reason over, and interact with internal documents using natural language while preserving privacy, traceability, access control, and operational reliability.

# 21. Session Completion Rules

At the end of every Codex session, always provide a concise summary of the work completed.

The summary should include:

## Changes Made
- Files created
- Files modified
- New folders added
- Major implementation decisions
- Any assumptions made

## Verification
- Confirm whether the project builds/runs (if applicable).
- Mention any untested functionality.
- List any TODOs remaining.

## Suggested Next Steps
Recommend the next logical engineering task(s) without implementing them.

## Suggested Commit Message
Always suggest one Git commit message.

Requirements:
- Use Conventional Commit format.
- Keep it medium detail.
- Clearly summarize the engineering work completed.
- Mention the primary feature or architectural change.
- Do not make the commit automatically.

Example:

feat(rag): scaffold notebook-based enterprise RAG research workspace

- establish initial repository structure
- add notebook experimentation framework
- add project development rules
- prepare directories for data, references, and future frontend

## Engineering Notes

If important architectural decisions were made during the session, explain why they were chosen and what alternatives were considered.

## Risks

Highlight anything that could become technical debt, reduce performance, or complicate future development.

## Future Refactoring Opportunities

If code written during this session should later be modularized or improved, explicitly mention it.

## Questions for the Next Session

List any open engineering questions that should be resolved before proceeding.

## 22. Reference Notebook Usage

The notebooks inside `references/` are **learning resources only**.

They demonstrate RAG concepts and algorithms but are **not** the architectural standard for this project.

Many of these notebooks use cloud-based services such as:

- ChatOpenAI
- OpenAI Embeddings
- OpenAI API
- Cohere APIs
- LangSmith

These are used only because they were the original implementations.

### When implementing any technique from the reference notebooks:

- Do **not** copy cloud-specific code.
- Preserve the underlying RAG technique, not the implementation.
- Replace cloud components with local, open-source alternatives.
- Assume the final system must run completely offline.
- Do not introduce API keys or hosted inference unless explicitly instructed for a temporary experiment.

Preferred local replacements include:

| Cloud Example | Preferred Local Alternative |
|---------------|-----------------------------|
| ChatOpenAI | Ollama / vLLM / llama.cpp |
| OpenAI Embeddings | BGE / E5 / Qwen Embeddings / Nomic Embed |
| Cohere Reranker | BGE Reranker / Local Cross-Encoder |
| LangSmith | Local logging and evaluation |

### Important

If a reference notebook contains cloud-specific code, **reinterpret the algorithm using the project's approved local stack rather than reproducing the original implementation.**

The reference notebooks should be treated as conceptual guides, not implementation templates.
