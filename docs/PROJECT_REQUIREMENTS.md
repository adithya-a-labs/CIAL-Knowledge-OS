# CIAL KnowledgeOS — Project Requirements

This file is the single source of truth for project requirements. Detailed implementation guidance lives in `PROJECT_RULES.md` and `NOTEBOOK_GUIDELINES.md`; if those documents conflict with this file, this file takes precedence.

## 1. Product Goal

Build a secure, enterprise-grade Knowledge OS that allows CIAL employees to search, reason over, and interact with internal documents while preserving privacy, traceability, access control, and operational reliability.

## 2. Deployment and Infrastructure

- The production system must run fully on CIAL-controlled, on-premise infrastructure.
- Core functionality must work offline without cloud services or hosted inference.
- Prefer local databases, self-hosted services, and hardware-aware components.
- Experiments should run on a laptop where practical and target 12 GB VRAM for initial GPU testing.
- Cloud services may be used only when explicitly approved for a temporary, non-production experiment.

## 3. Models and AI Components

- Use local, open-source LLMs and embedding models wherever possible.
- Support local runtimes such as Ollama, llama.cpp, vLLM, and Hugging Face inference.
- Prefer BGE, E5, Qwen, or Nomic embeddings and local cross-encoder or BGE rerankers.
- Do not make OpenAI, Claude, Gemini, Cohere, Groq, Cerebras, or similar hosted APIs a core dependency.
- Keep agent use minimal, role-specific, inspectable, and replaceable with deterministic code where appropriate.

## 4. Retrieval and Answering

- Retrieve and inspect evidence before generation.
- Use hybrid retrieval: vector search, keyword/BM25 search, metadata filters, and reranking.
- Improve retrieval quality before increasing model size or context length.
- Do not pass entire documents to the LLM.
- Keep prompts and context concise; track token usage and latency where possible.
- Generated answers must be grounded in retrieved evidence and include traceable citations.
- Citations must retain source file, page number when available, chunk ID, and relevant metadata.
- Return an explicit safe-failure response when evidence is missing, weak, outdated, or conflicting.

## 5. Document Metadata

Index relevant metadata, including:

- department
- document type
- asset or system
- location
- date
- version
- access level
- source file
- page number
- owner or responsible team

Retrieval should apply metadata filters wherever possible.

## 6. Security and Data Privacy

- Enforce role- and department-aware access control during retrieval.
- Keep personal workspace documents isolated from organization-wide repositories.
- Do not upload CIAL documents or sensitive data to external services.
- External document processing requires explicit approval.
- Keep raw documents in controlled local storage.
- Use non-sensitive sample documents during early experimentation.

## 7. Experimentation and Evaluation

- The current development stage is notebook-based experimentation; avoid premature backend, UI, and deployment complexity.
- Notebooks must follow `NOTEBOOK_GUIDELINES.md` and expose intermediate retrieval and generation outputs.
- Evaluate correctness, citation accuracy, retrieval relevance, hallucination risk, latency, token usage, and local hardware feasibility.
- Test failure cases including vague or multipart questions, missing or conflicting information, outdated or duplicate documents, scanned PDFs, long manuals, tables, and domain terminology.
- Keep experimental code modular enough to migrate into ingestion, chunking, embeddings, vector store, retrieval, reranking, generation, verification, and evaluation modules.

## 8. Reference Notebooks

- Notebooks in `references/` are conceptual learning resources, not architectural templates.
- Preserve the underlying RAG technique while replacing cloud-specific code with approved local alternatives.
- Do not introduce API keys or hosted inference from a reference notebook unless explicitly approved for a temporary experiment.

## 9. Observability and Reliability

- Keep every pipeline stage inspectable.
- Expose the original query, rewritten query, retrieved chunks, reranked chunks, final context, generated answer, citations, and verifier output where applicable.
- Handle insufficient, contradictory, missing, and outdated evidence explicitly.
- Keep workflows reproducible, debuggable, and auditable.

## 10. Requirement Tracking

- Update this file whenever development introduces or discovers a requirement, constraint, architecture decision, technical rule, feature expectation, deployment rule, or workflow preference.
- Do not rely on chat history as the record of a requirement.
- Add new requirements immediately and update changed requirements in place instead of duplicating them.
- Record unclear requirements under **Pending Clarifications** until resolved.
- Keep entries concise, structured, and actionable.
- Before ending every coding session, check whether new requirements were introduced and update this file when necessary.
- Mention requirement changes in the session summary.
- End every coding session with a medium-detailed Conventional Commit message suggestion; do not commit automatically.

## 11. Python Dependency Management

- Record every successful Python package installation, uninstall, or upgrade in the applicable dependency file immediately.
- Update `requirements.txt` after every project virtual-environment `pip install`, uninstall, or upgrade.
- When dependencies are grouped, also update the relevant file, such as `requirements-dev.txt`.
- Keep dependency entries alphabetized where practical, free of duplicates, and pinned unless the project intentionally documents an unpinned policy.
- Remove dependencies that are no longer required.
- Verify that the application still runs after dependency changes.
- Report all dependency additions, removals, and upgrades in the session summary.

## 12. Project State Synchronization

- No implementation may leave repository documentation, dependency declarations, configuration examples, or operational instructions out of sync.
- Update this file whenever project requirements change.
- Update Python dependency files whenever packages are installed, uninstalled, or upgraded.
- Update `.env.example` whenever environment variables are added, removed, renamed, or materially changed.
- Update `README.md` whenever setup, usage, or architecture changes.
- Update architecture documentation after major structural changes.
- Update API documentation when endpoints or interfaces change.
- Update database migration documentation when schemas change.
- Update Docker and Compose files when deployment requirements change.
- Update configuration examples when configuration behavior changes.
- Update changelogs or development notes when significant features are completed.
- Before ending a coding session, review the change set and synchronize every affected repository artifact.

## 13. Model Agnosticism & Local AI Deployment

### Core Principle

Knowledge OS must remain **model-agnostic**. The platform shall never depend on a single LLM provider, vendor, or model family. Every AI component must be designed behind a common abstraction layer so that models can be replaced without requiring application-level changes.

### Local-First AI

All AI inference must execute on infrastructure owned and controlled by the organization.

No cloud-hosted inference providers (OpenAI, Anthropic, Google Gemini API, AWS Bedrock, Azure OpenAI, etc.) shall be required for any core platform functionality.

All prompts, retrieved documents, embeddings, intermediate reasoning, and generated responses must remain within the organization's internal infrastructure.

### Supported Model Families

Knowledge OS should support local deployment of open-weight models from multiple vendors, including but not limited to:

* Meta (Llama)
* Google (Gemma)
* Microsoft (Phi)
* Mistral AI
* Qwen
* DeepSeek

The architecture must not make assumptions that favor any individual model family.

### Model Abstraction Layer

Every LLM integration must communicate through a unified interface.

Changing the active model should require only a configuration change rather than modifications to business logic.

Future integrations should support multiple local inference engines, including:

* Ollama (development)
* vLLM (production)
* Additional local inference runtimes as required

### Development Policy

During notebook development and experimentation:

* Multiple models should be benchmarked against the same retrieval pipeline.
* Model-specific prompt engineering should be minimized.
* Retrieval quality should remain independent of the selected LLM.
* Any notebook should be executable with different supported models by changing configuration only.

### Production Philosophy

Knowledge OS is an AI platform, not an application tied to a specific language model.

Organizations should be free to choose the model that best satisfies their requirements for performance, hardware availability, licensing, security, multilingual capability, and cost, without requiring changes to the surrounding platform.

This principle must remain true throughout the lifetime of the project.

## 14. Reusable Experiment Architecture

- Keep notebooks as lightweight learning and orchestration layers.
- Put reusable ingestion, chunking, embedding, storage, retrieval, generation,
  benchmarking, and visualization code under `src/cial_knowledge_os`.
- Keep notebook cells short, inspectable, rerunnable, and free of large reusable
  function or class implementations.
- Reuse the same `src` APIs from future notebooks, evaluation code, and backend
  services.
- Keep local sample fixtures separate from ignored runtime and real-document data.

## Pending Clarifications

None currently recorded.
