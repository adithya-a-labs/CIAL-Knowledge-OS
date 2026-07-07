# CIAL Knowledge OS — Project Requirements

This file is the single source of truth for project requirements. Detailed
implementation guidance lives in `PROJECT_RULES.md` and
`NOTEBOOK_GUIDELINES.md`; the verified implementation status and roadmap live in
`CURRENT_STATE.md`. If those documents conflict with this file, this file takes
precedence.

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

- The completed Phase 1 and Phase 2 baselines are dense-only. Phase 3 hybrid
  retrieval and Phase 4 reranking/evidence selection are implemented but await
  full benchmark qualification.
- Retrieve and inspect evidence before generation.
- The implemented architecture uses hybrid vector and keyword/BM25 retrieval
  plus local reranking. Retrieval-time metadata filters remain planned.
- Improve retrieval quality before increasing model size or context length.
- Do not pass entire documents to the LLM.
- Keep prompts and context concise; track token usage and latency where possible.
- Generated answers must be grounded in retrieved evidence and include traceable citations.
- Citations must retain source file, page number when available, chunk ID, and relevant metadata.
- Return an explicit safe-failure response when evidence is missing, weak, outdated, or conflicting.

## 5. Document Metadata

Index relevant metadata, including:

- source filename
- absolute source path
- path relative to the configured knowledge root
- category from the first knowledge-root folder
- collection from the second knowledge-root folder, when present
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

### Canonical Knowledge Repository

- Configure the enterprise corpus through `KnowledgeOSConfig.knowledge_root`,
  which defaults to `project_root / "data" / "files"`.
- Recursively discover documents below the configured root; do not scatter
  hardcoded corpus paths through notebooks, runners, ingestion, or tests.
- Organize enterprise files by category and optional collection, for example
  `cybersecurity/nist/`, `aviation/icao/`, `engineering/electrical/`, `hr/`,
  and `legal/`.
- Route discovery, ingestion validation, dataset scanning, reporting, and future
  frontend settings through one Enterprise File Format Registry. Do not duplicate
  hardcoded format lists in notebooks, runners, loaders, or reports.
- Preserve legacy Phase 1--4 source metadata and export contracts while adding
  taxonomy and path metadata.
- Do not search `data/pdf/` during runtime ingestion. A missing or empty
  canonical root must remain empty rather than silently switching corpora.
- Provide a safe migration utility that copies to `data/files/legacy_pdf/` by
  default, supports `--dry-run`, and moves only with explicit `--move`.

### Enterprise File Format Registry

- Maintain one backend registry with these support statuses:
  `SUPPORTED_NOW`, `OCR_SUPPORTED`, `RECOGNIZED_FUTURE_SUPPORT`, and
  `UNSUPPORTED`.
- Process only `SUPPORTED_NOW` and `OCR_SUPPORTED` files. Recognized future
  formats must be logged, skipped, reported, and never extracted, chunked,
  embedded, or indexed until a parser is implemented.
- Currently processed document formats are PDF, DOCX, DOC, XLSX, XLS, CSV, PPTX,
  PPT, TXT, Markdown, HTML, JSON, XML, and YAML.
- PNG, JPG/JPEG, and TIFF are `OCR_SUPPORTED`; image text must pass through OCR
  before entering the existing chunking, embedding, and indexing pipeline.
- Recognized future categories are email and communication files, archives,
  source code, configuration and DevOps files, audio/video, and CAD/engineering
  formats.
- The registry must expose category name, category description, format label,
  extensions, support status, ingestion enablement, OCR requirement, user-facing
  message, and backend notes for reports and future upload/settings UI.
- Dataset readiness scans must report totals, extension distribution, category
  coverage, support status distribution, future-support examples, unsupported
  examples, skipped files, and sample filenames by extension.
- Phase 4 CSV, XLSX, HTML, metrics, and notebook diagnostics must include
  Enterprise File Format Readiness without breaking existing batch-answer
  outputs.

### OCR Subsystem

- OCR is a first-class ingestion component for supported image formats, not a
  generic text loader. Images must be validated, preprocessed, OCR-extracted,
  cleaned, then sent into the same chunking, embedding, and indexing pipeline as
  other extracted text.
- The ingestion pipeline must depend on an abstract OCR engine interface. The
  default backend is Tesseract through `pytesseract` and Pillow, configured by
  `ocr_enabled`, `ocr_engine`, `ocr_preprocessing`, and `ocr_language`.
- OCR preprocessing should include EXIF orientation correction, grayscale
  conversion, contrast enhancement, small-image resizing where useful, and
  denoising.
- OCR metadata must include engine, engine version, preprocessing steps,
  extraction time, extracted character and word counts, image dimensions, source
  format, and OCR status.
- OCR failures must log and skip the failed image without crashing the remaining
  ingestion run. Reports must summarize OCR success, failure, success rate,
  processing time, extracted text volume, and engine used.

## 6. Security and Data Privacy

- Enforce role- and department-aware access control during retrieval.
- Keep personal workspace documents isolated from organization-wide repositories.
- Do not upload CIAL documents or sensitive data to external services.
- External document processing requires explicit approval.
- Keep raw documents in controlled local storage.
- Use non-sensitive sample documents during early experimentation.
- Never generate synthetic sample documents implicitly during normal pipeline execution.
- Load an existing sample directory normally, but require an explicit configuration opt-in or setup utility to create demonstration fixtures.

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

## 13. Phase 2 Query and Context Construction

- Treat Phase 2 as completed and frozen.
- Preserve Notebook 01 and all Basic RAG APIs as a frozen Phase 1 milestone.
- Keep Phase 2 reusable logic under `src/cial_knowledge_os`; Notebook 02 only orchestrates experiments.
- Use configurable top-10 retrieval per query variant without changing Phase 1's top-3 default.
- Support inspectable original, deterministically rewritten, keyword-expanded,
  and domain-reformulated queries. AI/LLM-based rewrite is not part of the
  completed Phase 2 implementation.
- Merge multi-query retrieval evidence and deduplicate by `(source, page, chunk_id)` before citations, context formatting, or generation.
- Support configurable source-relative neighbor expansion, contiguous chunk merging, and bounded context compression.
- Preserve document, page, chunk ID, similarity score, and nested metadata through every retrieval and context stage.
- Use the explicit Phase 2 safe-failure response when indexed evidence is insufficient.
- Reuse the Phase 1 batch exporter for Phase 2 without changing Notebook 01 or removing existing CSV columns.
- Ensure every Phase 2 batch row runs the complete transformed-query, multi-query retrieval, context construction, generation, and citation workflow.
- Append query variants, retrieval-stage counts, final context sizes, semantic answer status, and a concise retrieval trace to Phase 2 CSV exports.
- Provide reusable pandas tables and matplotlib plots for query variants, retrieval comparisons, duplicate frequency, neighbor provenance, context-stage counts, final citation quality, and batch retrieval traces.
- Include source and page distributions, score diagnostics by query variant, context compression and section-balance views, batch answer-status counts, and per-question latency diagnostics.
- Generate Phase 2 diagnostics from real pipeline trace data; keep visualization logic out of Notebook 02 and avoid dashboard frameworks.
- Maintain extension boundaries for later hybrid retrieval, local reranking, and bounded agentic workflows without implementing them in Phase 2.

## 14. Model Agnosticism & Local AI Deployment

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

## 15. Reusable Experiment Architecture

- Keep notebooks as lightweight learning and orchestration layers.
- Put reusable ingestion, chunking, embedding, storage, retrieval, generation,
  benchmarking, and visualization code under `src/cial_knowledge_os`.
- Keep notebook cells short, inspectable, rerunnable, and free of large reusable
  function or class implementations.
- Reuse the same `src` APIs from future notebooks, evaluation code, and backend
  services.
- Keep local sample fixtures separate from ignored runtime and real-document data.

## 16. Batch Question-Answer Export

- Provide a reusable source API that accepts notebook-defined question lists and
  exports grounded answers, citations, retrieval scores, and timing metrics to CSV.
- Keep question iteration, failure isolation, metrics collection, directory
  creation, version numbering, and file writing out of notebooks.
- Store exports under the repository-local `outputs/batch_answers/` hierarchy and
  never overwrite an earlier version.
- Keep batch retrieval and generation offline, local-only, model-agnostic, and
  implemented through existing pipeline abstractions.
- Record a failed row and continue when an individual question cannot be answered.

## 17. Phase Isolation and Backward Compatibility

- Do not modify completed Phase 1 or Phase 2 notebooks.
- Treat Notebook 01, Notebook 02, and the Phase 2 automated-evaluation notebook
  as frozen, reproducible baselines.
- Add new phase capabilities through new notebooks and reusable modules.
- Keep existing notebooks runnable.
- Phase 3 may replace internal architecture when useful, but must preserve
  external contracts: **new architecture internally, same contracts
  externally**.

## 18. Configuration Policy

- Do not hardcode paths, model names, output folders, retrieval modes, token
  budgets, or artifact filenames in notebooks or business logic.
- Define operational choices in typed configuration or explicit API parameters.
- Validate configuration at system boundaries and serialize the effective
  configuration with each reproducible run.
- Keep defaults centralized and avoid duplicated hidden constants.

## 19. Phase 3 Implementation Contract

- Phase 3 must compare hybrid retrieval with the frozen Phase 2 dense baseline.
- Add BM25 lexical retrieval and Reciprocal Rank Fusion.
- Add tokenizer-aware context budgeting.
- Use `tiktoken` as the primary tokenizer for Phase 3 counting, truncation,
  budgeting, remaining-capacity calculation, evaluation, and reporting.
- Route every token operation through the shared token-management module; do not
  estimate tokens from characters or duplicate token logic.
- Keep the token codec injectable so a future model-specific tokenizer does not
  require pipeline, evaluation, or reporting changes.
- Add clickable citation exports and per-run CSV, XLSX, and standalone HTML
  reports.
- Add a `RunManager` that writes isolated run artifacts below
  `outputs/batch_answers/03_Hybrid_Retrieval/run_<timestamp>/`.
- Extend the existing `outputs/` hierarchy; do not add a top-level `artifacts/`
  directory.
- Keep Phase 3 itself free of reranking so it remains the reproducible hybrid
  comparison baseline; implement reranking only through the additive Phase 4
  pipeline.

## 20. Phase 4 Implementation Contract

- Apply a configurable local cross-encoder after Phase 3 RRF and before context
  construction. Keep loading lazy and cache-first; permit one-time developer
  staging while allowing enterprise deployments to prohibit downloads. Do not
  average raw dense, BM25, RRF, and reranker scores.
- Select the minimum strong evidence set using configurable maximum count,
  reranker threshold, source diversity, redundancy reduction, and token budget.
- Treat the reranker threshold as a confidence signal rather than a hard
  no-evidence boundary. Keep a configurable evidence floor and adaptive
  top-ranked fallback whenever non-empty usable candidates exist.
- Distinguish weak evidence from no evidence and label weak-evidence answers
  with caution instead of silently starving the generation context.
- Reuse Phase 3 token management, citations, evaluation interfaces, reporting
  schema, and `RunManager` rather than creating competing implementations.
- Record candidate, selected, and final-context tokens; reduction percentage;
  discarded chunks/reasons; evidence quality; latency; citations; answer; and
  artifact paths.
- Additive Phase 4 settings/reporting must surface enterprise file-format
  readiness and OCR processing summaries from the registry and OCR subsystem.
- Provide deterministic mock reranking and dependency injection so automated
  tests do not require real model weights or Ollama.
- Write compatible run bundles below
  `outputs/batch_answers/04_Reranking_and_Evidence_Selection/run_<timestamp>/`.
- Label Phase 4 as implemented but not benchmark-qualified until the unchanged
  frozen Phase 3 versus Phase 4 comparison is completed.
- Defer visual document understanding, multimodal retrieval, and contradiction
  detection to Phase 4.5; do not imply they are implemented.

## 21. Structured Logging and Failure Handling

- Use configurable structured logging for indexing, dense and BM25 retrieval,
  hybrid fusion, token budgeting, report generation, and evaluation.
- Keep intentional notebook display separate from pipeline logging.
- Fail actionably for empty lexical corpora, unavailable tokenizers, invalid
  configuration, missing benchmarks, corrupt documents, and token overflow.
- Isolate per-question failures in batch and evaluation workflows.

## 22. Retrieval Extensibility and Cache Reuse

- Depend on a small retriever contract rather than retrieval-mode conditionals
  spread across business logic.
- Add future retrieval methods through new implementations and composition.
- Reuse loaded documents, chunks, embeddings, dense indexes, and unchanged BM25
  token caches; do not recompute them for every query or sweep configuration.
- Preserve retriever-specific rank and score provenance after fusion.
- Persist corpus fingerprints and chunk counts in
  `data/indexes/document_manifest.json`.
- Skip unchanged documents, replace changed document points, remove deleted
  points, and expose additive indexing diagnostics.
- Rebuild BM25 from the complete post-update corpus whenever files change.
- Support explicit full rebuild and legacy full-processing configuration
  without changing retrieval, reranking, or evidence-selection semantics.

## Pending Clarifications

None currently recorded.
