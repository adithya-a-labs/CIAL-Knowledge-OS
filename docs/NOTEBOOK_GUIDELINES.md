# Notebook Guidelines

## Standard Notebook Structure

Every notebook in the `notebooks/` directory must follow this fixed engineering structure:

1. Objective
2. Theory
3. Architecture
4. Implementation
5. Visualization
6. Benchmark
7. Advantages
8. Limitations
9. Enterprise Considerations
10. What we'll improve in the next notebook

Each notebook must be treated as both an experiment and an engineering design document. The structure above is mandatory for notebooks under `notebooks/` and should remain local-first, offline-friendly, and model-agnostic. Do not introduce cloud-based assumptions, hosted inference requirements, or vendor-specific notebook flows.

Use the sections as follows:

- **Objective** should clearly state what RAG capability is being tested.
- **Theory** should explain the underlying concept before code is written.
- **Architecture** should show where the technique fits into the Knowledge OS pipeline.
- **Implementation** should contain clean, reproducible code that can run with local components.
- **Visualization** should make the pipeline or outputs easier to understand.
- **Benchmark** should measure latency, retrieval quality, model behavior, or resource usage where applicable.
- **Advantages** and **Limitations** should honestly document tradeoffs.
- **Enterprise Considerations** should explain relevance to a fully local, on-prem CIAL deployment.
- **What we'll improve in the next notebook** should explain how the next notebook improves the current one.

Each notebook must also preserve the following experiment checklist within that structure:

## 1. Objective

Explain what this notebook tests.

## 2. Setup

Imports, paths, model configuration, and local runtime notes.

## 3. Data Loading

Load only small test documents first.

## 4. Processing

Cleaning, chunking, metadata extraction.

## 5. Retrieval Experiment

Show retrieved chunks, scores, and metadata.

## 6. Generation Experiment

Generate answer only after retrieval is inspected.

## 7. Evaluation

Check:
- relevance
- correctness
- hallucination risk
- citation quality
- token usage
- latency

## 8. Observations

Write what worked and what failed.

## 9. Next Steps

Decide whether the technique should move into the final pipeline.

## Model Independence Reminder

- All notebooks must remain model-agnostic.
- Never hardcode prompts, APIs, or logic around a single model family.
- Every notebook should be executable with any supported local model by changing configuration only.
- Notebook experiments should compare multiple models whenever practical.

General rules:

- Prefer small, understandable experiments.
- Never hide intermediate outputs.
- Do not use cloud APIs.
- Do not use real sensitive CIAL documents during early testing.
- Keep experiments reproducible.
- Keep prompts short.
- Track token usage where possible.
- Prefer local OSS models.
- Prefer local embeddings.
- Prefer citation-backed answers.
