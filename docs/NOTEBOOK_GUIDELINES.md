# Notebook Guidelines

Each notebook must follow this structure:

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
