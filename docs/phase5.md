# Phase 5 — Adaptive Agentic Response Planning

Phase 5 is an optional response-planning and validation layer over the stable
Phase 4 evidence engine. It does not retrieve, rerank, fuse, select, or expand
evidence. Phase 4 remains responsible for those deterministic operations.

## Execution model

The pipeline calls Phase 4 once, converts its response into a typed
`AgentState`, and runs query analysis, response planning, prompt composition,
drafting, criticism, compliance review, risk review, evidence verification,
and deterministic consensus. Consensus can accept, reject, or permit one
revision. Unsupported and insufficient-evidence statuses are preserved.

Agents must not retrieve new material, invent evidence, bypass Phase 4 answer
statuses, use cloud APIs, or create unbounded revision loops.

## Multimodal evidence

`Evidence` accepts text, tables, figures, images, screenshots, diagrams, OCR,
and scanned regions. Image paths, OCR text, captions, bounding boxes, scores,
and arbitrary modality metadata are optional. Existing Phase 4 text chunks are
adapted automatically and require no migration.

Visual verification is optional. When visual evidence and an assigned
vision-capable local model are both present, `EvidenceVerifier` may inspect the
images. With no visual evidence its status is `not_applicable`; with visual
evidence but no configured vision model it is `not_configured`.

## Local model configuration

```yaml
phase5:
  enabled: true
  max_revision_loops: 1
  model_profiles:
    fast_text:
      provider: ollama
      model: qwen2.5:7b-instruct
      capabilities: [text, structured_json]
    reasoning_text:
      provider: ollama
      model: phi4:14b
      capabilities: [text, structured_json]
    vision_local:
      provider: ollama
      model: llava:13b
      capabilities: [vision, text]
  agents:
    query_analyzer: {model_profile: fast_text}
    response_planner: {model_profile: fast_text}
    draft_generator: {model_profile: reasoning_text}
    critic_agent: {model_profile: reasoning_text}
    compliance_agent: {model_profile: fast_text}
    risk_agent: {model_profile: reasoning_text}
    evidence_verifier:
      model_profile: fast_text
      vision_model_profile: vision_local
```

The older per-agent `model`, `fallback_model`, and `temperature` shape is also
accepted and translated into model profiles.

All model clients are injected into `ModelRouter`. This supports deterministic
tests and keeps inference local. No cloud client is provided.

## Decision Intelligence Dashboard

Phase 5 HTML is self-contained and works offline. Each answer includes a
computed recommendation, readiness score, evidence strength, verification,
risk matrix, latency, consensus flow, critic findings, citation coverage,
source diversity, and modality mix. Charts use HTML and CSS only. Theme
selection supports light, dark, and system modes and persists in local storage.

The dashboard is diagnostic decision support, not a certification. In
particular, citation presence is not proof that a claim is semantically
entailed. Production use should calibrate thresholds against CIAL benchmarks
and retain human review for aviation safety, security, and compliance actions.

## Use

```python
from cial_knowledge_os import ModelRouter, Phase5Pipeline

router = ModelRouter(config, clients={"ollama": local_client})
pipeline = Phase5Pipeline(
    phase4_pipeline=phase4_pipeline,
    config=config,
    model_router=router,
)
result = pipeline.answer("What controls should be prioritized?")
```

Set `phase5.enabled` to `false`, or omit it, to return the Phase 4 response
unchanged.

## Optional live command center

Phase 5 can publish execution events to a fully local FastAPI/SSE dashboard.
This observer is not required for batch or notebook execution. See
[Phase 5 Live Command Center](phase5-live-command-center.md) for startup,
metrics, GPU limitations, and privacy guarantees.
