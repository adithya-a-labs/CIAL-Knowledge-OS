# Phase 5 Live Command Center

The Phase 5 Agentic Command Center is an optional local web dashboard for
watching one or more Phase 5 runs. It observes structured pipeline events; it
does not participate in retrieval, generation, validation, or consensus.
Normal Phase 4 and Phase 5 execution does not import or require FastAPI.

## Start the command center

Activate the project environment and run:

```powershell
python -m cial_knowledge_os.live.command_center
```

The server binds to `127.0.0.1` and prints:

```text
Phase 5 Agentic Command Center: http://127.0.0.1:8765
```

The equivalent Phase 5 runner command is:

```powershell
python -m cial_knowledge_os.orchestration.phase5_runner --live
```

These commands start a standalone dashboard. To connect an actual batch run,
give the pipeline an `EventBus`, or let `Phase5Runner` create one:

```python
from cial_knowledge_os import Phase5Pipeline, Phase5Runner
from cial_knowledge_os.live import EventBus

bus = EventBus()
pipeline = Phase5Pipeline(
    phase4_pipeline=phase4_pipeline,
    config=config,
    model_router=model_router,
    event_bus=bus,
)
runner = Phase5Runner(pipeline, output_dir)
runner.run(questions, live=True)
```

`live=True` starts the server in a daemon thread and prints the local URL.
Leaving it at the default `False` follows the existing batch path.

## Dashboard panels

- **Run status:** run ID, question, active stage/agent, answer status, elapsed
  time, progress, and revision state.
- **Pipeline and agents:** pending, running, completed, failed, or skipped
  states, model, fallback use, latency, warnings, and errors.
- **Evidence:** selected count, score-based sufficiency diagnostic, distinct
  sources, and modality distribution.
- **Answer quality:** verification, unsupported claims, citation mismatches,
  critic findings, compliance, risk, consensus, revision, and final status.
- **Device telemetry:** CPU, RAM, disk, process memory, optional GPU/VRAM,
  current model, model latency, generated-token estimate, and throughput.
- **Live log and answer:** timestamped event stream, draft/final answer, and
  citations.

Evidence sufficiency and readiness are operational diagnostics, not calibrated
certifications. Verification rate is the proportion of claims accepted by the
configured verifier. Token counts are shown only when the local model adapter
reports them.

## GPU telemetry

The collector invokes the local `nvidia-smi` executable with a two-second
timeout. If the executable, driver, or NVIDIA GPU is absent, GPU and VRAM are
shown as unavailable while CPU, memory, disk, and process telemetry continue.
Multi-GPU values are aggregated and individual device names remain available.
No GPU management APIs are called.

## Privacy and offline guarantees

- The server binds to loopback by default.
- Events are held in bounded process memory.
- Browser assets are packaged with the project.
- Server-Sent Events stay between the Python process and local browser.
- No CDN, analytics, cloud telemetry, or external service is used.
- The dashboard does not write telemetry to disk.

Binding to another host exposes operational data to that network interface and
should be an explicit deployment decision.

## Missing optional dependencies

Install `requirements.txt` if live mode reports that FastAPI, Uvicorn, or
psutil is unavailable. Non-live Phase 5 workflows remain usable without the
dashboard server.
