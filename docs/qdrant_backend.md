# Qdrant Backend Operations

CIAL Knowledge OS supports two fully local Qdrant modes. `embedded` remains the
default for frozen notebooks, development, and small demos. `server` is opt-in
and recommended for large corpora or concurrent processes. Neither mode uses
Qdrant Cloud.

## Start and inspect server mode

From the repository root:

```powershell
docker compose -f docker-compose.qdrant.yml up -d
docker compose -f docker-compose.qdrant.yml ps
curl.exe http://localhost:6333/healthz
python scripts/check_qdrant_health.py --url http://localhost:6333 --collection cial_phase4
```

The Compose file uses the Docker named volume `cial_qdrant_storage`. This avoids
Windows bind-mount rename and permission failures during Qdrant segment
optimization.

Enable server mode explicitly:

```python
config = Phase4Config(
    qdrant_mode="server",
    qdrant_url="http://localhost:6333",
    qdrant_batch_size=32,
    qdrant_upsert_wait=True,
)
```

Run non-throwing deployment preflight before a large local job:

```python
from cial_knowledge_os.config import Phase4Config
from cial_knowledge_os.infra import run_preflight

report = run_preflight(
    Phase4Config(qdrant_mode="server"),
    embedding_dimension=1024,
    generation_enabled=True,
)
print(report)
```

The report contains `passed`, `warnings`, `errors`, and per-component `checks`.

## First backend switch

The document manifest can refer to vectors held in the previous backend. After
switching from embedded to server mode, either migrate the collection with
`scripts/migrate_embedded_qdrant_to_server.py` or set
`FORCE_REBUILD_INDEX=True` for one complete server rebuild. Set it back to
`False` after that successful run; leaving it enabled recreates the collection
on every run.

## Stop and restart

```powershell
docker compose -f docker-compose.qdrant.yml down
docker compose -f docker-compose.qdrant.yml up -d
```

`down` does not delete the named volume. Do not add `--volumes` unless permanent
deletion is explicitly intended.

## Troubleshooting

- `MemoryError` during upsert: retain `QDRANT_BATCH_SIZE=32` in server mode,
  close memory-heavy processes, and confirm available RAM. Reducing the value
  further lowers serialization memory.
- Red optimizer: retrieval may continue, but storage must be repaired before
  production use. Run the health script, inspect `docker logs cial-qdrant`,
  back up the volume, and rebuild or restore it.
- Collection missing after a backend switch: perform the one-time rebuild or
  migration described above. The pipeline intentionally refuses to trust an
  unchanged manifest when the active collection is absent.
- Dimension mismatch: use the embedding model that created the collection, or
  rebuild into a new/recreated collection.
- Unreachable server: run `docker compose -f docker-compose.qdrant.yml ps` and
  verify `http://localhost:6333/healthz`.
