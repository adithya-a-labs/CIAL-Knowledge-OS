# Local Docker Deployment Foundation

This repository is a development and evaluation workspace. Docker Compose
currently starts the local Qdrant dependency; it is the foundation for a later
versioned, offline application bundle rather than a complete production stack.

Enterprise documents are ingested recursively from `data/files/`. Embedded
Qdrant remains supported for notebooks and small demonstrations. For large
corpora and concurrent processes, opt into server Qdrant.

## Qdrant lifecycle

```powershell
scripts\start_qdrant.bat
python scripts/check_qdrant_health.py --url http://localhost:6333 --collection cial_phase4
scripts\stop_qdrant.bat
```

Equivalent commands:

```powershell
docker compose -f docker-compose.qdrant.yml up -d
docker compose -f docker-compose.qdrant.yml ps
docker compose -f docker-compose.qdrant.yml down
```

The service stores data in the named volume `cial_qdrant_storage`. Compose
`down` preserves it.

EOF runs in the Python process and does not require another container or
network service. Run traces and progress snapshots remain on the project host
under `outputs/runs/<run_id>/`. If the Python runner is later containerized,
mount that directory to persistent local storage; no telemetry endpoint or
external dashboard should be configured.

## Backup and restore the named volume

Stop Qdrant before taking a filesystem-level backup:

```powershell
docker compose -f docker-compose.qdrant.yml down
docker run --rm -v cial_qdrant_storage:/source -v "${PWD}:/backup" alpine tar czf /backup/cial_qdrant_storage.tar.gz -C /source .
```

Restore only into an empty replacement volume:

```powershell
docker volume create cial_qdrant_storage
docker run --rm -v cial_qdrant_storage:/target -v "${PWD}:/backup" alpine sh -c "cd /target && tar xzf /backup/cial_qdrant_storage.tar.gz"
docker compose -f docker-compose.qdrant.yml up -d
```

Validate the restored collection with the health script before enabling
ingestion. Volume names may be Compose-project-prefixed; confirm the effective
name with `docker volume ls` and `docker compose config`.

Cloning the Git repository is not the final enterprise deployment method. A
clone does not pin approved model artifacts, wheel dependencies, configuration,
document data, secrets, or database backups as one auditable unit. Production
deployment should consume a signed, versioned offline release bundle.
