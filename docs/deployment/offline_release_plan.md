# Offline Release Bundle Plan

The final CIAL deployment should be delivered as a versioned, checksum-verified
offline bundle. The source repository remains the development and benchmark
workspace.

Each approved bundle should contain:

- a source or wheel build pinned to a release identifier;
- an offline Python wheelhouse and locked dependency manifest;
- approved embedding, reranker, tokenizer, and Ollama model artifacts;
- Docker image archives and Compose configuration for local services;
- configuration templates with server Qdrant remaining opt-in;
- migration, preflight, health, backup, and restore utilities;
- release notes, checksums, licenses, and rollback instructions.

The installation procedure should load container images without a registry,
install Python packages with `--no-index`, stage models on organization-managed
storage, run preflight, restore or rebuild indexes, and execute an offline smoke
suite. Runtime egress should be disabled and verified.

Document corpora and Qdrant backups are deployment data, not release-source
artifacts. They should be transferred and retained under CIAL data-governance
controls. `data/files/` is the ingestion root after deployment.

Release acceptance should record bundle checksums, host/GPU requirements,
Qdrant collection health, model identities, test results, backup compatibility,
and the exact rollback version. This avoids treating an unpinned repository
clone as an enterprise release.
