# ScyllaDB End-to-End Workload

Spins up a resource-bounded, disposable ScyllaDB container and runs a real
put/get/delete workload against `ScyllaDBBackend`, reporting results straight
from LMCache's own Prometheus metrics rather than ad hoc timers.

## What It Does

1. `launch_scylla_e2e.sh` tears down any previous run and starts a fresh
   ScyllaDB container (RAM-disk-backed, capped CPU/memory).
2. `scylla_e2e_demo.py` connects a real `ScyllaDBBackend`, adds a batch of KV
   cache chunks, serves them back (verifying data integrity), deletes a
   subset, and prints a metrics summary pulled from LMCache's own
   `PrometheusLogger`/`LMCStatsMonitor` (read/write bytes and request counts,
   `get_blocking_failed_count`, `put_failed_count`, per-chunk get latency
   histogram).

## Usage

```bash
benchmarks/scylla_e2e/launch_scylla_e2e.sh
#   optional args: [container-name] [host-port] [scylla-mem-mb]
#   e.g. benchmarks/scylla_e2e/launch_scylla_e2e.sh lmcache-scylla-e2e 9042 3500

SCYLLA_HOST=127.0.0.1 SCYLLA_PORT=9042 \
    python benchmarks/scylla_e2e/scylla_e2e_demo.py
```

`SCYLLA_E2E_MODE` (default `both`) selects `put`, `get`, or `both` --
`put` writes and exits, `get` reads back a *previously written* batch
(same `SCYLLA_E2E_NUM_CHUNKS`, deterministic content regenerated from each
chunk's index, no state shared between the two processes), `both` does
both in one process (see the module docstring for the full env var
reference, including `SCYLLA_E2E_NUM_CHUNKS`/`SCYLLA_E2E_CONCURRENCY`).

Edit the constants at the top of `scylla_e2e_demo.py` (`NUM_LAYERS`,
`NUM_TOKENS`, `HIDDEN_DIM`) to try different shapes.
