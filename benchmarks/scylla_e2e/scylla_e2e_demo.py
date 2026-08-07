# SPDX-License-Identifier: Apache-2.0
"""
End-to-end benchmark for ScyllaDBBackend against a real ScyllaDB instance.

Runs a put and/or get phase, reporting throughput and latency percentiles
for each phase independently, from LMCache's own Prometheus metrics.

Usage:
    benchmarks/scylla_e2e/launch_scylla_e2e.sh
    SCYLLA_HOST=127.0.0.1 SCYLLA_PORT=9042 \
        python benchmarks/scylla_e2e/scylla_e2e_demo.py

Environment variables:
    SCYLLA_E2E_MODE (default "both"):
        put  -- write SCYLLA_E2E_NUM_CHUNKS chunks and exit. Data persists
                in ScyllaDB (subject to ttl_seconds, currently 600s) for a
                later, separate "get" run to read back.
        get  -- read back SCYLLA_E2E_NUM_CHUNKS chunks previously written
                by a "put" run using the *same* SCYLLA_E2E_NUM_CHUNKS
                value, and verify their contents -- regenerated
                deterministically from each chunk's index, so no state
                needs to be shared between the two processes.
        both -- put then immediately get, in one process (the original
                combined round-trip behavior).
        churn -- repeatedly put, get, and delete a rolling batch of
                SCYLLA_E2E_CHURN_BATCH_SIZE chunks, for
                SCYLLA_E2E_CHURN_CYCLES cycles. Each cycle deletes what it
                wrote, keeping the *logical* dataset small, but CQL
                deletes are tombstones -- physical space is only reclaimed
                by compaction, later -- so cumulative disk usage over many
                cycles is NOT bounded by one batch (confirmed: a 100x30
                run filled the tmpfs and started failing writes around
                cycle 15). Still subject to the tmpfs ceiling below;
                pick batch_size/cycles with that in mind.
    SCYLLA_E2E_NUM_CHUNKS (default 200): total chunks/samples over the
        whole run. Each chunk is ~8.4MB, and while this no longer drives
        *this script's* peak memory (see SCYLLA_E2E_CONCURRENCY), it still
        drives ScyllaDB's own storage footprint: all num_chunks stay
        written until ttl_seconds expiry (or "both" mode's delete phase),
        regardless of concurrency. launch_scylla_e2e.sh's container has an
        8GB tmpfs -- raising this too far risks "No space left on device"
        (measured on a 4GB tmpfs: 500 resident chunks, ~4.2GB raw before
        SSTable/compaction overhead, exhausted it and degraded the
        cluster mid-run; scale expectations accordingly for the 8GB tmpfs).
    SCYLLA_E2E_CONCURRENCY (default 10): how many chunks are in flight at
        once, processed in successive waves until SCYLLA_E2E_NUM_CHUNKS is
        exhausted. Each wave's results are verified and released before
        the next one starts, so peak memory is bounded by this value, not
        by the total chunk count -- see _local_cpu_size_gb(). All waves
        feed the same Prometheus histograms, so percentiles reported at
        the end reflect every sample across every wave.

        This also dominates measured latency: each whole-chunk put/get
        already fans out into NUM_LAYERS internal concurrent CQL ops, so
        concurrency=N means up to N * NUM_LAYERS CQL ops in flight against
        the cluster at once. On launch_scylla_e2e.sh's 4-shard container,
        latency scales roughly linearly with this value once it exceeds
        the cluster's real capacity (measured: ~180ms p50 get at
        concurrency=1 vs ~9.8s at concurrency=100, on identical data) --
        so a high value here measures queueing delay under self-inflicted
        overload, not a representative per-request latency.
    SCYLLA_E2E_CHURN_BATCH_SIZE (default 100): chunks per cycle in "churn"
        mode. Ignored otherwise.
    SCYLLA_E2E_CHURN_CYCLES (default 20): number of put/get/delete cycles
        in "churn" mode. Ignored otherwise.
"""

# Standard
from typing import NamedTuple
import asyncio
import os
import re
import threading
import time

# Third Party
from prometheus_client import REGISTRY, generate_latest
import torch

# First Party
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.scylla_backend import ScyllaDBBackend

NUM_LAYERS = 32
NUM_HEADS = 32
HEAD_SIZE = 128
HIDDEN_DIM = NUM_HEADS * HEAD_SIZE  # 4096
NUM_TOKENS = 16
CHUNK_SHAPE = torch.Size([2, NUM_LAYERS, NUM_TOKENS, HIDDEN_DIM])
CHUNK_BYTES = CHUNK_SHAPE.numel() * 2  # bfloat16 = 2 bytes/element
DELETE_FRACTION = 0.4
MODEL_NAME = "e2e-demo-model"
VALID_MODES = ("put", "get", "both", "churn")


class BenchSession(NamedTuple):
    backend: ScyllaDBBackend
    local_cpu: LocalCPUBackend
    loop: asyncio.AbstractEventLoop
    loop_thread: threading.Thread
    metadata: LMCacheMetadata
    config: LMCacheEngineConfig


def make_metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name=MODEL_NAME,
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(NUM_LAYERS, 2, NUM_TOKENS, NUM_HEADS, HEAD_SIZE),
        use_mla=False,
        role="worker",
    )


_CONTENT_POOL_SIZE = 8
_content_pool: list[torch.Tensor] = []


def _pool_tensor(seed: int) -> torch.Tensor:
    """Return one of _CONTENT_POOL_SIZE pre-generated content patterns.

    Lazily built once per process, from the same fixed seeds (0..
    _CONTENT_POOL_SIZE-1) every time -- so a later, separate "get" process
    still needs no shared state with the "put" process that wrote the
    data, preserving make_chunk/expected_tensor's existing cross-process
    contract. Regenerating ~4M random floats (torch.Generator().manual_seed
    + uniform_) per *chunk* -- as opposed to once per pool entry -- was
    ~14ms of GIL-bound CPU work competing with the event loop/driver
    reactor threads, and dominated reported p99 (confirmed: removing it
    dropped measured get p99 by ~10x). Distinct chunk_hash values sharing
    a content pattern is fine for verification; a small pool (not 1) still
    catches a real content mixup, since most chunks differ from their
    neighbors.
    """
    if not _content_pool:
        for s in range(_CONTENT_POOL_SIZE):
            generator = torch.Generator().manual_seed(s)
            _content_pool.append(
                torch.empty(CHUNK_SHAPE, dtype=torch.bfloat16).uniform_(
                    -1, 1, generator=generator
                )
            )
    return _content_pool[seed % _CONTENT_POOL_SIZE]


def make_chunk(seed: int) -> MemoryObj:
    """Allocate one chunk, filled by copying a pooled content pattern.

    See :func:`_pool_tensor`. Still a fresh MemoryObj per call (each is
    used and released once by the caller), just a cheap copy instead of
    regenerating random content.
    """
    allocator = AdHocMemoryAllocator(device="cpu")
    obj = allocator.allocate([CHUNK_SHAPE], [torch.bfloat16], fmt=MemoryFormat.KV_T2D)
    if obj is None or obj.tensor is None:
        raise RuntimeError(f"failed to allocate chunk for seed {seed}")
    obj.tensor.copy_(_pool_tensor(seed))
    return obj


def expected_tensor(seed: int) -> torch.Tensor:
    """Return the exact tensor :func:`make_chunk` would write for *seed*."""
    return _pool_tensor(seed)


def _local_cpu_size_gb(mode: str, concurrency: int) -> float:
    """Size the LocalCPUBackend pinned pool for this mode/concurrency.

    Pinned host memory is first-touched/zero-filled at *construction*
    time regardless of subsequent use (the same eager-allocation cost
    documented for tests/conftest.py's memory_allocator fixture, issue
    #4295), so this should be sized to what each mode actually needs, not
    a fixed "safe" number. "get"/"both" modes hold up to *concurrency*
    results in this pool simultaneously -- one wave's worth
    (batched_get_blocking returns a whole wave before do_get starts
    releasing it) -- regardless of the total chunk count, since later
    waves reuse the same pool space; "put"-only mode never touches this
    pool at all (puts are backed by their own AdHocMemoryAllocator chunk).
    """
    if mode == "put":
        return 0.1
    needed_gb = (concurrency * CHUNK_BYTES) / 1e9
    return needed_gb + 0.1


def connect(host: str, port: int, mode: str, concurrency: int) -> BenchSession:
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    local_cpu = LocalCPUBackend(
        LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            max_local_cpu_size=_local_cpu_size_gb(mode, concurrency),
        ),
        make_metadata(),
        dst_device="cpu",
    )

    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        extra_config={
            "scylla": {
                "contact_points": [host],
                "port": port,
                "keyspace": "lmcache_e2e_demo",
                "local_dc": "datacenter1",
                "ttl_seconds": 600,
                "table_compression": os.environ.get("SCYLLA_E2E_COMPRESSION", ""),
                "wire_compression": os.environ.get(
                    "SCYLLA_E2E_WIRE_COMPRESSION", "false"
                ).lower()
                == "true",
            }
        },
    )
    # Reused below to look up the same PrometheusLogger instance ScyllaDBBackend
    # itself registered with (PrometheusLogger.GetOrCreate keys off metadata).
    metadata = make_metadata()

    backend = ScyllaDBBackend(
        dst_device="cpu",
        config=config,
        metadata=metadata,
        local_cpu_backend=local_cpu,
        loop=loop,
    )
    return BenchSession(backend, local_cpu, loop, loop_thread, metadata, config)


def close(session: BenchSession) -> None:
    session.backend.close()
    session.local_cpu.close()
    session.loop.call_soon_threadsafe(session.loop.stop)
    session.loop_thread.join(timeout=5.0)


def _print_latency_stats(metric_name: str, label: str) -> None:
    """Print sample count / mean / P50 / P99 for one Prometheus histogram.

    Scrapes the process-global registry (the caller must have already
    triggered a stats push via ``PrometheusLogger.log_prometheus``).
    Percentiles are interpolated from the histogram's bucket boundaries;
    if some samples fall past the highest finite bucket, that percentile
    is reported as ">"  the highest bucket rather than a falsely precise
    number.
    """
    metrics_text = generate_latest(REGISTRY).decode()
    buckets: list[tuple[float, float]] = []
    count: float | None = None
    total_sum: float | None = None
    for line in metrics_text.splitlines():
        if line.startswith("#") or metric_name not in line:
            continue
        if f"{metric_name}_bucket" in line:
            m = re.search(r'le="([^"]+)"[^}]*}\s+([\d.]+)', line)
            if m and m.group(1) != "+Inf":
                buckets.append((float(m.group(1)), float(m.group(2))))
        elif f"{metric_name}_count" in line:
            count = float(line.rsplit(" ", 1)[-1])
        elif f"{metric_name}_sum" in line:
            total_sum = float(line.rsplit(" ", 1)[-1])

    if not count or total_sum is None:
        print(f"  {label}: no samples recorded")
        return

    buckets.sort()

    def quantile(q: float) -> str:
        target = q * count
        prev_le, prev_cnt = 0.0, 0.0
        for le, cnt in buckets:
            if cnt >= target:
                if cnt == prev_cnt:
                    return f"{le:.0f}"
                frac = (target - prev_cnt) / (cnt - prev_cnt)
                return f"{prev_le + frac * (le - prev_le):.0f}"
            prev_le, prev_cnt = le, cnt
        return f">{buckets[-1][0]:.0f}" if buckets else "n/a"

    mean = total_sum / count
    print(
        f"  {label}: n={int(count)} mean={mean:.1f}ms "
        f"p50={quantile(0.50)}ms p99={quantile(0.99)}ms"
    )


def _flush_and_print_stats(session: BenchSession, metric_name: str, label: str) -> None:
    # scylla_backend.py's _setup_metrics() wires get_blocking_failed_count/
    # put_failed_count/remote_put_task_num as live-valued Prometheus gauges
    # (set_function callbacks, evaluated at scrape time -- no push needed).
    # Byte/latency counters (num_remote_read_bytes, remote_time_to_put, ...)
    # instead flow through LMCStatsMonitor and only reach their gauges when
    # something calls PrometheusLogger.log_prometheus(); in the full engine
    # that's driven by a periodic LMCacheStatsLogger thread this standalone
    # script never starts, so trigger one push manually before scraping.
    stats_monitor = LMCStatsMonitor.GetOrCreate()
    prometheus_logger = PrometheusLogger.GetOrCreate(
        session.metadata, config=session.config
    )
    prometheus_logger.log_prometheus(stats_monitor.get_stats_and_clear())
    _print_latency_stats(metric_name, label)


def _waves(keys: list[CacheEngineKey], concurrency: int) -> list[list[CacheEngineKey]]:
    return [keys[i : i + concurrency] for i in range(0, len(keys), concurrency)]


def _put_chunks(
    session: BenchSession,
    keys: list[CacheEngineKey],
    concurrency: int,
    progress: bool = True,
) -> None:
    """Put *keys* wave by wave. Core of :func:`do_put`, minus its
    throughput print/stats flush -- reused per-cycle by :func:`do_churn`.
    """
    for wave_num, wave_keys in enumerate(_waves(keys, concurrency), start=1):
        chunks = [make_chunk(key.chunk_hash) for key in wave_keys]
        session.backend.batched_submit_put_task(wave_keys, chunks)
        deadline = time.monotonic() + 60.0
        for key in wave_keys:
            while session.backend.exists_in_put_tasks(key):
                if time.monotonic() > deadline:
                    raise TimeoutError(f"put of {key} did not complete in time")
                time.sleep(0.02)
        # All puts in this wave have completed (the wait above confirms it),
        # so the backend's own hold on `chunks` (ref_count_up/down in
        # batched_submit_put_task's completion callback) is already
        # released; drop this script's hold too before the next wave.
        for chunk in chunks:
            chunk.ref_count_down()
        if progress:
            print(f"  wave {wave_num}: {len(wave_keys)} chunks stored", flush=True)


def do_put(
    session: BenchSession,
    keys: list[CacheEngineKey],
    concurrency: int,
    total_mb: float,
) -> None:
    num_chunks = len(keys)
    print(
        f"=== Adding {num_chunks} chunks ({CHUNK_BYTES / 1e6:.1f} MB each, "
        f"{total_mb:.1f} MB total, concurrency={concurrency}) ==="
    )
    t0 = time.perf_counter()
    _put_chunks(session, keys, concurrency)
    put_elapsed = time.perf_counter() - t0

    print(
        f"Stored {num_chunks} chunks in {put_elapsed:.2f}s "
        f"({total_mb / put_elapsed:.1f} MB/s)"
    )
    _flush_and_print_stats(session, "lmcache:remote_time_to_put", "Put latency")


def _get_chunks(
    session: BenchSession,
    keys: list[CacheEngineKey],
    concurrency: int,
    progress: bool = True,
) -> tuple[int, int]:
    """Get *keys* wave by wave, verifying against :func:`expected_tensor`.
    Core of :func:`do_get`, minus its throughput print/stats flush --
    reused per-cycle by :func:`do_churn`.

    :return: (ok_count, mismatch_count) across every wave.
    """
    ok = 0
    mismatch = 0
    for wave_num, wave_keys in enumerate(_waves(keys, concurrency), start=1):
        results = session.backend.batched_get_blocking(wave_keys)
        for key, r in zip(wave_keys, results, strict=True):
            orig = expected_tensor(key.chunk_hash)
            if r is not None and r.tensor is not None and torch.equal(r.tensor, orig):
                ok += 1
            else:
                mismatch += 1
                print(f"  MISMATCH or MISS for chunk {key.chunk_hash}")
            if r is not None:
                r.ref_count_down()
        if progress:
            print(f"  wave {wave_num}: {len(wave_keys)} chunks served", flush=True)
    return ok, mismatch


def do_get(
    session: BenchSession,
    keys: list[CacheEngineKey],
    concurrency: int,
    total_mb: float,
) -> int:
    num_chunks = len(keys)
    print(f"=== Serving {num_chunks} chunks back (concurrency={concurrency}) ===")
    t0 = time.perf_counter()
    ok, _mismatch = _get_chunks(session, keys, concurrency)
    get_elapsed = time.perf_counter() - t0

    print(
        f"Retrieved {ok}/{num_chunks} chunks correctly in {get_elapsed:.2f}s "
        f"({total_mb / get_elapsed:.1f} MB/s)"
    )
    _flush_and_print_stats(session, "lmcache:remote_time_to_get_sync", "Get latency")
    return ok


def do_delete(session: BenchSession, keys: list[CacheEngineKey]) -> tuple[int, int]:
    num_to_delete = max(1, int(len(keys) * DELETE_FRACTION))
    print(f"=== Deleting {num_to_delete} of {len(keys)} chunks ===")
    to_delete = keys[:num_to_delete]
    t0 = time.perf_counter()
    for key in to_delete:
        session.backend.remove(key)
    del_elapsed = time.perf_counter() - t0
    still_present = sum(1 for key in to_delete if session.backend.contains(key))
    remaining_present = sum(
        1 for key in keys[num_to_delete:] if session.backend.contains(key)
    )
    print(f"Deleted {num_to_delete} chunks in {del_elapsed:.2f}s")
    print(f"  Deleted chunks still present: {still_present} (expect 0)")
    print(
        f"  Remaining {len(keys) - num_to_delete} chunks still present: "
        f"{remaining_present} (expect {len(keys) - num_to_delete})"
    )
    return still_present, remaining_present


def do_churn(
    session: BenchSession, batch_size: int, cycles: int, concurrency: int
) -> tuple[int, int]:
    """Put, get, and delete a rolling batch of chunks for *cycles* cycles.

    Each cycle uses fresh chunk_hash values and deletes them before the
    next starts, bounding the *logical* dataset to one batch -- but not
    disk usage (deletes are tombstones, reclaimed later by compaction);
    see the module docstring's "churn" mode.

    :return: (total_ok, total_mismatch) across every cycle's get phase.
    """
    total_chunks = batch_size * cycles
    print(
        f"=== Churning {cycles} cycles x {batch_size} chunks "
        f"({total_chunks} total, concurrency={concurrency}) ==="
    )
    total_ok = 0
    total_mismatch = 0
    t0 = time.perf_counter()
    for cycle in range(cycles):
        keys = [
            CacheEngineKey(
                model_name=MODEL_NAME,
                world_size=1,
                worker_id=0,
                chunk_hash=cycle * batch_size + i,
                dtype=torch.bfloat16,
            )
            for i in range(batch_size)
        ]
        _put_chunks(session, keys, concurrency, progress=False)
        ok, mismatch = _get_chunks(session, keys, concurrency, progress=False)
        total_ok += ok
        total_mismatch += mismatch
        for key in keys:
            session.backend.remove(key)
        if (cycle + 1) % max(1, cycles // 10) == 0 or cycle == cycles - 1:
            print(f"  cycle {cycle + 1}/{cycles} done", flush=True)
    elapsed = time.perf_counter() - t0

    total_mb = CHUNK_BYTES * total_chunks / 1e6
    print(
        f"Churned {total_chunks} chunks (put+get+delete) in {elapsed:.2f}s "
        f"({total_mb / elapsed:.1f} MB/s put+get)"
    )
    _flush_and_print_stats(
        session, "lmcache:remote_time_to_put", "Put latency (cumulative)"
    )
    _flush_and_print_stats(
        session, "lmcache:remote_time_to_get_sync", "Get latency (cumulative)"
    )
    return total_ok, total_mismatch


def main() -> None:
    host = os.environ.get("SCYLLA_HOST", "127.0.0.1")
    port = int(os.environ.get("SCYLLA_PORT", "9042"))
    mode = os.environ.get("SCYLLA_E2E_MODE", "both")
    if mode not in VALID_MODES:
        raise ValueError(f"SCYLLA_E2E_MODE must be one of {VALID_MODES}, got {mode!r}")
    num_chunks = int(os.environ.get("SCYLLA_E2E_NUM_CHUNKS", "200"))
    concurrency = int(os.environ.get("SCYLLA_E2E_CONCURRENCY", "10"))
    churn_batch_size = int(os.environ.get("SCYLLA_E2E_CHURN_BATCH_SIZE", "100"))
    churn_cycles = int(os.environ.get("SCYLLA_E2E_CHURN_CYCLES", "20"))

    session = connect(host, port, mode, concurrency)
    size_desc = (
        f"batch_size={churn_batch_size}, cycles={churn_cycles}"
        if mode == "churn"
        else f"num_chunks={num_chunks}"
    )
    print(
        f"Connected to ScyllaDB at {host}:{port} "
        f"(mode={mode}, {size_desc}, concurrency={concurrency}).\n"
    )

    ok = num_chunks
    total_mismatch = 0
    still_present = remaining_present = 0
    if mode == "churn":
        ok, total_mismatch = do_churn(
            session, churn_batch_size, churn_cycles, concurrency
        )
        print()
    else:
        keys = [
            CacheEngineKey(
                model_name=MODEL_NAME,
                world_size=1,
                worker_id=0,
                chunk_hash=i,
                dtype=torch.bfloat16,
            )
            for i in range(num_chunks)
        ]
        total_mb = CHUNK_BYTES * num_chunks / 1e6

        if mode in ("put", "both"):
            do_put(session, keys, concurrency, total_mb)
            print()
        if mode in ("get", "both"):
            ok = do_get(session, keys, concurrency, total_mb)
            print()
        if mode == "both":
            still_present, remaining_present = do_delete(session, keys)
            print()

    print("=== Backend status ===")
    print(f"is_connected: {session.backend.is_connected()}")
    print(f"ping error code (0=ok): {session.backend.ping()}")

    metrics_text = generate_latest(REGISTRY).decode()
    of_interest = (
        "remote_put_task_num",
        "get_blocking_failed_count",
        "put_failed_count",
        "remote_write",
        "remote_read",
        "remote_time_to_get",
        "remote_time_to_put",
        "remote_ping",
    )
    print("\n=== Prometheus metrics (lmcache: prefixed, filtered to this run) ===")
    for line in metrics_text.splitlines():
        if line.startswith("#"):
            continue
        if any(name in line for name in of_interest):
            print(f"  {line}")

    close(session)

    if mode == "churn":
        assert total_mismatch == 0, f"{total_mismatch} chunk(s) mismatched or missing"
    if mode in ("get", "both"):
        assert ok == num_chunks, "not all chunks retrieved correctly"
    if mode == "both":
        num_to_delete = max(1, int(num_chunks * DELETE_FRACTION))
        assert still_present == 0, "deleted chunks still readable"
        assert remaining_present == num_chunks - num_to_delete, (
            "non-deleted chunks went missing"
        )
    print("\nEnd-to-end run PASSED.")


if __name__ == "__main__":
    main()
