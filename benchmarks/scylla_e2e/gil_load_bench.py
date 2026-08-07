# SPDX-License-Identifier: Apache-2.0
"""Measures ScyllaDBBackend get latency while a "noise" thread holds the GIL,
comparing Python's default sys.setswitchinterval() against a configured one
(the ``scylla.gil_switch_interval_secs`` backend config).

Usage:
    benchmarks/scylla_e2e/launch_scylla_e2e.sh
    uv run python -m benchmarks.scylla_e2e.gil_load_bench \
        --budgets 0 15 30 --n 15 --switch-interval 0.0001
"""

# Standard
from typing import List, NamedTuple, Optional
import argparse
import asyncio
import sys
import threading
import time

# Third Party
import torch

# Local
from benchmarks.scylla_e2e.scylla_e2e_demo import (
    CHUNK_BYTES,
    make_chunk,
    make_metadata,
)

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.scylla_backend import ScyllaDBBackend

MODEL_NAME = "gil-load-bench"
_KEY_POOL_SIZE = 24
_DEFAULT_SWITCH_INTERVAL = 0.005  # sys.getswitchinterval()'s own CPython default


class BenchSession(NamedTuple):
    backend: ScyllaDBBackend
    local_cpu: LocalCPUBackend
    loop: asyncio.AbstractEventLoop
    loop_thread: threading.Thread


def _make_key(chunk_hash: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name=MODEL_NAME,
        world_size=1,
        worker_id=0,
        chunk_hash=chunk_hash,
        dtype=torch.bfloat16,
    )


def _connect(
    host: str, port: int, keyspace: str, switch_interval_secs: Optional[float]
) -> BenchSession:
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    metadata = make_metadata()
    local_cpu = LocalCPUBackend(
        LMCacheEngineConfig.from_defaults(
            chunk_size=256, max_local_cpu_size=(3 * CHUNK_BYTES) / 1e9
        ),
        metadata,
        dst_device="cpu",
    )
    scylla_cfg: dict = {
        "contact_points": [host],
        "port": port,
        "keyspace": keyspace,
        "local_dc": "datacenter1",
        "ttl_seconds": 600,
        "table_compression": "",
        "wire_compression": False,
    }
    if switch_interval_secs is not None:
        scylla_cfg["gil_switch_interval_secs"] = switch_interval_secs
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256, extra_config={"scylla": scylla_cfg}
    )
    backend = ScyllaDBBackend(
        dst_device="cpu",
        config=config,
        metadata=metadata,
        local_cpu_backend=local_cpu,
        loop=loop,
    )
    return BenchSession(backend, local_cpu, loop, loop_thread)


def _close(session: BenchSession) -> None:
    session.backend.close()
    session.local_cpu.close()
    session.loop.call_soon_threadsafe(session.loop.stop)
    session.loop_thread.join(timeout=5.0)
    session.loop.close()


_CLIP = (1 << 16) - 1


def gil_work(target_ms: float, data: bytes) -> float:
    """Bytewise pure-Python loop that holds the GIL for ~target_ms."""
    view = memoryview(data)
    total = len(view)
    step = 1 << 16
    acc = 0
    start = time.perf_counter()
    while True:
        for off in range(0, total, step):
            for b in view[off : off + step]:
                acc = (acc + (b >> 1)) & _CLIP
            if (time.perf_counter() - start) * 1e3 >= target_ms:
                return (time.perf_counter() - start) * 1e3


class NoiseThread:
    """Continuously holds the GIL via *work*, one call per iteration --
    mimicking an engine thread busy with CPU-bound work while the ScyllaDB
    driver's reactor tries to deliver query responses."""

    def __init__(self, budget_ms: float, data: bytes) -> None:
        self._budget_ms = budget_ms
        self._data = data
        self._stop = threading.Event()
        self._iters = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            gil_work(self._budget_ms, self._data)
            self._iters += 1

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30.0)

    @property
    def iters(self) -> int:
        return self._iters


def _write_pool(session: BenchSession) -> None:
    keys = [_make_key(i) for i in range(_KEY_POOL_SIZE)]
    chunks = [make_chunk(i) for i in range(_KEY_POOL_SIZE)]
    session.backend.batched_submit_put_task(keys, chunks)
    deadline = time.monotonic() + 60.0
    for key in keys:
        while session.backend.exists_in_put_tasks(key):
            if time.monotonic() > deadline:
                raise TimeoutError(f"put of {key} did not complete in time")
            time.sleep(0.02)
    for chunk in chunks:
        chunk.ref_count_down()


def run_get_campaign(
    session: BenchSession, n: int, noise: Optional[NoiseThread]
) -> dict:
    backend = session.backend
    keys = [_make_key(i) for i in range(_KEY_POOL_SIZE)]
    lats: List[float] = []
    for i in range(n):
        key = keys[i % _KEY_POOL_SIZE]
        t0 = time.perf_counter()
        objs = backend.batched_get_blocking([key])
        lats.append((time.perf_counter() - t0) * 1e3)
        obj = objs[0] if objs else None
        if obj is not None:
            obj.ref_count_down()
    lats_sorted = sorted(lats)

    def q(frac: float) -> float:
        return lats_sorted[min(len(lats_sorted) - 1, int(frac * len(lats_sorted)))]

    return {
        "n": len(lats),
        "mean_ms": sum(lats) / len(lats),
        "p50_ms": q(0.50),
        "p99_ms": q(0.99),
        "noise_iters": noise.iters if noise is not None else 0,
    }


def _print_cell(tag: str, s: dict) -> None:
    print(
        f"  {tag:<28} n={s['n']:<4} mean={s['mean_ms']:8.1f}ms "
        f"p50={s['p50_ms']:8.1f}ms p99={s['p99_ms']:8.1f}ms "
        f"noise_iters={s['noise_iters']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9042)
    parser.add_argument("--keyspace", default="lmcache_gil_load_bench")
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=[0.0, 15.0],
        help="GIL-work budgets (ms) the noise thread spends per iteration.",
    )
    parser.add_argument("--n", type=int, default=15, help="iterations per cell")
    parser.add_argument(
        "--switch-interval",
        type=float,
        default=0.0001,
        help="gil_switch_interval_secs to compare against Python's default.",
    )
    parser.add_argument("--data-mb", type=float, default=CHUNK_BYTES / 1e6)
    args = parser.parse_args()

    data = bytes(int(args.data_mb * 1e6))
    print(
        f"gil_load_bench: host={args.host}:{args.port} budgets={args.budgets} "
        f"n={args.n} switch_interval={args.switch_interval} "
        f"(default={_DEFAULT_SWITCH_INTERVAL})",
        flush=True,
    )

    results: dict[str, dict] = {}
    for budget in args.budgets:
        for switch_interval in (None, args.switch_interval):
            label = "default" if switch_interval is None else "tuned"
            tag = f"budget={budget:g}/{label}"
            session = _connect(args.host, args.port, args.keyspace, switch_interval)
            try:
                _write_pool(session)
                for _ in range(3):
                    run_get_campaign(session, 1, None)  # warm-up
                noise = NoiseThread(budget, data) if budget > 0 else None
                if noise is not None:
                    noise.start()
                try:
                    s = run_get_campaign(session, args.n, noise)
                finally:
                    if noise is not None:
                        noise.stop()
                results[tag] = s
                _print_cell(tag, s)
            finally:
                _close(session)
                # Don't let one cell's interval leak into the next.
                sys.setswitchinterval(_DEFAULT_SWITCH_INTERVAL)

    print("\ncomparison (tuned vs default switch interval, same budget cell):")
    for budget in args.budgets:
        default = results.get(f"budget={budget:g}/default")
        tuned = results.get(f"budget={budget:g}/tuned")
        if default is None or tuned is None:
            continue
        ratio = (
            tuned["p50_ms"] / default["p50_ms"] if default["p50_ms"] else float("nan")
        )
        print(
            f"  budget={budget:g}: default p50={default['p50_ms']:8.1f}ms "
            f"tuned p50={tuned['p50_ms']:8.1f}ms ratio={ratio:.3f}"
        )


if __name__ == "__main__":
    main()
