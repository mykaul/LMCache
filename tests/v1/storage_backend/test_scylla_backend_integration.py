# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved

# Standard
from typing import Any
import asyncio
import os
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.scylla_backend import ScyllaDBBackend
from tests.v1.utils import create_test_memory_obj

_HIDDEN_DIM = 1024  # 8 heads * 128 head_size
_NUM_LAYERS = 16
_NUM_TOKENS = 8
_TEST_MEM_SHAPE = torch.Size([2, _NUM_LAYERS, _NUM_TOKENS, _HIDDEN_DIM])
_TEST_KV_SHAPE = (_NUM_LAYERS, 2, _NUM_TOKENS, 8, 128)


def create_test_metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=_TEST_KV_SHAPE,
        use_mla=False,
        role="worker",
    )


def create_test_config(
    contact_points: list[str] | None = None,
    ttl_seconds: int = 600,
) -> LMCacheEngineConfig:
    extras: dict[str, Any] = {
        "scylla": {
            "contact_points": contact_points or ["127.0.0.1"],
            "port": int(os.environ.get("SCYLLA_PORT", "9042")),
            "keyspace": os.environ.get("SCYLLA_KEYSPACE", "test_lmcache"),
            "local_dc": os.environ.get("SCYLLA_LOCAL_DC", "datacenter1"),
            "consistency_level": "LOCAL_ONE",
            "ttl_seconds": ttl_seconds,
            "table_compression": "LZ4WithDictsCompressor",
        }
    }
    return LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        extra_config=extras,
    )


def _is_scylla_reachable(host: str, port: int) -> bool:
    # Standard
    import socket

    try:
        with socket.create_connection((host, port), timeout=3.0):
            return True
    except OSError:
        return False


def _wait_for_puts(
    backend: ScyllaDBBackend, keys: list[CacheEngineKey], timeout: float = 10.0
) -> None:
    """
    Wait for one or more fire-and-forget puts to finish.

    ``batched_submit_put_task`` dispatches each key's write onto the
    backend's event loop without blocking the caller (matching
    ``StorageManager.batched_put``'s documented non-blocking contract), so
    tests that assert on a put's effects must wait for completion first.
    """
    deadline = time.monotonic() + timeout
    for key in keys:
        while backend.exists_in_put_tasks(key):
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for put of {key} to complete")
            time.sleep(0.01)


@pytest.fixture(scope="module")
def async_loop():
    loop = asyncio.new_event_loop()
    th = threading.Thread(target=loop.run_forever, daemon=True)
    th.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    th.join(timeout=5.0)
    loop.close()


@pytest.fixture
def local_cpu_backend():
    # Test KV chunks are ~512 KiB each (see _TEST_MEM_SHAPE); 0.1 GB leaves
    # comfortable headroom for the batched tests (~2.5 MiB at once) without
    # pre-allocating an unnecessarily large pinned buffer per test.
    backend = LocalCPUBackend(
        LMCacheEngineConfig.from_defaults(chunk_size=256, max_local_cpu_size=0.1),
        create_test_metadata(),
        dst_device="cpu",
    )
    yield backend
    backend.close()


@pytest.mark.scylla
@pytest.mark.no_shared_allocator
class TestScyllaDBIntegration:
    @pytest.fixture(autouse=True)
    def check_scylla(self):
        host = os.environ.get("SCYLLA_HOST", "127.0.0.1")
        port = int(os.environ.get("SCYLLA_PORT", "9042"))
        if not _is_scylla_reachable(host, port):
            pytest.skip(
                f"ScyllaDB not reachable at {host}:{port}. "
                "Start with: docker run -d --name scylla "
                "-p 9042:9042 scylladb/scylla:2026.2"
            )
        self._scylla_host = host
        self._scylla_port = port

    @pytest.fixture
    def backend(self, async_loop, local_cpu_backend):
        config = create_test_config(
            contact_points=[self._scylla_host],
        )
        metadata = create_test_metadata()
        backend = ScyllaDBBackend(
            dst_device="cpu",
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=async_loop,
        )
        yield backend
        backend.close()

    def test_put_get_roundtrip(self, backend):
        key = CacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=42,
            dtype=torch.bfloat16,
        )
        mem_obj = create_test_memory_obj(shape=_TEST_MEM_SHAPE)
        assert mem_obj.tensor is not None
        mem_obj.tensor.fill_(3.14)
        original = mem_obj.tensor.clone()

        backend.batched_submit_put_task([key], [mem_obj])
        _wait_for_puts(backend, [key])
        assert backend.contains(key)

        result = backend.get_blocking(key)
        assert result is not None
        assert result.tensor is not None
        assert torch.equal(result.tensor, original)

    def test_missing_key(self, backend):
        key = CacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=999,
            dtype=torch.bfloat16,
        )
        assert not backend.contains(key)
        assert backend.get_blocking(key) is None

    def test_remove(self, backend):
        key = CacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=100,
            dtype=torch.bfloat16,
        )
        mem_obj = create_test_memory_obj(shape=_TEST_MEM_SHAPE)
        backend.batched_submit_put_task([key], [mem_obj])
        _wait_for_puts(backend, [key])
        assert backend.contains(key)

        backend.remove(key)
        assert not backend.contains(key)

    def test_remove_deletes_every_layer_partition(self, backend):
        """
        Each layer is its own ScyllaDB partition (composite partition key
        ``(chunk_hash, layer_id)``, no clustering column), so there is no
        single ``chunk_hash``-scoped delete that covers a whole chunk --
        a whole-chunk ``remove`` must fan out and delete every layer's
        partition individually. ``contains(key)`` only checks layer 0 (see
        :meth:`ScyllaDBBackend._contains_async`), so this checks every
        layer explicitly to catch a remove that only clears layer 0.
        """
        key = CacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=101,
            dtype=torch.bfloat16,
        )
        mem_obj = create_test_memory_obj(shape=_TEST_MEM_SHAPE)
        backend.batched_submit_put_task([key], [mem_obj])
        _wait_for_puts(backend, [key])
        assert backend.contains(key)

        backend.remove(key)

        for layer_id in range(_NUM_LAYERS):
            layer_key = LayerCacheEngineKey(
                model_name="test_model",
                world_size=1,
                worker_id=0,
                chunk_hash=101,
                dtype=torch.bfloat16,
                layer_id=layer_id,
            )
            assert not backend.contains(layer_key), (
                f"layer {layer_id} still present after whole-chunk remove"
            )

    def test_contains_layer_key(self, backend):
        key = CacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=102,
            dtype=torch.bfloat16,
        )
        mem_obj = create_test_memory_obj(shape=_TEST_MEM_SHAPE)
        backend.batched_submit_put_task([key], [mem_obj])
        _wait_for_puts(backend, [key])

        layer_key = LayerCacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=102,
            dtype=torch.bfloat16,
            layer_id=0,
        )
        assert backend.contains(layer_key)

        missing_layer = LayerCacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=102,
            dtype=torch.bfloat16,
            layer_id=_NUM_LAYERS + 1,
        )
        assert not backend.contains(missing_layer)

    def test_partial_layer_removal_makes_whole_chunk_read_a_miss(self, backend):
        """
        Removing a single layer must not corrupt the other layers' rows
        (each lives in its own partition), but a whole-chunk
        ``get_blocking`` fans out one read per layer and requires all of
        them to be present (see
        :meth:`ScyllaDBBackend._fetch_all_layers_async`) -- so once any
        layer is gone, the whole chunk must read back as a clean miss
        instead of silently reconstructing a wrong/misaligned tensor.
        """
        key = CacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=103,
            dtype=torch.bfloat16,
        )
        mem_obj = create_test_memory_obj(shape=_TEST_MEM_SHAPE)
        backend.batched_submit_put_task([key], [mem_obj])
        _wait_for_puts(backend, [key])
        assert backend.get_blocking(key) is not None

        layer_key = LayerCacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=103,
            dtype=torch.bfloat16,
            layer_id=1,
        )
        backend.remove(layer_key)

        # Layer 0 (the `contains` proxy for a whole-chunk key) is
        # untouched, in its own partition...
        assert backend.contains(key)
        # ...but layer 1's partition is gone...
        assert not backend.contains(layer_key)
        # ...so the whole-chunk fetch must now report a miss.
        assert backend.get_blocking(key) is None

    def test_batched_operations(self, backend):
        keys = [
            CacheEngineKey(
                "test_model",
                world_size=1,
                worker_id=0,
                chunk_hash=i,
                dtype=torch.bfloat16,
            )
            for i in range(10, 15)
        ]
        objs = [create_test_memory_obj(shape=_TEST_MEM_SHAPE) for _ in range(5)]
        backend.batched_submit_put_task(keys, objs)
        _wait_for_puts(backend, keys)
        results = backend.batched_get_blocking(keys)
        assert len(results) == 5
        for r in results:
            assert r is not None

    def test_ttl_expiry(self, local_cpu_backend, async_loop):
        """
        ``ttl_seconds`` must actually cause a row to stop being readable
        once it elapses -- not just be an accepted, inert config value.
        Uses a short TTL (a few seconds, not the config's usual
        minutes/hours) so this stays fast; per-cell CQL TTL
        (``INSERT ... USING TTL``, see ``_CQL_INSERT``) drops an expired
        row's *visibility* the instant its TTL elapses (no need to wait
        for background compaction to physically reclaim it), so polling
        briefly past the deadline is sufficient. This regresses a real bug:
        an earlier version used ScyllaDB's per-row TTL column feature
        instead, which -- confirmed against ScyllaDB's own docs -- only
        expires *eventually* via a cluster-wide background sweep (default
        24h), not at read time, so rows stayed fully readable long after
        their configured TTL.
        """
        ttl_seconds = 2
        config = create_test_config(
            contact_points=[self._scylla_host], ttl_seconds=ttl_seconds
        )
        metadata = create_test_metadata()
        backend = ScyllaDBBackend(
            dst_device="cpu",
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=async_loop,
        )
        try:
            key = CacheEngineKey(
                model_name="test_model",
                world_size=1,
                worker_id=0,
                chunk_hash=300,
                dtype=torch.bfloat16,
            )
            mem_obj = create_test_memory_obj(shape=_TEST_MEM_SHAPE)
            backend.batched_submit_put_task([key], [mem_obj])
            _wait_for_puts(backend, [key])

            assert backend.contains(key)
            assert backend.get_blocking(key) is not None

            deadline = time.monotonic() + ttl_seconds + 8.0
            while backend.contains(key):
                if time.monotonic() > deadline:
                    raise AssertionError(
                        f"key still present {ttl_seconds + 8.0}s after a "
                        f"{ttl_seconds}s TTL; expiry did not take effect"
                    )
                time.sleep(0.5)

            assert backend.get_blocking(key) is None
        finally:
            backend.close()

    def test_model_isolation(self, backend, local_cpu_backend, async_loop):
        key_a = CacheEngineKey(
            "model_alpha",
            world_size=1,
            worker_id=0,
            chunk_hash=200,
            dtype=torch.bfloat16,
        )
        key_b = CacheEngineKey(
            "model_beta",
            world_size=1,
            worker_id=0,
            chunk_hash=200,
            dtype=torch.bfloat16,
        )

        mem_obj = create_test_memory_obj(shape=_TEST_MEM_SHAPE)
        backend.batched_submit_put_task([key_a], [mem_obj])
        _wait_for_puts(backend, [key_a])

        assert backend.contains(key_a)
        assert not backend.contains(key_b)
