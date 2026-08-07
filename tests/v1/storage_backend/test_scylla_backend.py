# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Optional
from unittest.mock import MagicMock, patch
import asyncio
import re
import sys
import threading
import time
import types

# Create mock cassandra modules BEFORE any other imports, so that
# scylla_backend's import guard finds them and sets _SCYLLA_AVAILABLE = True.
_mock_cassandra_modules: dict[str, Any] = {}
for _name in [
    "cassandra",
    "cassandra.cluster",
    "cassandra.connection",
    "cassandra.policies",
    "cassandra.query",
]:
    _mod = types.ModuleType(_name)
    _mock_cassandra_modules[_name] = _mod
    sys.modules[_name] = _mod

# Also mock out lmcache.c_ops so that memory_management can be imported
_cops_mod = types.ModuleType("lmcache.c_ops")
_cops_mod.alloc_shm_pinned_ptr = None
_cops_mod.free_shm_pinned_ptr = None
_cops_mod.alloc_hugepage_pinned_numa_ptr = None
_cops_mod.free_hugepage_pinned_numa_ptr = None
_cops_mod.alloc_pinned_numa_ptr = None
_cops_mod.free_pinned_numa_ptr = None
_cops_mod.alloc_hugepage_pinned_ptr = None
_cops_mod.free_hugepage_pinned_ptr = None
_cops_mod.alloc_pinned_ptr = None
_cops_mod.free_pinned_ptr = None
sys.modules["lmcache.c_ops"] = _cops_mod

# Wire sub-modules into parent
_mock_cassandra_modules["cassandra"].cluster = _mock_cassandra_modules[
    "cassandra.cluster"
]

# Populate mock attributes each module-level consumer expects
_cass_cluster = _mock_cassandra_modules["cassandra.cluster"]
_cass_connection = _mock_cassandra_modules["cassandra.connection"]
_cass_query = _mock_cassandra_modules["cassandra.query"]
_cass_policies = _mock_cassandra_modules["cassandra.policies"]
_cass_top = _mock_cassandra_modules["cassandra"]

_cass_cluster.Cluster = type("MockClusterImport", (), {})
_cass_cluster.Session = type("MockSessionImport", (), {})
_cass_cluster.EXEC_PROFILE_DEFAULT = object()
_cass_cluster.ExecutionProfile = lambda **kw: None
# Real Exception subclasses: scylla_backend.py catches these by type in its
# retry logic (_RETRYABLE_EXCEPTIONS), so they must be genuine exception
# classes, not bare `type(...)` placeholders.
_cass_top.Unavailable = type("MockUnavailable", (Exception,), {})
_cass_top.ReadTimeout = type("MockReadTimeout", (Exception,), {})
_cass_top.WriteTimeout = type("MockWriteTimeout", (Exception,), {})
_cass_top.OperationTimedOut = type("MockOperationTimedOut", (Exception,), {})
_cass_cluster.NoHostAvailable = type("MockNoHostAvailable", (Exception,), {})
_cass_connection.ConnectionBusy = type("MockConnectionBusy", (Exception,), {})
_cass_query.ConsistencyLevel = type(
    "MockConsistencyLevel",
    (),
    {
        "LOCAL_ONE": 0,
    },
)
_cass_query.PreparedStatement = type("MockPreparedStatement", (), {})
_cass_query.BatchStatement = type("MockBatchStatement", (), {})
_cass_query.BatchType = type("MockBatchType", (), {})
_cass_query.tuple_factory = lambda *args, **kwargs: None
_cass_policies.DCAwareRoundRobinPolicy = lambda **kw: None
_cass_policies.RackAwareRoundRobinPolicy = lambda **kw: None
_cass_policies.TokenAwarePolicy = lambda *a, **kw: None
_cass_policies.ConstantSpeculativeExecutionPolicy = lambda *a, **kw: None

# Now it's safe to import the scylla backend — _SCYLLA_AVAILABLE will be True
# Third Party
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

# First Party
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey  # noqa: E402
from lmcache.v1.config import LMCacheEngineConfig  # noqa: E402
from lmcache.v1.metadata import LMCacheMetadata  # noqa: E402
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend  # noqa: E402
from lmcache.v1.storage_backend.scylla_backend import (  # noqa: E402
    ScyllaDBBackend,
    _is_local_connection_busy,
)
from tests.v1.utils import create_test_memory_obj  # noqa: E402


def create_test_config(
    extra_overrides: dict[str, Any] | None = None,
) -> LMCacheEngineConfig:
    extras: dict[str, Any] = {
        "scylla": {
            "contact_points": ["127.0.0.1"],
            "port": 9042,
            "keyspace": "test_lmcache",
            "local_dc": "datacenter1",
            "consistency_level": "LOCAL_ONE",
            "ttl_seconds": 86400,
            "table_compression": "LZ4WithDictsCompressor",
        }
    }
    if extra_overrides:
        _merge_dict(extras, extra_overrides)
    return LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        extra_config=extras,
    )


def _merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base:
            _merge_dict(base[k], v)
        else:
            base[k] = v


def create_test_metadata(
    kv_shape: tuple[int, int, int, int, int] = (16, 2, 8, 8, 128),
) -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
        use_mla=False,
        role="worker",
    )


def create_test_key(key_id: int = 0) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=key_id,
        dtype=torch.bfloat16,
    )


# hidden_dim = 8 * 128 = 1024; num_layers = 16; num_tokens = 8
_TEST_MEM_SHAPE = torch.Size([2, 16, 8, 1024])


def _wait_for_puts(
    backend: ScyllaDBBackend, keys: list[CacheEngineKey], timeout: float = 5.0
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


@pytest.fixture
def mock_in_memory_scylla():
    """Return a mocked Cluster whose session stores rows in a dict.

    Mirrors the real ``cassandra.cluster`` API surface (``Cluster.connect``,
    ``Session.execute``, ``Session.execute_async`` returning a
    ``ResponseFuture``-like object, ``Session.prepare``) rather than the
    fictional ``connect_async``/``execute_async``-as-coroutine/
    ``prepare_async`` methods the production code used to (incorrectly)
    assume existed.
    """

    store: dict[str, dict[tuple[int, int], tuple[bytes, bytes]]] = {}
    executed_cql: list[str] = []

    def _get_mock_table(stmt: object, fallback: str) -> str:
        if isinstance(stmt, MagicMock) and "_mock_table" in stmt.__dict__:
            return stmt.__dict__["_mock_table"]
        return fallback

    class MockResponseFuture:
        """Mimics ``cassandra.cluster.ResponseFuture``'s callback API."""

        def __init__(self, rows: list, error: Optional[Exception] = None):
            self._rows = rows
            self._error = error
            # This mock never produces more than one page of results.
            self.has_more_pages = False

        def add_callbacks(self, callback, errback) -> None:
            if self._error is not None:
                errback(self._error)
            else:
                callback(self._rows)

        def start_fetching_next_page(self) -> None:
            """No-op: this mock never produces more than one page."""

    class MockSession:
        default_timeout = 30.0

        def _compute_rows(self, stmt, params=None) -> list:
            # _ensure_keyspace_exists() passes a raw CQL string (no
            # PreparedStatement wrapping the DDL); everything else goes
            # through self.prepare() first, producing a MagicMock with a
            # `_mock_cql` attribute.
            cql = stmt if isinstance(stmt, str) else getattr(stmt, "_mock_cql", "")
            executed_cql.append(cql)
            table = _get_mock_table(
                stmt, "kv_chunks_test_model_world1_worker0_bfloat16"
            )
            tbl = store.setdefault(table, {})

            if "INSERT INTO" in cql:
                chunk_hash, layer_id, k_blob, v_blob, _ttl_seconds = params
                tbl[(chunk_hash, layer_id)] = (k_blob, v_blob)
                return []

            if "LIMIT 1" in cql:
                chunk_hash, layer_id = params
                return [chunk_hash] if (chunk_hash, layer_id) in tbl else []

            if "ORDER BY layer_id ASC" in cql:
                (chunk_hash,) = params
                matched = sorted(
                    [(k, v) for k, v in tbl.items() if k[0] == chunk_hash],
                    key=lambda x: x[0][1],
                )
                rows = []
                for (ch, lid), (kdata, vdata) in matched:
                    row = MagicMock()
                    row.chunk_hash = ch
                    row.layer_id = lid
                    row.k_data = kdata
                    row.v_data = vdata
                    rows.append(row)
                return rows

            if cql.startswith("SELECT") and "AND layer_id" in cql:
                chunk_hash, layer_id = params
                item = tbl.get((chunk_hash, layer_id))
                if item is not None:
                    # Session.row_factory is tuple_factory in production
                    # (see scylla_backend._connect); match its shape here.
                    kdata, vdata = item
                    return [(kdata, vdata)]
                return []

            if cql.startswith("DELETE") and "AND layer_id" in cql:
                chunk_hash, layer_id = params
                tbl.pop((chunk_hash, layer_id), None)
                return []

            if cql.startswith("DELETE"):
                (chunk_hash,) = params
                keys_to_delete = [k for k in tbl if k[0] == chunk_hash]
                for k in keys_to_delete:
                    tbl.pop(k, None)
                return []

            return []

        def execute(self, stmt, params=None) -> list:
            return self._compute_rows(stmt, params)

        def execute_async(self, stmt, params=None) -> MockResponseFuture:
            try:
                return MockResponseFuture(self._compute_rows(stmt, params))
            except Exception as e:
                return MockResponseFuture([], error=e)

        def prepare(self, cql):
            stmt = MagicMock()
            stmt._mock_cql = cql
            stmt.consistency_level = None
            m = re.search(r"kv_chunks_\S+", cql)
            if m:
                stmt._mock_table = m.group(0)
            return stmt

        def set_keyspace(self, keyspace: str) -> None:
            """No-op: `store` above is a flat dict of table->rows, not
            scoped per-keyspace, so there is nothing to switch."""

    class MockCluster:
        def connect(self, keyspace=None):
            return MockSession()

    mock_cluster = MockCluster()
    mock_cluster.executed_cql = executed_cql
    mock_cluster.store = store
    return mock_cluster


@pytest.fixture
def backend(async_loop, local_cpu_backend, mock_in_memory_scylla):
    config = create_test_config()
    metadata = create_test_metadata()
    with patch(
        "lmcache.v1.storage_backend.scylla_backend.Cluster",
        return_value=mock_in_memory_scylla,
    ):
        backend = ScyllaDBBackend(
            dst_device="cpu",
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=async_loop,
        )
        yield backend
        backend.close()


@pytest.mark.no_shared_allocator
class TestScyllaDBBackend:
    def test_worker_id_isolation(self, backend):
        """Different worker_id values must not collide, even when
        chunk_hash and model_name match (see ScyllaDBBackend._table_name)."""
        key_worker0 = create_test_key(200)
        key_worker1 = CacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=1,
            chunk_hash=200,
            dtype=torch.bfloat16,
        )

        mem_obj = create_test_memory_obj(shape=_TEST_MEM_SHAPE)
        backend.batched_submit_put_task([key_worker0], [mem_obj])
        _wait_for_puts(backend, [key_worker0])

        assert backend.contains(key_worker0)
        assert not backend.contains(key_worker1)

    def test_gil_switch_interval_not_set_by_default(
        self, async_loop, local_cpu_backend, mock_in_memory_scylla
    ):
        """Process-global sys.setswitchinterval must not be touched unless
        gil_switch_interval_secs is explicitly configured."""
        config = create_test_config()
        metadata = create_test_metadata()
        with (
            patch(
                "lmcache.v1.storage_backend.scylla_backend.Cluster",
                return_value=mock_in_memory_scylla,
            ),
            patch("sys.setswitchinterval") as mock_set_interval,
        ):
            backend = ScyllaDBBackend(
                dst_device="cpu",
                config=config,
                metadata=metadata,
                local_cpu_backend=local_cpu_backend,
                loop=async_loop,
            )
            mock_set_interval.assert_not_called()
            backend.close()

    def test_gil_switch_interval_configured(
        self, async_loop, local_cpu_backend, mock_in_memory_scylla
    ):
        """gil_switch_interval_secs, when set, must be forwarded verbatim
        to sys.setswitchinterval (see the class docstring: this is a
        process-global setting, not scoped to this backend)."""
        config = create_test_config({"scylla": {"gil_switch_interval_secs": 0.0001}})
        metadata = create_test_metadata()
        with (
            patch(
                "lmcache.v1.storage_backend.scylla_backend.Cluster",
                return_value=mock_in_memory_scylla,
            ),
            patch("sys.setswitchinterval") as mock_set_interval,
        ):
            backend = ScyllaDBBackend(
                dst_device="cpu",
                config=config,
                metadata=metadata,
                local_cpu_backend=local_cpu_backend,
                loop=async_loop,
            )
            mock_set_interval.assert_called_once_with(0.0001)
            backend.close()

    def test_sanitize_model(self):
        assert ScyllaDBBackend._sanitize_model("llama-3.1-8b") == "llama_3_1_8b"
        assert ScyllaDBBackend._sanitize_model("123model") == "m_123model"
        assert ScyllaDBBackend._sanitize_model("simple") == "simple"

    def test_creates_keyspace_if_not_exists(self, backend, mock_in_memory_scylla):
        assert any(
            "CREATE KEYSPACE IF NOT EXISTS test_lmcache" in cql
            for cql in mock_in_memory_scylla.executed_cql
        )

    def test_connect_uses_rack_aware_policy_when_local_rack_configured(
        self, async_loop, local_cpu_backend, mock_in_memory_scylla
    ):
        """local_rack must select RackAwareRoundRobinPolicy, not
        DCAwareRoundRobinPolicy -- the latter has no local_rack parameter
        at all and raises TypeError if passed one."""
        config = create_test_config({"scylla": {"local_rack": "rack1"}})
        metadata = create_test_metadata()
        with (
            patch(
                "lmcache.v1.storage_backend.scylla_backend.Cluster",
                return_value=mock_in_memory_scylla,
            ),
            patch(
                "lmcache.v1.storage_backend.scylla_backend.RackAwareRoundRobinPolicy"
            ) as mock_rack_policy,
            patch(
                "lmcache.v1.storage_backend.scylla_backend.DCAwareRoundRobinPolicy"
            ) as mock_dc_policy,
        ):
            backend = ScyllaDBBackend(
                dst_device="cpu",
                config=config,
                metadata=metadata,
                local_cpu_backend=local_cpu_backend,
                loop=async_loop,
            )
            mock_rack_policy.assert_called_once_with(
                local_dc="datacenter1", local_rack="rack1"
            )
            mock_dc_policy.assert_not_called()
            backend.close()

    def test_put_get_roundtrip(self, backend):
        key = create_test_key(1)
        mem_obj = create_test_memory_obj(shape=_TEST_MEM_SHAPE)
        assert mem_obj.tensor is not None
        mem_obj.tensor.fill_(3.14)
        original = mem_obj.tensor.clone()

        backend.batched_submit_put_task([key], [mem_obj])
        _wait_for_puts(backend, [key])
        result = backend.get_blocking(key)

        assert result is not None
        assert result.tensor is not None
        assert torch.equal(result.tensor, original)

    def test_put_sends_zero_copy_views_of_the_source_tensor(
        self, backend, mock_in_memory_scylla
    ):
        """PUT must hand the driver views into the source tensor's own
        memory, not copies -- measured (isolated micro-benchmark, no
        network) to be faster than copying here: spreading many per-layer
        memcpys across self._blob_executor's threads costs more in
        GIL/thread-pool contention than doing them on the single thread
        that calls execute_async."""
        key = create_test_key(50)
        mem_obj = create_test_memory_obj(shape=_TEST_MEM_SHAPE)
        assert mem_obj.tensor is not None
        tensor_bytes = mem_obj.tensor.view(torch.uint8).numpy().ravel()

        backend.batched_submit_put_task([key], [mem_obj])
        _wait_for_puts(backend, [key])

        stored = next(
            item
            for table in mock_in_memory_scylla.store.values()
            for (chunk_hash, _layer_id), item in table.items()
            if chunk_hash == key.chunk_hash
        )
        k_blob, v_blob = stored
        assert np.shares_memory(k_blob, tensor_bytes)
        assert np.shares_memory(v_blob, tensor_bytes)

    def test_contains(self, backend):
        key = create_test_key(2)
        assert not backend.contains(key)

        mem_obj = create_test_memory_obj(shape=_TEST_MEM_SHAPE)
        backend.batched_submit_put_task([key], [mem_obj])
        _wait_for_puts(backend, [key])
        assert backend.contains(key)

    def test_contains_layer_key(self, backend):
        key = create_test_key(3)
        mem_obj = create_test_memory_obj(shape=_TEST_MEM_SHAPE)
        backend.batched_submit_put_task([key], [mem_obj])
        _wait_for_puts(backend, [key])

        layer_key = LayerCacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=3,
            dtype=torch.bfloat16,
            layer_id=0,
        )
        assert backend.contains(layer_key)

        missing_layer = LayerCacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=3,
            dtype=torch.bfloat16,
            layer_id=999,
        )
        assert not backend.contains(missing_layer)

    def test_remove(self, backend):
        key = create_test_key(4)
        mem_obj = create_test_memory_obj(shape=_TEST_MEM_SHAPE)
        backend.batched_submit_put_task([key], [mem_obj])
        _wait_for_puts(backend, [key])
        assert backend.contains(key)

        backend.remove(key)
        assert not backend.contains(key)

    def test_remove_layer_key(self, backend):
        key = create_test_key(5)
        mem_obj = create_test_memory_obj(shape=_TEST_MEM_SHAPE)
        backend.batched_submit_put_task([key], [mem_obj])
        _wait_for_puts(backend, [key])

        # Remove layer 1; layer 0 should still exist, so contains(key) is True
        layer_key = LayerCacheEngineKey(
            model_name="test_model",
            world_size=1,
            worker_id=0,
            chunk_hash=5,
            dtype=torch.bfloat16,
            layer_id=1,
        )
        backend.remove(layer_key)
        assert backend.contains(key)

        # The removed layer should no longer be found via its own key
        assert not backend.contains(layer_key)

    def test_missing_key_returns_none(self, backend):
        key = create_test_key(999)
        result = backend.get_blocking(key)
        assert result is None

    def test_get_blocking_handles_corrupted_row_gracefully(
        self, backend, mock_in_memory_scylla
    ):
        """
        A row with wrong-length blob data (e.g. a partial write, driver
        bug, or bit-rot) must not crash get_blocking() -- it should be
        treated as a failed get, matching every other error path in this
        backend, and reported through get_blocking_failed_count.
        """
        key = create_test_key(50)
        mem_obj = create_test_memory_obj(shape=_TEST_MEM_SHAPE)
        backend.batched_submit_put_task([key], [mem_obj])
        _wait_for_puts(backend, [key])

        table = next(iter(mock_in_memory_scylla.store))
        k_data, v_data = mock_in_memory_scylla.store[table][(key.chunk_hash, 0)]
        mock_in_memory_scylla.store[table][(key.chunk_hash, 0)] = (
            k_data[:-1],
            v_data,
        )

        initial_failed_count = backend.get_blocking_failed_count
        assert backend.get_blocking(key) is None
        assert backend.get_blocking_failed_count == initial_failed_count + 1

    def test_batched_put_get(self, backend):
        keys = [create_test_key(i) for i in range(5)]
        objs = [create_test_memory_obj(shape=_TEST_MEM_SHAPE) for _ in range(5)]

        backend.batched_submit_put_task(keys, objs)
        _wait_for_puts(backend, keys)

        for key in keys:
            assert backend.contains(key)

        results = backend.batched_get_blocking(keys)
        assert len(results) == 5
        for r in results:
            assert r is not None

    def test_exists_in_put_tasks(self, backend):
        key = create_test_key(10)
        assert not backend.exists_in_put_tasks(key)
        # After put completes, the key is removed from put_tasks
        backend.batched_submit_put_task(
            [key], [create_test_memory_obj(shape=_TEST_MEM_SHAPE)]
        )
        _wait_for_puts(backend, [key])
        assert not backend.exists_in_put_tasks(key)

    def test_pin_unpin(self, backend):
        key = create_test_key(11)
        assert backend.pin(key) is True
        assert backend.unpin(key) is True

    def test_per_model_isolation(self, backend, local_cpu_backend, async_loop):
        key_a = CacheEngineKey(
            model_name="model_alpha",
            world_size=1,
            worker_id=0,
            chunk_hash=100,
            dtype=torch.bfloat16,
        )
        key_b = CacheEngineKey(
            model_name="model_beta",
            world_size=1,
            worker_id=0,
            chunk_hash=100,
            dtype=torch.bfloat16,
        )

        mem_obj = create_test_memory_obj(shape=_TEST_MEM_SHAPE)
        backend.batched_submit_put_task([key_a], [mem_obj])
        _wait_for_puts(backend, [key_a])

        assert backend.contains(key_a)
        assert not backend.contains(key_b)

    def test_get_allocator_backend(self, backend, local_cpu_backend):
        result = backend.get_allocator_backend()
        assert result is local_cpu_backend

    def test_batched_get_non_blocking(self, backend, async_loop):
        keys = [create_test_key(i) for i in range(20, 23)]
        objs = [create_test_memory_obj(shape=_TEST_MEM_SHAPE) for _ in range(3)]
        backend.batched_submit_put_task(keys, objs)
        _wait_for_puts(backend, keys)

        future = asyncio.run_coroutine_threadsafe(
            backend.batched_get_non_blocking("lookup_id", keys), async_loop
        )
        results = future.result(timeout=5.0)

        assert len(results) == 3
        for mem_obj in results:
            assert mem_obj is not None

    def test_batched_get_non_blocking_isolates_per_key_failure(
        self, backend, async_loop
    ):
        """
        A transient failure retrieving one key (e.g. a driver error that
        survives all retries) must not fail the whole batch: the other
        keys' results are still returned, the failing key is dropped, and
        the failure is counted in ``get_blocking_failed_count`` -- matching
        the isolation semantics of ``batched_get_blocking``.
        """
        good_key = create_test_key(30)
        bad_key = create_test_key(31)
        mem_obj = create_test_memory_obj(shape=_TEST_MEM_SHAPE)
        # Deterministic content: torch.equal() below is NaN-unsafe (NaN !=
        # NaN), and create_test_memory_obj's buffer is uninitialized.
        mem_obj.tensor.fill_(3.14)
        backend.batched_submit_put_task([good_key], [mem_obj])
        _wait_for_puts(backend, [good_key])

        real_get_blocking_async = backend._get_blocking_async

        async def flaky_get_blocking_async(key):
            if key == bad_key:
                raise RuntimeError("simulated transient ScyllaDB failure")
            return await real_get_blocking_async(key)

        initial_failed_count = backend.get_blocking_failed_count
        with patch.object(
            backend,
            "_get_blocking_async",
            side_effect=flaky_get_blocking_async,
        ):
            future = asyncio.run_coroutine_threadsafe(
                backend.batched_get_non_blocking("lookup_id", [good_key, bad_key]),
                async_loop,
            )
            results = future.result(timeout=5.0)

        assert len(results) == 1
        assert results[0] is not None
        assert torch.equal(results[0].tensor, mem_obj.tensor)
        assert backend.get_blocking_failed_count == initial_failed_count + 1

    def test_close(self, backend):
        backend.close()
        assert not backend.is_connected()

    def test_close_waits_for_in_flight_get(self, backend, local_cpu_backend):
        """close() must drain in-flight gets, not just puts -- otherwise a
        concurrent get races the session teardown it triggers."""
        key = create_test_key(61)
        mem_obj = create_test_memory_obj(shape=_TEST_MEM_SHAPE)
        backend.batched_submit_put_task([key], [mem_obj])
        _wait_for_puts(backend, [key])

        real_allocate = local_cpu_backend.allocate
        release_get = threading.Event()

        def delayed_allocate(*args, **kwargs):
            release_get.wait(timeout=5.0)
            return real_allocate(*args, **kwargs)

        result: dict = {}
        with patch.object(local_cpu_backend, "allocate", side_effect=delayed_allocate):
            get_thread = threading.Thread(
                target=lambda: result.__setitem__("value", backend.get_blocking(key))
            )
            get_thread.start()
            time.sleep(0.05)  # let the get start and register itself in-flight

            close_thread = threading.Thread(target=backend.close)
            close_thread.start()
            time.sleep(0.1)
            assert close_thread.is_alive(), (
                "close() must not return while a get is still in flight"
            )

            release_get.set()
            get_thread.join(timeout=5.0)
            close_thread.join(timeout=5.0)

        assert not close_thread.is_alive()
        assert result["value"] is not None

    def test_ensure_table_async_raises_after_close_even_if_table_cached(
        self, backend, async_loop
    ):
        """_ensure_table_async must fail fast once closed, matching
        _ensure_table's check -- even when the table is already cached, so
        a caller reached only through an async path (e.g.
        batched_get_non_blocking, which never checks _closed itself)
        doesn't silently proceed against a closed session."""
        table = backend._table_name(create_test_key(0))
        backend._tables_created.add(table)
        backend.close()
        future = asyncio.run_coroutine_threadsafe(
            backend._ensure_table_async(table), async_loop
        )
        with pytest.raises(RuntimeError, match="closed"):
            future.result(timeout=5.0)

    @pytest.mark.parametrize(
        "errors,expected",
        [
            ({"host1": _cass_connection.ConnectionBusy("busy")}, True),
            (
                {
                    "host1": _cass_connection.ConnectionBusy("busy"),
                    "host2": _cass_connection.ConnectionBusy("busy"),
                },
                True,
            ),
            (
                {
                    "host1": _cass_connection.ConnectionBusy("busy"),
                    "host2": _cass_top.Unavailable("down"),
                },
                False,
            ),
            ({}, False),
        ],
    )
    def test_is_local_connection_busy_classification(self, errors, expected):
        exc = _cass_cluster.NoHostAvailable("no host available")
        exc.errors = errors
        assert _is_local_connection_busy(exc) is expected

    def test_is_local_connection_busy_rejects_other_exception_types(self):
        assert _is_local_connection_busy(_cass_top.Unavailable("down")) is False

    async def _retry_and_capture_delays(
        self, backend, exc_factory, fail_count: int
    ) -> list[float]:
        delays: list[float] = []
        attempts = {"n": 0}

        async def flaky_op():
            attempts["n"] += 1
            if attempts["n"] <= fail_count:
                raise exc_factory()
            return "ok"

        async def fake_sleep(seconds: float) -> None:
            delays.append(seconds)

        with patch(
            "lmcache.v1.storage_backend.scylla_backend.asyncio.sleep",
            side_effect=fake_sleep,
        ):
            result = await backend._retry_async(flaky_op, "test-op")
        assert result == "ok"
        return delays

    def test_retry_async_uses_short_delay_for_connection_busy(
        self, backend, async_loop
    ):
        def make_exc():
            exc = _cass_cluster.NoHostAvailable("no host available")
            exc.errors = {"host1": _cass_connection.ConnectionBusy("busy")}
            return exc

        future = asyncio.run_coroutine_threadsafe(
            self._retry_and_capture_delays(backend, make_exc, fail_count=1),
            async_loop,
        )
        delays = future.result(timeout=5.0)
        assert delays == [backend._connection_busy_retry_delay]

    def test_retry_async_uses_cluster_delay_for_genuine_transient_error(
        self, backend, async_loop
    ):
        future = asyncio.run_coroutine_threadsafe(
            self._retry_and_capture_delays(
                backend, lambda: _cass_top.Unavailable("down"), fail_count=1
            ),
            async_loop,
        )
        delays = future.result(timeout=5.0)
        assert delays == [backend._operation_retry_delay]
