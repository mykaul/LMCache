# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any
from unittest.mock import MagicMock, patch
import asyncio
import sys
import threading
import time
import types

# Create mock scylla (python-rs-driver) modules BEFORE any other imports, so
# that scylla_backend's import guard finds them and sets
# _SCYLLA_AVAILABLE = True.
#
# Reuse any modules another scylla test file already stubbed (checked via
# "scylla" itself, not each submodule) instead of unconditionally
# overwriting them: scylla_backend.py's `from scylla.errors import
# RequestTimeoutError` etc. only runs on the *first* import across the
# whole pytest process and binds to whichever objects were in sys.modules
# at that moment. If this file's module body still replaced them
# unconditionally, its own local mock classes (e.g. RequestTimeoutError)
# would no longer be the same objects scylla_backend._RETRYABLE_EXCEPTIONS
# actually checks against, breaking any test raising the mocked exception
# type -- overwriting is not just redundant, it silently desyncs identity.
_already_stubbed = "scylla" in sys.modules
_mock_scylla_modules: dict[str, Any] = {}
for _name in [
    "scylla",
    "scylla.enums",
    "scylla.errors",
    "scylla.execution_profile",
    "scylla.policies",
    "scylla.policies.load_balancing",
    "scylla.session",
    "scylla.session_builder",
    "scylla.statement",
]:
    if _already_stubbed:
        _mock_scylla_modules[_name] = sys.modules[_name]
    else:
        _mod = types.ModuleType(_name)
        _mock_scylla_modules[_name] = _mod
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

_scylla_enums = _mock_scylla_modules["scylla.enums"]
_scylla_errors = _mock_scylla_modules["scylla.errors"]
_scylla_execution_profile = _mock_scylla_modules["scylla.execution_profile"]
_scylla_load_balancing = _mock_scylla_modules["scylla.policies.load_balancing"]
_scylla_session = _mock_scylla_modules["scylla.session"]
_scylla_session_builder = _mock_scylla_modules["scylla.session_builder"]
_scylla_statement = _mock_scylla_modules["scylla.statement"]

if not _already_stubbed:
    # Wire sub-modules into parent
    _mock_scylla_modules["scylla"].enums = _scylla_enums
    _mock_scylla_modules["scylla"].errors = _scylla_errors
    _mock_scylla_modules["scylla"].execution_profile = _scylla_execution_profile
    _mock_scylla_modules["scylla"].policies = _mock_scylla_modules["scylla.policies"]
    _mock_scylla_modules["scylla.policies"].load_balancing = _scylla_load_balancing
    _mock_scylla_modules["scylla"].session = _scylla_session
    _mock_scylla_modules["scylla"].session_builder = _scylla_session_builder
    _mock_scylla_modules["scylla"].statement = _scylla_statement

    # Populate mock attributes each module-level consumer expects
    _scylla_enums.Consistency = type(
        "MockConsistency",
        (),
        {"LocalOne": 1, "Quorum": 2, "All": 3},
    )
    _scylla_enums.Compression = type("MockCompression", (), {"Lz4": 1, "Snappy": 2})
    # Real Exception subclasses: scylla_backend.py catches these by type in
    # its retry logic (_RETRYABLE_EXCEPTIONS), so they must be genuine
    # exception classes, not bare `type(...)` placeholders.
    _scylla_errors.ExecuteError = type("MockExecuteError", (Exception,), {})
    _scylla_errors.PrepareError = type("MockPrepareError", (Exception,), {})
    _scylla_errors.RequestTimeoutError = type(
        "MockRequestTimeoutError", (Exception,), {}
    )
    _scylla_errors.SessionConnectionError = type(
        "MockSessionConnectionError", (Exception,), {}
    )
    _scylla_execution_profile.ExecutionProfile = lambda **kw: None

    class _MockNodeLocationPreference:
        """Mimics ``scylla.policies.load_balancing.NodeLocationPreference``."""

        @staticmethod
        def datacenter(name: str):
            return ("dc", name)

        @staticmethod
        def datacenter_and_rack(dc: str, rack: str):
            return ("dc_rack", dc, rack)

    _scylla_load_balancing.DefaultPolicy = lambda **kw: None
    _scylla_load_balancing.NodeLocationPreference = _MockNodeLocationPreference
    _scylla_session.Session = type("MockSessionImport", (), {})
    _scylla_session_builder.SessionBuilder = type("MockSessionBuilderImport", (), {})
    _scylla_statement.PreparedStatement = type("MockPreparedStatementImport", (), {})

# Now it's safe to import the scylla backend — _SCYLLA_AVAILABLE will be True
# Third Party
import pytest  # noqa: E402
import torch  # noqa: E402

# First Party
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey  # noqa: E402
from lmcache.v1.config import LMCacheEngineConfig  # noqa: E402
from lmcache.v1.metadata import LMCacheMetadata  # noqa: E402
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend  # noqa: E402
from lmcache.v1.storage_backend.scylla_backend import ScyllaDBBackend  # noqa: E402
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
            "consistency_level": "LocalOne",
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
    """Return a mocked SessionBuilder whose session stores rows in a dict.

    Mirrors the real ``scylla`` (python-rs-driver) API surface --
    ``SessionBuilder().connect()``, ``Session.execute`` (async, returning a
    ``RequestResult``-like object with ``iter_current_page()``),
    ``Session.prepare`` (async) -- rather than the old ``cassandra`` driver's
    callback-based ``execute_async``/``add_callbacks``.
    """

    store: dict[str, dict[tuple[int, int], tuple[bytes, bytes]]] = {}
    executed_cql: list[str] = []

    def _get_mock_table(stmt: object, fallback: str) -> str:
        if isinstance(stmt, MagicMock) and "_mock_table" in stmt.__dict__:
            return stmt.__dict__["_mock_table"]
        return fallback

    class MockRequestResult:
        """Mimics ``scylla.results.RequestResult`` for a single, final page."""

        def __init__(self, rows: list):
            self._rows = rows

        def iter_current_page(self):
            return iter(self._rows)

    class MockSession:
        def _compute_rows(self, stmt, params=None) -> list:
            # _ensure_keyspace_exists_async() passes a raw CQL string (no
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

            if cql.startswith("SELECT") and "AND layer_id" in cql:
                chunk_hash, layer_id = params
                item = tbl.get((chunk_hash, layer_id))
                if item is not None:
                    # Default row factory is dict-of-columns in production
                    # (see scylla_backend._reconstruct_tensor); match here.
                    kdata, vdata = item
                    return [{"k_data": kdata, "v_data": vdata}]
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

        async def execute(
            self, stmt, params=None, *, factory=None, paging_state=None, paged=True
        ) -> MockRequestResult:
            return MockRequestResult(self._compute_rows(stmt, params))

        async def prepare(self, cql):
            stmt = MagicMock()
            stmt._mock_cql = cql
            stmt.with_consistency.return_value = stmt
            stmt.set_is_idempotent.return_value = stmt
            match = None
            for word in cql.split():
                if word.startswith("kv_chunks_"):
                    match = word
                    break
            if match:
                stmt._mock_table = match
            return stmt

        async def use_keyspace(
            self, keyspace: str, case_sensitive: bool = False
        ) -> None:
            """No-op: `store` above is a flat dict of table->rows, not
            scoped per-keyspace, so there is nothing to switch."""

    class MockSessionBuilder:
        def contact_points(self, contact_points):
            return self

        def execution_profile(self, execution_profile):
            return self

        def compression(self, compression):
            return self

        def connection_timeout(self, timeout):
            return self

        async def connect(self):
            return MockSession()

    mock_session_builder = MockSessionBuilder()
    mock_session_builder.executed_cql = executed_cql
    mock_session_builder.store = store
    return mock_session_builder


@pytest.fixture
def backend(async_loop, local_cpu_backend, mock_in_memory_scylla):
    config = create_test_config()
    metadata = create_test_metadata()
    with patch(
        "lmcache.v1.storage_backend.scylla_backend.SessionBuilder",
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
                "lmcache.v1.storage_backend.scylla_backend.SessionBuilder",
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
                "lmcache.v1.storage_backend.scylla_backend.SessionBuilder",
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

    def test_connect_uses_rack_scoped_preference_when_local_rack_configured(
        self, async_loop, local_cpu_backend, mock_in_memory_scylla
    ):
        """local_rack must select
        NodeLocationPreference.datacenter_and_rack(), not
        NodeLocationPreference.datacenter() -- the latter has no rack
        parameter at all."""
        config = create_test_config({"scylla": {"local_rack": "rack1"}})
        metadata = create_test_metadata()
        with (
            patch(
                "lmcache.v1.storage_backend.scylla_backend.SessionBuilder",
                return_value=mock_in_memory_scylla,
            ),
            patch(
                "lmcache.v1.storage_backend.scylla_backend"
                ".NodeLocationPreference.datacenter_and_rack"
            ) as mock_dc_rack,
            patch(
                "lmcache.v1.storage_backend.scylla_backend"
                ".NodeLocationPreference.datacenter"
            ) as mock_dc_only,
        ):
            backend = ScyllaDBBackend(
                dst_device="cpu",
                config=config,
                metadata=metadata,
                local_cpu_backend=local_cpu_backend,
                loop=async_loop,
            )
            mock_dc_rack.assert_called_once_with("datacenter1", "rack1")
            mock_dc_only.assert_not_called()
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

    def test_put_sends_bytes_matching_the_source_tensor(
        self, backend, mock_in_memory_scylla
    ):
        """PUT must hand the driver real ``bytes`` of each layer's K/V slice
        (the blob serializer rejects anything else, e.g. an ``ndarray`` --
        see ``ScyllaDBBackend._split_kv``), byte-for-byte identical to the
        source tensor's own memory at that layer's offset."""
        key = create_test_key(50)
        mem_obj = create_test_memory_obj(shape=_TEST_MEM_SHAPE)
        assert mem_obj.tensor is not None
        tensor_bytes = mem_obj.tensor.view(torch.uint8).numpy().ravel()

        backend.batched_submit_put_task([key], [mem_obj])
        _wait_for_puts(backend, [key])

        layer_id, stored = next(
            (layer_id, item)
            for table in mock_in_memory_scylla.store.values()
            for (chunk_hash, layer_id), item in table.items()
            if chunk_hash == key.chunk_hash
        )
        k_blob, v_blob = stored
        assert isinstance(k_blob, bytes)
        assert isinstance(v_blob, bytes)
        num_layers = _TEST_MEM_SHAPE[1]
        bytes_per_layer = len(k_blob)
        k_off = layer_id * bytes_per_layer
        v_off = (num_layers + layer_id) * bytes_per_layer
        assert k_blob == tensor_bytes[k_off : k_off + bytes_per_layer].tobytes()
        assert v_blob == tensor_bytes[v_off : v_off + bytes_per_layer].tobytes()

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

    def test_retry_async_uses_operation_delay_for_transient_error(
        self, backend, async_loop
    ):
        """_retry_async uses a single retry budget (operation_retry_delay)
        for every retryable error -- the old driver's separate, shorter
        budget for local send-buffer backpressure has no equivalent here
        (see the class's _RETRYABLE_EXCEPTIONS/_retry_async docstrings)."""
        future = asyncio.run_coroutine_threadsafe(
            self._retry_and_capture_delays(
                backend,
                lambda: _scylla_errors.RequestTimeoutError("timed out"),
                fail_count=1,
            ),
            async_loop,
        )
        delays = future.result(timeout=5.0)
        assert delays == [backend._operation_retry_delay]
