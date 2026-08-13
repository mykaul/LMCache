# SPDX-License-Identifier: Apache-2.0

# Standard
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError,
)
from typing import (
    Any,
    Callable,
    Coroutine,
    List,
    NamedTuple,
    Optional,
    Sequence,
    TypeVar,
)
import asyncio
import functools
import re
import sys
import threading
import time

# Third Party
import torch

try:
    # Third Party
    from scylla.enums import Compression, Consistency
    from scylla.errors import (
        ExecuteError,
        PrepareError,
        RequestTimeoutError,
        SessionConnectionError,
    )
    from scylla.execution_profile import ExecutionProfile
    from scylla.policies.load_balancing import DefaultPolicy, NodeLocationPreference
    from scylla.session import Session
    from scylla.session_builder import SessionBuilder
    from scylla.statement import PreparedStatement

    _SCYLLA_AVAILABLE = True
    # Transient, retryable conditions: the coordinator/replica(s) were
    # temporarily unavailable or too slow, or the client couldn't reach the
    # cluster at all. Anything else (e.g. InvalidRequest, syntax errors,
    # AuthenticationFailed) indicates a permanent/logical error and must
    # not be retried. Deliberately narrower than ``UseKeyspaceError``
    # (``RequestTimeoutError``'s actual parent in this driver's hierarchy):
    # that parent also covers permanent config errors like
    # ``BadKeyspaceNameError``, which must not be retried.
    _RETRYABLE_EXCEPTIONS: tuple = (
        ExecuteError,
        PrepareError,
        RequestTimeoutError,
        SessionConnectionError,
    )
except ImportError:
    _SCYLLA_AVAILABLE = False
    _RETRYABLE_EXCEPTIONS = ()


class _RetryBudget(NamedTuple):
    """Backoff state for the retry policy in :meth:`ScyllaDBBackend._retry_async`.

    Bundles the current backoff delay, the retry ceiling, and how many
    retries remain into one immutable value; :meth:`next` derives each
    subsequent attempt's state, so the delay-doubling-with-cap arithmetic
    lives in exactly one place.
    """

    delay: float
    retries_left: int
    max_delay: float

    def next(self) -> "_RetryBudget":
        """Return this budget's state after one retry attempt is consumed.

        The delay doubles each attempt, capped at ``max_delay``. Returns a
        new object (NamedTuples are immutable) rather than mutating self.
        """
        return _RetryBudget(
            delay=min(self.delay * 2, self.max_delay),
            retries_left=self.retries_left - 1,
            max_delay=self.max_delay,
        )


# First Party
from lmcache import torch_device_type  # noqa: E402
from lmcache.logging import init_logger  # noqa: E402
from lmcache.observability import LMCStatsMonitor, PrometheusLogger  # noqa: E402
from lmcache.utils import (  # noqa: E402
    TORCH_DTYPE_TO_STR_DTYPE,
    CacheEngineKey,
    LayerCacheEngineKey,
)
from lmcache.v1.config import LMCacheEngineConfig  # noqa: E402
from lmcache.v1.memory_management import MemoryFormat, MemoryObj  # noqa: E402
from lmcache.v1.metadata import LMCacheMetadata  # noqa: E402
from lmcache.v1.storage_backend.abstract_backend import (  # noqa: E402
    AllocatorBackendInterface,
    StoragePluginInterface,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend  # noqa: E402

logger = init_logger(__name__)


_CQL_CREATE_KEYSPACE = (
    "CREATE KEYSPACE IF NOT EXISTS {keyspace} WITH replication = "
    "{{'class': 'NetworkTopologyStrategy', '{local_dc}': {replication_factor}}}"
)

_CQL_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS {table} (
        chunk_hash bigint,
        layer_id tinyint,
        k_data blob,
        v_data blob,
        PRIMARY KEY ((chunk_hash, layer_id))
    )
"""

_CQL_CREATE_TABLE_WITH_COMPRESSION = (
    _CQL_CREATE_TABLE
    + "  WITH compression = {{'sstable_compression': '{compressor_class}'}};"
)

# Real per-cell USING TTL: the cluster enforces expiry at read time, with
# no reliance on a background sweep.
_CQL_INSERT = (
    "INSERT INTO {table} (chunk_hash, layer_id, k_data, v_data) "
    "VALUES (?, ?, ?, ?) USING TTL ?"
)

_CQL_SELECT_ONE_LAYER = (
    "SELECT k_data, v_data FROM {table} WHERE chunk_hash = ? AND layer_id = ?"
)

_CQL_EXISTS = (
    "SELECT chunk_hash FROM {table} WHERE chunk_hash = ? AND layer_id = ? LIMIT 1"
)

_CQL_DELETE_ONE = "DELETE FROM {table} WHERE chunk_hash = ? AND layer_id = ?"


_T = TypeVar("_T")


class ScyllaDBBackend(StoragePluginInterface):
    """
    ScyllaDB-backed storage backend for LMCache.

    Stores KV caches in ScyllaDB using per-model tables with per-cell CQL
    TTL (``INSERT ... USING TTL``) -- a row is fully invisible to reads the
    instant its TTL elapses, not on any later background sweep (see
    ``_CQL_INSERT``). MLA (Multi-head Latent Attention) models are not
    supported: the table schema and hidden-dim calculation assume the
    standard ``(num_layers, 2, num_tokens, num_heads, head_size)``
    ``kv_shape`` layout.

    Each layer of a chunk is its own CQL partition -- the table's primary
    key is the composite ``((chunk_hash, layer_id))``, with no clustering
    column. This deliberately trades away the single-partition, single
    round-trip "read all layers of a chunk" query a plain
    ``PRIMARY KEY (chunk_hash, layer_id)`` (partition key + clustering key)
    schema would allow, in exchange for two things that matter more at
    model scale: partition size stays small and bounded (a single layer's
    row, not ``num_layers`` of them) regardless of ``chunk_size``/
    ``hidden_dim``/dtype, and a popular shared prefix's traffic spreads
    across ``num_layers`` replica sets instead of hammering one. Whole-chunk
    reads/writes/deletes fan out into ``num_layers`` concurrent per-layer
    CQL operations instead of a single range query; this is intentional
    and assumes an I/O-bound workload where that fan-out is cheap relative
    to the network/storage work it overlaps.

    There is deliberately no client-side cap on how many CQL requests can
    be in flight at once: ScyllaDB has its own admission control, and a
    static client-side number can't know a given cluster's real capacity
    (too conservative on a big cluster, still not enough on a small one).
    Transient overload is instead handled where it actually surfaces --
    see :meth:`_retry_async` and ``operation_max_retries`` below.

    TODO: puts are already documented (see :meth:`batched_submit_put_task`)
    as non-blocking, best-effort work done for *future* requests' benefit,
    while gets sit on the latency-critical serving path for the *current*
    request. Those aren't the same priority, and ScyllaDB's Service Levels
    (per-role ``SHARES``, i.e. workload prioritization) exist specifically
    to express that -- e.g. a lower-share role for the connection issuing
    puts, so it yields cluster resources to concurrent gets under load.
    Not implemented: needs cluster-side service level/role provisioning,
    which is a separate decision from anything this class can do alone.

    Configuration is passed via ``config.extra_config["scylla"]`` as a dict
    with the following optional keys:

    - ``contact_points`` (list[str], default ``["127.0.0.1"]``)
    - ``port`` (int, default ``9042``)
    - ``keyspace`` (str, default ``"lmcache"``): created automatically
      (``CREATE KEYSPACE IF NOT EXISTS``) if it does not already exist,
      using ``NetworkTopologyStrategy`` scoped to ``local_dc`` -- see
      ``keyspace_replication_factor``.
    - ``local_dc`` (str, default ``"datacenter1"``)
    - ``local_rack`` (str or None)
    - ``keyspace_replication_factor`` (int, default ``1``): replication
      factor for ``local_dc`` when auto-creating ``keyspace``. Only
      relevant the first time this backend connects to a given keyspace;
      to use a multi-DC replication topology, pre-create the keyspace
      manually instead.
    - ``ttl_seconds`` (int, default ``86400``)
    - ``table_compression`` (str, default ``""``): the literal SSTable
      compressor class name to pass to CQL's ``WITH compression = {
      'sstable_compression': ...}`` (e.g. ``"LZ4Compressor"``,
      ``"LZ4WithDictsCompressor"``, ``"ZstdCompressor"``,
      ``"ZstdWithDictsCompressor"``), just like ``consistency_level``
      below is the literal name of a driver enum member. An empty string
      disables compression (ScyllaDB requires the clause -- omitting it
      falls back to the cluster's own default). Not validated client-side --
      an invalid name surfaces as a CQL ``InvalidRequest`` error on table
      creation.
    - ``wire_compression`` (bool, default ``False``): whether the driver
      negotiates native-protocol (lz4) compression. Off by default -- KV
      tensor bytes are already dense, so compression only costs CPU.
    - ``consistency_level`` (str, default ``"LocalOne"``): a
      ``scylla.enums.Consistency`` member name, e.g. ``LocalOne``,
      ``Quorum``, ``All``.
    - ``timeout_secs`` (float, default ``30.0``)
    - ``max_connect_retries`` (int, default ``3``)
    - ``connect_retry_delay`` (float, default ``1.0``)
    - ``operation_max_retries`` (int, default ``2``): number of *additional*
      attempts (beyond the first) for a single CQL operation when it fails
      with a transient error (``ExecuteError``, ``PrepareError``,
      ``RequestTimeoutError``, ``SessionConnectionError``).
    - ``operation_retry_delay`` (float, default ``0.05``): initial delay in
      seconds (50ms) between operation retries; doubles after each attempt,
      capped at ``operation_retry_max_delay`` (default ``5.0``).
    - ``gil_switch_interval_secs`` (float, default ``None``): if set, calls
      ``sys.setswitchinterval()`` with this value. Process-global, not
      scoped to this backend -- see benchmarks/scylla_e2e/gil_load_bench.py
      for measured impact under GIL contention.

    Not currently supported by ``python-rs-driver`` (dropped when this
    backend was ported off the old ``cassandra``-based driver, not
    carried forward as a no-op): per-request speculative execution, and
    explicit socket send/receive buffer sizing.
    """

    def __init__(
        self,
        dst_device: str = torch_device_type,
        config: Optional[LMCacheEngineConfig] = None,
        metadata: Optional[LMCacheMetadata] = None,
        local_cpu_backend: Optional[LocalCPUBackend] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        super().__init__(dst_device=dst_device)
        if not _SCYLLA_AVAILABLE:
            raise ImportError(
                "python-rs-driver (package `scylla`) is required for "
                "ScyllaDBBackend. Not yet on PyPI -- build it from "
                "https://github.com/scylladb/python-rs-driver (maturin)."
            )

        self.config = config
        self.metadata = metadata
        self.local_cpu_backend = local_cpu_backend
        self.loop = loop

        extra = config.extra_config if config is not None else {}
        scylla_cfg = extra.get("scylla", {}) if isinstance(extra, dict) else {}

        self._contact_points: list[str] = scylla_cfg.get(
            "contact_points", ["127.0.0.1"]
        )
        self._port: int = scylla_cfg.get("port", 9042)
        self._keyspace: str = scylla_cfg.get("keyspace", "lmcache")
        self._local_dc: str = scylla_cfg.get("local_dc", "datacenter1")
        self._local_rack: Optional[str] = scylla_cfg.get("local_rack", None)
        self._ttl_seconds: int = scylla_cfg.get("ttl_seconds", 86400)
        self._keyspace_replication_factor: int = int(
            scylla_cfg.get("keyspace_replication_factor", 1)
        )

        self._compressor_class: str = str(scylla_cfg.get("table_compression", ""))
        self._wire_compression: bool = bool(scylla_cfg.get("wire_compression", False))

        cl_name = str(scylla_cfg.get("consistency_level", "LocalOne"))
        if not hasattr(Consistency, cl_name):
            raise ValueError(
                f"Invalid scylla.consistency_level={cl_name!r}; must be a "
                f"valid scylla.enums.Consistency name (e.g. "
                f"LocalOne, Quorum, All)"
            )
        self._consistency: "Consistency" = getattr(Consistency, cl_name)

        self._timeout: float = float(scylla_cfg.get("timeout_secs", 30))
        self._max_connect_retries: int = int(scylla_cfg.get("max_connect_retries", 3))
        self._connect_retry_delay: float = float(
            scylla_cfg.get("connect_retry_delay", 1.0)
        )
        self._operation_max_retries: int = int(
            scylla_cfg.get("operation_max_retries", 2)
        )
        self._operation_retry_delay: float = float(
            scylla_cfg.get("operation_retry_delay", 0.05)
        )
        self._operation_retry_max_delay: float = float(
            scylla_cfg.get("operation_retry_max_delay", 5.0)
        )
        self._gil_switch_interval_secs: Optional[float] = scylla_cfg.get(
            "gil_switch_interval_secs", None
        )
        if self._gil_switch_interval_secs is not None:
            # Process-global (see docstring), not scoped to this backend.
            sys.setswitchinterval(self._gil_switch_interval_secs)
            logger.info(
                "ScyllaDBBackend set process-wide GIL switch interval to "
                "%.6fs (default is 0.005s)",
                self._gil_switch_interval_secs,
            )

        # Local put task tracking (dedup)
        self._put_tasks: set[CacheEngineKey] = set()
        self._put_tasks_lock = threading.Lock()

        # In-flight get count, so close() can drain gets the same way it
        # drains puts via _put_tasks.
        self._active_gets = 0
        self._active_gets_lock = threading.Lock()

        # Dedicated pool for blob copy work (split/reconstruct) so it never
        # starves the shared default executor's prepare round-trips. Exactly
        # one worker: these copies hold the GIL end to end, so extra threads
        # add no throughput and only convoy on it (measured, 32MiB chunks at
        # concurrency 30: same wall clock, per-chunk p99 9.8ms -> 133ms
        # going 1 -> 8). Only a driver that took buffer-protocol blobs --
        # removing the bytes materialization -- would change that.
        self._blob_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="lmcache-scylla-blob",
        )

        # Table existence cache (avoids redundant CREATE TABLE), plus a
        # per-table in-flight dedup so N concurrent first-time accesses to
        # the same never-seen table don't each independently fire a
        # redundant (if harmless/idempotent) CREATE TABLE.
        self._tables_created: set[str] = set()
        self._tables_lock = threading.Lock()
        self._table_creation_locks: dict[str, asyncio.Lock] = {}

        # Prepared-statement cache, keyed by CQL text. The driver performs a
        # network round-trip on every Session.prepare() call and does not
        # deduplicate repeated identical queries itself, so callers are
        # expected to cache and reuse PreparedStatement objects.
        self._prepared_statements: dict[str, "PreparedStatement"] = {}
        self._prepared_statements_lock = threading.Lock()

        # Driver resources
        self._session: Optional["Session"] = None
        self._closed = False

        # Cached hidden_dim/num_layers from metadata
        if metadata is not None and metadata.kv_shape is not None:
            if metadata.use_mla:
                raise ValueError(
                    "ScyllaDBBackend does not support MLA (Multi-head "
                    "Latent Attention) models: the table schema and "
                    "hidden-dim calculation assume the standard "
                    "(num_layers, 2, num_tokens, num_heads, head_size) "
                    "kv_shape layout."
                )
            # kv_shape: (num_layers, 2, num_tokens, num_heads, head_size)
            self._num_layers: int = metadata.kv_shape[0]
            self._hidden_dim: int = metadata.kv_shape[3] * metadata.kv_shape[4]
        else:
            raise ValueError(
                "ScyllaDBBackend requires metadata with kv_shape "
                "to determine hidden_dim"
            )

        # Failure counters for observability / health-check consumption.
        # _get_blocking_failed_count is incremented from multiple call
        # paths/threads (unlike _put_failed_count, only ever touched from
        # the event loop thread), so += 1 alone can lose increments.
        self._get_blocking_failed_count = 0
        self._get_blocking_failed_count_lock = threading.Lock()
        self._put_failed_count = 0

        self._connect()

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Register Prometheus gauges for this backend (best-effort)."""
        if self.metadata is None:
            return
        try:
            prometheus_logger = PrometheusLogger.GetOrCreate(
                self.metadata,
                config=self.config,
            )
            prometheus_logger.remote_put_task_num.set_function(
                lambda: len(self._put_tasks)
            )
            prometheus_logger.get_blocking_failed_count.set_function(
                lambda: self._get_blocking_failed_count
            )
            prometheus_logger.put_failed_count.set_function(
                lambda: self._put_failed_count
            )
        except Exception as e:
            logger.warning(
                "Failed to set up Prometheus metrics for ScyllaDBBackend: %s", e
            )

    def _connect(self) -> None:
        """Connect to ScyllaDB with retries.

        Dispatches :meth:`_connect_async` onto ``self.loop`` via
        :meth:`_sync_execute` and blocks this (the calling) thread for the
        result -- ``SessionBuilder.connect()``/``Session.execute()`` are
        coroutines in this driver, unlike the old driver's synchronous
        ``Cluster.connect()``.
        """
        if self._closed:
            return
        # Bounds the whole retry loop: each attempt is itself capped by
        # connection_timeout(self._timeout), plus a retry_delay sleep
        # between attempts.
        overall_timeout = self._max_connect_retries * (
            self._timeout + self._connect_retry_delay
        )
        self._sync_execute(self._connect_async(), timeout=overall_timeout)

    async def _connect_async(self) -> None:
        """Async core of :meth:`_connect`."""
        last_exc: Exception = RuntimeError("Failed to connect to ScyllaDB")
        for attempt in range(1, self._max_connect_retries + 1):
            try:
                # RackAwareRoundRobinPolicy-equivalent: DefaultPolicy takes a
                # single NodeLocationPreference, rack-scoped only if
                # local_rack is configured (mirrors the old driver's
                # dc-only vs. dc+rack policy choice).
                location = (
                    NodeLocationPreference.datacenter_and_rack(
                        self._local_dc, self._local_rack
                    )
                    if self._local_rack is not None
                    else NodeLocationPreference.datacenter(self._local_dc)
                )
                profile = ExecutionProfile(
                    timeout=self._timeout,
                    consistency=self._consistency,
                    load_balancing_policy=DefaultPolicy(
                        node_location_preference=location, token_aware=True
                    ),
                )
                # Only Lz4/Snappy are exposed (no plain bool toggle); Lz4
                # matches the old driver's default when compression was on.
                compression = Compression.Lz4 if self._wire_compression else None
                builder = (
                    SessionBuilder()
                    .contact_points([(cp, self._port) for cp in self._contact_points])
                    .execution_profile(profile)
                    .compression(compression)
                    .connection_timeout(self._timeout)
                )
                # Connect without a keyspace first: the configured keyspace
                # may not exist yet, and binding to a nonexistent keyspace
                # up front fails outright (unlike CREATE TABLE IF NOT EXISTS
                # below). _ensure_keyspace_exists_async() creates it if
                # needed, then use_keyspace() binds this session to it.
                self._session = await builder.connect()
                await self._ensure_keyspace_exists_async()
                await self._session.use_keyspace(self._keyspace)
                logger.info(
                    "Connected to ScyllaDB at %s:%d/%s (attempt %d)",
                    self._contact_points,
                    self._port,
                    self._keyspace,
                    attempt,
                )
                return
            except Exception as e:
                last_exc = e
                logger.warning(
                    "ScyllaDB connection attempt %d/%d failed: %s",
                    attempt,
                    self._max_connect_retries,
                    e,
                )
                self._cleanup_resources()
                if attempt < self._max_connect_retries:
                    await asyncio.sleep(self._connect_retry_delay)

        logger.error(
            "Failed to connect to ScyllaDB after %d attempts",
            self._max_connect_retries,
        )
        raise last_exc

    async def _ensure_keyspace_exists_async(self) -> None:
        """Create the configured keyspace if it does not already exist.

        Uses ``NetworkTopologyStrategy`` scoped to ``local_dc`` (matching
        the DC-aware load balancing policy this backend already uses), so
        a single-DC deployment -- the common case for a co-located LMCache
        ScyllaDB -- needs no manual keyspace provisioning. A multi-DC
        keyspace can still be pre-created manually with whatever
        replication topology is appropriate before pointing this backend
        at it; ``CREATE KEYSPACE IF NOT EXISTS`` is then a no-op.

        Called from :meth:`_connect_async` on a session not yet bound to a
        keyspace (``use_keyspace`` fails if the keyspace doesn't exist yet,
        so this must run first).
        """
        cql = _CQL_CREATE_KEYSPACE.format(
            keyspace=self._keyspace,
            local_dc=self._local_dc,
            replication_factor=self._keyspace_replication_factor,
        )
        await self._safe_session.execute(cql)

    def _cleanup_resources(self) -> None:
        """Release this backend's ScyllaDB session.

        The new driver exposes no explicit ``close()``/``shutdown()`` on
        ``Session`` (unlike the old driver's ``Cluster.shutdown()``), so
        dropping the reference is the whole cleanup.
        """
        self._session = None

    @staticmethod
    def _sanitize_model(name: str) -> str:
        """Sanitize a model name for use as a CQL table name component."""
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        if not safe:
            safe = "unknown"
        elif safe[0].isdigit():
            safe = "m_" + safe
        return safe

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _resolve_table_name(
        model_name: str, world_size: int, worker_id: int, dtype: torch.dtype
    ) -> str:
        """Compute the table name for one (model, world_size, worker_id,
        dtype) combination. Cached: bounded by distinct combinations per
        process (see ``_prepared_statements``), not by request volume.
        """
        model = ScyllaDBBackend._sanitize_model(model_name)
        dtype_str = TORCH_DTYPE_TO_STR_DTYPE[dtype]
        return f"kv_chunks_{model}_world{world_size}_worker{worker_id}_{dtype_str}"

    def _table_name(self, key: CacheEngineKey) -> str:
        """Return the CQL table name for the given key."""
        return self._resolve_table_name(
            key.model_name, key.world_size, key.worker_id, key.dtype
        )

    def _table_for(self, key: CacheEngineKey) -> str:
        """Return the CQL table name for *key*, creating the table if needed.

        For use from plain synchronous call sites (arbitrary caller
        threads). Do not call this from a coroutine already running on
        ``self.loop`` -- use :meth:`_table_for_async` instead, or it will
        self-deadlock (see :meth:`_ensure_table`).

        :param key: The cache key whose table name is being resolved.
        :return: The CQL table name, guaranteed to exist in ScyllaDB.
        """
        table = self._table_name(key)
        self._ensure_table(table)
        return table

    async def _table_for_async(self, key: CacheEngineKey) -> str:
        """Async variant of :meth:`_table_for`.

        Use this from coroutines already running on ``self.loop`` (e.g.
        :meth:`_contains_async`, :meth:`_get_blocking_async`).

        :param key: The cache key whose table name is being resolved.
        :return: The CQL table name, guaranteed to exist in ScyllaDB.
        """
        table = self._table_name(key)
        await self._ensure_table_async(table)
        return table

    @property
    def _safe_loop(self) -> asyncio.AbstractEventLoop:
        if self.loop is None:
            raise RuntimeError("Event loop is required for ScyllaDBBackend")
        return self.loop

    @property
    def _safe_session(self) -> "Session":
        # Single read of self._session: close() can concurrently null it
        # out from another thread between a check and a second read, which
        # would let a None slip through as if it were a live Session.
        session = self._session
        if session is None:
            raise RuntimeError("ScyllaDB session is not connected")
        return session

    def _ensure_table(self, table: str) -> None:
        """Create the CQL table if it does not already exist.

        For use from plain synchronous call sites only. This dispatches
        onto ``self.loop`` and blocks the *calling* thread for the result;
        calling it from a coroutine already running on that same loop
        would self-deadlock (the loop's thread would be blocked waiting
        for work it can only run on... its own, now-blocked, thread). Use
        :meth:`_ensure_table_async` from async call sites instead.
        """
        if self._closed:
            raise RuntimeError("ScyllaDBBackend is closed")
        if table not in self._tables_created:
            self._sync_execute(self._ensure_table_async(table))

    async def _ensure_table_async(self, table: str) -> None:
        """Async core of :meth:`_ensure_table`.

        A per-table ``asyncio.Lock`` de-duplicates concurrent first-time
        creation attempts for the *same* table (avoiding a thundering herd
        of redundant, if harmless/idempotent, ``CREATE TABLE`` statements
        when many coroutines race to access a never-seen-before table at
        once). The lock is only ever acquired/released on this backend's
        single event loop thread, so a plain dict (no extra guard) is safe
        for tracking per-table locks.
        """
        if self._closed:
            raise RuntimeError("ScyllaDBBackend is closed")
        if table in self._tables_created:
            return
        # No clause at all means "use ScyllaDB's server-side default
        # compressor", not "disabled" -- always pass the clause explicitly.
        cql = _CQL_CREATE_TABLE_WITH_COMPRESSION.format(
            table=table,
            compressor_class=self._compressor_class,
        )
        lock = self._table_creation_locks.setdefault(table, asyncio.Lock())
        async with lock:
            if table in self._tables_created:
                return
            await self._execute(cql)
            with self._tables_lock:
                self._tables_created.add(table)
            self._table_creation_locks.pop(table, None)
        logger.debug("Ensured table exists: %s", table)

    def _prepare(self, cql: str) -> "PreparedStatement":
        """Prepare a CQL statement and set the configured consistency level.

        Sync entry point for callers on a thread other than the event
        loop's own (currently just :meth:`remove`'s ``LayerCacheEngineKey``
        branch). Async call sites should call :meth:`_prepare_async`
        directly instead -- routing through here would self-deadlock
        (:meth:`_sync_execute` blocks the calling thread waiting for the
        loop to run the coroutine, which doesn't work if that thread *is*
        the loop).

        :param cql: The CQL text to prepare.
        :return: A cached or newly-prepared statement with ``consistency``
            set.
        """
        # Lock-free read: dict.get is GIL-atomic; the lock guards only the write.
        cached = self._prepared_statements.get(cql)
        if cached is None:
            cached = self._sync_execute(self._prepare_async(cql))
        return cached

    async def _prepare_async(self, cql: str) -> "PreparedStatement":
        """Async core of :meth:`_prepare`; call this directly from coroutines
        already running on the event loop.

        ``PreparedStatement`` objects are cached client-side, keyed by CQL
        text: the driver performs a network round-trip on every
        ``Session.prepare()`` call and does not deduplicate repeated
        identical queries itself. This cache is intentionally unbounded:
        the number of distinct CQL strings is bounded by the number of
        distinct (model, world_size, worker_id, dtype) combinations a
        given process handles, not by request volume.

        On a cache miss, retries transient failures the same as
        :meth:`_execute` (see :meth:`_retry_async`) -- ``Session.prepare()``
        dispatches over the same connections as any other request and can
        hit the same transient conditions. ``Session.prepare()`` is
        natively async in this driver, so unlike the old driver's blocking
        call, no executor offload is needed here.

        :param cql: The CQL text to prepare.
        :return: A cached or newly-prepared statement with
            ``consistency`` and ``is_idempotent`` set.
        """
        # Lock-free read (see _prepare); the lock guards only the miss write.
        cached = self._prepared_statements.get(cql)
        if cached is not None:
            return cached

        async def _do_prepare() -> "PreparedStatement":
            return await self._safe_session.prepare(cql)

        stmt = await self._retry_async(_do_prepare, "prepare")
        # PreparedStatement is immutable here (unlike the old driver's
        # in-place attribute mutation) -- with_consistency/set_is_idempotent
        # each return a new statement, so the result must be reassigned.
        # All statements here are primary-key upserts/selects/deletes, so
        # duplicate execution is always safe -- hence is_idempotent=True.
        stmt = stmt.with_consistency(self._consistency).set_is_idempotent(True)
        with self._prepared_statements_lock:
            self._prepared_statements[cql] = stmt
        return stmt

    def _sync_execute(
        self, coro: Coroutine[Any, Any, _T], timeout: Optional[float] = None
    ) -> _T:
        """Execute a coroutine on the backend's event loop synchronously."""
        loop = self._safe_loop
        if timeout is None:
            timeout = self._timeout
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise

    async def _retry_async(
        self, op: Callable[[], Coroutine[Any, Any, _T]], op_name: str
    ) -> _T:
        """
        Retry *op* (a zero-argument coroutine factory) on transient errors.

        Only retries exceptions representing transient cluster conditions
        (see ``_RETRYABLE_EXCEPTIONS``); anything else propagates
        immediately. All CQL statements issued by this backend (idempotent
        CREATE-TABLE-IF-NOT-EXISTS, primary-key-based INSERT/upsert,
        SELECT, DELETE) are safe to retry blindly.

        Uses a single retry budget (``operation_retry_delay``, doubling,
        capped at ``operation_retry_max_delay``, up to ``operation_max_retries``
        attempts) for every retryable error. The old driver's separate,
        shorter-backoff budget for purely local send-buffer backpressure
        (``ConnectionBusy``) has no equivalent here: this driver's flat
        error hierarchy exposes no per-host cause the way the old driver's
        ``NoHostAvailable.errors`` did, so that distinction is dropped.

        :param op: A zero-argument callable returning a fresh coroutine
            (must be callable more than once, since a coroutine object
            cannot be awaited twice).
        :param op_name: A short label used in retry log messages.
        :return: Whatever *op* returns.
        :raises Exception: The last exception encountered, once the retry
            budget is exhausted.
        """
        # Lazy: common no-error path allocates nothing.
        budget: Optional[_RetryBudget] = None
        while True:
            try:
                return await op()
            except _RETRYABLE_EXCEPTIONS as e:
                if budget is None:
                    budget = _RetryBudget(
                        delay=self._operation_retry_delay,
                        retries_left=self._operation_max_retries,
                        max_delay=self._operation_retry_max_delay,
                    )
                if budget.retries_left <= 0:
                    raise
                retries_left = budget.retries_left
                delay = budget.delay
                budget = budget.next()
                logger.warning(
                    "ScyllaDB %s failed with a transient error, "
                    "retrying in %.3fs (%d retries left): %s",
                    op_name,
                    delay,
                    retries_left - 1,
                    e,
                )
                await asyncio.sleep(delay)

    async def _execute(
        self, stmt: Any, params: Optional[Sequence[Any]] = None
    ) -> list[Any]:
        """
        Execute a CQL statement and return all its result rows.

        ``Session.execute()`` is natively a coroutine in this driver, so no
        callback-to-future bridging is needed (unlike the old driver's
        callback-based ``ResponseFuture``/``add_callbacks``). Every
        statement this backend issues is a point lookup/upsert/delete on a
        single partition key -- never more than one row -- so paging is
        disabled (``paged=False``) rather than draining pages that never
        exist. Transient failures are retried automatically (see
        :meth:`_retry_async`).

        :param stmt: A CQL string or prepared ``Statement``.
        :param params: Positional bind parameters, if any.
        :return: All result rows (a DDL/schema-change statement, e.g.
            ``CREATE TABLE``, returns an empty list).
        :raises Exception: Whatever exception the driver raises for the
            query (e.g. timeouts, unavailable replicas), after retries.
        """

        async def _do_execute() -> list[Any]:
            result = await self._safe_session.execute(stmt, params, paged=False)
            return list(result.iter_current_page())

        return await self._retry_async(_do_execute, "execute")

    # ---- Key-level helpers ----

    def _split_kv(self, tensor: torch.Tensor) -> List[tuple[int, bytes, bytes]]:
        """Split a [2, L, T, H] tensor into per-layer (k_blob, v_blob) bytes.

        Requires a contiguous tensor (as does the previous per-layer
        ``.numpy()`` path). Takes a single flat uint8 view over the whole
        buffer and slices it by byte offsets of the contiguous [2, L, T, H]
        layout -- layer *i*'s K at ``i * bytes_per_layer``, its V at
        ``(L + i) * bytes_per_layer`` -- instead of materializing 2L
        separate tensor-slice/numpy-view objects per chunk.

        Unlike the old driver's blob codec (which called ``bytes(val)`` on
        whatever it was given), this driver's blob serializer only accepts
        ``bytes`` -- an ``ndarray`` raises ``SerializationError``. So each
        slice is materialized here via ``.tobytes()``, already off the
        event loop (this whole method runs on ``self._blob_executor``).
        """
        layers = tensor.shape[1]
        hidden_elements = tensor.shape[2] * tensor.shape[3]
        bytes_per_layer = hidden_elements * tensor.dtype.itemsize
        # ravel(): slice by byte offset, not by the [2, L, T, H] first axis.
        flat = tensor.view(torch.uint8).numpy().ravel()
        # Final size is known upfront, so index-assign into a pre-sized list
        # instead of growing it one append() at a time.
        results: list[tuple[int, bytes, bytes]] = [(0, b"", b"")] * layers
        for layer in range(layers):
            k_off = layer * bytes_per_layer
            v_off = (layers + layer) * bytes_per_layer
            results[layer] = (
                layer,
                flat[k_off : k_off + bytes_per_layer].tobytes(),
                flat[v_off : v_off + bytes_per_layer].tobytes(),
            )
        return results

    def _reconstruct_tensor(
        self,
        rows: list[Any],
        target_u8: torch.Tensor,
        num_layers: int,
        bytes_per_layer: int,
    ) -> None:
        """
        Fill a pre-allocated [2, L, T, H] KV buffer from CQL query rows.

        Copies each layer's K and V directly into *target_u8* (the KV
        tensor's flat memory) with one copy per row, indexing it by byte
        offsets derived from the contiguous [2, L, T, H] layout -- layer
        *i*'s K at ``i * bytes_per_layer`` and its V at
        ``(num_layers + i) * bytes_per_layer`` -- avoiding the per-layer
        ``view``/``flatten`` object allocation the old
        ``target[0, i].flatten().view(torch.uint8)`` path incurred.
        Callers must pass *rows* built from ``num_layers`` separate
        per-layer point reads (see :meth:`_get_blocking_async`), one per
        ``layer_id`` in ``0..num_layers-1`` order -- row identity is
        therefore already known from the query that produced it, not
        re-derived from a ``layer_id`` column here.

        :param rows: Per-layer rows from :meth:`_fetch_all_layers_async`,
            in ``layer_id`` order.
        :param target_u8: Pre-allocated contiguous uint8 buffer of at
            least ``num_layers * 2 * bytes_per_layer`` bytes (the KV
            tensor's flat memory).
        :param num_layers: Number of layers (the tensor's second dim).
        :param bytes_per_layer: Byte size of one layer's K (equals its
            V's size): ``num_tokens * hidden_dim * element_size``.
        :raises ValueError: If any row's K or V byte length differs from
            ``bytes_per_layer``.
        """
        for i, row in enumerate(rows):
            # Default row factory is dict-of-columns: no row_factory config
            # equivalent to the old driver's tuple_factory in this driver.
            k_data, v_data = row["k_data"], row["v_data"]
            # Check raw byte length (O(1)) before frombuffer, which warns on bytes.
            if len(k_data) != bytes_per_layer:
                raise ValueError(
                    f"k_data length {len(k_data)} bytes != expected "
                    f"{bytes_per_layer} bytes for layer {i}"
                )
            if len(v_data) != bytes_per_layer:
                raise ValueError(
                    f"v_data length {len(v_data)} bytes != expected "
                    f"{bytes_per_layer} bytes for layer {i}"
                )
            k_u8 = torch.frombuffer(k_data, dtype=torch.uint8)
            v_u8 = torch.frombuffer(v_data, dtype=torch.uint8)
            k_offset = i * bytes_per_layer
            v_offset = (num_layers + i) * bytes_per_layer
            target_u8[k_offset : k_offset + bytes_per_layer].copy_(k_u8)
            target_u8[v_offset : v_offset + bytes_per_layer].copy_(v_u8)

    async def _fetch_all_layers_async(
        self, table: str, key: CacheEngineKey
    ) -> Optional[list[Any]]:
        """
        Fetch every layer's row for a whole-chunk key via ``num_layers``
        concurrent per-layer point reads.

        Each layer lives in its own partition (see the class docstring), so
        there is no single range query that returns "all layers of this
        chunk" -- this is the fan-out that replaces it. Row *i* in the
        returned list is always layer *i*'s row, since it comes from a
        query that explicitly asked for ``layer_id = i``.

        Each per-layer query goes through :meth:`_execute` (and so gets its
        own :meth:`_retry_async` retry budget). Unlike a plain
        ``asyncio.gather``, this stops and cancels the rest as soon as any
        one layer reports a miss/failure, matching the old callback-based
        code's early-exit -- this sits on the latency-critical get path
        (see the class docstring), so a chunk that is already a known miss
        should not still wait on its slowest sibling layer query.

        :param table: The CQL table name.
        :param key: The whole-chunk key being fetched (for logging only).
        :return: A list of ``self._num_layers`` rows in ``layer_id`` order,
            or ``None`` if any layer is missing or its query failed after
            exhausting retries.
        """
        stmt = await self._prepare_async(_CQL_SELECT_ONE_LAYER.format(table=table))
        # Final size (self._num_layers) is known upfront, so index-assign
        # into a pre-sized list instead of growing it one append() at a
        # time.
        rows: list[Any] = [None] * self._num_layers
        tasks = {
            asyncio.ensure_future(
                self._execute(stmt, (key.chunk_hash, layer_id))
            ): layer_id
            for layer_id in range(self._num_layers)
        }
        pending = set(tasks)
        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    layer_id = tasks[task]
                    exc = task.exception()
                    if exc is not None:
                        with self._get_blocking_failed_count_lock:
                            self._get_blocking_failed_count += 1
                        logger.warning(
                            "ScyllaDB layer %d fetch failed for key %s: %s; "
                            "treating chunk as a miss",
                            layer_id,
                            key,
                            exc,
                        )
                        return None
                    result = task.result()
                    if not result:
                        logger.debug(
                            "ScyllaDB layer %d missing for key %s; "
                            "treating chunk as a miss",
                            layer_id,
                            key,
                        )
                        return None
                    rows[layer_id] = result[0]
        finally:
            if pending:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        return rows

    # ---- StorageBackendInterface methods ----

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """
        Check whether *key* exists in ScyllaDB.

        For a :class:`LayerCacheEngineKey`, checks that specific layer's
        row. For a whole-chunk :class:`CacheEngineKey`, checks only layer
        0's row, as a proxy for the whole chunk: :meth:`_put_one_async`
        writes (or rolls back) all layers of a chunk atomically, so under
        normal operation layer 0 being present implies the rest of the
        chunk is too. This is *not* a full guarantee, though -- a layer
        removed individually via :meth:`remove` (with a
        :class:`LayerCacheEngineKey`) after the chunk was written breaks
        that invariant for the remaining layers, without affecting layer
        0. Use :meth:`get_blocking`, which fetches and requires every
        layer, if you need a real full-chunk existence guarantee.

        :param key: The key to check.
        :param pin: Ignored (pin is a no-op for ScyllaDB).
        :return: ``True`` if the checked row (layer 0, for whole-chunk
            keys) exists.
        """
        if self._closed:
            return False
        try:
            return self._sync_execute(self._contains_async(key))
        except Exception as e:
            logger.warning("contains(%s) failed: %s", key, e)
            return False

    async def _contains_async(self, key: CacheEngineKey) -> bool:
        """Async core of :meth:`contains`, reused by the batched async path.

        For a whole-chunk key (not a :class:`LayerCacheEngineKey`), checks
        only ``layer_id=0`` -- see :meth:`contains` for why that is a
        proxy for the whole chunk rather than a per-layer check.
        """
        table = await self._table_for_async(key)
        layer_id = key.layer_id if isinstance(key, LayerCacheEngineKey) else 0
        stmt = await self._prepare_async(_CQL_EXISTS.format(table=table))
        rows = await self._execute(stmt, [key.chunk_hash, layer_id])
        return len(rows) > 0

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        Check whether *key* has a pending put operation.

        :param key: Must be a full :class:`CacheEngineKey` (not
            :class:`LayerCacheEngineKey`); layer keys share the same
            hash/equality as their parent.
        :return: ``True`` if the key is currently being written.
        """
        with self._put_tasks_lock:
            return key in self._put_tasks

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Store a batch of KV caches in ScyllaDB without blocking the caller.

        Skips keys that already have a pending put. Each key's write is
        dispatched onto this backend's event loop and *not* awaited here
        (matching ``StorageManager.batched_put``'s documented non-blocking
        contract); ``on_complete_callback`` fires later, from the event
        loop thread, once that key's write actually completes.

        Within one key, all layers are written concurrently. If any layer
        fails, the whole chunk is rolled back (best-effort delete of any
        rows that did succeed) and
        ``on_complete_callback`` is *not* invoked for that key -- a partial
        write is never left silently readable as if it were complete (see
        :meth:`_put_one_async`).

        :param keys: Sequence of cache keys.
        :param objs: Parallel list of memory objects containing KV tensors
            of shape ``[2, L, T, H]``.
        :param transfer_spec: Ignored.
        :param on_complete_callback: Optional callback invoked for each
            key that finishes writing *successfully* (all layers).
        """
        if not self.is_connected():
            logger.warning("ScyllaDB session not available, skipping put")
            return

        loop = self._safe_loop

        for key, memory_obj in zip(keys, objs, strict=True):
            # Check-and-reserve atomically under one lock acquisition --
            # checking exists_in_put_tasks() and adding separately left a
            # window where two concurrent callers could both pass the check
            # for the same key before either reserved it.
            with self._put_tasks_lock:
                if key in self._put_tasks:
                    continue
                self._put_tasks.add(key)

            tensor = memory_obj.tensor
            if tensor is None:
                logger.warning("MemoryObj has no tensor for key %s", key)
                with self._put_tasks_lock:
                    self._put_tasks.discard(key)
                continue

            try:
                memory_obj.ref_count_up()
            except Exception as e:
                logger.error("ref_count_up failed for key %s, skipping put: %s", key, e)
                with self._put_tasks_lock:
                    self._put_tasks.discard(key)
                continue

            submit_time = time.perf_counter()
            future = asyncio.run_coroutine_threadsafe(
                self._put_one_async(key, tensor), loop
            )
            future.add_done_callback(
                functools.partial(
                    self._on_batched_put_done,
                    key,
                    memory_obj,
                    submit_time,
                    on_complete_callback,
                )
            )

    def _on_batched_put_done(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        submit_time: float,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
        future: "Future[bool]",
    ) -> None:
        """
        Completion callback for one key's fire-and-forget put.

        Runs on the event loop thread once *future* resolves. Invokes
        ``on_complete_callback`` only if the write genuinely succeeded (see
        :meth:`_put_one_async`); a partial or total failure increments
        ``put_failed_count`` and is logged, but never reports success. On
        success, also records submit-to-completion latency (mirroring
        :meth:`_get_blocking_async`'s ``update_interval_remote_time_to_get_sync``)
        so put latency shows up in the same Prometheus histograms as get.

        :param submit_time: ``time.perf_counter()`` value captured when this
            key's write was dispatched, used to compute end-to-end latency.
        """
        memory_obj.ref_count_down()
        with self._put_tasks_lock:
            self._put_tasks.discard(key)
        try:
            succeeded = future.result()
        except Exception as e:
            succeeded = False
            self._put_failed_count += 1
            logger.error("ScyllaDB put failed for key %s: %s", key, e)

        if succeeded:
            elapsed_ms = (time.perf_counter() - submit_time) * 1000
            self.stats_monitor.update_interval_remote_time_to_put(elapsed_ms)
            if on_complete_callback is not None:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning("on_complete_callback failed for key %s: %s", key, e)

    async def _put_one_async(
        self,
        key: CacheEngineKey,
        tensor: torch.Tensor,
    ) -> bool:
        """
        Write all layers of one chunk; roll back and return ``False`` on
        any partial failure.

        A "partial write" means: a chunk consists of multiple independent
        per-layer CQL rows, each its own partition (one ``INSERT`` per
        layer, dispatched concurrently). If, say, 3 of 16 layer-inserts
        fail (e.g. a transient timeout on just those requests) while the
        other 13 succeed, those 13 rows are *already durably committed* to
        ScyllaDB -- there is no implicit transaction spanning all 16
        inserts. Left alone, a later whole-chunk read would find some
        layers present and others missing (see
        :meth:`_fetch_all_layers_async`, which treats any missing layer as
        a full miss) rather than silently reconstructing a wrong/misaligned
        KV tensor -- but that still leaves stray, never-cleaned-up rows
        sitting in ScyllaDB until TTL expiry. This method prevents that: on
        any per-layer failure, it deletes whatever rows for this chunk did
        get written (best-effort) so the chunk is either fully present or
        fully absent, never half-written.

        :return: ``True`` if every layer was written successfully.
        """
        table = await self._table_for_async(key)
        stmt = await self._prepare_async(_CQL_INSERT.format(table=table))
        # Offload: puts are fire-and-forget, and tensor is kept alive by the
        # caller's ref_count_up.
        loop = asyncio.get_running_loop()
        per_layer = await loop.run_in_executor(
            self._blob_executor, self._split_kv, tensor
        )

        # Summarize now so nothing below needs the blobs again: dropping this
        # list lets each blob die with its own INSERT's frame instead of all
        # 2L (32MiB/chunk) living until the slowest sibling returns.
        layer_ids = [layer_id for layer_id, _, _ in per_layer]
        session_bytes_written = sum(len(k) + len(v) for _, k, v in per_layer)
        tasks = [
            self._execute(
                stmt,
                (key.chunk_hash, layer_id, k_blob, v_blob, self._ttl_seconds),
            )
            for layer_id, k_blob, v_blob in per_layer
        ]
        del per_layer
        results = await asyncio.gather(*tasks, return_exceptions=True)

        failed = [
            (layer_id, exc)
            for layer_id, exc in zip(layer_ids, results, strict=True)
            if isinstance(exc, BaseException)
        ]
        if failed:
            logger.error(
                "ScyllaDB put PARTIALLY failed for key %s: %d/%d layers "
                "failed (%s); rolling back the whole chunk",
                key,
                len(failed),
                len(layer_ids),
                failed,
            )
            await self._rollback_partial_write_async(key, table, layer_ids)
            return False

        self.stats_monitor.update_interval_remote_write_metrics(session_bytes_written)
        return True

    async def _rollback_partial_write_async(
        self, key: CacheEngineKey, table: str, layer_ids: Sequence[int]
    ) -> None:
        """Best-effort delete of a partially-written chunk's rows.

        Each layer is its own partition (see the class docstring), so
        there is no single ``chunk_hash``-scoped delete that covers the
        whole chunk -- this deletes each attempted layer's row
        individually, concurrently.
        """
        try:
            stmt = await self._prepare_async(_CQL_DELETE_ONE.format(table=table))
            await asyncio.gather(
                *(
                    self._execute(stmt, (key.chunk_hash, layer_id))
                    for layer_id in layer_ids
                )
            )
            logger.warning("Rolled back partially-written chunk for key %s", key)
        except Exception as e:
            logger.error(
                "Failed to roll back partially-written chunk for key %s "
                "(rows may remain inconsistent until TTL expiry): %s",
                key,
                e,
            )

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Retrieve a KV cache from ScyllaDB, blocking until complete.

        Allocates a :class:`MemoryObj` from ``local_cpu_backend`` and fills
        it with data from ScyllaDB.

        :param key: The key to retrieve.
        :return: A :class:`MemoryObj` containing the KV tensor, or ``None``
            if the key does not exist or an error occurs.
        """
        if self.local_cpu_backend is None:
            logger.warning("local_cpu_backend is None in get_blocking")
            return None
        if not self.is_connected():
            return None

        try:
            return self._sync_execute(self._get_blocking_async(key))
        except Exception as e:
            with self._get_blocking_failed_count_lock:
                self._get_blocking_failed_count += 1
            logger.warning("get_blocking(%s) failed: %s", key, e)
            return None

    async def _get_blocking_async(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Async core of :meth:`get_blocking`, reused by the batched async path.

        Guards ``local_cpu_backend is None`` here (rather than relying on
        callers) because this coroutine is also reached from
        :meth:`_batched_get_blocking_async` and :meth:`batched_get_non_blocking`,
        neither of which check it before dispatching -- a scheduler-only
        backend (see :meth:`get_allocator_backend`) must fail this per-key
        lookup gracefully, not crash the whole batch.

        Tracks itself in ``self._active_gets`` for the duration of the
        call, so :meth:`close` can wait for in-flight gets the same way it
        already waits for in-flight puts (``self._put_tasks``).
        """
        with self._active_gets_lock:
            self._active_gets += 1
        try:
            if self.local_cpu_backend is None:
                logger.warning("local_cpu_backend is None in _get_blocking_async")
                return None
            start = time.perf_counter()
            table = await self._table_for_async(key)
            element_size = key.dtype.itemsize

            if isinstance(key, LayerCacheEngineKey):
                stmt = await self._prepare_async(
                    _CQL_SELECT_ONE_LAYER.format(table=table)
                )
                layer_rows = await self._execute(stmt, [key.chunk_hash, key.layer_id])
                if not layer_rows:
                    return None
                rows: list[Any] = layer_rows
                layer_count = 1
            else:
                chunk_rows = await self._fetch_all_layers_async(table, key)
                if chunk_rows is None:
                    return None
                rows = chunk_rows
                layer_count = self._num_layers

            # Default row factory is dict-of-columns (see _reconstruct_tensor).
            first_row_k_data = rows[0]["k_data"]
            # Size from byte length directly; avoids a frombuffer (and its warning).
            num_elements = len(first_row_k_data) // element_size
            num_tokens = num_elements // self._hidden_dim
            shape = torch.Size([2, layer_count, num_tokens, self._hidden_dim])

            mem_obj = self.local_cpu_backend.allocate(
                [shape], [key.dtype], fmt=MemoryFormat.KV_2LTD
            )
            if mem_obj is None:
                logger.error("Memory allocation failed for key %s", key)
                return None

            try:
                target_u8 = torch.frombuffer(
                    mem_obj.byte_array,
                    dtype=torch.uint8,
                    count=mem_obj.byte_array.nbytes,
                )

                if isinstance(key, LayerCacheEngineKey):
                    k_u8 = torch.frombuffer(first_row_k_data, dtype=torch.uint8)
                    v_bytes = torch.frombuffer(rows[0]["v_data"], dtype=torch.uint8)
                    offset = num_tokens * self._hidden_dim * element_size
                    target_u8[:offset].copy_(k_u8)
                    target_u8[offset : offset + len(v_bytes)].copy_(v_bytes)
                else:
                    # Offload the copies so the loop stays free to drain CQL
                    # completions. This does NOT run in parallel with other
                    # chunks' copies -- copy_() holds the GIL for its whole
                    # memcpy (measured) -- it only keeps the blocking memcpy
                    # off the event loop thread.
                    bytes_per_layer = num_tokens * self._hidden_dim * element_size
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        self._blob_executor,
                        self._reconstruct_tensor,
                        rows,
                        target_u8,
                        layer_count,
                        bytes_per_layer,
                    )
                    # Copied out: drop the driver's row blobs (32MiB/chunk)
                    # now instead of holding them to coroutine exit.
                    rows = []
                    del chunk_rows
            except Exception:
                # Release the pool-allocated object on failure so reconstruction
                # errors don't leak it and silently shrink the bounded pool.
                mem_obj.ref_count_down()
                raise

            elapsed_ms = (time.perf_counter() - start) * 1000
            self.stats_monitor.update_interval_remote_time_to_get_sync(elapsed_ms)
            self.stats_monitor.update_interval_remote_read_metrics(
                mem_obj.byte_array.nbytes
            )
            return mem_obj
        finally:
            with self._active_gets_lock:
                self._active_gets -= 1

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """
        Retrieve multiple KV caches concurrently.

        :param keys: List of keys to retrieve.
        :return: List of :class:`MemoryObj` or ``None`` for missing/errored
            keys, in the same order as *keys*.
        """
        if not self.is_connected():
            return [None] * len(keys)
        try:
            return self._sync_execute(self._batched_get_blocking_async(keys))
        except Exception as e:
            logger.warning("batched_get_blocking failed: %s", e)
            return [None] * len(keys)

    async def _batched_get_blocking_async(
        self, keys: List[CacheEngineKey]
    ) -> List[Optional[MemoryObj]]:
        """Async core of :meth:`batched_get_blocking`: fetches all keys
        concurrently via the same non-blocking bridge used everywhere else
        in this backend, isolating per-key failures from one another."""
        raw_results = await asyncio.gather(
            *(self._get_blocking_async(key) for key in keys),
            return_exceptions=True,
        )
        # Always exactly len(keys) entries (None for failures), so index-
        # assign into a pre-sized list instead of growing it via append().
        results: List[Optional[MemoryObj]] = [None] * len(keys)
        for i, (key, result) in enumerate(zip(keys, raw_results, strict=True)):
            if isinstance(result, BaseException):
                with self._get_blocking_failed_count_lock:
                    self._get_blocking_failed_count += 1
                logger.warning("get_blocking(%s) failed: %s", key, result)
            else:
                results[i] = result
        return results

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """
        Return the number of consecutive keys that exist, starting from
        index 0.  Stops at the first miss.

        Checks are dispatched concurrently (see
        :meth:`_batched_contains_core_async`); this only affects how fast
        the batch completes, not which keys are checked -- semantics match
        the sequential version exactly.

        :param keys: Ordered list of keys.
        :param pin: Ignored.
        :return: Number of consecutive hits from the start of the list.
        """
        if self._closed:
            return 0
        try:
            return self._sync_execute(self._batched_contains_core_async(keys))
        except Exception as e:
            logger.warning("batched_contains failed: %s", e)
            return 0

    async def _batched_contains_core_async(self, keys: Sequence[CacheEngineKey]) -> int:
        """Shared async core for :meth:`batched_contains` and
        :meth:`batched_async_contains`."""
        exists_flags = await asyncio.gather(
            *(self._contains_async(key) for key in keys),
            return_exceptions=True,
        )
        hit = 0
        for exists in exists_flags:
            if not exists or isinstance(exists, BaseException):
                break
            hit += 1
        return hit

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """
        Async variant of :meth:`batched_contains`.

        Checks all keys concurrently via :meth:`_contains_async` (each
        bridged onto the driver's own I/O reactor, not a thread pool), then
        returns the number of consecutive hits starting from index 0 to
        match :meth:`batched_contains`'s semantics.
        """
        return await self._batched_contains_core_async(keys)

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        """
        Non-blocking batched retrieval.

        Retrieves all keys concurrently via :meth:`_get_blocking_async`
        (each bridged onto the driver's own I/O reactor, not a thread pool).
        A failure on one key (e.g. a transient error surviving retries) is
        isolated to that key -- it is logged, counted in
        :attr:`get_blocking_failed_count`, and dropped from the result,
        exactly like :meth:`_batched_get_blocking_async` -- rather than
        failing the whole batch.
        """
        raw_results = await asyncio.gather(
            *(self._get_blocking_async(key) for key in keys),
            return_exceptions=True,
        )
        # Pre-size and fill successes contiguously, then trim to the
        # filled region (dropped failures leave no gap in the result).
        results: list[Optional[MemoryObj]] = [None] * len(keys)
        hit = 0
        for key, result in zip(keys, raw_results, strict=True):
            if isinstance(result, BaseException):
                with self._get_blocking_failed_count_lock:
                    self._get_blocking_failed_count += 1
                logger.warning("get_blocking(%s) failed: %s", key, result)
            elif result is not None:
                results[hit] = result
                hit += 1
        return [m for m in results[:hit] if m is not None]

    def pin(self, key: CacheEngineKey) -> bool:
        """
        Pin a key in the cache.  No-op for ScyllaDB.

        :param key: Ignored.
        :return: Always ``True``.
        """
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        """
        Unpin a key in the cache.  No-op for ScyllaDB.

        :param key: Ignored.
        :return: Always ``True``.
        """
        return True

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """
        Remove a key (or specific layer) from ScyllaDB.

        :param key: The key to remove.  A :class:`LayerCacheEngineKey`
            removes only that layer; a full :class:`CacheEngineKey` removes
            all layers.
        :param force: Ignored.
        :return: ``True`` if the delete was submitted successfully.
        """
        if not self.is_connected():
            return False
        try:
            table = self._table_for(key)
            if isinstance(key, LayerCacheEngineKey):
                stmt = self._prepare(_CQL_DELETE_ONE.format(table=table))
                self._sync_execute(self._execute(stmt, [key.chunk_hash, key.layer_id]))
            else:
                self._sync_execute(self._remove_all_layers_async(key, table))
            return True
        except Exception as e:
            logger.warning("remove(%s) failed: %s", key, e)
            return False

    async def _remove_all_layers_async(self, key: CacheEngineKey, table: str) -> None:
        """Delete every layer's row for a whole-chunk key.

        Each layer is its own partition (see the class docstring), so
        there is no single ``chunk_hash``-scoped delete that covers the
        whole chunk -- this deletes each layer's row individually,
        concurrently.
        """
        stmt = await self._prepare_async(_CQL_DELETE_ONE.format(table=table))
        await asyncio.gather(
            *(
                self._execute(stmt, (key.chunk_hash, layer_id))
                for layer_id in range(self._num_layers)
            )
        )

    def get_allocator_backend(self) -> AllocatorBackendInterface:
        """
        Return the local CPU backend used for memory allocation.

        :return: The :class:`LocalCPUBackend` instance.
        :raises RuntimeError: If ``local_cpu_backend`` was not provided
            (e.g., in a scheduler-only role).
        """
        if self.local_cpu_backend is None:
            raise RuntimeError(
                "local_cpu_backend is required; should not be called in scheduler role"
            )
        return self.local_cpu_backend

    @property
    def get_blocking_failed_count(self) -> int:
        """Number of ``get_blocking``/batched-get failures observed so far
        (for health-check/observability consumption)."""
        return self._get_blocking_failed_count

    @property
    def put_failed_count(self) -> int:
        """Number of put failures (including rolled-back partial writes)
        observed so far (for health-check/observability consumption)."""
        return self._put_failed_count

    def is_connected(self) -> bool:
        """
        Whether this backend currently holds a live ScyllaDB session.

        Public accessor for health checks, so they never need to reach
        into the private ``_session``/``_closed`` attributes directly.
        """
        return self._session is not None and not self._closed

    def ping(self) -> int:
        """
        Lightweight connectivity check for health monitoring.

        Issues a trivial query against ``system.local`` and waits (up to a
        short fixed timeout) for it to complete.

        :return: ``0`` on success; a nonzero error code if the backend is
            closed/disconnected, or the query fails or times out.
        """
        if not self.is_connected():
            return 1
        try:
            self._sync_execute(
                self._execute("SELECT now() FROM system.local"), timeout=5.0
            )
            return 0
        except TimeoutError:
            logger.warning("ScyllaDB ping timed out")
            return 2
        except Exception as e:
            logger.warning("ScyllaDB ping failed: %s", e)
            return 3

    def close(self) -> None:
        """
        Close the backend, waiting up to ``timeout_secs`` for pending put
        tasks and in-flight gets to finish before releasing the ScyllaDB
        session.

        If the wait times out with puts or gets still in flight, those
        operations are abandoned (a put's rows may be partially or fully
        missing; a get raises once the session it depends on is gone) and
        a warning is logged; the backend is closed regardless.
        """
        if self._closed:
            return
        self._closed = True
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            with self._put_tasks_lock:
                puts_done = not self._put_tasks
            with self._active_gets_lock:
                gets_done = self._active_gets == 0
            if puts_done and gets_done:
                break
            time.sleep(0.05)
        else:
            with self._put_tasks_lock:
                pending_puts = len(self._put_tasks)
            with self._active_gets_lock:
                pending_gets = self._active_gets
            if pending_puts or pending_gets:
                logger.warning(
                    "ScyllaDB backend closing with %d put(s) and %d get(s) "
                    "still pending after %.1fs timeout; those operations "
                    "are abandoned.",
                    pending_puts,
                    pending_gets,
                    self._timeout,
                )
        self._cleanup_resources()
        self._blob_executor.shutdown(wait=False, cancel_futures=True)
        logger.info("ScyllaDB backend closed.")
