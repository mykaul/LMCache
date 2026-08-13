ScyllaDB
========

Overview
--------

The ScyllaDB storage plugin stores KV cache chunks in a `ScyllaDB
<https://www.scylladb.com/>`_ cluster, using one CQL table per
``(model, world_size, worker_id, dtype)`` combination and per-cell CQL TTL
(``INSERT ... USING TTL``) for automatic expiry -- a row is invisible to
reads the instant its TTL elapses, not on some later background sweep.
(ScyllaDB also has a newer, separate "per-row TTL" column feature; it isn't
used here since it's built on Alternator's TTL implementation and only
expires *eventually*, on a cluster-wide background sweep interval, not
read-time-immediately -- see
`the per-row TTL docs <https://docs.scylladb.com/manual/stable/cql/cql-extensions.html>`_.)

Each layer of a chunk is stored as its own CQL partition (composite primary
key ``(chunk_hash, layer_id)``, no clustering column). This keeps partition
size small and bounded regardless of ``chunk_size``/``hidden_dim``/dtype, and
spreads a popular shared prefix's traffic across ``num_layers`` replica sets
instead of hammering one -- at the cost of a whole-chunk read/write/delete
fanning out into ``num_layers`` concurrent per-layer CQL operations instead of
a single range query.

The configured keyspace is created automatically (``CREATE KEYSPACE IF NOT
EXISTS``) using ``NetworkTopologyStrategy`` scoped to ``local_dc``; tables are
likewise created automatically per model/world_size/worker_id/dtype the first
time they're needed.

.. note::

   MLA (Multi-head Latent Attention) models are not supported: the table
   schema and hidden-dim calculation assume the standard
   ``(num_layers, 2, num_tokens, num_heads, head_size)`` ``kv_shape`` layout.

There is deliberately no client-side cap on how many CQL requests can be in
flight at once -- ScyllaDB has its own admission control, and a static
client-side number can't know a given cluster's real capacity. Transient
overload is instead handled where it actually surfaces, via retry with
backoff (``operation_max_retries``, below).

.. note::

   **Future work**: puts are non-blocking, best-effort writes made for
   *future* requests' benefit, while gets sit on the latency-critical
   serving path for the *current* request -- those aren't the same
   priority. ScyllaDB's Service Levels (per-role ``SHARES``, i.e. workload
   prioritization) exist specifically to express that, e.g. a lower-share
   role for the connection issuing puts. Not implemented yet: it needs
   cluster-side service level/role provisioning, a separate decision from
   anything this backend can do unilaterally.

Configuration
-------------

.. code-block:: yaml

   local_cpu: true
   max_local_cpu_size: 10.0

   storage_plugins: ["scylla"]
   extra_config:
     storage_plugin.scylla.module_path: lmcache.v1.storage_backend.scylla_backend
     storage_plugin.scylla.class_name: ScyllaDBBackend

     scylla:
       contact_points: ["127.0.0.1"]
       port: 9042
       keyspace: "lmcache"
       local_dc: "datacenter1"
       ttl_seconds: 86400
       table_compression: "LZ4WithDictsCompressor"

Install the driver on your serving machine. ``python-rs-driver`` (package
``scylla``) is not yet published to PyPI; build it from
https://github.com/scylladb/python-rs-driver with ``maturin`` (see that
repo's ``pyproject.toml``/``Makefile``).

Configuration Reference
------------------------

Configure the following options inside the ``scylla`` mapping under
``extra_config`` (i.e. ``extra_config.scylla.<key>``):

.. list-table:: ScyllaDB Configuration Parameters
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter Key
     - Type
     - Default
     - Description
   * - ``contact_points``
     - list[str]
     - ``["127.0.0.1"]``
     - Initial ScyllaDB node(s) to connect to for cluster discovery.
   * - ``port``
     - int
     - ``9042``
     - CQL native protocol port.
   * - ``keyspace``
     - str
     - ``"lmcache"``
     - Keyspace to use; created automatically if it does not already exist
       (see ``keyspace_replication_factor``).
   * - ``local_dc``
     - str
     - ``"datacenter1"``
     - Datacenter name used for DC-aware load balancing and as the
       replication target when auto-creating ``keyspace``.
   * - ``local_rack``
     - str
     - None
     - Optional rack name for rack-aware load balancing within ``local_dc``.
   * - ``keyspace_replication_factor``
     - int
     - ``1``
     - Replication factor for ``local_dc`` when auto-creating ``keyspace``.
       Only relevant the first time this backend connects to a given
       keyspace. To use a multi-DC replication topology, pre-create the
       keyspace manually instead.
   * - ``ttl_seconds``
     - int
     - ``86400``
     - CQL TTL applied to every cell of a row (``INSERT ... USING TTL``).
       Rows (and therefore chunks) expire -- and stop being readable --
       automatically after this many seconds.
   * - ``table_compression``
     - str
     - ``"LZ4WithDictsCompressor"``
     - Literal SSTable compressor class name passed straight into CQL's
       ``WITH compression = {'sstable_compression': ...}`` (e.g.
       ``"LZ4Compressor"``, ``"LZ4WithDictsCompressor"``,
       ``"ZstdCompressor"``, ``"ZstdWithDictsCompressor"``) -- just like
       ``consistency_level`` below is the literal name of a driver enum
       member. Empty string disables compression entirely. Not validated
       client-side; an invalid name surfaces as a CQL error on table
       creation.
   * - ``consistency_level``
     - str
     - ``"LocalOne"``
     - Any ``scylla.enums.Consistency`` member name (e.g. ``LocalOne``,
       ``Quorum``, ``All``).
   * - ``timeout_secs``
     - float
     - ``30.0``
     - Default per-request timeout used for the session and for waiting on
       individual CQL operations.
   * - ``max_connect_retries``
     - int
     - ``3``
     - Number of connection attempts before giving up at startup.
   * - ``connect_retry_delay``
     - float
     - ``1.0``
     - Delay in seconds between connection attempts.
   * - ``operation_max_retries``
     - int
     - ``2``
     - Number of *additional* attempts (beyond the first) for a single CQL
       operation when it fails with a transient error (``ExecuteError``,
       ``PrepareError``, ``RequestTimeoutError``, ``SessionConnectionError``).
   * - ``operation_retry_delay``
     - float
     - ``0.05``
     - Initial delay in seconds between operation retries; doubles after
       each attempt, capped at ``operation_retry_max_delay`` (``5.0``).

Health Monitoring
-----------------

When a ScyllaDB backend is configured, LMCache's health monitor
automatically registers a matching health check that pings the cluster and
tracks ``get_blocking``/put failure counts. It shares the same top-level
(not nested under ``scylla``) ``extra_config`` keys as other backends'
health checks:

.. list-table:: Health Check Configuration Parameters
   :widths: 25 15 15 45
   :header-rows: 1

   * - Parameter Key
     - Type
     - Default
     - Description
   * - ``fallback_policy``
     - str
     - ``"RECOMPUTE"``
     - ``"RECOMPUTE"`` skips all cache operations while unhealthy;
       ``"LOCAL_CPU"`` bypasses ScyllaDB and serves from local CPU cache
       instead.
   * - ``get_blocking_failed_threshold``
     - int
     - ``10``
     - Number of ``get_blocking`` failures within one health-check interval
       that marks the backend unhealthy.
   * - ``waiting_time_for_recovery``
     - float
     - ``300.0``
     - Minimum seconds to wait after a failure before probing for recovery.

Limitations
-----------

- MLA models are not supported (see the note above).
- Tensor parallelism/world_size and worker_id are encoded into the table
  name, so each ``(model, world_size, worker_id, dtype)`` combination gets
  its own table; nothing needs to be provisioned manually.
- A whole-chunk ``contains()`` check only checks layer 0's row as a proxy
  for the whole chunk (writes/rollbacks are atomic across all layers under
  normal operation); use a full ``get_blocking`` if you need a guarantee
  covering every layer.

Troubleshooting
---------------

- **ImportError: python-rs-driver is required**: build the ``scylla``
  package from https://github.com/scylladb/python-rs-driver (not yet on
  PyPI).
- **Connection fails on every attempt**: verify ``contact_points``/``port``
  are reachable and that ``local_dc`` matches the target cluster's actual
  datacenter name (mismatches surface as ``SessionConnectionError``).
- **Server warning about low replication factor**: expected for a
  single-node development cluster with the default
  ``keyspace_replication_factor: 1``; raise it (and provision enough nodes
  in ``local_dc``) for production use.
