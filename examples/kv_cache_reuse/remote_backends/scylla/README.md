# ScyllaDB

LMCache can use [ScyllaDB](https://www.scylladb.com/) as a remote backend
storage. Each layer of a chunk is stored as its own CQL partition, with
per-cell TTL for automatic expiry -- see the
[full configuration reference](../../../../docs/source/kv_cache/storage_backends/scylla.rst)
for how it's laid out.

## Quick Start

1. Install the driver. `python-rs-driver` (package `scylla`) is not yet on
   PyPI; build it from
   [scylladb/python-rs-driver](https://github.com/scylladb/python-rs-driver)
   with `maturin`.

2. Start a local, disposable ScyllaDB cluster (RAM-backed, resource-capped --
   for local testing, not production):

   ```bash
   benchmarks/scylla_e2e/launch_scylla_e2e.sh
   #   optional args: [container-name] [host-port] [scylla-mem-mb]
   #   e.g. benchmarks/scylla_e2e/launch_scylla_e2e.sh lmcache-scylla 9042 3500
   ```

3. Start vLLM with LMCache pointed at it, using [`scylla.yaml`](scylla.yaml):

   ```bash
   LMCACHE_CONFIG_FILE=scylla.yaml \
   vllm serve mistralai/Mistral-7B-Instruct-v0.2 \
       --port 8000 \
       --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
   ```

For a real cluster instead of the local disposable one, edit `scylla.yaml`'s
`contact_points`/`local_dc` to match it.

## Test with a Sample Request

Send the same prompt twice. The first call is a cache miss that populates
ScyllaDB; the second reuses the stored KV cache instead of recomputing it:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [
            {"role": "user", "content": "Write a segment tree implementation in python"}
        ],
        "max_tokens": 150
    }'
```

Run it a second time and check vLLM's logs for a cache hit (LMCache logs a
retrieved-token count for hits), or watch the `lmcache:` Prometheus metrics
(`remote_time_to_get_sync_count`, `remote_time_to_put_sync_count`) increment
against the ScyllaDB backend.

## Additional Resources

- [ScyllaDB documentation](https://www.scylladb.com/product/technology/)
- [Full ScyllaDB backend configuration reference](../../../../docs/source/kv_cache/storage_backends/scylla.rst)
  (all config keys, health monitoring, limitations, troubleshooting)
- [python-rs-driver on GitHub](https://github.com/scylladb/python-rs-driver)
