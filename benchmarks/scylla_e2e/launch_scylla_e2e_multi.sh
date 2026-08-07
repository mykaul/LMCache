#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Launch a 2-node ScyllaDB container cluster (RAM-backed tmpfs storage) for
# exercising ScyllaDBBackend's sub-chunk/salting across two nodes.
#
# Why loopback-IP addressing (not per-node ports):
#   - The Python driver assigns every discovered peer the cluster's default
#     native port; a peer's *address* is its row's rpc_address (i.e. its
#     broadcast_rpc_address). Scylla's system.peers exposes no
#     native_transport_port column, so per-node ports are unrecoverable.
#   - Container IPs (10.90.0.x) are not reachable from the host under rootless
#     podman. So each node advertises a distinct host-loopback address at the
#     SAME native port: A->127.0.0.1:9042, B->127.0.0.2:9042. The host
#     publishes each loopback IP to the matching container, so the driver sees
#     and reaches BOTH coordinators and token-aware routing spreads requests.
#
# Layout:
#   - Both containers share a podman bridge network with STATIC IPs
#     (10.90.0.2 = node A, 10.90.0.3 = node B) for inter-node (storage/gossip)
#     traffic on port 7000.
#   - Native (CQL) transport binds rpc_address = the container's bridge IP
#     (Scylla derives it from --listen-address). Rootless port-forwarding
#     dials the bridge IP, so transport MUST bind it -- binding loopback would
#     make the host unable to reach the container. The in-container health
#     check therefore targets the bridge IP, not 127.0.0.1.
#   - Each node publishes native-transport-port 9042 to a distinct host-loopback
#     IP (A=127.0.0.1, B=127.0.0.2) AND advertises it as broadcast_rpc_address.
#     The driver builds each peer's endpoint from peer.rpc_address + the
#     cluster default port, so it reaches both coordinators.
#   - Storage is RAM-backed tmpfs, one allocation per node, sized for a single
#     e2e run (keyspace RF=1 spreads a 4GB dataset ~2GB/node). Tear the whole
#     cluster down between runs so tmpfs never accumulates SStables.
#
# Usage: ./launch_scylla_e2e_multi.sh
# Env:   SCYLLA_MEM_MB (default 1400), CONTAINER_CPUS (default 4),
#        TMPFS_SIZE_MB (default 3584). Static IPs are fixed at 10.90.0.2/.3.
#
# Notes on sizing:
#   - Scylla's default commitlog pool is huge (~1.5GB here); cap it so the
#     tmpfs budget goes to real data, not commitlog segments.
#   - A 4GB RF=1 dataset spreads ~2GB/node of SSTables; a 3GB tmpfs per node
#     leaves room for commitlog (512MB) and system tables.
set -euo pipefail

NET="lmcache-scylla-e2e-net"
NAME_A="lmcache-scylla-e2e-a"
NAME_B="lmcache-scylla-e2e-b"
IP_A="10.90.0.2"
IP_B="10.90.0.3"
LOOPBACK_A="127.0.0.1"
LOOPBACK_B="127.0.0.2"
NATIVE_PORT="9042"
IMAGE="docker.io/scylladb/scylla:2026.2"
SCYLLA_MEM_MB="${SCYLLA_MEM_MB:-1400}"
CONTAINER_CPUS="${CONTAINER_CPUS:-4}"
TMPFS_SIZE_MB="${TMPFS_SIZE_MB:-3072}"
COMMITLOG_MB=512
IO_PROPERTIES_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/io_properties.yaml"

MARGIN_MB=256
CONTAINER_MEM_MB=$(( SCYLLA_MEM_MB + MARGIN_MB + TMPFS_SIZE_MB ))

# real = 2*(RSS + TMPFS). Guard the host the same way the single-node
#   launcher does, but against both containers' cgroup caps. The cgroup cap is
#   a limit, not a reservation, so a 2GB headroom covers transient spikes on
#   top of the caps.
HOST_HEADROOM_MB=2048
mem_available_kb=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo || echo "")
if [ -n "$mem_available_kb" ]; then
  mem_available_mb=$(( mem_available_kb / 1024 ))
  if [ "$(( CONTAINER_MEM_MB * 2 ))" -gt "$(( mem_available_mb - HOST_HEADROOM_MB ))" ]; then
    echo "ERROR: two containers would reserve $(( CONTAINER_MEM_MB * 2 ))M, but only" \
         "${mem_available_mb}M is available on the host (need ${HOST_HEADROOM_MB}M" \
         "memory). Lower SCYLLA_MEM_MB/CONTAINER_CPUS/TMPFS_SIZE_MB or free" \
         "host memory, then retry." >&2
    exit 1
  fi
fi

podman network exists "$NET" || podman network create --subnet 10.90.0.0/24 "$NET"

launch_node() {
  local name=$1 ip=$2 loopback=$3 seeds=$4
  podman rm -f --time 0 "$name" >/dev/null 2>&1 || true
  podman run -d --name "$name" --network "$NET" --ip "$ip" \
    --cpus="$CONTAINER_CPUS" --memory="${CONTAINER_MEM_MB}m" \
    --health-startup-cmd "cqlsh ${ip} ${NATIVE_PORT} -e \"SELECT now() FROM system.local\" || exit 1" \
    --health-startup-interval 2s --health-startup-retries 40 \
    --health-startup-timeout 5s --health-startup-success 1 \
    --health-cmd "cqlsh ${ip} ${NATIVE_PORT} -e \"SELECT now() FROM system.local\" || exit 1" \
    --health-interval 30s --health-retries 30 --health-timeout 5s --health-start-period 10s \
    -p "${loopback}:${NATIVE_PORT}:${NATIVE_PORT}" \
    --tmpfs "/var/lib/scylla:size=${TMPFS_SIZE_MB}m,mode=1777" \
    -v "${IO_PROPERTIES_FILE}:/etc/scylla.d/io_properties.yaml:ro" \
    "$IMAGE" \
    --smp "$CONTAINER_CPUS" --memory "${SCYLLA_MEM_MB}M" --developer-mode 1 \
    --unsafe-bypass-fsync 1 --overprovisioned 1 \
    --commitlog-total-space-in-mb "$COMMITLOG_MB" \
    --native-transport-port "$NATIVE_PORT" \
    --listen-address "$ip" \
    --broadcast-rpc-address "$loopback" \
    --seeds "$seeds"
}

# Node A is the seed and forms the cluster on its own static IP.
launch_node "$NAME_A" "$IP_A" "$LOOPBACK_A" "$IP_A"
echo "Waiting for node $NAME_A ($IP_A) to become healthy..."
for i in $(seq 1 90); do
  status=$(podman inspect --format '{{.State.Health.Status}}' "$NAME_A" 2>/dev/null || echo "")
  [ "$status" = "healthy" ] && { echo "  $NAME_A healthy after ~$((i * 2))s."; break; }
  [ "$status" = "unhealthy" ] && { echo "  node A unhealthy. Logs:" >&2; podman logs --tail 40 "$NAME_A" >&2; exit 1; }
  sleep 2
done

# Node B joins A's cluster, seeded by A's static IP.
launch_node "$NAME_B" "$IP_B" "$LOOPBACK_B" "$IP_A"
echo "Waiting for node $NAME_B ($IP_B) to join..."
for i in $(seq 1 90); do
  status=$(podman inspect --format '{{.State.Health.Status}}' "$NAME_B" 2>/dev/null || echo "")
  [ "$status" = "healthy" ] && { echo "  $NAME_B healthy after ~$((i * 2))s."; break; }
  [ "$status" = "unhealthy" ] && { echo "  node B unhealthy:" >&2; podman logs --tail 40 "$NAME_B" >&2; exit 1; }
  sleep 2
done

echo "Verifying 2-node view from node A..."
podman exec "$NAME_A" cqlsh "$IP_A" "$NATIVE_PORT" -e "SELECT peer, rpc_address FROM system.peers;" 2>&1 \
  | head -8
podman exec "$NAME_A" cqlsh "$IP_A" "$NATIVE_PORT" -e "SELECT key, rpc_address FROM system.local;" 2>&1 \
  | head -5

echo "Cluster up: A=${LOOPBACK_A}:${NATIVE_PORT} (seed, ${IP_A}), B=${LOOPBACK_B}:${NATIVE_PORT} (${IP_B})."
echo "Point the demo at SCYLLA_HOST=127.0.0.1 SCYLLA_PORT=${NATIVE_PORT}; it discovers B at ${LOOPBACK_B} automatically."