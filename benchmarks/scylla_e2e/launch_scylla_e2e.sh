#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Launch a resource-bounded, RAM-disk-backed ScyllaDB container for
# exercising ScyllaDBBackend end-to-end, without risking a system-wide OOM.
#
# Usage: ./launch_scylla_e2e.sh [container-name] [host-port] [scylla-mem-mb] [cpus]
set -euo pipefail

NAME="${1:-lmcache-scylla-e2e}"
HOST_PORT="${2:-9042}"
SCYLLA_MEM_MB="${3:-1500}"
CONTAINER_CPUS="${4:-4}"
IMAGE="docker.io/scylladb/scylla:2026.2"

# Container cgroup cap must exceed Scylla's --memory by a small margin:
# the gap between Scylla's --memory allocation arena and its true RSS, plus
# the sidecar processes (supervisord, node_exporter -- measured ~40MB RSS
# total, NOT the ~2GB the old comment claimed). tmpfs is RAM-backed, so its
# *contents* are added on top, separately -- without that, the cgroup cap
# could OOM the container well before tmpfs's own `size=` ever rejects a
# write (confirmed: this used to just add a fixed margin, ignoring
# TMPFS_SIZE_GB entirely). A real 200-chunk put/get/churn run peaked at
# ~4.2GB cgroup usage against a ~10GB cap, so ~256MB of margin is generous
# headroom for the Scylla RSS gap + sidecars.
#
# The larger the tmpfs, the more room Scylla has for commitlog + SSTables
# before "No space left on device" mid-run (measured: 4GB tmpfs filled with
# only 500 resident chunks); size it for the dataset, not for the margin.
MARGIN_PCT=10
MARGIN_FLOOR_MB=256

SCYLLA_SMP="$CONTAINER_CPUS" # shard count -- matches CONTAINER_CPUS
TMPFS_SIZE_GB="${TMPFS_SIZE_GB:-12}"

margin_pct_mb=$(( SCYLLA_MEM_MB * MARGIN_PCT / 100 ))
margin_mb=$(( margin_pct_mb > MARGIN_FLOOR_MB ? margin_pct_mb : MARGIN_FLOOR_MB ))
CONTAINER_MEM_MB=$(( SCYLLA_MEM_MB + margin_mb + TMPFS_SIZE_GB * 1024 ))

# Bridge networking + explicit port, not --network host, to avoid colliding
# with any other local ScyllaDB container.

# --overprovisioned: matches scylla-ccm's own default for non-pinned
# containers.
#
# --developer-mode skips iotune calibration, defaulting to a degenerate
# IO scheduler config (max-io-requests=1). io_properties.yaml below is a
# static calibration (measured ~8-9GB/s, ~1.8-2M IOPS on tmpfs) so Scylla
# still sees realistic RAM-disk IO limits.
#
# --unsafe-bypass-fsync: safe only because this container is ephemeral and
# tmpfs-backed. Never carry this into anything that needs to survive a crash.
IO_PROPERTIES_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/io_properties.yaml"

# mode=1777 on the tmpfs mount: rootless podman's UID remapping means the
# image's unprivileged `scylla` user won't own a bare tmpfs mount otherwise.

# Seastar's AIO reactor can exhaust the host-wide fs.aio-max-nr default;
# raise it if needed. Best-effort -- this script doesn't otherwise need root.
AIO_MAX_NR_TARGET=1048576
current_aio_max_nr=$(cat /proc/sys/fs/aio-max-nr 2>/dev/null || echo 0)
if [ "$current_aio_max_nr" -lt "$AIO_MAX_NR_TARGET" ]; then
  if sudo -n sysctl -w fs.aio-max-nr="$AIO_MAX_NR_TARGET" >/dev/null 2>&1; then
    echo "Raised fs.aio-max-nr $current_aio_max_nr -> $AIO_MAX_NR_TARGET"
  else
    echo "WARNING: fs.aio-max-nr is $current_aio_max_nr (< $AIO_MAX_NR_TARGET)" \
         "and couldn't raise it (no passwordless sudo) -- continuing anyway." >&2
  fi
fi

echo "Sizing: Scylla --memory ${SCYLLA_MEM_MB}M + margin ${margin_mb}M" \
     "+ tmpfs ${TMPFS_SIZE_GB}g = container cgroup cap ${CONTAINER_MEM_MB}M" \
     "(margin = max(${MARGIN_PCT}% => ${margin_pct_mb}M, floor => ${MARGIN_FLOOR_MB}M))"

# Cgroup cap bounds only this container; tmpfs is still real host RAM and
# can trigger a system-wide OOM kill elsewhere. Reserve headroom for the rest
# of the machine.
HOST_HEADROOM_MB=4096
mem_available_kb=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo 2>/dev/null || echo "")
if [ -n "$mem_available_kb" ]; then
  mem_available_mb=$(( mem_available_kb / 1024 ))
  if [ "$CONTAINER_MEM_MB" -gt $(( mem_available_mb - HOST_HEADROOM_MB )) ]; then
    echo "ERROR: container would reserve ${CONTAINER_MEM_MB}M, but only" \
         "${mem_available_mb}M is available on the host (need" \
         "${HOST_HEADROOM_MB}M headroom left over for everything else)." >&2
    echo "Lower SCYLLA_MEM_MB/CONTAINER_CPUS (shrinks the margin) or" \
         "TMPFS_SIZE_GB in this script, or free up host memory, then retry." >&2
    exit 1
  fi
else
  echo "WARNING: couldn't read /proc/meminfo to check available host memory" \
       "-- continuing without the pre-flight OOM check." >&2
fi

echo "Removing any previous '$NAME' container..."
podman rm -f --time 0 "$NAME" >/dev/null 2>&1 || true

echo "Starting $NAME (cpus=$CONTAINER_CPUS mem=${CONTAINER_MEM_MB}M, tmpfs=${TMPFS_SIZE_GB}g) on port $HOST_PORT..."
# --health-cmd runs cqlsh inside the container so "healthy" means CQL is
# actually serving, not just a port check. 19042 is Scylla's shard-aware
# port. Fast health-startup-* checks (2s) until the first success, then a
# slow steady-state check (30s) -- cqlsh is a fresh Python process each
# time and podman's --cpuset-cpus puts it on the same cores as Scylla, so
# a 2s-forever check measurably competes with Scylla during a benchmark.
podman run -d --name "$NAME" \
  --cpus="$CONTAINER_CPUS" --memory="${CONTAINER_MEM_MB}m" \
  --health-startup-cmd 'cqlsh -e "SELECT now() FROM system.local" || exit 1' \
  --health-startup-interval 2s --health-startup-retries 30 \
  --health-startup-timeout 5s --health-startup-success 1 \
  --health-cmd 'cqlsh -e "SELECT now() FROM system.local" || exit 1' \
  --health-interval 30s --health-retries 30 --health-timeout 5s \
  --health-start-period 10s \
  -p "${HOST_PORT}:9042" -p 19042:19042 \
  --tmpfs "/var/lib/scylla:size=${TMPFS_SIZE_GB}g,mode=1777" \
  -v "${IO_PROPERTIES_FILE}:/etc/scylla.d/io_properties.yaml:ro" \
  "$IMAGE" \
  --smp "$SCYLLA_SMP" --memory "${SCYLLA_MEM_MB}M" --developer-mode 1 \
  --unsafe-bypass-fsync 1 --overprovisioned 1 \
  --io-properties-file /etc/scylla.d/io_properties.yaml

echo "Waiting for ScyllaDB to report healthy..."
for i in $(seq 1 60); do
  status=$(podman inspect --format '{{.State.Health.Status}}' "$NAME" 2>/dev/null || echo "")
  if [ "$status" = "healthy" ]; then
    echo "Ready after ~$((i * 2))s."
    exit 0
  fi
  if [ "$status" = "unhealthy" ]; then
    echo "Container reported unhealthy. Recent logs:" >&2
    podman logs --tail 40 "$NAME" >&2
    exit 1
  fi
  sleep 2
done

echo "Timed out waiting for ScyllaDB to become healthy. Recent logs:" >&2
podman logs --tail 40 "$NAME" >&2
exit 1
