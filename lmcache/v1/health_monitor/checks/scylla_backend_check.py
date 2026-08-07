# SPDX-License-Identifier: Apache-2.0
"""
Health check for ScyllaDBBackend.
"""

# Standard
from typing import TYPE_CHECKING, List, Optional
import time

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.v1.health_monitor.base import (
    FailureCheckResult,
    GetBlockingFailureTracker,
    HealthCheck,
    resolve_fallback_policy,
)
from lmcache.v1.health_monitor.constants import (
    DEFAULT_FALLBACK_POLICY,
    DEFAULT_GET_BLOCKING_FAILED_THRESHOLD,
    DEFAULT_WAITING_TIME_FOR_RECOVERY,
    FALLBACK_POLICY_CONFIG_KEY,
    GET_BLOCKING_FAILED_THRESHOLD_CONFIG_KEY,
    PING_GENERIC_ERROR_CODE,
    PING_TIMEOUT_ERROR_CODE,
    WAITING_TIME_FOR_RECOVERY_CONFIG_KEY,
    FallbackPolicy,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.manager import LMCacheManager
    from lmcache.v1.storage_backend.scylla_backend import ScyllaDBBackend

logger = init_logger(__name__)


class ScyllaBackendHealthCheck(HealthCheck):
    """
    Health check for ScyllaDBBackend by pinging ScyllaDB and monitoring
    ``get_blocking``/put failure counters.

    ``ScyllaDBBackend`` implements ``StoragePluginInterface`` rather than
    ``RemoteBackend``, so it is not covered by ``RemoteBackendHealthCheck``
    (which only discovers ``RemoteBackend`` instances). This check mirrors
    ``RemoteBackendHealthCheck``'s logic using the public
    ``get_blocking_failed_count``/``put_failed_count``/``ping()``/
    ``is_connected()`` surface ``ScyllaDBBackend`` exposes for exactly this
    purpose.

    Fallback Policies:
        - RECOMPUTE (default): Skip all cache operations when ScyllaDB is
          unhealthy
        - LOCAL_CPU: Bypass ScyllaDB and use local CPU with hot_cache
          enabled
    """

    def __init__(self, backend: "ScyllaDBBackend"):
        self.backend = backend
        fallback_policy_value = (
            backend.config.get_extra_config_value(
                FALLBACK_POLICY_CONFIG_KEY, DEFAULT_FALLBACK_POLICY.value
            )
            if backend.config is not None
            else DEFAULT_FALLBACK_POLICY.value
        )
        self._fallback_policy = resolve_fallback_policy(
            fallback_policy_value, context=str(backend)
        )

        self._failure_tracker = GetBlockingFailureTracker()
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()
        self._backend_name: Optional[str] = None

    @classmethod
    def create(cls, manager: "LMCacheManager") -> List[HealthCheck]:
        """
        Create ScyllaBackendHealthCheck instances from a LMCacheManager.

        Finds all ScyllaDBBackend instances in the storage manager and
        creates a health check for each one.
        """
        # First Party
        from lmcache.v1.storage_backend.scylla_backend import ScyllaDBBackend

        instances: List[HealthCheck] = []

        engine = manager.lmcache_engine
        if engine is None or engine.storage_manager is None:
            return instances

        for backend_name, backend in engine.storage_manager.storage_backends.items():
            if isinstance(backend, ScyllaDBBackend):
                check = cls(backend)
                check._backend_name = backend_name
                instances.append(check)
                logger.info(f"Created {check} for {backend_name}")

        return instances

    def name(self) -> str:
        return f"ScyllaBackendHealthCheck({self._backend_name})"

    @property
    def fallback_policy(self) -> FallbackPolicy:
        """Return the fallback policy for this health check."""
        return self._fallback_policy

    def get_bypass_backend_name(self) -> Optional[str]:
        """Return the backend name to bypass when this check fails."""
        return self._backend_name

    def _get_waiting_time_for_recovery(self) -> float:
        if self.backend.config is None:
            return DEFAULT_WAITING_TIME_FOR_RECOVERY
        return self.backend.config.get_extra_config_value(
            WAITING_TIME_FOR_RECOVERY_CONFIG_KEY, DEFAULT_WAITING_TIME_FOR_RECOVERY
        )

    def _get_failed_threshold(self) -> int:
        if self.backend.config is None:
            return DEFAULT_GET_BLOCKING_FAILED_THRESHOLD
        return self.backend.config.get_extra_config_value(
            GET_BLOCKING_FAILED_THRESHOLD_CONFIG_KEY,
            DEFAULT_GET_BLOCKING_FAILED_THRESHOLD,
        )

    def check(self) -> bool:
        """
        Perform a health check on the ScyllaDB backend:

        - If ``get_blocking_failed_count`` increased by at least the
          configured threshold since the last check, mark unhealthy and
          wait at least ``waiting_time_for_recovery`` seconds before
          re-checking (mirrors ``RemoteBackendHealthCheck``).
        - Otherwise, issue a lightweight ``ping()`` and record its latency
          and error code.

        :return: ``True`` if the backend is healthy, ``False`` otherwise.
        """
        if not self.backend.is_connected():
            logger.warning("ScyllaDBBackend is not connected.")
            return False

        failure_check = self._failure_tracker.evaluate(
            current_failed_count=self.backend.get_blocking_failed_count,
            threshold=self._get_failed_threshold(),
            waiting_time_for_recovery=self._get_waiting_time_for_recovery(),
            recovery_probe=lambda: self.backend.ping() == 0,
        )
        if failure_check is FailureCheckResult.UNHEALTHY:
            return False

        start = time.perf_counter()
        error_code = self.backend.ping()
        latency = (time.perf_counter() - start) * 1000
        self._stats_monitor.update_remote_ping_latency(latency)
        self._stats_monitor.update_remote_ping_error_code(error_code)

        if error_code == 1:
            # Disconnected -- already covered by is_connected() above, but
            # ping() can also observe this transiently.
            return False
        if error_code == 2:
            self._stats_monitor.update_remote_ping_error_code(PING_TIMEOUT_ERROR_CODE)
            logger.warning("ScyllaDB ping timed out")
            return False
        if error_code != 0:
            self._stats_monitor.update_remote_ping_error_code(PING_GENERIC_ERROR_CODE)
            logger.warning(f"ScyllaDB ping failed with error code: {error_code}")
            return False

        return True
