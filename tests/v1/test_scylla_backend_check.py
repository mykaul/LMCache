# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ScyllaBackendHealthCheck.

Uses a MagicMock(spec=ScyllaDBBackend) standing in for the real backend
(which requires a live ScyllaDB cluster) -- these tests exercise the health
check's own decision logic (discovery, ping-error-code handling, and the
shared GetBlockingFailureTracker threshold/recovery window), not the
backend's connectivity itself (covered by test_scylla_backend*.py).
"""

# Standard
from unittest.mock import MagicMock
import sys
import time
import types

# Third Party
import pytest

# Stub the scylla (python-rs-driver) package BEFORE importing scylla_backend.
# Whichever test module imports scylla_backend first in a given pytest
# process determines _SCYLLA_AVAILABLE for the rest of it (module-level
# imports only execute once) -- if this file ran first without stubbing,
# scylla_backend would see no driver installed, bind none of its driver
# symbols, and every other scylla test file's patch() targets (e.g.
# "scylla_backend.SessionBuilder") would fail with AttributeError. Mirrors
# tests/v1/storage_backend/test_scylla_backend.py's stubbing so import
# order across the test suite doesn't matter. This file's own tests never
# touch the driver directly (they mock ScyllaDBBackend itself), so the
# stub contents are irrelevant here -- only their presence matters.
if "scylla" not in sys.modules:
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
        sys.modules[_name] = types.ModuleType(_name)
    sys.modules["scylla"].enums = sys.modules["scylla.enums"]
    sys.modules["scylla"].errors = sys.modules["scylla.errors"]
    sys.modules["scylla"].execution_profile = sys.modules["scylla.execution_profile"]
    sys.modules["scylla"].policies = sys.modules["scylla.policies"]
    sys.modules["scylla.policies"].load_balancing = sys.modules[
        "scylla.policies.load_balancing"
    ]
    sys.modules["scylla"].session = sys.modules["scylla.session"]
    sys.modules["scylla"].session_builder = sys.modules["scylla.session_builder"]
    sys.modules["scylla"].statement = sys.modules["scylla.statement"]
    sys.modules["scylla.enums"].Consistency = type(
        "MockConsistency", (), {"LocalOne": 1, "Quorum": 2, "All": 3}
    )
    sys.modules["scylla.enums"].Compression = type(
        "MockCompression", (), {"Lz4": 1, "Snappy": 2}
    )
    sys.modules["scylla.errors"].ExecuteError = type(
        "MockExecuteError", (Exception,), {}
    )
    sys.modules["scylla.errors"].PrepareError = type(
        "MockPrepareError", (Exception,), {}
    )
    sys.modules["scylla.errors"].RequestTimeoutError = type(
        "MockRequestTimeoutError", (Exception,), {}
    )
    sys.modules["scylla.errors"].SessionConnectionError = type(
        "MockSessionConnectionError", (Exception,), {}
    )
    sys.modules["scylla.execution_profile"].ExecutionProfile = lambda **kw: None
    sys.modules["scylla.policies.load_balancing"].DefaultPolicy = lambda **kw: None
    sys.modules["scylla.policies.load_balancing"].NodeLocationPreference = type(
        "MockNodeLocationPreference",
        (),
        {
            "datacenter": staticmethod(lambda name: ("dc", name)),
            "datacenter_and_rack": staticmethod(lambda dc, rack: ("dc_rack", dc, rack)),
        },
    )
    sys.modules["scylla.session"].Session = type("MockSessionImport", (), {})
    sys.modules["scylla.session_builder"].SessionBuilder = type(
        "MockSessionBuilderImport", (), {}
    )
    sys.modules["scylla.statement"].PreparedStatement = type(
        "MockPreparedStatementImport", (), {}
    )

# First Party
from lmcache.v1.config import LMCacheEngineConfig  # noqa: E402
from lmcache.v1.health_monitor.checks.scylla_backend_check import (  # noqa: E402
    ScyllaBackendHealthCheck,
)
from lmcache.v1.health_monitor.constants import FallbackPolicy  # noqa: E402
from lmcache.v1.storage_backend.scylla_backend import ScyllaDBBackend  # noqa: E402

pytestmark = pytest.mark.no_shared_allocator


def make_config(extra_overrides: dict | None = None) -> LMCacheEngineConfig:
    extra_config = {"fallback_policy": "recompute"}
    if extra_overrides:
        extra_config.update(extra_overrides)
    return LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        extra_config=extra_config,
    )


@pytest.fixture
def mock_backend():
    backend = MagicMock(spec=ScyllaDBBackend)
    backend.config = make_config()
    backend.is_connected.return_value = True
    backend.ping.return_value = 0
    backend.get_blocking_failed_count = 0
    return backend


@pytest.fixture
def mock_manager(mock_backend):
    engine = MagicMock()
    engine.storage_manager.storage_backends = {"ScyllaDBBackend": mock_backend}
    manager = MagicMock()
    manager.lmcache_engine = engine
    return manager


class TestScyllaBackendHealthCheckCreate:
    def test_discovers_scylla_backend(self, mock_manager, mock_backend):
        instances = ScyllaBackendHealthCheck.create(mock_manager)
        assert len(instances) == 1
        assert instances[0].backend is mock_backend
        assert instances[0].get_bypass_backend_name() == "ScyllaDBBackend"

    def test_ignores_non_scylla_backends(self, mock_manager):
        mock_manager.lmcache_engine.storage_manager.storage_backends = {
            "LocalCPUBackend": MagicMock()
        }
        assert ScyllaBackendHealthCheck.create(mock_manager) == []

    def test_returns_empty_when_no_engine(self):
        manager = MagicMock()
        manager.lmcache_engine = None
        assert ScyllaBackendHealthCheck.create(manager) == []

    def test_returns_empty_when_no_storage_manager(self):
        manager = MagicMock()
        manager.lmcache_engine.storage_manager = None
        assert ScyllaBackendHealthCheck.create(manager) == []


class TestScyllaBackendHealthCheckCheck:
    def test_healthy_when_connected_and_ping_succeeds(self, mock_backend):
        check = ScyllaBackendHealthCheck(mock_backend)
        assert check.check() is True

    def test_unhealthy_when_disconnected(self, mock_backend):
        mock_backend.is_connected.return_value = False
        check = ScyllaBackendHealthCheck(mock_backend)
        assert check.check() is False
        # A disconnected backend should not even attempt a ping.
        mock_backend.ping.assert_not_called()

    @pytest.mark.parametrize("error_code", [1, 2, 3])
    def test_unhealthy_on_nonzero_ping_error_code(self, mock_backend, error_code):
        mock_backend.ping.return_value = error_code
        check = ScyllaBackendHealthCheck(mock_backend)
        assert check.check() is False

    def test_unhealthy_after_threshold_failures_then_recovers(self, mock_backend):
        mock_backend.config = make_config(
            {
                "get_blocking_failed_threshold": 3,
                "waiting_time_for_recovery": 0.1,
            }
        )
        check = ScyllaBackendHealthCheck(mock_backend)
        assert check.check() is True

        mock_backend.get_blocking_failed_count = 5
        assert check.check() is False

        # Still within the recovery window: stays unhealthy even though a
        # ping right now would succeed.
        assert check.check() is False

        time.sleep(0.15)
        assert check.check() is True


class TestScyllaBackendHealthCheckFallbackPolicy:
    def test_defaults_to_recompute(self, mock_backend):
        check = ScyllaBackendHealthCheck(mock_backend)
        assert check.fallback_policy == FallbackPolicy.RECOMPUTE

    def test_resolves_local_cpu_from_config(self, mock_backend):
        mock_backend.config = make_config({"fallback_policy": "local_cpu"})
        check = ScyllaBackendHealthCheck(mock_backend)
        assert check.fallback_policy == FallbackPolicy.LOCAL_CPU

    def test_handles_none_config(self, mock_backend):
        mock_backend.config = None
        check = ScyllaBackendHealthCheck(mock_backend)
        assert check.fallback_policy == FallbackPolicy.RECOMPUTE
