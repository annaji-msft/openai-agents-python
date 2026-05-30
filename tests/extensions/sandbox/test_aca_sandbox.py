"""Smoke tests for the Azure Container Apps Sandboxes provider.

These tests do not contact Azure. They verify that the public API imports, the
polymorphic options/state registries dispatch on the ``aca_sandbox`` discriminator,
and that ``create()`` forwards to ``begin_create_sandbox`` with the expected kwargs.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from agents.extensions.sandbox.aca_sandbox import (
    ACASandboxClient,
    ACASandboxClientOptions,
    ACASandboxSession,
    ACASandboxSessionState,
    ACASandboxVolumeMount,
)
from agents.sandbox.manifest import Manifest
from agents.sandbox.session.sandbox_client import BaseSandboxClientOptions
from agents.sandbox.session.sandbox_session_state import SandboxSessionState
from agents.sandbox.snapshot import NoopSnapshot


def test_backend_id_is_stable() -> None:
    assert ACASandboxClient.backend_id == "aca_sandbox"


def test_options_defaults() -> None:
    opts = ACASandboxClientOptions()
    assert opts.type == "aca_sandbox"
    assert opts.disk == "ubuntu"
    assert opts.exposed_ports == ()
    assert opts.volume_mounts == ()


def test_options_polymorphic_parse() -> None:
    payload = {"type": "aca_sandbox", "disk": "ubuntu", "exposed_ports": [8080]}
    parsed = BaseSandboxClientOptions.parse(payload)
    assert isinstance(parsed, ACASandboxClientOptions)
    assert parsed.disk == "ubuntu"
    assert parsed.exposed_ports == (8080,)


def test_volume_mount_polymorphic_parse() -> None:
    payload = {
        "type": "aca_sandbox_volume_mount",
        "volume_name": "shared",
        "mountpoint": "/workspace/shared",
        "read_only": True,
    }
    parsed = BaseSandboxClientOptions.parse(payload)
    assert isinstance(parsed, ACASandboxVolumeMount)
    assert parsed.volume_name == "shared"
    assert parsed.mountpoint == "/workspace/shared"
    assert parsed.read_only is True


def _make_state() -> ACASandboxSessionState:
    return ACASandboxSessionState(
        type="aca_sandbox",
        session_id=uuid4(),
        snapshot=NoopSnapshot(id="noop-test"),
        manifest=Manifest(),
        sandbox_id="sb-test-1",
        sandbox_group="sample-group",
        resource_group="sample-rg",
        subscription_id="00000000-0000-0000-0000-000000000000",
        region="westus2",
    )


def test_state_polymorphic_parse() -> None:
    raw = _make_state().model_dump(mode="json")
    parsed = SandboxSessionState.parse(raw)
    assert isinstance(parsed, ACASandboxSessionState)
    assert parsed.sandbox_id == "sb-test-1"
    assert parsed.region == "westus2"


def _bare_client() -> ACASandboxClient:
    client = ACASandboxClient.__new__(ACASandboxClient)
    object.__setattr__(client, "_group_client", None)
    object.__setattr__(client, "_instrumentation", None)
    object.__setattr__(client, "_dependencies", None)
    return client


def test_client_deserialize_session_state_roundtrip() -> None:
    raw = _make_state().model_dump(mode="json")
    restored = _bare_client().deserialize_session_state(raw)
    assert isinstance(restored, ACASandboxSessionState)
    assert restored.sandbox_id == "sb-test-1"


def test_session_from_state_constructs_inner() -> None:
    inner = ACASandboxSession.from_state(_make_state())
    assert isinstance(inner, ACASandboxSession)
    assert inner.state.sandbox_id == "sb-test-1"
    assert inner.supports_pty() is False
    assert inner.supports_docker_volume_mounts() is False


def test_region_recovery_from_endpoint() -> None:
    client = _bare_client()
    object.__setattr__(
        client,
        "_group_client",
        MagicMock(_endpoint="", endpoint="https://management.westus2.azuredevcompute.io"),
    )
    assert client._region_from_endpoint() == "westus2"


def test_create_calls_begin_create_sandbox_with_expected_kwargs() -> None:
    async def _coro() -> None:
        fake_sandbox = MagicMock()
        fake_sandbox.sandbox_id = "sb-fake"
        fake_poller = MagicMock()
        fake_poller.result = AsyncMock(return_value=fake_sandbox)
        fake_gc = MagicMock()
        fake_gc.begin_create_sandbox = AsyncMock(return_value=fake_poller)
        fake_gc.subscription_id = "sub-x"
        fake_gc.resource_group = "rg-x"
        fake_gc.sandbox_group = "grp-x"
        fake_gc.endpoint = "https://management.westus2.azuredevcompute.io"

        client = ACASandboxClient(fake_gc)
        session = await client.create(
            options=ACASandboxClientOptions(
                disk="ubuntu",
                cpu="1000m",
                memory="2048Mi",
                labels={"tenant": "demo"},
                exposed_ports=(8080,),
            ),
        )
        await session.aclose()  # tear down the wrapper without starting the inner session

        fake_gc.begin_create_sandbox.assert_awaited_once()
        kwargs = fake_gc.begin_create_sandbox.await_args.kwargs
        assert kwargs["disk"] == "ubuntu"
        assert kwargs["cpu"] == "1000m"
        assert kwargs["memory"] == "2048Mi"
        assert kwargs["ports"] == [8080]
        labels = kwargs["labels"]
        assert labels["tenant"] == "demo"
        assert "session-id" in labels

    asyncio.run(_coro())


def test_create_with_snapshot_id_strips_create_only_fields() -> None:
    async def _coro() -> None:
        fake_sandbox = MagicMock()
        fake_sandbox.sandbox_id = "sb-restored"
        fake_poller = MagicMock()
        fake_poller.result = AsyncMock(return_value=fake_sandbox)
        fake_gc = MagicMock()
        fake_gc.begin_create_sandbox = AsyncMock(return_value=fake_poller)
        fake_gc.subscription_id = "sub-x"
        fake_gc.resource_group = "rg-x"
        fake_gc.sandbox_group = "grp-x"
        fake_gc.endpoint = "https://management.westus2.azuredevcompute.io"

        client = ACASandboxClient(fake_gc)
        session = await client.create(
            options=ACASandboxClientOptions(snapshot_id="snap-123", labels={"tenant": "demo"}),
        )
        await session.aclose()

        kwargs = fake_gc.begin_create_sandbox.await_args.kwargs
        assert kwargs["snapshot_id"] == "snap-123"
        # Snapshot restore replays captured state, so create-only fields are stripped.
        assert "disk" not in kwargs
        assert "labels" not in kwargs
        assert "ports" not in kwargs

    asyncio.run(_coro())
