"""Smoke tests for the Azure Container Apps Sandboxes provider.

These tests do not contact Azure. They verify that the public API imports, the
polymorphic options/state registries dispatch on the ``aca_sandbox`` discriminator,
and that ``create()`` forwards to ``begin_create_sandbox`` with the expected kwargs.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from agents.extensions.sandbox.aca_sandbox import (
    ACASandboxClient,
    ACASandboxClientOptions,
    ACASandboxGroupVolumeMount,
    ACASandboxGroupVolumeMountStrategy,
    ACASandboxSession,
    ACASandboxSessionState,
    ACASandboxVolumeMount,
)
from agents.sandbox.entries import BaseEntry
from agents.sandbox.errors import (
    ExecTransportError,
    InvalidManifestPathError,
    WorkspaceStartError,
)
from agents.sandbox.manifest import Environment, Manifest
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


def test_group_volume_mount_entry_polymorphic_parse() -> None:
    payload = {
        "type": "aca_sandbox_group_volume_mount",
        "volume_name": "shared",
        "mount_path": "/workspace/shared",
        "mount_strategy": {"type": "aca_sandbox_group_volume"},
    }
    parsed = BaseEntry.parse(payload)
    assert isinstance(parsed, ACASandboxGroupVolumeMount)
    assert parsed.volume_name == "shared"
    assert isinstance(parsed.mount_strategy, ACASandboxGroupVolumeMountStrategy)


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
        fake_sandbox.mkdir = AsyncMock()
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


def test_create_forwards_native_volume_mounts() -> None:
    async def _coro() -> None:
        fake_sandbox = MagicMock()
        fake_sandbox.sandbox_id = "sb-fake"
        fake_sandbox.mkdir = AsyncMock()
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
                volume_mounts=(
                    ACASandboxVolumeMount(
                        volume_name="shared-memory",
                        mountpoint="/workspace/memories",
                        read_only=True,
                    ),
                ),
            ),
        )
        await session.aclose()

        volumes = fake_gc.begin_create_sandbox.await_args.kwargs["volumes"]
        assert len(volumes) == 1
        assert volumes[0].volume_name == "shared-memory"
        assert volumes[0].mountpoint == "/workspace/memories"
        assert volumes[0].read_only is True

    asyncio.run(_coro())


def test_create_forwards_manifest_group_volume_mounts() -> None:
    async def _coro() -> None:
        fake_sandbox = MagicMock()
        fake_sandbox.sandbox_id = "sb-fake"
        fake_sandbox.mkdir = AsyncMock()
        fake_poller = MagicMock()
        fake_poller.result = AsyncMock(return_value=fake_sandbox)
        fake_gc = MagicMock()
        fake_gc.begin_create_sandbox = AsyncMock(return_value=fake_poller)
        fake_gc.subscription_id = "sub-x"
        fake_gc.resource_group = "rg-x"
        fake_gc.sandbox_group = "grp-x"
        fake_gc.endpoint = "https://management.westus2.azuredevcompute.io"
        manifest = Manifest(
            entries={
                "memories": ACASandboxGroupVolumeMount(
                    volume_name="shared-memory",
                    mount_strategy=ACASandboxGroupVolumeMountStrategy(),
                    read_only=False,
                )
            }
        )

        client = ACASandboxClient(fake_gc)
        session = await client.create(manifest=manifest)
        await session.aclose()

        volumes = fake_gc.begin_create_sandbox.await_args.kwargs["volumes"]
        assert len(volumes) == 1
        assert volumes[0].volume_name == "shared-memory"
        assert volumes[0].mountpoint == "/workspace/memories"
        assert volumes[0].read_only is False

    asyncio.run(_coro())


def test_create_forwards_manifest_environment() -> None:
    async def _coro() -> None:
        fake_sandbox = MagicMock()
        fake_sandbox.sandbox_id = "sb-fake"
        fake_sandbox.mkdir = AsyncMock()
        fake_poller = MagicMock()
        fake_poller.result = AsyncMock(return_value=fake_sandbox)
        fake_gc = MagicMock()
        fake_gc.begin_create_sandbox = AsyncMock(return_value=fake_poller)
        fake_gc.subscription_id = "sub-x"
        fake_gc.resource_group = "rg-x"
        fake_gc.sandbox_group = "grp-x"
        fake_gc.endpoint = "https://management.westus2.azuredevcompute.io"

        manifest = Manifest(
            environment=Environment(value={"OPENAI_API_KEY": "sk-test", "FOO": "bar"})
        )

        client = ACASandboxClient(fake_gc)
        session = await client.create(manifest=manifest)
        await session.aclose()

        kwargs = fake_gc.begin_create_sandbox.await_args.kwargs
        assert kwargs["environment"] == {"OPENAI_API_KEY": "sk-test", "FOO": "bar"}

    asyncio.run(_coro())


def test_create_omits_environment_when_manifest_env_is_empty() -> None:
    async def _coro() -> None:
        fake_sandbox = MagicMock()
        fake_sandbox.sandbox_id = "sb-fake"
        fake_sandbox.mkdir = AsyncMock()
        fake_poller = MagicMock()
        fake_poller.result = AsyncMock(return_value=fake_sandbox)
        fake_gc = MagicMock()
        fake_gc.begin_create_sandbox = AsyncMock(return_value=fake_poller)
        fake_gc.subscription_id = "sub-x"
        fake_gc.resource_group = "rg-x"
        fake_gc.sandbox_group = "grp-x"
        fake_gc.endpoint = "https://management.westus2.azuredevcompute.io"

        client = ACASandboxClient(fake_gc)
        session = await client.create()
        await session.aclose()

        kwargs = fake_gc.begin_create_sandbox.await_args.kwargs
        assert "environment" not in kwargs

    asyncio.run(_coro())


def test_create_with_snapshot_id_strips_create_only_fields() -> None:
    async def _coro() -> None:
        fake_sandbox = MagicMock()
        fake_sandbox.sandbox_id = "sb-restored"
        fake_sandbox.mkdir = AsyncMock()
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
            manifest=Manifest(environment=Environment(value={"FOO": "bar"})),
            options=ACASandboxClientOptions(snapshot_id="snap-123", labels={"tenant": "demo"}),
        )
        await session.aclose()

        kwargs = fake_gc.begin_create_sandbox.await_args.kwargs
        assert kwargs["snapshot_id"] == "snap-123"
        # Snapshot restore replays captured state, so create-only fields are stripped.
        assert "disk" not in kwargs
        assert "labels" not in kwargs
        assert "ports" not in kwargs
        assert "environment" not in kwargs

    asyncio.run(_coro())


def test_resolve_exposed_port_parses_backend_url() -> None:
    async def _coro() -> None:
        fake_sandbox = MagicMock()
        fake_sandbox.add_port = AsyncMock(
            return_value=MagicMock(url="https://abc.example.test:8443/?token=redacted")
        )
        inner = ACASandboxSession(
            state=_make_state().model_copy(update={"exposed_ports": (8080,)}),
            sandbox_client=fake_sandbox,
        )

        endpoint = await inner.resolve_exposed_port(8080)

        assert endpoint.host == "abc.example.test"
        assert endpoint.port == 8443
        assert endpoint.tls is True
        assert endpoint.query == "token=redacted"
        assert endpoint.url_for("http") == "https://abc.example.test:8443/?token=redacted"

    asyncio.run(_coro())


def test_exec_uses_workspace_root_as_working_directory() -> None:
    async def _coro() -> None:
        fake_sandbox = MagicMock()
        fake_sandbox.exec = AsyncMock(return_value=MagicMock(stdout=b"ok", stderr=b"", exit_code=0))
        inner = ACASandboxSession(state=_make_state(), sandbox_client=fake_sandbox)

        result = await inner.exec("pwd")

        assert result.stdout == b"ok"
        fake_sandbox.exec.assert_awaited_once_with(
            "sh -lc pwd",
            working_directory="/workspace",
        )

    asyncio.run(_coro())


def test_exec_quotes_single_shell_false_command() -> None:
    async def _coro() -> None:
        fake_sandbox = MagicMock()
        fake_sandbox.exec = AsyncMock(return_value=MagicMock(stdout=b"", stderr=b"", exit_code=127))
        inner = ACASandboxSession(state=_make_state(), sandbox_client=fake_sandbox)

        await inner.exec("echo safe; uname", shell=False)

        fake_sandbox.exec.assert_awaited_once_with(
            "'echo safe; uname'",
            working_directory="/workspace",
        )

    asyncio.run(_coro())


def test_read_rejects_absolute_path_outside_workspace() -> None:
    async def _coro() -> None:
        fake_sandbox = MagicMock()
        fake_sandbox.read_file = AsyncMock()
        inner = ACASandboxSession(state=_make_state(), sandbox_client=fake_sandbox)

        with pytest.raises(InvalidManifestPathError):
            await inner.read(Path("/tmp/config.json"))

        fake_sandbox.read_file.assert_not_awaited()

    asyncio.run(_coro())


def test_write_rejects_absolute_path_outside_workspace() -> None:
    async def _coro() -> None:
        fake_sandbox = MagicMock()
        fake_sandbox.write_file = AsyncMock()
        inner = ACASandboxSession(state=_make_state(), sandbox_client=fake_sandbox)

        with pytest.raises(InvalidManifestPathError):
            await inner.write(Path("/tmp/config.json"), io.BytesIO(b"secret"))

        fake_sandbox.write_file.assert_not_awaited()

    asyncio.run(_coro())


# ---- hardening: exit-code fidelity, env prefix, resume, options contract --------------


def test_exec_internal_raises_when_exit_code_missing() -> None:
    async def _coro() -> None:
        # A backend that omits exit_code from its exec result must NOT be silently
        # treated as success: that would let a failed tar/mkdir during persist or
        # hydrate corrupt the workspace.
        bad_result = MagicMock(spec=["stdout", "stderr"])
        bad_result.stdout = b""
        bad_result.stderr = b"boom"
        fake_sandbox = MagicMock()
        fake_sandbox.exec = AsyncMock(return_value=bad_result)
        inner = ACASandboxSession(state=_make_state(), sandbox_client=fake_sandbox)

        with pytest.raises(ExecTransportError):
            await inner.exec("pwd")

    asyncio.run(_coro())


def _state_with_env(env: dict[str, str]) -> ACASandboxSessionState:
    return ACASandboxSessionState(
        type="aca_sandbox",
        session_id=uuid4(),
        snapshot=NoopSnapshot(id="noop-test"),
        manifest=Manifest(environment=Environment(value=dict(env))),
        sandbox_id="sb-test-1",
        sandbox_group="sample-group",
        resource_group="sample-rg",
        subscription_id="00000000-0000-0000-0000-000000000000",
        region="westus2",
    )


def test_exec_prepends_env_when_manifest_env_present() -> None:
    async def _coro() -> None:
        fake_sandbox = MagicMock()
        fake_sandbox.exec = AsyncMock(return_value=MagicMock(stdout=b"", stderr=b"", exit_code=0))
        inner = ACASandboxSession(
            state=_state_with_env({"OPENAI_API_KEY": "sk-test", "FOO": "bar"}),
            sandbox_client=fake_sandbox,
        )

        await inner.exec("pwd")

        fake_sandbox.exec.assert_awaited_once()
        joined = fake_sandbox.exec.await_args.args[0]
        # The env -- K=V prefix must appear before the user's argv so capabilities
        # (Memory, Skills) that add env entries after create() see them in exec.
        assert joined.startswith("env -- ")
        assert "OPENAI_API_KEY=sk-test" in joined
        assert "FOO=bar" in joined
        assert joined.endswith(" pwd") or joined.endswith(" sh -lc pwd")

    asyncio.run(_coro())


def test_exec_omits_env_prefix_when_manifest_env_empty() -> None:
    async def _coro() -> None:
        fake_sandbox = MagicMock()
        fake_sandbox.exec = AsyncMock(return_value=MagicMock(stdout=b"", stderr=b"", exit_code=0))
        inner = ACASandboxSession(state=_make_state(), sandbox_client=fake_sandbox)

        await inner.exec("pwd")

        joined = fake_sandbox.exec.await_args.args[0]
        # No env prefix when the manifest has no env entries.
        assert not joined.startswith("env -- ")

    asyncio.run(_coro())


def test_resume_raises_typed_error_on_sandbox_not_found() -> None:
    pytest.importorskip("azure.core")
    from azure.core.exceptions import ResourceNotFoundError

    async def _coro() -> None:
        fake_sb = MagicMock()
        fake_sb.ensure_running = AsyncMock(side_effect=ResourceNotFoundError("gone"))
        fake_sb.mkdir = AsyncMock()
        fake_gc = MagicMock()
        fake_gc.get_sandbox_client = MagicMock(return_value=fake_sb)
        fake_gc.endpoint = "https://management.westus2.azuredevcompute.io"

        client = ACASandboxClient(fake_gc)
        with pytest.raises(WorkspaceStartError) as excinfo:
            await client.resume(_make_state())

        # Context should carry a reason field consumers can branch on.
        assert excinfo.value.context.get("reason") == "sandbox_not_found"
        assert excinfo.value.context.get("sandbox_id") == "sb-test-1"

    asyncio.run(_coro())


def test_resume_does_not_mark_preserved_when_workspace_root_not_ready() -> None:
    async def _coro() -> None:
        fake_sb = MagicMock()
        fake_sb.ensure_running = AsyncMock()
        fake_sb.mkdir = AsyncMock()
        fake_gc = MagicMock()
        fake_gc.get_sandbox_client = MagicMock(return_value=fake_sb)
        fake_gc.endpoint = "https://management.westus2.azuredevcompute.io"

        client = ACASandboxClient(fake_gc)
        # workspace_root_ready defaults to False; assert preserved flag stays False.
        state = _make_state()
        assert state.workspace_root_ready is False
        session = await client.resume(state)

        inner = getattr(session, "_inner", session)
        assert inner._workspace_state_preserved_on_start() is False

        await session.aclose()

    asyncio.run(_coro())


def test_resume_marks_preserved_when_workspace_root_ready() -> None:
    async def _coro() -> None:
        fake_sb = MagicMock()
        fake_sb.ensure_running = AsyncMock()
        fake_sb.mkdir = AsyncMock()
        fake_gc = MagicMock()
        fake_gc.get_sandbox_client = MagicMock(return_value=fake_sb)
        fake_gc.endpoint = "https://management.westus2.azuredevcompute.io"

        client = ACASandboxClient(fake_gc)
        state = _make_state().model_copy(update={"workspace_root_ready": True})
        session = await client.resume(state)

        inner = getattr(session, "_inner", session)
        assert inner._workspace_state_preserved_on_start() is True

        await session.aclose()

    asyncio.run(_coro())


@pytest.mark.parametrize(
    "option_kwargs,expected",
    [
        pytest.param(
            {"disk": "ubuntu"},
            {"disk": "ubuntu"},
            id="disk",
        ),
        pytest.param(
            {"disk_id": "img-123"},
            {"disk_id": "img-123"},
            id="disk_id-supersedes-disk",
        ),
        pytest.param(
            {"cpu": "1500m"},
            {"cpu": "1500m"},
            id="cpu",
        ),
        pytest.param(
            {"memory": "4096Mi"},
            {"memory": "4096Mi"},
            id="memory",
        ),
        pytest.param(
            {"exposed_ports": (8080, 9090)},
            {"ports": [8080, 9090]},
            id="exposed_ports",
        ),
        pytest.param(
            {"polling_interval_seconds": 7.0},
            {"polling_interval": 7},
            id="polling_interval_seconds",
        ),
        pytest.param(
            {"polling_timeout_seconds": 600.0},
            {"polling_timeout": 600},
            id="polling_timeout_seconds",
        ),
    ],
)
def test_create_forwards_advertised_option_knobs(
    option_kwargs: dict[str, Any], expected: dict[str, Any]
) -> None:
    async def _coro() -> None:
        fake_sandbox = MagicMock()
        fake_sandbox.sandbox_id = "sb-fake"
        fake_sandbox.mkdir = AsyncMock()
        fake_poller = MagicMock()
        fake_poller.result = AsyncMock(return_value=fake_sandbox)
        fake_gc = MagicMock()
        fake_gc.begin_create_sandbox = AsyncMock(return_value=fake_poller)
        fake_gc.subscription_id = "sub-x"
        fake_gc.resource_group = "rg-x"
        fake_gc.sandbox_group = "grp-x"
        fake_gc.endpoint = "https://management.westus2.azuredevcompute.io"

        client = ACASandboxClient(fake_gc)
        session = await client.create(options=ACASandboxClientOptions(**option_kwargs))
        await session.aclose()

        kwargs = fake_gc.begin_create_sandbox.await_args.kwargs
        for key, value in expected.items():
            assert kwargs[key] == value, (
                f"begin_create_sandbox kwargs[{key!r}]={kwargs.get(key)!r} != {value!r}"
            )
        # disk_id and disk are mutually exclusive: when disk_id is provided we must
        # not also pass a raw `disk` so the backend uses the explicit image.
        if "disk_id" in option_kwargs:
            assert "disk" not in kwargs

    asyncio.run(_coro())


def test_session_state_serialization_roundtrip_preserves_fields() -> None:
    state = ACASandboxSessionState(
        type="aca_sandbox",
        session_id=uuid4(),
        snapshot=NoopSnapshot(id="noop-rt"),
        manifest=Manifest(environment=Environment(value={"K": "V"})),
        exposed_ports=(8080, 9090),
        workspace_root_ready=True,
        sandbox_id="sb-rt",
        sandbox_group="grp-rt",
        resource_group="rg-rt",
        subscription_id="11111111-1111-1111-1111-111111111111",
        region="eastus2",
    )
    payload = state.model_dump(mode="json")
    parsed = SandboxSessionState.parse(payload)
    assert isinstance(parsed, ACASandboxSessionState)
    assert parsed.sandbox_id == "sb-rt"
    assert parsed.sandbox_group == "grp-rt"
    assert parsed.resource_group == "rg-rt"
    assert parsed.subscription_id == "11111111-1111-1111-1111-111111111111"
    assert parsed.region == "eastus2"
    assert parsed.exposed_ports == (8080, 9090)
    assert parsed.workspace_root_ready is True
    assert parsed.manifest.environment is not None
