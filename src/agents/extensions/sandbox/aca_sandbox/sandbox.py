"""
Azure Container Apps Sandboxes (https://learn.microsoft.com/azure/container-apps/) implementation.

This module provides an Azure Container Apps Sandboxes-backed sandbox client and session
implementation backed by ``azure.containerapps.sandbox.aio.SandboxGroupClient`` and the
per-sandbox handle returned from its long-running create operation.

The ``azure-containerapps-sandbox`` dependency is optional and installed via the
``aca-sandbox`` extra, so package-level exports guard the import of this module. Within
this module the Azure SDK imports are normal so users with the extra installed get full
type navigation.
"""

from __future__ import annotations

import asyncio
import io
import logging
import shlex
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import ConfigDict, Field

from ....sandbox.errors import (
    ExecTimeoutError,
    ExecTransportError,
    SandboxError,
    WorkspaceReadNotFoundError,
)
from ....sandbox.manifest import Manifest
from ....sandbox.session.base_sandbox_session import BaseSandboxSession
from ....sandbox.session.dependencies import Dependencies
from ....sandbox.session.manager import Instrumentation
from ....sandbox.session.sandbox_client import BaseSandboxClient, BaseSandboxClientOptions
from ....sandbox.session.sandbox_session import SandboxSession
from ....sandbox.session.sandbox_session_state import SandboxSessionState
from ....sandbox.snapshot import SnapshotBase, SnapshotSpec, resolve_snapshot
from ....sandbox.types import ExecResult, ExposedPortEndpoint, User

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from azure.containerapps.sandbox.aio import (
        SandboxClient as _AzureSandboxClient,
        SandboxGroupClient as _AzureSandboxGroupClient,
    )

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------------

DEFAULT_ACA_DISK = "ubuntu"
"""Default public disk image used when no ``disk`` / ``disk_id`` is supplied."""

DEFAULT_ACA_WORKSPACE_ROOT = "/workspace"
"""Workspace root mounted inside an Azure Container Apps Sandbox."""

DEFAULT_ACA_POLLING_INTERVAL_SECONDS = 3.0
DEFAULT_ACA_POLLING_TIMEOUT_SECONDS = 300.0


# ----------------------------------------------------------------------------------
# Internal helpers (paths + exception remap)
# ----------------------------------------------------------------------------------


def _to_posix(path: Path | str) -> str:
    """Return ``path`` as an absolute POSIX string anchored at ``/`` or the workspace root."""

    if isinstance(path, Path):
        text = path.as_posix()
    else:
        text = str(path).replace("\\", "/")

    if not text:
        return DEFAULT_ACA_WORKSPACE_ROOT

    if text.startswith("/"):
        return text

    return f"{DEFAULT_ACA_WORKSPACE_ROOT}/{text.lstrip('/')}"


def _ensure_bytes(value: str | bytes | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8", errors="replace")


def _remap_azure_exception(
    error: Exception,
    *,
    context: str | None = None,
    command: list[str | Path] | None = None,
    path: str | Path | None = None,
    timeout_s: float | None = None,
) -> Exception:
    """Remap an Azure SDK exception onto the sandbox-native error family.

    ``SandboxError`` instances pass through unchanged so callers can ``raise ... from``
    without double-wrapping. The Azure SDK imports are guarded so this module stays
    importable even when ``azure-core`` is not installed.
    """

    if isinstance(error, SandboxError):
        return error

    cmd_list: list[str | Path] = list(command) if command is not None else []
    ctx_dict: dict[str, Any] | None = {"context": context} if context else None
    msg = (
        f"{context}: {type(error).__name__}: {error}"
        if context
        else (f"{type(error).__name__}: {error}")
    )

    def _path_for_error() -> Path:
        if path is None:
            return Path("?")
        return Path(str(path)) if not isinstance(path, Path) else path

    try:
        from azure.core.exceptions import (
            AzureError,
            ClientAuthenticationError,
            HttpResponseError,
            ResourceNotFoundError,
            ServiceRequestError,
            ServiceResponseError,
        )
    except ImportError:  # pragma: no cover - azure-core ships with the extra
        AzureError = Exception  # type: ignore[assignment, misc]
        ClientAuthenticationError = Exception  # type: ignore[assignment, misc]
        HttpResponseError = Exception  # type: ignore[assignment, misc]
        ResourceNotFoundError = Exception  # type: ignore[assignment, misc]
        ServiceRequestError = Exception  # type: ignore[assignment, misc]
        ServiceResponseError = Exception  # type: ignore[assignment, misc]

    if isinstance(error, TimeoutError):
        return ExecTimeoutError(
            command=cmd_list,
            timeout_s=timeout_s,
            context=ctx_dict,
            cause=error,
        )

    if isinstance(error, ResourceNotFoundError):
        return WorkspaceReadNotFoundError(
            path=_path_for_error(),
            context=ctx_dict,
            cause=error,
        )

    if isinstance(error, ClientAuthenticationError):
        return ExecTransportError(
            command=cmd_list,
            context=ctx_dict,
            cause=error,
            message=msg,
        )

    if isinstance(error, HttpResponseError):
        status: Any = getattr(error, "status_code", None)
        if status == 404:
            return WorkspaceReadNotFoundError(
                path=_path_for_error(),
                context=ctx_dict,
                cause=error,
            )
        return ExecTransportError(
            command=cmd_list,
            context=ctx_dict,
            cause=error,
            message=msg,
        )

    if isinstance(error, ServiceRequestError | ServiceResponseError | AzureError):
        return ExecTransportError(
            command=cmd_list,
            context=ctx_dict,
            cause=error,
            message=msg,
        )

    return error


# ----------------------------------------------------------------------------------
# Models — volume mount, options, session state
# ----------------------------------------------------------------------------------


class ACASandboxVolumeMount(BaseSandboxClientOptions):
    """Mount an existing sandbox-group volume into the sandbox at create time.

    The named volume must already exist on the parent sandbox group; this struct
    is the per-create wiring that attaches it to a path inside the sandbox.
    """

    type: Literal["aca_sandbox_volume_mount"] = "aca_sandbox_volume_mount"
    volume_name: str = Field(..., description="Name of the existing sandbox-group volume to mount.")
    mountpoint: str = Field(
        ..., description="Absolute path inside the sandbox where the volume is mounted."
    )
    read_only: bool | None = Field(
        default=None,
        description="Mount read-only; default (``None``) defers to the backend default.",
    )


class ACASandboxClientOptions(BaseSandboxClientOptions):
    """Per-create options for :class:`ACASandboxClient`.

    Fields map onto :meth:`SandboxGroupClient.begin_create_sandbox`. ``polling_*``
    fields control how long the client waits for sandbox lifecycle long-running
    operations.
    """

    type: Literal["aca_sandbox"] = "aca_sandbox"

    disk: str = Field(
        default=DEFAULT_ACA_DISK,
        description="Public disk image name used when ``disk_id`` is not set.",
    )
    disk_id: str | None = Field(
        default=None,
        description="Private disk image ID; overrides ``disk`` when set.",
    )
    snapshot_id: str | None = Field(
        default=None,
        description=(
            "Snapshot ID to restore from. When set the backend reuses the snapshot's "
            "captured disk/cpu/memory/volumes/ports/labels and the other create-time "
            "fields below are ignored."
        ),
    )
    cpu: str | None = Field(default=None, description="CPU request (e.g. ``'1000m'``).")
    memory: str | None = Field(default=None, description="Memory request (e.g. ``'2048Mi'``).")
    labels: dict[str, str] | None = Field(
        default=None,
        description="Labels stamped on the created sandbox for observability/cleanup.",
    )
    exposed_ports: tuple[int, ...] = Field(
        default_factory=tuple,
        description="TCP ports to publish on create.",
    )
    volume_mounts: tuple[ACASandboxVolumeMount, ...] = Field(
        default_factory=tuple,
        description=("Existing sandbox-group volumes to mount into this sandbox at create time."),
    )
    polling_interval_seconds: float = Field(
        default=DEFAULT_ACA_POLLING_INTERVAL_SECONDS,
        gt=0.0,
        description="Poll interval (seconds) for sandbox lifecycle long-running operations.",
    )
    polling_timeout_seconds: float = Field(
        default=DEFAULT_ACA_POLLING_TIMEOUT_SECONDS,
        gt=0.0,
        description="Polling deadline (seconds) for sandbox lifecycle long-running operations.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class ACASandboxSessionState(SandboxSessionState):
    """JSON-serializable identity of an Azure Container Apps sandbox.

    Use :meth:`ACASandboxClient.resume` (or :meth:`SandboxSessionState.parse`) to
    reattach to an existing sandbox by its ``sandbox_id``.
    """

    type: Literal["aca_sandbox"] = "aca_sandbox"

    sandbox_id: str
    """Sandbox UUID assigned by the backend."""

    sandbox_group: str
    """Name of the parent sandbox group."""

    resource_group: str
    """Azure resource group that owns the sandbox group."""

    subscription_id: str
    """Azure subscription that owns the resource group."""

    region: str
    """Azure region of the sandbox group (used to reconstruct the data-plane endpoint)."""


# ----------------------------------------------------------------------------------
# Session
# ----------------------------------------------------------------------------------


class ACASandboxSession(BaseSandboxSession):
    """Sandbox session backed by an :class:`azure.containerapps.sandbox.aio.SandboxClient`."""

    state: ACASandboxSessionState

    def __init__(
        self,
        *,
        state: ACASandboxSessionState,
        sandbox_client: _AzureSandboxClient | None = None,
        group_client: _AzureSandboxGroupClient | None = None,
    ) -> None:
        self.state = state
        # ``create()`` supplies the sandbox client straight from the LRO poller.
        # ``resume()`` keeps a reference to the group client so the per-sandbox handle
        # can be lazily rebuilt if the underlying network handle expires.
        self._sb = sandbox_client
        self._gc = group_client
        self._running = sandbox_client is not None

    @classmethod
    def from_state(cls, state: ACASandboxSessionState) -> ACASandboxSession:
        return cls(state=state)

    # ---- capabilities ----------------------------------------------------------------

    def supports_pty(self) -> bool:
        return False

    def supports_docker_volume_mounts(self) -> bool:
        return False

    # ---- lifecycle -------------------------------------------------------------------

    async def running(self) -> bool:
        return self._running and self._sb is not None

    async def _prepare_backend_workspace(self) -> None:
        # The runtime disk attaches the workspace root automatically; ensure the
        # directory exists in case the configured disk image is bare.
        if self._sb is None:
            return
        try:
            await self._sb.mkdir(_to_posix(DEFAULT_ACA_WORKSPACE_ROOT))
        except Exception as exc:  # noqa: BLE001 - best-effort, exec layer logs failures
            logger.debug("_prepare_backend_workspace mkdir: %s", exc)

    async def _after_start(self) -> None:
        self._running = True

    async def _after_start_failed(self) -> None:
        self._running = False

    async def _after_shutdown(self) -> None:
        # Local shutdown only. Backend cleanup happens in ``ACASandboxClient.delete``
        # so resumed sessions can reattach to a sandbox that survived an earlier
        # local close.
        self._running = False

    async def _resolve_exposed_port(self, port: int) -> ExposedPortEndpoint:
        if self._sb is None:
            raise ExecTransportError(
                command=[],
                message="Sandbox client is not bound; call create() first.",
            )
        try:
            published = await self._sb.add_port(port)
        except Exception as exc:  # noqa: BLE001 - remap to sandbox-native error
            raise _remap_azure_exception(exc, context=f"add_port({port})") from exc
        host = str(getattr(published, "host", None) or getattr(published, "url", "") or "")
        return ExposedPortEndpoint(host=host, port=port)

    # ---- IO --------------------------------------------------------------------------

    async def _exec_internal(
        self,
        *command: str | Path,
        timeout: float | None = None,
    ) -> ExecResult:
        if self._sb is None:
            raise ExecTransportError(
                command=list(command),
                message="Sandbox client is not bound; call create() first.",
            )

        joined = command[0] if len(command) == 1 else shlex.join(str(part) for part in command)
        coro = self._sb.exec(str(joined))
        try:
            result = await (asyncio.wait_for(coro, timeout=timeout) if timeout else coro)
        except asyncio.TimeoutError as exc:
            raise ExecTimeoutError(
                command=list(command),
                timeout_s=timeout,
                cause=exc,
            ) from exc
        except Exception as exc:  # noqa: BLE001 - remap to sandbox-native error
            raise _remap_azure_exception(
                exc,
                context=f"exec({str(joined)[:120]!r})",
                command=list(command),
            ) from exc

        return ExecResult(
            stdout=_ensure_bytes(getattr(result, "stdout", b"")),
            stderr=_ensure_bytes(getattr(result, "stderr", b"")),
            exit_code=int(getattr(result, "exit_code", 0) or 0),
        )

    async def read(self, path: Path, *, user: str | User | None = None) -> io.IOBase:
        if self._sb is None:
            raise ExecTransportError(
                command=[],
                message="Sandbox client is not bound; call create() first.",
            )
        target = _to_posix(path)
        try:
            data = await self._sb.read_file(target)
        except Exception as exc:  # noqa: BLE001 - remap to sandbox-native error
            raise _remap_azure_exception(
                exc, context=f"read_file({target!r})", path=target
            ) from exc
        return io.BytesIO(data if isinstance(data, bytes) else _ensure_bytes(data))

    async def write(
        self,
        path: Path,
        data: io.IOBase,
        *,
        user: str | User | None = None,
    ) -> None:
        if self._sb is None:
            raise ExecTransportError(
                command=[],
                message="Sandbox client is not bound; call create() first.",
            )
        target = _to_posix(path)
        payload = data.read()
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        try:
            await self._sb.write_file(target, payload, create_dirs=True)
        except Exception as exc:  # noqa: BLE001 - remap to sandbox-native error
            raise _remap_azure_exception(
                exc, context=f"write_file({target!r})", path=target
            ) from exc

    # ---- workspace persistence -------------------------------------------------------

    async def persist_workspace(self) -> io.IOBase:
        # Workspace persistence will move to the backend-native snapshot/restore flow
        # once it lands in the Azure SDK. For now return an empty buffer so the
        # framework's snapshot bookkeeping has a no-op payload to round-trip.
        return io.BytesIO(b"")

    async def hydrate_workspace(self, data: io.IOBase) -> None:
        # Counterpart to ``persist_workspace``: drain the buffer so callers can free
        # it, then no-op. Workspace state is rematerialized from the manifest on each
        # start.
        try:
            data.read()
        except Exception:  # noqa: BLE001 - best-effort drain
            pass


# ----------------------------------------------------------------------------------
# Client
# ----------------------------------------------------------------------------------


class ACASandboxClient(BaseSandboxClient[ACASandboxClientOptions]):
    """Sandbox client backed by an :class:`azure.containerapps.sandbox.aio.SandboxGroupClient`.

    Construct the group client (with credentials and target sandbox group) in the
    calling application and hand it to this client. The client owns the per-sandbox
    lifecycle and bridges the agents-SDK contract onto the Azure data-plane.
    """

    backend_id: str = "aca_sandbox"
    supports_default_options: bool = True

    def __init__(
        self,
        group_client: _AzureSandboxGroupClient,
        *,
        instrumentation: Instrumentation | None = None,
        dependencies: Dependencies | None = None,
    ) -> None:
        super().__init__()
        self._group_client = group_client
        self._instrumentation = instrumentation or Instrumentation()
        self._dependencies = dependencies

    @property
    def group_client(self) -> _AzureSandboxGroupClient:
        return self._group_client

    # ---- create ----------------------------------------------------------------------

    async def create(
        self,
        *,
        snapshot: SnapshotSpec | SnapshotBase | None = None,
        manifest: Manifest | None = None,
        options: ACASandboxClientOptions | None = None,
    ) -> SandboxSession:
        opts = options or ACASandboxClientOptions()
        resolved_manifest = manifest or Manifest()
        session_id = uuid.uuid4()
        snapshot_instance = resolve_snapshot(snapshot, str(session_id))

        labels: dict[str, str] = dict(opts.labels) if opts.labels else {}
        labels.setdefault("session-id", str(session_id))

        # When restoring from a captured snapshot the backend rejects fields that
        # would contradict the snapshot's pre-recorded state (disk, cpu, memory,
        # ports, volumes, labels). Only the snapshot id and polling controls are
        # forwarded in that case.
        restoring_from_snapshot = opts.snapshot_id is not None

        create_kwargs: dict[str, Any] = {
            "polling_interval": int(opts.polling_interval_seconds),
            "polling_timeout": int(opts.polling_timeout_seconds),
        }
        if restoring_from_snapshot:
            create_kwargs["snapshot_id"] = opts.snapshot_id
        else:
            create_kwargs["labels"] = labels
            if opts.disk_id is not None:
                create_kwargs["disk_id"] = opts.disk_id
            else:
                create_kwargs["disk"] = opts.disk
            if opts.cpu is not None:
                create_kwargs["cpu"] = opts.cpu
            if opts.memory is not None:
                create_kwargs["memory"] = opts.memory
            if opts.exposed_ports:
                create_kwargs["ports"] = list(opts.exposed_ports)
            if opts.volume_mounts:
                from azure.containerapps.sandbox import SandboxVolume

                create_kwargs["volumes"] = [
                    SandboxVolume(
                        volume_name=mount.volume_name,
                        mountpoint=mount.mountpoint,
                        read_only=mount.read_only,
                    )
                    for mount in opts.volume_mounts
                ]

        try:
            poller = await self._group_client.begin_create_sandbox(**create_kwargs)
            sandbox_client = await poller.result()
        except Exception as exc:  # noqa: BLE001 - remap to sandbox-native error
            raise _remap_azure_exception(exc, context="begin_create_sandbox") from exc

        gc_subscription = str(getattr(self._group_client, "subscription_id", "") or "")
        gc_resource_group = str(getattr(self._group_client, "resource_group", "") or "")
        gc_sandbox_group = str(getattr(self._group_client, "sandbox_group", "") or "")
        gc_region = self._region_from_endpoint()

        state = ACASandboxSessionState(
            type="aca_sandbox",
            session_id=session_id,
            manifest=resolved_manifest,
            snapshot=snapshot_instance,
            exposed_ports=opts.exposed_ports,
            sandbox_id=str(getattr(sandbox_client, "sandbox_id", "") or ""),
            sandbox_group=gc_sandbox_group,
            resource_group=gc_resource_group,
            subscription_id=gc_subscription,
            region=gc_region,
        )

        inner = ACASandboxSession(
            state=state,
            sandbox_client=sandbox_client,
            group_client=self._group_client,
        )
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    # ---- delete ----------------------------------------------------------------------

    async def delete(self, session: SandboxSession) -> SandboxSession:
        inner = session._inner
        if not isinstance(inner, ACASandboxSession):
            raise TypeError("ACASandboxClient.delete expects an ACASandboxSession")

        # Close local resources first so dependencies/streams shut down cleanly,
        # then release the backend sandbox through the long-running delete
        # operation.
        try:
            await inner.shutdown()
        except Exception as exc:  # noqa: BLE001 - best-effort local teardown
            logger.debug("inner.shutdown() during delete: %s", exc)

        sandbox_id = inner.state.sandbox_id
        try:
            poller = await self._group_client.begin_delete_sandbox(sandbox_id)
            await poller.result()
        except Exception as exc:  # noqa: BLE001 - remap to sandbox-native error
            raise _remap_azure_exception(
                exc, context=f"begin_delete_sandbox({sandbox_id!r})"
            ) from exc

        inner._sb = None
        inner._running = False
        return session

    # ---- resume ----------------------------------------------------------------------

    async def resume(self, state: SandboxSessionState) -> SandboxSession:
        if not isinstance(state, ACASandboxSessionState):
            raise TypeError("ACASandboxClient.resume expects an ACASandboxSessionState")

        try:
            sandbox_client = self._group_client.get_sandbox_client(state.sandbox_id)
            # Wake the sandbox if it was auto-suspended; harmless if already running.
            await sandbox_client.ensure_running()
        except Exception as exc:  # noqa: BLE001 - remap to sandbox-native error
            raise _remap_azure_exception(
                exc, context=f"resume(sandbox_id={state.sandbox_id!r})"
            ) from exc

        inner = ACASandboxSession(
            state=state,
            sandbox_client=sandbox_client,
            group_client=self._group_client,
        )
        # Tell the framework that preserved backend state is intact so it can skip a
        # full manifest re-apply if the workspace root is still ready.
        inner._set_start_state_preserved(True)
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    def deserialize_session_state(self, payload: dict[str, object]) -> SandboxSessionState:
        instance = ACASandboxSessionState.model_validate(payload)
        return cast(SandboxSessionState, instance)

    # ---- helpers ---------------------------------------------------------------------

    def _region_from_endpoint(self) -> str:
        # The data-plane endpoint follows the form
        # ``https://management.<region>.azuredevcompute.io``; recover the region
        # from the configured endpoint for serialization so a resume on a different
        # host can rebuild the endpoint URL.
        endpoint = getattr(self._group_client, "_endpoint", "") or getattr(
            self._group_client, "endpoint", ""
        )
        endpoint = str(endpoint)
        if "management." in endpoint:
            try:
                return endpoint.split("management.", 1)[1].split(".", 1)[0]
            except IndexError:
                return ""
        return ""


__all__ = [
    "ACASandboxClient",
    "ACASandboxClientOptions",
    "ACASandboxSession",
    "ACASandboxSessionState",
    "ACASandboxVolumeMount",
    "DEFAULT_ACA_DISK",
    "DEFAULT_ACA_WORKSPACE_ROOT",
    "DEFAULT_ACA_POLLING_INTERVAL_SECONDS",
    "DEFAULT_ACA_POLLING_TIMEOUT_SECONDS",
]
