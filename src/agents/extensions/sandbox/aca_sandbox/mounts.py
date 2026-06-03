"""Mount strategies for Azure Container Apps sandboxes."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

from pydantic import Field

from ....sandbox.entries.mounts.base import Mount, MountStrategyBase
from ....sandbox.errors import MountConfigError
from ....sandbox.materialization import MaterializedFile
from ....sandbox.session.base_sandbox_session import BaseSandboxSession
from ....sandbox.types import FileMode, Permissions
from ....sandbox.workspace_paths import sandbox_path_str


class ACASandboxGroupVolumeMount(Mount):
    """Mount an existing ACA sandbox-group volume into the sandbox workspace.

    The volume must already exist on the Azure Container Apps sandbox group.
    This entry is translated into the ACA SDK's ``SandboxVolume`` create-time
    option by :class:`ACASandboxClient`.
    """

    type: Literal["aca_sandbox_group_volume_mount"] = "aca_sandbox_group_volume_mount"
    volume_name: str
    mount_strategy: MountStrategyBase = Field(
        default_factory=lambda: ACASandboxGroupVolumeMountStrategy()
    )

    def model_post_init(self, context: object, /) -> None:
        _ = context
        default_permissions = Permissions(
            owner=FileMode.ALL,
            group=FileMode.READ | FileMode.EXEC,
            other=FileMode.READ | FileMode.EXEC,
        )
        if (
            self.permissions.owner != default_permissions.owner
            or self.permissions.group != default_permissions.group
            or self.permissions.other != default_permissions.other
        ):
            warnings.warn(
                "Mount permissions are not enforced. "
                "Please configure access in the cloud provider instead; "
                "mount-level permissions can be unreliable.",
                stacklevel=2,
            )
            self.permissions.owner = default_permissions.owner
            self.permissions.group = default_permissions.group
            self.permissions.other = default_permissions.other
        self.permissions.directory = True
        self.mount_strategy.validate_mount(self)


class ACASandboxGroupVolumeMountStrategy(MountStrategyBase):
    """Attach an existing ACA sandbox-group volume during sandbox creation."""

    type: Literal["aca_sandbox_group_volume"] = "aca_sandbox_group_volume"

    def validate_mount(self, mount: Mount) -> None:
        if not isinstance(mount, ACASandboxGroupVolumeMount):
            raise MountConfigError(
                message=(
                    "ACASandboxGroupVolumeMountStrategy requires an "
                    "ACASandboxGroupVolumeMount entry"
                ),
                context={"mount_type": mount.type},
            )

    def supports_native_snapshot_detach(self, mount: Mount) -> bool:
        _ = mount
        return False

    async def activate(
        self,
        mount: Mount,
        session: BaseSandboxSession,
        dest: Path,
        base_dir: Path,
    ) -> list[MaterializedFile]:
        self._assert_aca_session(session)
        _ = (mount, dest, base_dir)
        return []

    async def deactivate(
        self,
        mount: Mount,
        session: BaseSandboxSession,
        dest: Path,
        base_dir: Path,
    ) -> None:
        self._assert_aca_session(session)
        _ = (mount, dest, base_dir)

    async def teardown_for_snapshot(
        self,
        mount: Mount,
        session: BaseSandboxSession,
        path: Path,
    ) -> None:
        self._assert_aca_session(session)
        _ = (mount, path)

    async def restore_after_snapshot(
        self,
        mount: Mount,
        session: BaseSandboxSession,
        path: Path,
    ) -> None:
        self._assert_aca_session(session)
        _ = (mount, path)

    def build_docker_volume_driver_config(
        self,
        mount: Mount,
    ) -> tuple[str, dict[str, str], bool] | None:
        _ = mount
        return None

    @staticmethod
    def mountpoint_for(mount: ACASandboxGroupVolumeMount, fallback: Path) -> str:
        return sandbox_path_str(mount.mount_path or fallback)

    @staticmethod
    def _assert_aca_session(session: BaseSandboxSession) -> None:
        if type(session).__name__ != "ACASandboxSession":
            raise MountConfigError(
                message="ACA sandbox group volume mounts require an ACASandboxSession",
                context={"session_type": type(session).__name__},
            )


__all__ = [
    "ACASandboxGroupVolumeMount",
    "ACASandboxGroupVolumeMountStrategy",
]
