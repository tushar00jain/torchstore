# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NIXL transport for StorageVolume tensor transfers."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import socket
import time
import uuid
import weakref
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch

from torchstore.transport.buffers import TransportBuffer, TransportCache
from torchstore.transport.types import Request

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.buffers import TransportContext

try:
    from nixl import nixl_agent as _nixl_agent
    from nixl import nixl_agent_config as _nixl_agent_config
except ImportError:
    try:
        from nixl_cu13 import nixl_agent as _nixl_agent
        from nixl_cu13 import nixl_agent_config as _nixl_agent_config
    except ImportError:
        try:
            from nixl_cu12 import nixl_agent as _nixl_agent
            from nixl_cu12 import nixl_agent_config as _nixl_agent_config
        except ImportError:
            _nixl_agent = None
            _nixl_agent_config = None


logger = logging.getLogger(__name__)


def _backend() -> str:
    return os.environ.get("TORCHSTORE_NIXL_BACKEND", "UCX")


def _configure_bundled_ucx_modules() -> None:
    """Point UCX at plugins bundled beside the CUDA-specific NIXL wheel."""
    if "UCX_MODULE_DIR" in os.environ or _nixl_agent is None:
        return
    package_name = _nixl_agent.__module__.partition(".")[0]
    try:
        package = importlib.import_module(package_name)
        package_file = getattr(package, "__file__", None)
        if package_file is None:
            return
        module_dir = (
            Path(package_file).resolve().parent.parent
            / f"{package_name}.libs"
            / "ucx"
        )
        if module_dir.is_dir():
            os.environ["UCX_MODULE_DIR"] = str(module_dir)
    except Exception:
        logger.debug("Could not locate bundled NIXL UCX modules", exc_info=True)


@cache
def nixl_available() -> bool:
    """Return whether the optional NIXL transport was explicitly enabled."""
    return (
        os.environ.get("TORCHSTORE_NIXL_ENABLED", "0") == "1"
        and _nixl_agent is not None
        and _nixl_agent_config is not None
    )


@dataclass
class _Registration:
    descriptors: Any
    storage_ref: weakref.ReferenceType[Any]


class NixlAgentCache(TransportCache):
    """Process-local NIXL agent and reusable memory registrations."""

    def __init__(self) -> None:
        if _nixl_agent is None or _nixl_agent_config is None:
            raise RuntimeError("NIXL is not installed")

        _configure_bundled_ucx_modules()
        self.backend = _backend()
        name = (
            f"torchstore-{socket.gethostname()[:24]}-{os.getpid()}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        config = _nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=False,
            backends=[self.backend],
        )
        self.agent = _nixl_agent(name, config)
        self._registrations: dict[tuple[int, int], _Registration] = {}
        self._remote_agents: dict[str, bytes] = {}

    def register(self, tensor: torch.Tensor) -> Any:
        """Register a contiguous tensor once for the lifetime of its storage."""
        key = (tensor.data_ptr(), tensor.nbytes)
        existing = self._registrations.get(key)
        if existing is not None:
            return existing.descriptors

        descriptors = self.agent.register_memory(tensor, backends=[self.backend])
        storage_ref = weakref.ref(
            tensor.untyped_storage(), lambda _ref, _key=key: self._evict(_key)
        )
        self._registrations[key] = _Registration(descriptors, storage_ref)
        return descriptors

    def _evict(self, key: tuple[int, int]) -> None:
        registration = self._registrations.pop(key, None)
        if registration is None:
            return
        try:
            self.agent.deregister_memory(
                registration.descriptors, backends=[self.backend]
            )
        except Exception:
            logger.warning("Failed to deregister NIXL memory", exc_info=True)

    def add_remote_agent(self, name: str, metadata: bytes) -> str:
        """Load new remote metadata, replacing an older copy when needed."""
        previous_metadata = self._remote_agents.get(name)
        if previous_metadata == metadata:
            return name
        if previous_metadata is not None:
            self.agent.remove_remote_agent(name)
        remote_name = self.agent.add_remote_agent(metadata)
        if isinstance(remote_name, bytes):
            remote_name = remote_name.decode()
        if remote_name != name:
            raise RuntimeError(
                f"NIXL metadata named agent {remote_name!r}, expected {name!r}"
            )
        self._remote_agents[name] = metadata
        return remote_name

    def clear(self) -> None:
        for key in list(self._registrations):
            self._evict(key)
        for remote_name in self._remote_agents:
            try:
                self.agent.remove_remote_agent(remote_name)
            except Exception:
                logger.warning(
                    "Failed to remove NIXL remote agent %s",
                    remote_name,
                    exc_info=True,
                )
        self._remote_agents.clear()


@dataclass
class NixlRequestContext:
    """Serializable state for one tensor or object in a request batch."""

    remote_descriptors: bytes | None = None
    tensor: torch.Tensor | None = None
    shape: torch.Size | None = None
    dtype: torch.dtype | None = None
    is_object: bool = False
    objects: Any = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["tensor"] = None
        return state


class NixlTransportBuffer(TransportBuffer):
    """Move StorageVolume tensors with NIXL's UCX backend."""

    supports_batch_puts = True
    supports_batch_gets = True

    def __init__(self, storage_volume_ref: "StorageVolumeRef") -> None:
        super().__init__(storage_volume_ref)
        if not nixl_available():
            raise RuntimeError(
                "NIXL transport is unavailable. Install nixl and set "
                "TORCHSTORE_NIXL_ENABLED=1."
            )
        self._client_metadata: bytes | None = None
        self._client_agent_name: str | None = None
        self._contexts: list[NixlRequestContext] = []

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["storage_volume_ref"] = None
        return state

    def _client_cache(self) -> NixlAgentCache:
        return self.storage_volume_ref.transport_context.get(NixlAgentCache)

    def _register_client_tensor(self, tensor: torch.Tensor) -> NixlRequestContext:
        assert tensor.is_contiguous()
        cache = self._client_cache()
        cache.register(tensor)
        descriptors = cache.agent.get_xfer_descs(tensor)
        return NixlRequestContext(
            remote_descriptors=cache.agent.get_serialized_descs(descriptors),
            tensor=tensor,
            shape=tensor.shape,
            dtype=tensor.dtype,
        )

    def _publish_client_metadata(self) -> None:
        if any(not context.is_object for context in self._contexts):
            cache = self._client_cache()
            self._client_agent_name = cache.agent.name
            self._client_metadata = cache.agent.get_agent_metadata()

    async def _pre_put_hook(self, requests: list[Request]) -> None:
        """Allocate RDMA memory for put (transport already set up)."""
        self._contexts = []
        for request in requests:
            if request.is_object:
                self._contexts.append(
                    NixlRequestContext(is_object=True, objects=request.objects)
                )
                continue

            tensor = request.tensor_val
            assert tensor is not None
            if not tensor.is_contiguous():
                logger.warning(
                    "NIXL PUT received a non-contiguous tensor for key=%s; "
                    "staging a contiguous CPU copy",
                    request.key,
                )
                tensor = tensor.cpu().contiguous()
            self._contexts.append(self._register_client_tensor(tensor))
        self._publish_client_metadata()

    async def _pre_get_hook(self, requests: list[Request]) -> None:
        """Fetch metadata if needed and allocate RDMA buffers."""
        # 1. fetch metadata in a single batch, preserving order
        meta_requests = [req.meta_only() for req in requests if req.tensor_val is None]
        meta_results = (
            await self.storage_volume_ref.volume.get_meta.call_one(meta_requests)
            if meta_requests
            else []
        )
        meta_iterator = iter(meta_results)

        # 2. build contexts
        self._contexts = []
        for request in requests:
            tensor = request.tensor_val
            if tensor is None:
                meta = next(meta_iterator)
                if isinstance(meta, str) or meta is None:
                    self._contexts.append(NixlRequestContext(is_object=True))
                    continue
                shape, dtype = meta
                if request.tensor_slice is not None:
                    shape = request.tensor_slice.local_shape
                tensor = torch.empty(shape, dtype=dtype, device="cpu")

            self._contexts.append(self._register_client_tensor(tensor))
        self._publish_client_metadata()

    async def _transfer(
        self,
        ctx: "TransportContext",
        operation: str,
        transfers: list[tuple[Request, torch.Tensor, bytes]],
    ) -> None:
        if not transfers:
            return

        cache = ctx.get(NixlAgentCache)
        remote_agent = self._connect_client(cache)
        local_descs = None
        remote_descs = None
        handle = None
        try:
            for request, tensor, serialized_remote_descs in transfers:
                cache.register(tensor)
                tensor_local_descs = cache.agent.get_xfer_descs(tensor)
                tensor_remote_descs = cache.agent.deserialize_descs(
                    serialized_remote_descs
                )

                if local_descs is None:
                    local_descs = tensor_local_descs
                    remote_descs = tensor_remote_descs
                else:
                    if local_descs.getType() != tensor_local_descs.getType():
                        raise RuntimeError(
                            "NIXL cannot batch a mix of CPU and GPU tensors on "
                            "the StorageVolume side"
                        )
                    if remote_descs.getType() != tensor_remote_descs.getType():
                        raise RuntimeError(
                            "NIXL cannot batch a mix of CPU and GPU tensors on "
                            "the client side; CPU-to-GPU transfers are supported"
                        )
                    for index in range(tensor_local_descs.descCount()):
                        local_descs.append(tensor_local_descs[index])
                    for index in range(tensor_remote_descs.descCount()):
                        remote_descs.append(tensor_remote_descs[index])

            assert local_descs is not None and remote_descs is not None
            handle = cache.agent.initialize_xfer(
                operation,
                local_descs,
                remote_descs,
                remote_agent,
                backends=[cache.backend],
            )
            status = cache.agent.transfer(handle)

            deadline = time.monotonic() + float(
                os.environ.get("TORCHSTORE_NIXL_TIMEOUT_S", "60")
            )
            while status == "PROC" and time.monotonic() < deadline:
                await asyncio.sleep(0)
                status = cache.agent.check_xfer_state(handle)
            if status == "PROC":
                raise TimeoutError("NIXL transfer timed out")
            if status != "DONE":
                raise RuntimeError(f"NIXL transfer failed with status {status!r}")
        except Exception as error:
            raise RuntimeError(
                f"NIXL {operation} failed for keys="
                f"{[request.key for request, _, _ in transfers]!r}"
            ) from error
        finally:
            if handle is not None:
                cache.agent.release_xfer_handle(handle)

    def _connect_client(self, cache: NixlAgentCache) -> str:
        if self._client_agent_name is None or self._client_metadata is None:
            raise RuntimeError("NIXL request is missing client agent metadata")
        return cache.add_remote_agent(
            self._client_agent_name, self._client_metadata
        )

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        """Called by storage volume. Read from client's source RdmaMemory (put)."""
        results: list[Any] = []
        transfers: list[tuple[Request, torch.Tensor, bytes]] = []

        for (request, tensor), request_context in zip(
            entries, self._contexts, strict=True
        ):
            if request_context.is_object:
                results.append(request_context.objects)
                continue

            if tensor is None:
                tensor = torch.empty(
                    request_context.shape,
                    dtype=request_context.dtype,
                    device="cpu",
                )
            self._assert_valid_tensor(
                tensor, request_context.dtype, request_context.shape
            )
            assert request_context.remote_descriptors is not None
            transfers.append(
                (request, tensor, request_context.remote_descriptors)
            )
            results.append(tensor)
        await self._transfer(ctx, "READ", transfers)
        return results

    async def handle_get_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> None:
        transfers: list[tuple[Request, torch.Tensor, bytes]] = []

        for (request, data), request_context in zip(
            entries, self._contexts, strict=True
        ):
            if not isinstance(data, torch.Tensor):
                request_context.is_object = True
                request_context.objects = data
                continue

            self._assert_valid_tensor(
                data,
                request_context.dtype,
                request_context.shape,
                must_be_contiguous=False,
            )
            source = data if data.is_contiguous() else data.contiguous()
            assert request_context.remote_descriptors is not None
            transfers.append(
                (request, source, request_context.remote_descriptors)
            )
        await self._transfer(ctx, "WRITE", transfers)

    async def _handle_storage_volume_response(
        self,
        requests: list[Request],
        transport_buffer: TransportBuffer,
    ) -> list[Any]:
        assert isinstance(transport_buffer, NixlTransportBuffer)
        results: list[Any] = []
        for client_context, volume_context in zip(
            self._contexts, transport_buffer._contexts, strict=True
        ):
            results.append(
                volume_context.objects
                if volume_context.is_object
                else client_context.tensor
            )
        return results

    async def drop(self) -> None:
        self._client_metadata = None
        self._contexts = []
