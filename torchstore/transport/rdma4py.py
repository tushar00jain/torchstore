# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RDMA transport backed by the ``ibverbs`` package from rdma4py.

The Monarch actor RPC is used only as a control plane.  Client and storage
volume cache connected multi-QP endpoints and weakly-owned memory registrations
across request batches.  The storage volume performs one-sided reads for PUTs
and one-sided writes for GETs through rdma4py's continuously-refilled scheduler.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
import weakref
from dataclasses import dataclass
from functools import cache
from typing import Any, TYPE_CHECKING

import torch

from torchstore.direct_transport import DirectTransport
from torchstore.transport.buffers import TransportBuffer, TransportCache
from torchstore.transport.types import Request

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.buffers import TransportContext

try:
    import ibverbs as ib
except (ImportError, OSError):
    ib = None


logger = logging.getLogger(__name__)

_MAX_SGE_BYTES = (1 << 32) - 1
_DEFAULT_QUEUE_DEPTH = 128
_DEFAULT_QP_COUNT = 1
_DEFAULT_READ_CHUNK_SIZE = _MAX_SGE_BYTES
_DEFAULT_TIMEOUT_SECONDS = 60.0


def _open_configured_factory() -> Any:
    if ib is None:
        raise RuntimeError(
            "rdma4py transport requires the 'ibverbs' package from rdma4py"
        )
    configured_port = os.environ.get("TORCHSTORE_RDMA4PY_PORT")
    configured_gid = os.environ.get("TORCHSTORE_RDMA4PY_GID_INDEX")
    return ib.RdmaTransportFactory.open(
        device_name=os.environ.get("TORCHSTORE_RDMA4PY_DEVICE"),
        port=int(configured_port) if configured_port is not None else None,
        gid_index=int(configured_gid) if configured_gid is not None else None,
    )


def rdma4py_enabled() -> bool:
    return os.environ.get("TORCHSTORE_RDMA4PY_ENABLED", "1") == "1"


@cache
def rdma4py_transport_available() -> bool:
    """Return whether rdma4py and an active RC-capable device are available."""
    if not rdma4py_enabled() or ib is None:
        return False
    factory = None
    try:
        factory = _open_configured_factory()
        return True
    except Exception as error:
        logger.info("rdma4py transport is not available: %s", error)
        return False
    finally:
        if factory is not None:
            factory.close()


def _queue_depth() -> int:
    """Return the maximum outstanding work requests per queue pair."""
    value = int(os.environ.get("TORCHSTORE_RDMA4PY_QUEUE_DEPTH", _DEFAULT_QUEUE_DEPTH))
    if value <= 0:
        raise ValueError("TORCHSTORE_RDMA4PY_QUEUE_DEPTH must be positive")
    return value


def _qp_count() -> int:
    value = int(os.environ.get("TORCHSTORE_RDMA4PY_QP_COUNT", _DEFAULT_QP_COUNT))
    if value <= 0:
        raise ValueError("TORCHSTORE_RDMA4PY_QP_COUNT must be positive")
    return value


def _read_chunk_size() -> int:
    """Return scheduler chunk bytes, bounded by the verbs SGE length field."""
    value = int(
        os.environ.get(
            "TORCHSTORE_RDMA4PY_CHUNK_SIZE_BYTES", _DEFAULT_READ_CHUNK_SIZE
        )
    )
    if value <= 0 or value > _MAX_SGE_BYTES:
        raise ValueError(
            "TORCHSTORE_RDMA4PY_CHUNK_SIZE_BYTES must be between 1 and 2**32 - 1"
        )
    return value


def _timeout_seconds() -> float:
    value = float(
        os.environ.get("TORCHSTORE_RDMA4PY_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS)
    )
    if value <= 0:
        raise ValueError("TORCHSTORE_RDMA4PY_TIMEOUT_SECONDS must be positive")
    return value


def _cuda_module() -> Any:
    import ibverbs.cuda as ibcuda

    return ibcuda


def _synchronize_cuda_source(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.current_stream(tensor.device).synchronize()


def _synchronize_cuda_sources(tensors: list[torch.Tensor]) -> None:
    by_device = {tensor.device: tensor for tensor in tensors if tensor.is_cuda}
    for tensor in by_device.values():
        _synchronize_cuda_source(tensor)


def _flush_cuda_destination(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        with torch.cuda.device(tensor.device):
            _cuda_module().flush_gpudirect_writes()


def _flush_cuda_destinations(tensors: list[torch.Tensor]) -> None:
    by_device = {tensor.device: tensor for tensor in tensors if tensor.is_cuda}
    for tensor in by_device.values():
        _flush_cuda_destination(tensor)


def _register_tensor(factory: Any, tensor: torch.Tensor, remote_access: int) -> Any:
    if tensor.is_cuda:
        # rdma4py's dma-buf export uses the current CUDA driver context.
        with torch.cuda.device(tensor.device):
            return factory.register_tensor(tensor, remote_access)
    return factory.register_tensor(tensor, remote_access)


def _release_registration_owner(registration: Any) -> None:
    """Let the cache's storage weakref, rather than the MR, own lifetime.

    rdma4py registrations normally retain the tensor to make one-off use safe.
    A persistent cache must release that strong reference after installing its
    storage weakref; otherwise the registration keeps the storage alive and the
    eviction callback can never run.
    """
    ib.release_tensor_owner(registration)


@dataclass(frozen=True)
class _FallbackRemoteMemory:
    """Serializable address/key pair for one registered client tensor."""

    address: int
    rkey: int
    nbytes: int

    @classmethod
    def from_registration(cls, registration: Any, nbytes: int):
        return cls(int(registration.addr), int(registration.rkey), int(nbytes))


RemoteMemory = (
    ib.RemoteMemory
    if ib is not None and hasattr(ib, "RemoteMemory")
    else _FallbackRemoteMemory
)


@dataclass
class Rdma4PyRequestContext:
    remote_memory: RemoteMemory | None = None
    shape: torch.Size | None = None
    dtype: torch.dtype | None = None
    is_object: bool = False
    objects: Any = None

    # Client-local state; stripped when the transport buffer crosses RPC.
    tensor: torch.Tensor | None = None
    result_tensor: torch.Tensor | None = None
    memory_region: Any = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["tensor"] = None
        state["result_tensor"] = None
        state["memory_region"] = None
        return state


class _Rdma4PyEndpoint:
    def __init__(self, queue_depth: int, qp_count: int, chunk_size: int):
        self.factory = None
        self.transport = None
        self.context = None
        self.pd = None
        self.scheduler = None
        self.queue_depth = queue_depth
        self.qp_count = qp_count
        self.chunk_size = chunk_size
        self.local_infos: tuple[bytes, ...] = ()

        try:
            self.factory = _open_configured_factory()
            self.context = self.factory.context
            self.pd = self.factory.pd
            self.transport = self.factory.create_transport(
                qp_count=qp_count,
                queue_depth=queue_depth,
                chunk_size=chunk_size,
            )
            self.scheduler = self.transport.scheduler
            connection_info = self.transport.bind()
            self.local_infos = (
                (connection_info,)
                if isinstance(connection_info, bytes)
                else tuple(connection_info)
            )
        except Exception:
            self.close()
            raise

    @property
    def connection_info(self) -> bytes | tuple[bytes, ...]:
        """Return the legacy single-QP or multi-QP wire representation."""
        if len(self.local_infos) == 1:
            return self.local_infos[0]
        return self.local_infos

    def connect(
        self, remote_infos: bytes | tuple[bytes, ...], incoming_access: int
    ) -> None:
        if self.transport is None:
            raise RuntimeError("rdma4py endpoint is closed")
        self.transport.connect(remote_infos, incoming_access=incoming_access)

    @property
    def usable(self) -> bool:
        return bool(
            self.transport is not None
            and self.transport.usable
        )

    async def read_many(
        self,
        requests: list[Any],
        *,
        timeout: float | None = None,
        on_complete: Any = None,
    ) -> None:
        if self.transport is None:
            raise RuntimeError("rdma4py endpoint is closed")
        await self.transport.read_many_async(
            requests,
            timeout=_timeout_seconds() if timeout is None else timeout,
            on_complete=on_complete,
        )

    async def write_many(self, requests: list[Any]) -> None:
        if self.transport is None:
            raise RuntimeError("rdma4py endpoint is closed")
        await self.transport.write_many_async(
            requests,
            timeout=_timeout_seconds(),
        )

    def close(self) -> None:
        self.close_scheduler()
        factory = self.factory
        if factory is not None:
            try:
                factory.close()
            except Exception:
                logger.warning("Failed to close rdma4py factory", exc_info=True)
        self.factory = None
        self.context = None
        self.pd = None

    def close_scheduler(self) -> None:
        transport = self.transport
        if transport is None:
            return
        try:
            transport.close()
        except Exception:
            logger.warning("Failed to close rdma4py transport", exc_info=True)
        self.transport = None
        self.scheduler = None


def _create_endpoint(
    queue_depth: int | None = None,
    qp_count: int | None = None,
    chunk_size: int | None = None,
) -> _Rdma4PyEndpoint:
    return _Rdma4PyEndpoint(
        _queue_depth() if queue_depth is None else queue_depth,
        _qp_count() if qp_count is None else qp_count,
        _read_chunk_size() if chunk_size is None else chunk_size,
    )


class Rdma4PyConnectionCache(TransportCache):
    """Persistent endpoints and weakref-safe registrations for one process."""

    def __init__(self) -> None:
        self._connections: dict[str, _Rdma4PyEndpoint] = {}
        self._pending_connections: dict[str, _Rdma4PyEndpoint] = {}
        self._peer_keys: dict[str, str] = {}
        self._registrations: dict[tuple[int, int, int, int], Any] = {}
        self._storage_refs: dict[tuple[int, int, int, int], weakref.ref[Any]] = {}

    def peer_key(self, volume_id: str) -> str:
        if volume_id not in self._peer_keys:
            self._peer_keys[volume_id] = uuid.uuid4().hex
        return self._peer_keys[volume_id]

    def put(self, key: str, endpoint: _Rdma4PyEndpoint) -> None:
        if self._connections.get(key) is endpoint:
            return
        if key in self._connections:
            self.discard(key)
        self._connections[key] = endpoint

    def put_if_absent(
        self, key: str, endpoint: _Rdma4PyEndpoint
    ) -> _Rdma4PyEndpoint:
        if self.contains(key):
            existing = self.get(key)
            if existing is not endpoint:
                self._close_endpoint(endpoint)
            return existing
        self._connections[key] = endpoint
        return endpoint

    def stage(self, handshake_id: str, endpoint: _Rdma4PyEndpoint) -> None:
        old = self._pending_connections.pop(handshake_id, None)
        if old is not None and old is not endpoint:
            self._close_endpoint(old)
        self._pending_connections[handshake_id] = endpoint

    def pending(self, handshake_id: str) -> _Rdma4PyEndpoint | None:
        endpoint = self._pending_connections.get(handshake_id)
        if endpoint is not None and not endpoint.usable:
            self.discard_pending(handshake_id)
            return None
        return endpoint

    def promote(self, handshake_id: str, peer_key: str) -> None:
        endpoint = self._pending_connections.pop(handshake_id, None)
        if endpoint is None:
            return
        self.put_if_absent(peer_key, endpoint)

    def get(self, key: str) -> _Rdma4PyEndpoint:
        try:
            endpoint = self._connections[key]
        except KeyError as error:
            raise RuntimeError(
                f"Missing rdma4py connection for request {key}"
            ) from error
        if not endpoint.usable:
            self.discard(key)
            raise RuntimeError(f"rdma4py connection {key} is no longer usable")
        return endpoint

    def contains(self, key: str) -> bool:
        endpoint = self._connections.get(key)
        if endpoint is None:
            return False
        if endpoint.usable:
            return True
        self.discard(key)
        return False

    def get_or_register(
        self, endpoint: _Rdma4PyEndpoint, tensor: torch.Tensor, access: int
    ) -> Any:
        if endpoint.pd is None:
            raise RuntimeError("rdma4py endpoint is closed")
        nbytes = tensor.numel() * tensor.element_size()
        key = (id(endpoint.pd), tensor.data_ptr(), nbytes, int(access))
        registration = self._registrations.get(key)
        if registration is not None:
            return registration
        if endpoint.factory is None:
            raise RuntimeError("rdma4py endpoint is closed")
        registration = _register_tensor(endpoint.factory, tensor, access)
        self._registrations[key] = registration
        self._storage_refs[key] = weakref.ref(
            tensor.untyped_storage(), lambda _ref, _key=key: self._evict(_key)
        )
        _release_registration_owner(registration)
        return registration

    def _evict(self, key: tuple[int, int, int, int]) -> None:
        registration = self._registrations.pop(key, None)
        self._storage_refs.pop(key, None)
        if registration is not None:
            registration.close()

    def _evict_pd(self, pd: Any) -> None:
        pd_id = id(pd)
        for key in [key for key in self._registrations if key[0] == pd_id]:
            self._evict(key)

    def _close_endpoint(self, endpoint: _Rdma4PyEndpoint) -> None:
        endpoint.close_scheduler()
        self._evict_pd(endpoint.pd)
        endpoint.close()

    def discard(self, key: str) -> None:
        endpoint = self._connections.pop(key, None)
        if endpoint is not None:
            self._close_endpoint(endpoint)

    def discard_pending(self, handshake_id: str) -> bool:
        endpoint = self._pending_connections.pop(handshake_id, None)
        if endpoint is None:
            return False
        self._close_endpoint(endpoint)
        return True

    def clear(self) -> None:
        for key in list(self._connections):
            self.discard(key)
        for handshake_id in list(self._pending_connections):
            self.discard_pending(handshake_id)
        for key in list(self._registrations):
            self._evict(key)
        self._peer_keys.clear()


class _Rdma4PyDirectPeer:
    """Per-peer state owned by :class:`Rdma4PyDirectTransport`."""

    def __init__(
        self,
        remote_info: bytes | tuple[bytes, ...] | None = None,
        incoming_access: int = 0,
    ) -> None:
        if ib is None:
            raise RuntimeError(
                "Direct rdma4py transport requires the 'ibverbs' package"
            )
        self._cache = Rdma4PyConnectionCache()
        self._endpoint = _create_endpoint()
        self._cache.put("direct", self._endpoint)
        self._pending_reads: list[_PendingDirectRead] = []
        self._drain_task: asyncio.Task[None] | None = None
        if remote_info is None:
            return
        try:
            self._endpoint.connect(remote_info, incoming_access=incoming_access)
        except BaseException:
            self.close()
            raise

    @property
    def connection_info(self) -> bytes | tuple[bytes, ...]:
        return self._endpoint.connection_info

    def register(self, tensor: torch.Tensor) -> RemoteMemory:
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes == 0:
            return RemoteMemory(0, 0, 0)
        registration = self._cache.get_or_register(
            self._endpoint,
            tensor,
            ib.RdmaAccess.READ,
        )
        return RemoteMemory.from_registration(registration, nbytes)

    def connect(self, remote_info: bytes | tuple[bytes, ...]) -> None:
        try:
            self._endpoint.connect(remote_info, incoming_access=0)
        except BaseException:
            self.close()
            raise

    async def read_into(
        self, remote_buffer: RemoteMemory, tensor: torch.Tensor
    ) -> None:
        """Coalesce reads submitted in one event-loop turn into one batch."""
        future = asyncio.get_running_loop().create_future()
        self._pending_reads.append(_PendingDirectRead(remote_buffer, tensor, future))
        if self._drain_task is None:
            self._drain_task = asyncio.create_task(self._drain_pending_reads())
        await future

    async def _drain_pending_reads(self) -> None:
        await asyncio.sleep(0)
        try:
            while self._pending_reads:
                pending, self._pending_reads = self._pending_reads, []
                try:
                    await self._read_many(pending)
                except BaseException as error:
                    for read in pending:
                        if not read.future.done():
                            read.future.set_exception(error)
                else:
                    for read in pending:
                        if not read.future.done():
                            read.future.set_result(None)
        finally:
            self._drain_task = None

    async def _read_many(self, reads: list[_PendingDirectRead]) -> None:
        requests = []
        cuda_destinations: list[torch.Tensor] = []
        for read in reads:
            nbytes = read.tensor.numel() * read.tensor.element_size()
            if read.remote_buffer.nbytes != nbytes:
                raise ValueError(
                    f"Direct rdma4py size mismatch: "
                    f"{read.remote_buffer.nbytes} != {nbytes}"
                )
            if nbytes == 0:
                continue
            registration = self._cache.get_or_register(
                self._endpoint,
                read.tensor,
                ib.RdmaAccess.NONE,
            )
            requests.append(
                ib.RdmaTransferRequest.read(
                    registration,
                    read.remote_buffer,
                )
            )
            if read.tensor.is_cuda:
                cuda_destinations.append(read.tensor)

        await self._endpoint.read_many(
            requests,
            on_complete=lambda: _flush_cuda_destinations(cuda_destinations),
        )

    def close(self) -> None:
        if self._drain_task is not None and not self._drain_task.done():
            self._drain_task.cancel()
        error = RuntimeError("Direct rdma4py connection closed")
        for read in self._pending_reads:
            if not read.future.done():
                read.future.set_exception(error)
        self._pending_reads.clear()
        self._cache.clear()


@dataclass
class _PendingDirectRead:
    remote_buffer: RemoteMemory
    tensor: torch.Tensor
    future: asyncio.Future[None]


class Rdma4PyDirectTransport(DirectTransport):
    """Direct rdma4py transport for source and destination roles."""

    def __init__(self, store: Any, key: str) -> None:
        from torchstore.transport import TransportType

        super().__init__(store, key, TransportType.Rdma4Py)

    def create_source_connection(self, connection_info, _device):
        return _Rdma4PyDirectPeer(
            connection_info,
            incoming_access=ib.RdmaAccess.READ,
        )

    def create_destination_connection(self, _device):
        return _Rdma4PyDirectPeer()


class Rdma4PyTransportBuffer(TransportBuffer):
    """TorchStore transport using rdma4py's low-level ibverbs bindings."""

    supports_batch_puts = True
    supports_batch_gets = True

    def __init__(self, storage_volume_ref: "StorageVolumeRef"):
        super().__init__(storage_volume_ref)
        if ib is None:
            raise RuntimeError(
                "Rdma4Py transport requires the 'ibverbs' package from rdma4py"
            )
        self.queue_depth = _queue_depth()
        self.qp_count = _qp_count()
        self.chunk_size = _read_chunk_size()
        cache = storage_volume_ref.transport_context.get(Rdma4PyConnectionCache)
        self.peer_key = cache.peer_key(storage_volume_ref.volume_id)
        self.handshake_id = uuid.uuid4().hex
        self.client_qp_infos: tuple[bytes, ...] = ()
        self._handshake_phase = "connect"
        self._server_connection_pending = False
        self._request_succeeded = False
        self._endpoint_published = False
        self._using_cached_endpoint = False
        self._endpoint: _Rdma4PyEndpoint | None = None
        self._contexts: list[Rdma4PyRequestContext] = []

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["storage_volume_ref"] = None
        state["_endpoint"] = None
        return state

    def requires_handshake(self, requests: list[Request]) -> bool:
        self._request_succeeded = False
        cache = self.storage_volume_ref.transport_context.get(
            Rdma4PyConnectionCache
        )
        if cache.contains(self.peer_key):
            self._endpoint = cache.get(self.peer_key)
            self._endpoint_published = True
            self._using_cached_endpoint = True
            return False
        self._endpoint = _create_endpoint(
            self.queue_depth, self.qp_count, self.chunk_size
        )
        self.client_qp_infos = self._endpoint.local_infos
        self._endpoint_published = False
        self._using_cached_endpoint = False
        return True

    async def _pre_handshake(self) -> None:
        if self._endpoint is None:
            raise RuntimeError("rdma4py client endpoint was not initialized")

    async def _post_handshake(
        self, handshake_results: list[Any], requests: list[Request]
    ) -> None:
        if self._endpoint is None:
            raise RuntimeError("rdma4py client endpoint was not initialized")
        if len(handshake_results) != 1:
            raise RuntimeError("rdma4py handshake expected one server endpoint set")
        self._server_connection_pending = True
        incoming_access = ib.RdmaAccess.READ | ib.RdmaAccess.WRITE
        self._endpoint.connect(tuple(handshake_results[0]), incoming_access)

    async def recv_handshake(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        connection_cache = ctx.get(Rdma4PyConnectionCache)
        if self._handshake_phase == "abort":
            if not connection_cache.discard_pending(self.handshake_id):
                connection_cache.discard(self.peer_key)
            return []
        if self._handshake_phase != "connect":
            raise RuntimeError(
                f"Unknown rdma4py handshake phase: {self._handshake_phase}"
            )
        if not self.client_qp_infos:
            raise RuntimeError("rdma4py handshake is missing client QP information")
        endpoint = _create_endpoint(self.queue_depth, self.qp_count, self.chunk_size)
        try:
            endpoint.connect(self.client_qp_infos, incoming_access=0)
            connection_cache.stage(self.handshake_id, endpoint)
        except Exception:
            endpoint.close()
            raise
        return [endpoint.local_infos]

    def _register_client_tensor(
        self, tensor: torch.Tensor, remote_access: int
    ) -> Rdma4PyRequestContext:
        if self._endpoint is None:
            raise RuntimeError("rdma4py endpoint is not connected")
        self._assert_valid_tensor(tensor, tensor.dtype, tensor.shape)
        nbytes = tensor.numel() * tensor.element_size()
        memory_region = None
        remote_memory = RemoteMemory(0, 0, 0)
        if nbytes:
            memory_region = self.storage_volume_ref.transport_context.get(
                Rdma4PyConnectionCache
            ).get_or_register(self._endpoint, tensor, remote_access)
            remote_memory = RemoteMemory.from_registration(memory_region, nbytes)
        return Rdma4PyRequestContext(
            remote_memory=remote_memory,
            tensor=tensor,
            result_tensor=tensor,
            memory_region=memory_region,
            shape=tensor.shape,
            dtype=tensor.dtype,
        )

    async def _pre_put_hook(self, requests: list[Request]) -> None:
        self._contexts = []
        for request in requests:
            if request.is_object:
                self._contexts.append(
                    Rdma4PyRequestContext(is_object=True, objects=request.objects)
                )
                continue
            tensor = request.tensor_val
            if tensor is None:
                raise ValueError(f"PUT request {request.key!r} has no tensor")
            if not tensor.is_contiguous():
                logger.warning(
                    "PUT called with non-contiguous tensor (key=%s); staging a "
                    "contiguous CPU copy",
                    request.key,
                )
                tensor = tensor.cpu().contiguous()
            self._contexts.append(
                self._register_client_tensor(tensor, ib.RdmaAccess.READ)
            )
        _synchronize_cuda_sources(
            [context.tensor for context in self._contexts if context.tensor is not None]
        )

    async def _pre_get_hook(self, requests: list[Request]) -> None:
        meta_requests = [
            request.meta_only() for request in requests if request.tensor_val is None
        ]
        meta_results = (
            await self.storage_volume_ref.volume.get_meta.call_one(meta_requests)
            if meta_requests
            else []
        )
        meta_iterator = iter(meta_results)

        self._contexts = []
        for request in requests:
            result_tensor = request.tensor_val
            if result_tensor is None:
                meta = next(meta_iterator)
                if isinstance(meta, str) or meta is None:
                    self._contexts.append(Rdma4PyRequestContext(is_object=True))
                    continue
                shape, dtype = meta
                if request.tensor_slice is not None:
                    shape = request.tensor_slice.local_shape
                tensor = torch.empty(shape, dtype=dtype, device="cpu")
                result_tensor = tensor
            elif result_tensor.is_contiguous():
                tensor = result_tensor
            else:
                tensor = torch.empty(
                    result_tensor.shape,
                    dtype=result_tensor.dtype,
                    device=result_tensor.device,
                )

            context = self._register_client_tensor(tensor, ib.RdmaAccess.WRITE)
            context.result_tensor = result_tensor
            self._contexts.append(context)

    def _register_server_tensor(
        self,
        cache: Rdma4PyConnectionCache,
        endpoint: _Rdma4PyEndpoint,
        tensor: torch.Tensor,
    ):
        nbytes = tensor.numel() * tensor.element_size()
        if not nbytes:
            return None
        return cache.get_or_register(endpoint, tensor, ib.RdmaAccess.NONE)

    def _server_endpoint(
        self, ctx: "TransportContext"
    ) -> tuple[_Rdma4PyEndpoint, bool]:
        cache = ctx.get(Rdma4PyConnectionCache)
        pending = cache.pending(self.handshake_id)
        if pending is not None:
            return pending, True
        return cache.get(self.peer_key), False

    async def handle_put_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        cache = ctx.get(Rdma4PyConnectionCache)
        endpoint, pending = self._server_endpoint(ctx)
        destinations_to_flush: list[torch.Tensor] = []
        rdma_requests: list[Any] = []
        results: list[Any] = []
        try:
            for (_, current_object), request_ctx in zip(
                entries, self._contexts, strict=True
            ):
                if request_ctx.is_object:
                    results.append(request_ctx.objects)
                    continue
                if request_ctx.remote_memory is None:
                    raise RuntimeError("rdma4py PUT is missing remote memory")

                tensor = current_object
                if tensor is None or not tensor.is_contiguous():
                    tensor = torch.empty(
                        request_ctx.shape, dtype=request_ctx.dtype, device="cpu"
                    )
                self._assert_valid_tensor(tensor, request_ctx.dtype, request_ctx.shape)
                registration = self._register_server_tensor(cache, endpoint, tensor)
                if registration is not None:
                    rdma_requests.append(
                        ib.RdmaTransferRequest.read(
                            registration,
                            request_ctx.remote_memory,
                        )
                    )
                    if tensor.is_cuda:
                        destinations_to_flush.append(tensor)
                results.append(tensor)

            await endpoint.read_many(rdma_requests)
            _flush_cuda_destinations(destinations_to_flush)
            if pending:
                cache.promote(self.handshake_id, self.peer_key)
            return results
        except BaseException:
            if pending:
                cache.discard_pending(self.handshake_id)
            else:
                cache.discard(self.peer_key)
            raise

    async def handle_get_request(
        self,
        ctx: "TransportContext",
        entries: list[tuple[Request, Any]],
    ) -> None:
        cache = ctx.get(Rdma4PyConnectionCache)
        endpoint, pending = self._server_endpoint(ctx)
        rdma_requests: list[Any] = []
        source_tensors: list[torch.Tensor] = []
        try:
            for (_, data), request_ctx in zip(entries, self._contexts, strict=True):
                if not isinstance(data, torch.Tensor):
                    request_ctx.is_object = True
                    request_ctx.objects = data
                    continue
                if request_ctx.remote_memory is None:
                    raise RuntimeError("rdma4py GET is missing remote memory")

                tensor = data
                if not tensor.is_contiguous():
                    tensor = tensor.cpu().contiguous()
                self._assert_valid_tensor(tensor, request_ctx.dtype, request_ctx.shape)
                source_tensors.append(tensor)
                registration = self._register_server_tensor(cache, endpoint, tensor)
                if registration is None:
                    continue
                rdma_requests.append(
                    ib.RdmaTransferRequest.write(
                        registration,
                        request_ctx.remote_memory,
                    )
                )

            _synchronize_cuda_sources(source_tensors)
            await endpoint.write_many(rdma_requests)
            if pending:
                cache.promote(self.handshake_id, self.peer_key)
        except BaseException:
            if pending:
                cache.discard_pending(self.handshake_id)
            else:
                cache.discard(self.peer_key)
            raise

    async def _handle_storage_volume_response(
        self, requests: list[Request], transport_buffer: "TransportBuffer"
    ) -> list[Any]:
        results: list[Any] = []
        destinations_to_flush: list[torch.Tensor] = []
        tensor_results: list[Rdma4PyRequestContext] = []
        for client_ctx, server_ctx in zip(
            self._contexts, transport_buffer._contexts, strict=True
        ):
            if server_ctx.is_object:
                results.append(server_ctx.objects)
                continue
            if client_ctx.tensor is None or client_ctx.result_tensor is None:
                raise RuntimeError("rdma4py GET lost its local destination tensor")
            destinations_to_flush.append(client_ctx.tensor)
            tensor_results.append(client_ctx)
            results.append(client_ctx.result_tensor)
        _flush_cuda_destinations(destinations_to_flush)
        for client_ctx in tensor_results:
            if client_ctx.result_tensor is not client_ctx.tensor:
                client_ctx.result_tensor.copy_(client_ctx.tensor)
        return results

    async def _post_request_success(self) -> None:
        if self._endpoint is not None and not self._endpoint_published:
            self._endpoint = self.storage_volume_ref.transport_context.get(
                Rdma4PyConnectionCache
            ).put_if_absent(self.peer_key, self._endpoint)
            self._endpoint_published = True
        self._request_succeeded = True
        self._server_connection_pending = False

    async def drop(self) -> None:
        for request_ctx in self._contexts:
            request_ctx.memory_region = None
            request_ctx.tensor = None
            request_ctx.result_tensor = None
        self._contexts = []
        if not self._request_succeeded and self.storage_volume_ref is not None:
            try:
                self._handshake_phase = "abort"
                await self.storage_volume_ref.volume.handshake.call_one(self, [])
            except Exception:
                logger.warning(
                    "Failed to abort pending rdma4py connection", exc_info=True
                )
            finally:
                self._server_connection_pending = False
            cache = self.storage_volume_ref.transport_context.get(
                Rdma4PyConnectionCache
            )
            if self._using_cached_endpoint:
                cache.discard(self.peer_key)
            elif self._endpoint is not None:
                cache._close_endpoint(self._endpoint)
            self._endpoint = None
            self._endpoint_published = False
        elif self._endpoint is not None and not self._endpoint_published:
            self.storage_volume_ref.transport_context.get(
                Rdma4PyConnectionCache
            )._close_endpoint(self._endpoint)
            self._endpoint = None
