# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Common interface and shared implementation for direct weight transports."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch

from torchstore.transport.types import Request, TensorSlice

if TYPE_CHECKING:
    from torchstore.transport import TransportType


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DirectConnectionRequest:
    """Destination connection information published through TorchStore."""

    backend: TransportType
    client_id: str
    connection_info: bytes | tuple[bytes, ...]


@dataclass(frozen=True)
class DirectTensorHandle:
    """Remote tensor registration and source sharding metadata."""

    remote_buffer: Any
    tensor_slice: TensorSlice
    source_rank: int


@dataclass(frozen=True)
class DirectConnectionResponse:
    """Source connection information and registered tensor handles."""

    backend: TransportType
    connection_info: bytes | tuple[bytes, ...]
    handles: dict[str, DirectTensorHandle]
    error: str | None = None


def _timeout_seconds() -> float:
    value = float(os.environ.get("TORCHSTORE_DIRECT_RDMA_TIMEOUT_SECONDS", "60"))
    if value <= 0:
        raise ValueError("TORCHSTORE_DIRECT_RDMA_TIMEOUT_SECONDS must be positive")
    return value


def _poll_interval_seconds() -> float:
    value = float(os.environ.get("TORCHSTORE_DIRECT_RDMA_POLL_SECONDS", "0.01"))
    if value <= 0:
        raise ValueError("TORCHSTORE_DIRECT_RDMA_POLL_SECONDS must be positive")
    return value


def _require_cuda_tensor(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError(
            f"Direct rdma4py/TorchComms weight sync requires CUDA tensors; "
            f"{name!r} is on {tensor.device}"
        )
    if not tensor.is_contiguous():
        raise ValueError(
            f"Direct RDMA transfer buffer {name!r} must be contiguous, got "
            f"strides={tensor.stride()}"
        )


def _cuda_synchronize(tensors: dict[str, torch.Tensor]) -> None:
    devices = {tensor.device for tensor in tensors.values() if tensor.is_cuda}
    for device in devices:
        torch.cuda.current_stream(device).synchronize()


class _ReadableDirectBuffer:
    def __init__(self, connection: Any, remote_buffer: Any) -> None:
        self._connection = connection
        self._remote_buffer = remote_buffer

    async def read_into(self, destination: torch.Tensor) -> None:
        await self._connection.read_into(self._remote_buffer, destination)

    async def drop(self) -> None:
        return None


class DirectTransportWeightSyncSource:
    """Serve direct GPU registrations for a connection-oriented transport."""

    def __init__(
        self,
        backend: TransportType,
        store: Any,
        key: str,
        connection_factory,
    ) -> None:
        self.backend = backend
        self.store = store
        self.key = key
        self._connection_factory = connection_factory
        self.rank = -1
        self._tensors: dict[str, torch.Tensor] = {}
        self._tensor_slices: dict[str, TensorSlice] = {}
        self._staging: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._connections: dict[str, Any] = {}
        self._processed_requests: set[str] = set()
        self._server_task: asyncio.Task | None = None

    async def register(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        rank: int,
        transfer_dtype: torch.dtype | None,
    ) -> None:
        from torchstore.direct_weight_sync import _prepare_source_state_dict

        self.rank = rank
        self._tensors, self._tensor_slices, self._staging = (
            _prepare_source_state_dict(
                state_dict,
                transfer_dtype,
                stage_noncontiguous=True,
            )
        )
        for name, tensor in self._tensors.items():
            _require_cuda_tensor(name, tensor)

        self.refresh()
        self._server_task = asyncio.create_task(self._serve_requests())
        logger.info(
            "[ts-direct-rdma] backend=%s role=source rank=%d tensors=%d",
            self.backend.name,
            self.rank,
            len(self._tensors),
        )

    def refresh(self) -> None:
        for staging, source in self._staging.values():
            staging.copy_(source)
        _cuda_synchronize(self._tensors)
        if self._server_task is not None and self._server_task.done():
            error = self._server_task.exception()
            if error is not None:
                raise RuntimeError("Direct RDMA source service failed") from error

    @property
    def _request_prefix(self) -> str:
        return f"{self.key}/direct/{self.backend.name}/rank_{self.rank}/requests/"

    def _response_key(self, client_id: str) -> str:
        return (
            f"{self.key}/direct/{self.backend.name}/rank_{self.rank}/"
            f"responses/{client_id}"
        )

    async def _serve_requests(self) -> None:
        device = next(iter(self._tensors.values())).device
        while True:
            for request_key in await self.store.keys(self._request_prefix):
                if request_key in self._processed_requests:
                    continue
                request = await self.store.get(request_key)
                if not isinstance(request, DirectConnectionRequest):
                    raise TypeError(
                        f"Invalid direct RDMA connection request: {type(request)}"
                    )
                if request.backend != self.backend:
                    raise ValueError(
                        f"Direct RDMA backend mismatch: {request.backend} != "
                        f"{self.backend}"
                    )
                connection = None
                try:
                    connection = self._connection_factory(
                        request.connection_info, device
                    )
                    handles = {
                        name: DirectTensorHandle(
                            remote_buffer=connection.register(tensor),
                            tensor_slice=self._tensor_slices[name],
                            source_rank=self.rank,
                        )
                        for name, tensor in self._tensors.items()
                    }
                    response = DirectConnectionResponse(
                        backend=self.backend,
                        connection_info=connection.connection_info,
                        handles=handles,
                    )
                    await self.store.put(
                        self._response_key(request.client_id), response
                    )
                except Exception as error:
                    if connection is not None:
                        connection.close()
                    await self.store.put(
                        self._response_key(request.client_id),
                        DirectConnectionResponse(
                            backend=self.backend,
                            connection_info=b"",
                            handles={},
                            error=f"{type(error).__name__}: {error}",
                        ),
                    )
                else:
                    assert connection is not None
                    self._connections[request.client_id] = connection
                await self.store.delete(request_key)
                self._processed_requests.add(request_key)
            await asyncio.sleep(_poll_interval_seconds())

    async def close(self) -> None:
        if self._server_task is not None:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
            self._server_task = None
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()


class DirectTransportWeightSyncDest:
    """Pull weights directly from source GPUs through connected peers."""

    def __init__(
        self,
        backend: TransportType,
        store: Any,
        key: str,
        connection_factory,
    ) -> None:
        self.backend = backend
        self.store = store
        self.key = key
        self._connection_factory = connection_factory
        self.client_id = f"{os.getpid()}-{uuid.uuid4().hex}"
        self._connections: dict[int, Any] = {}
        self._handles = None

    def _request_key(self, rank: int) -> str:
        return (
            f"{self.key}/direct/{self.backend.name}/rank_{rank}/"
            f"requests/{self.client_id}"
        )

    def _response_key(self, rank: int) -> str:
        return (
            f"{self.key}/direct/{self.backend.name}/rank_{rank}/"
            f"responses/{self.client_id}"
        )

    async def _wait_for_response(self, rank: int) -> DirectConnectionResponse:
        response_key = self._response_key(rank)
        deadline = time.monotonic() + _timeout_seconds()
        while not await self.store.exists(response_key):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for direct {self.backend.name} source "
                    f"rank {rank}"
                )
            await asyncio.sleep(_poll_interval_seconds())
        response = await self.store.get(response_key)
        await self.store.delete(response_key)
        if not isinstance(response, DirectConnectionResponse):
            raise TypeError(f"Invalid direct RDMA response: {type(response)}")
        if response.backend != self.backend:
            raise ValueError(
                f"Direct RDMA backend mismatch: {response.backend} != "
                f"{self.backend}"
            )
        if response.error is not None:
            raise RuntimeError(
                f"Direct {self.backend.name} source rank {rank} failed: "
                f"{response.error}"
            )
        return response

    async def _connect(
        self,
        num_ranks: int,
        user_state_dict: dict[str, torch.Tensor],
    ):
        from torchstore.direct_weight_sync import RDMAWeightHandle

        tensors = {
            name: Request.from_any(name, value).tensor_val
            for name, value in user_state_dict.items()
        }
        for name, tensor in tensors.items():
            _require_cuda_tensor(name, tensor)
        device = next(iter(tensors.values())).device

        all_handles = {}
        for rank in range(num_ranks):
            connection = self._connection_factory(device)
            request = DirectConnectionRequest(
                backend=self.backend,
                client_id=self.client_id,
                connection_info=connection.connection_info,
            )
            await self.store.put(self._request_key(rank), request)
            response = await self._wait_for_response(rank)
            connection.connect(response.connection_info)
            self._connections[rank] = connection
            logger.info(
                "[ts-direct-rdma] backend=%s role=destination source_rank=%d "
                "tensors=%d",
                self.backend.name,
                rank,
                len(response.handles),
            )
            for name, handle in response.handles.items():
                all_handles.setdefault(name, []).append(
                    RDMAWeightHandle(
                        rdma_buffer=_ReadableDirectBuffer(
                            connection, handle.remote_buffer
                        ),
                        tensor_slice=handle.tensor_slice,
                        source_rank=handle.source_rank,
                    )
                )
        return all_handles

    async def pull(
        self,
        *,
        num_ranks: int,
        user_state_dict: dict[str, torch.Tensor],
    ) -> None:
        from torchstore.direct_weight_sync import DirectWeightSyncDest

        if not self._connections:
            self._handles = await self._connect(num_ranks, user_state_dict)
        assert self._handles is not None
        await DirectWeightSyncDest().pull(self._handles, user_state_dict)

    async def close(self) -> None:
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()


class DirectTransport:
    """Base class for persistent direct state-dict transports."""

    def __init__(
        self,
        store: Any,
        key: str,
        transport_type: TransportType | None = None,
    ) -> None:
        self.store = store
        self.key = key
        self.transport_type = transport_type
        self._source = None
        self._destination = None

    def create_source_connection(self, connection_info, device):
        raise NotImplementedError

    def create_destination_connection(self, device):
        raise NotImplementedError

    async def put(
        self,
        state_dict: dict[str, torch.Tensor] | None,
        *,
        rank: int,
        world_size: int,
        transfer_dtype: torch.dtype | None,
    ) -> None:
        if self.transport_type is None:
            raise NotImplementedError
        if self._source is not None:
            self._source.refresh()
            return
        assert state_dict is not None, (
            "state_dict is required on first put_state_dict call with "
            "direct_rdma=True"
        )
        self._source = DirectTransportWeightSyncSource(
            self.transport_type,
            self.store,
            self.key,
            self.create_source_connection,
        )
        await self._source.register(
            state_dict,
            rank=rank,
            transfer_dtype=transfer_dtype,
        )
        if rank == 0:
            await self.store.put(
                f"{self.key}/direct/{self.transport_type.name}/num_ranks",
                world_size,
            )

    async def get(self, user_state_dict: dict[str, torch.Tensor]) -> None:
        if self.transport_type is None:
            raise NotImplementedError
        if self._destination is None:
            self._destination = DirectTransportWeightSyncDest(
                self.transport_type,
                self.store,
                self.key,
                self.create_destination_connection,
            )
        num_ranks = await self.store.get(
            f"{self.key}/direct/{self.transport_type.name}/num_ranks"
        )
        await self._destination.pull(
            num_ranks=num_ranks,
            user_state_dict=user_state_dict,
        )

    async def close(self) -> None:
        if self._source is not None:
            await self._source.close()
            self._source = None
        if self._destination is not None:
            await self._destination.close()
            self._destination = None


def create_direct_transport(
    transport_type: TransportType,
    store: Any,
    key: str,
) -> DirectTransport:
    """Create the direct transport selected by the TorchStore strategy."""
    from torchstore.direct_weight_sync import MonarchDirectTransport
    from torchstore.transport import TransportType
    from torchstore.transport.rdma4py import Rdma4PyDirectTransport
    from torchstore.transport.torchcomms.buffer import TorchCommsDirectTransport

    factories = {
        TransportType.MonarchRDMA: MonarchDirectTransport,
        TransportType.Rdma4Py: Rdma4PyDirectTransport,
        TransportType.TorchComms: TorchCommsDirectTransport,
    }
    try:
        factory = factories[transport_type]
    except KeyError as error:
        raise ValueError(
            f"Transport {transport_type.name} does not support direct weight sync"
        ) from error
    return factory(store, key)
