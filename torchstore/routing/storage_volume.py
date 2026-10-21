# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch

from torchstore.storage_volume import InMemoryStore, StorageImpl, StorageVolume
from torchstore.transport.buffers import TransportBuffer
from torchstore.transport.types import Request, TensorSlice
from torchstore.utils import get_destination_view, same_slice_geometry


@dataclass
class _StoredSegment:
    tensor_slice: TensorSlice
    tensor: torch.Tensor


@dataclass
class _RoutedTensor:
    requesters: dict[str, _StoredSegment]


class RoutingInMemoryStore(InMemoryStore):
    def _register_storage_layouts(self, request: Request) -> None:
        assert request.storage_layouts is not None
        stored = self.kv.get(request.key)
        routed = stored if isinstance(stored, _RoutedTensor) else _RoutedTensor({})
        for requester, (tensor_slice, dtype) in request.storage_layouts.items():
            segment = next(
                (
                    segment
                    for segment in routed.requesters.values()
                    if segment.tensor.dtype == dtype
                    and same_slice_geometry(segment.tensor_slice, tensor_slice)
                ),
                None,
            )
            if segment is None:
                segment = _StoredSegment(
                    tensor_slice,
                    torch.empty(
                        tensor_slice.local_shape,
                        dtype=dtype,
                        device="cpu",
                        pin_memory=torch.cuda.is_available(),
                    ),
                )
            routed.requesters[requester] = segment
        self.kv[request.key] = routed

    def _store_routed_tensor(self, request: Request, tensor: torch.Tensor) -> None:
        assert request.tensor_slice is not None
        routed = self.kv[request.key]
        assert isinstance(routed, _RoutedTensor)
        stored_tensors = set()
        for segment in routed.requesters.values():
            if id(segment.tensor) in stored_tensors:
                continue
            source = get_destination_view(
                tensor,
                request.tensor_slice,
                segment.tensor_slice,
                require_contiguous=False,
            )
            if source is None:
                raise ValueError(
                    f"stored layout for {request.key!r} is outside its publisher slice"
                )
            segment.tensor.copy_(source)
            stored_tensors.add(id(segment.tensor))

    def _extract_existing(self, request: Request) -> torch.Tensor | None:
        if isinstance(self.kv.get(request.key), _RoutedTensor):
            return None
        return super()._extract_existing(request)

    async def put(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> None:
        for request in requests:
            if request.storage_layouts:
                self._register_storage_layouts(request)
        await super().put(transport_buffer, requests)

    async def handshake(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> list[Any]:
        for request in requests:
            request.stable_storage = self._segment(request) is not None
        return await super().handshake(transport_buffer, requests)

    def _store(self, request: Request, data: Any) -> None:
        if isinstance(self.kv.get(request.key), _RoutedTensor):
            assert isinstance(data, torch.Tensor)
            self._store_routed_tensor(request, data)
        else:
            super()._store(request, data)

    def _segment(self, request: Request) -> _StoredSegment | None:
        stored = self.kv.get(request.key)
        if isinstance(stored, _RoutedTensor) and request.requester_id is not None:
            return stored.requesters.get(request.requester_id)
        return None

    async def get(
        self,
        transport_buffer: TransportBuffer,
        requests: list[Request],
    ) -> TransportBuffer:
        return await super().get(transport_buffer, requests)

    def _get_data(self, request: Request) -> Any:
        stored = self.kv[request.key]
        if isinstance(stored, _RoutedTensor):
            segment = self._segment(request)
            if segment is None:
                raise KeyError(
                    f"Key {request.key!r} requires a registered requester layout"
                )
            return segment.tensor
        return super()._get_data(request)

    def _get_meta(self, request: Request) -> tuple[torch.Size, torch.dtype] | str:
        stored = self.kv[request.key]
        if isinstance(stored, _RoutedTensor):
            segment = self._segment(request)
            if segment is None:
                raise KeyError(
                    f"Key {request.key!r} requires a registered requester layout"
                )
            return segment.tensor.shape, segment.tensor.dtype
        return super()._get_meta(request)


class RoutingStorageVolume(StorageVolume):
    def __init__(self, id_func) -> None:
        self.store: StorageImpl = RoutingInMemoryStore()
        self.volume_id: str = id_func()
