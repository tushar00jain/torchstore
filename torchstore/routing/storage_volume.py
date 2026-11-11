# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch
from monarch.actor import endpoint

from torchstore.storage_volume import InMemoryStore, StorageVolume
from torchstore.transport.types import Request, TensorSlice
from torchstore.utils import same_slice_geometry

_StorageLayouts = dict[
    str,
    tuple[torch.dtype, dict[str, TensorSlice]],
]


@dataclass
class _StoredSegment:
    tensor_slice: TensorSlice
    tensor: torch.Tensor


@dataclass
class _RoutedTensor:
    requesters: dict[str, _StoredSegment]


class RoutingInMemoryStore(InMemoryStore):
    def install_layouts(self, layouts: _StorageLayouts) -> None:
        for key, (dtype, requesters) in layouts.items():
            stored = self.kv.get(key)
            if stored is None:
                stored = _RoutedTensor({})
                self.kv[key] = stored
            elif not isinstance(stored, _RoutedTensor):
                raise ValueError(f"cannot install a routed layout over {key!r}")

            for requester, requester_slice in requesters.items():
                previous = stored.requesters.get(requester)
                if previous is not None:
                    if previous.tensor.dtype != dtype or not same_slice_geometry(
                        previous.tensor_slice, requester_slice
                    ):
                        raise ValueError(
                            f"conflicting layout for {key!r} requester {requester!r}"
                        )
                    continue
                segment = next(
                    (
                        candidate
                        for candidate in stored.requesters.values()
                        if candidate.tensor.dtype == dtype
                        and same_slice_geometry(
                            candidate.tensor_slice, requester_slice
                        )
                    ),
                    None,
                )
                if segment is None:
                    segment = _StoredSegment(
                        requester_slice,
                        torch.empty(
                            requester_slice.local_shape,
                            dtype=dtype,
                            device="cpu",
                            pin_memory=torch.cuda.is_available(),
                        ),
                    )
                stored.requesters[requester] = segment

    def _extract_existing(self, request: Request) -> torch.Tensor | None:
        stored = self.kv.get(request.key)
        if isinstance(stored, _RoutedTensor):
            segment = self._segment(request, stored)
            return segment.tensor if segment is not None else None
        return super()._extract_existing(request)

    def _store(self, request: Request, data: Any) -> None:
        stored = self.kv.get(request.key)
        if isinstance(stored, _RoutedTensor):
            assert isinstance(data, torch.Tensor)
            segment = self._segment(request, stored)
            if segment is None:
                raise KeyError(
                    f"Key {request.key!r} requires a registered requester layout"
                )
            if data.data_ptr() != segment.tensor.data_ptr():
                segment.tensor.copy_(data)
            return
        super()._store(request, data)

    def _segment(
        self,
        request: Request,
        stored: _RoutedTensor,
    ) -> _StoredSegment | None:
        if request.requester_id is None:
            return None
        return stored.requesters.get(request.requester_id)

    def _get_data(self, request: Request) -> Any:
        stored = self.kv[request.key]
        if isinstance(stored, _RoutedTensor):
            segment = self._segment(request, stored)
            if segment is None:
                raise KeyError(
                    f"Key {request.key!r} requires a registered requester layout"
                )
            return segment.tensor
        return super()._get_data(request)

    def _get_meta(self, request: Request) -> tuple[torch.Size, torch.dtype] | str:
        stored = self.kv[request.key]
        if isinstance(stored, _RoutedTensor):
            tensor = self._get_data(request)
            return tensor.shape, tensor.dtype
        return super()._get_meta(request)


class RoutingStorageVolume(StorageVolume):
    def __init__(self, id_func) -> None:
        self.store = RoutingInMemoryStore()
        self.volume_id: str = id_func()

    @endpoint
    async def install_layouts(self, layouts: _StorageLayouts) -> None:
        self.store.install_layouts(layouts)
