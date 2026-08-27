# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Hashable, Mapping, Sequence
from typing import TypeVar

from torchstore.controller import ObjectType, StorageInfo
from torchstore.transport.types import Request, TensorSlice
from torchstore.utils import get_slice_intersection


SliceRegion = tuple[tuple, tuple, tuple]
Region = tuple[str, SliceRegion | None]
SourceId = TypeVar("SourceId", bound=Hashable)


def _region(key: str, tensor_slice: TensorSlice | None) -> Region:
    if tensor_slice is None:
        return key, None
    return key, (
        tensor_slice.offsets,
        tensor_slice.local_shape,
        tensor_slice.global_shape,
    )


def _slice_regions(request: Request, offered: StorageInfo) -> set[Region]:
    regions = set()
    for stored_slice in offered.tensor_slices:
        if stored_slice is None:
            continue
        overlap = stored_slice
        if request.tensor_slice is not None:
            overlap = get_slice_intersection(stored_slice, request.tensor_slice)
        if overlap is not None:
            regions.add(_region(request.key, overlap))
    return regions


def _cover_regions(request: Request, offered: StorageInfo) -> set[Region]:
    if offered.object_type is ObjectType.TENSOR_SLICE:
        return _slice_regions(request, offered)
    return {_region(request.key, None)}


def _request_info(request: Request) -> StorageInfo:
    return StorageInfo(ObjectType.from_request(request), {request.tensor_slice})


def _overlaps(wanted: StorageInfo, offered: StorageInfo) -> bool:
    if offered.object_type is not ObjectType.TENSOR_SLICE:
        return True
    wanted_slices = wanted.tensor_slices
    if None in wanted_slices:
        return bool(offered.tensor_slices)
    return any(
        get_slice_intersection(offered_slice, wanted_slice) is not None
        for offered_slice in offered.tensor_slices
        for wanted_slice in wanted_slices
    )


def cover(
    requests: Sequence[Request],
    source_maps: Mapping[str, Mapping[SourceId, StorageInfo]],
    covered: set[Region] | None = None,
) -> list[SourceId]:
    """Per-key/per-slice greedy walk over ranked sources."""
    seen = set() if covered is None else covered
    chosen: dict[SourceId, None] = {}
    for request in requests:
        wanted = _request_info(request)
        for source, offered in source_maps.get(request.key, {}).items():
            if not _overlaps(wanted, offered):
                continue
            fresh = _cover_regions(request, offered) - seen
            if fresh:
                seen.update(fresh)
                chosen.setdefault(source, None)
            if offered.object_type is not ObjectType.TENSOR_SLICE:
                break
    return list(chosen)
