# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchstore.controller import ObjectType
from torchstore.transport.types import TensorSlice

Publication = tuple[int, str]


def order_live_volumes(
    volumes: set[str], prefer: tuple[str, ...] | None
) -> tuple[str, ...]:
    if prefer is None:
        return tuple(sorted(volumes))
    preferred = tuple(dict.fromkeys(volume for volume in prefer if volume in volumes))
    return preferred + tuple(sorted(volumes - set(preferred)))


@dataclass(frozen=True, order=True)
class RegionKey:
    global_shape: tuple
    offsets: tuple
    local_shape: tuple

    @classmethod
    def from_slice(cls, tensor_slice: TensorSlice) -> "RegionKey":
        return cls(
            global_shape=tuple(tensor_slice.global_shape),
            offsets=tuple(tensor_slice.offsets),
            local_shape=tuple(tensor_slice.local_shape),
        )

    def as_tensor_slice(self) -> TensorSlice:
        return TensorSlice(
            offsets=self.offsets,
            coordinates=None,
            global_shape=self.global_shape,
            local_shape=self.local_shape,
            mesh_shape=None,
        )


@dataclass(frozen=True)
class PlanOffer:
    source_region: RegionKey | None
    intersection: RegionKey | None
    storage_offsets: tuple | None
    destination_offsets: tuple | None
    lengths: tuple | None


@dataclass(frozen=True)
class BoundPlanOffer:
    geometry: PlanOffer
    live_volumes: tuple[str, ...]
    pending: tuple[Publication, ...]


@dataclass(frozen=True)
class SlicePlan:
    key: str
    object_type: ObjectType
    requested_region: RegionKey | None
    geometry_epoch: int
    availability: int
    offers: tuple[BoundPlanOffer, ...]
