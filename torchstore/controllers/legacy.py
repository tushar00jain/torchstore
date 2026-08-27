# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Mapping, Sequence

from torchstore.controller import ObjectType, StorageInfo
from torchstore.controllers.indexed.plans import (
    BoundPlanOffer,
    PlanOffer,
    Publication,
    RegionKey,
    SlicePlan,
    order_live_volumes,
)
from torchstore.transport.types import Request
from torchstore.utils import get_slice_intersection


def locate_slices_exhaustive(
    requests: Sequence[Request],
    directory: Mapping[str, dict[str, dict[int, StorageInfo]]],
    *,
    missing_ok: bool,
    require_fully_committed: bool,
    include_pending: bool,
    prefer: Sequence[str] | None,
    completeness: Callable[[str, dict[str, StorageInfo]], bool],
) -> dict[str, SlicePlan]:
    """Build logical plans by scanning every source recorded for each key."""
    plans = {}
    for request in requests:
        volume_map = directory.get(request.key)
        if volume_map is None:
            if missing_ok:
                continue
            raise KeyError(f"Unable to locate {request.key} in any storage volumes.")
        live = {volume: slot[0] for volume, slot in volume_map.items() if 0 in slot}
        if require_fully_committed and not completeness(request.key, live):
            raise KeyError(
                f"DTensor '{request.key}' is only partially committed. "
                "Not all shards have been stored yet. Please ensure all ranks "
                "complete their put() operations."
            )
        sources: list[tuple[Publication, StorageInfo]] = []
        for volume, slot in volume_map.items():
            for pub, info in slot.items():
                if pub == 0 or include_pending:
                    sources.append(((pub, volume), info))
        if not sources:
            if missing_ok:
                continue
            raise KeyError(f"Unable to locate {request.key} in any storage volumes.")
        object_type = sources[0][1].object_type
        requested_region = (
            None
            if request.tensor_slice is None
            else RegionKey.from_slice(request.tensor_slice)
        )
        if object_type is not ObjectType.TENSOR_SLICE:
            live_volumes = order_live_volumes(
                {source[1] for source, _ in sources if source[0] == 0},
                None if prefer is None else tuple(prefer),
            )
            pending = tuple(sorted(source for source, _ in sources if source[0] != 0))
            offers = (
                BoundPlanOffer(
                    PlanOffer(None, None, None, None, None), live_volumes, pending
                ),
            )
        else:
            grouped: dict[RegionKey, set[Publication]] = {}
            for source, info in sources:
                for stored_slice in info.tensor_slices:
                    assert stored_slice is not None
                    region = RegionKey.from_slice(stored_slice)
                    grouped.setdefault(region, set()).add(source)
            bound = []
            for region in sorted(grouped):
                intersection = region.as_tensor_slice()
                if request.tensor_slice is not None:
                    intersection = get_slice_intersection(
                        intersection, request.tensor_slice
                    )
                    if intersection is None:
                        continue
                    destination_offsets = tuple(
                        offset - base
                        for offset, base in zip(
                            intersection.offsets,
                            request.tensor_slice.offsets,
                            strict=True,
                        )
                    )
                else:
                    destination_offsets = intersection.offsets
                region_sources = grouped[region]
                intersection_region = RegionKey.from_slice(intersection)
                bound.append(
                    BoundPlanOffer(
                        PlanOffer(
                            region,
                            intersection_region,
                            tuple(
                                offset - base
                                for offset, base in zip(
                                    intersection.offsets,
                                    region.offsets,
                                    strict=True,
                                )
                            ),
                            destination_offsets,
                            intersection_region.local_shape,
                        ),
                        order_live_volumes(
                            {source[1] for source in region_sources if source[0] == 0},
                            None if prefer is None else tuple(prefer),
                        ),
                        tuple(
                            sorted(
                                source for source in region_sources if source[0] != 0
                            )
                        ),
                    )
                )
            offers = tuple(bound)
        plans[request.key] = SlicePlan(
            key=request.key,
            object_type=object_type,
            requested_region=requested_region,
            geometry_epoch=0,
            availability=0,
            offers=offers,
        )
    return plans
