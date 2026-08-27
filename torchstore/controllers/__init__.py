# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchstore.controller import Controller, ObjectType, StorageInfo
from torchstore.controllers.indexed import (
    BoundPlanOffer,
    IndexedController,
    IndexedDirectoryBackend,
    PlanOffer,
    RegionKey,
    SlicePlan,
)


def get_controller_class(
    name: str,
) -> type[Controller] | type[IndexedController]:
    if name == "legacy":
        return Controller
    if name == "indexed":
        return IndexedController
    raise ValueError(
        f"Unknown controller implementation {name!r}; expected 'legacy' or 'indexed'"
    )


__all__ = [
    "BoundPlanOffer",
    "IndexedController",
    "IndexedDirectoryBackend",
    "ObjectType",
    "PlanOffer",
    "RegionKey",
    "SlicePlan",
    "StorageInfo",
    "get_controller_class",
]
