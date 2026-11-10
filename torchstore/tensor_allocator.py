# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable

import torch

_MAX_STORAGE_BYTES = 256 * 1024 * 1024


def allocate_tensor_views(
    specs: Iterable[tuple[torch.Size | tuple[int, ...], torch.dtype]],
    *,
    pin_memory: bool,
) -> list[torch.Tensor]:
    """Allocate tensor views packed into bounded CPU storages."""
    layouts = []
    storage_sizes = []
    storage_bytes = 0
    storage_index = 0
    for shape, dtype in specs:
        itemsize = dtype.itemsize
        offset_bytes = (storage_bytes + itemsize - 1) // itemsize * itemsize
        numel = 1
        for size in shape:
            numel *= size
        tensor_bytes = numel * itemsize
        if storage_bytes and offset_bytes + tensor_bytes > _MAX_STORAGE_BYTES:
            storage_sizes.append(storage_bytes)
            storage_index += 1
            offset_bytes = 0
        layouts.append((shape, dtype, storage_index, offset_bytes // itemsize))
        storage_bytes = offset_bytes + tensor_bytes

    if layouts:
        storage_sizes.append(storage_bytes)

    storages = [
        torch.empty(
            size,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=pin_memory,
        ).untyped_storage()
        for size in storage_sizes
    ]
    return [
        torch.empty(0, dtype=dtype, device="cpu").set_(
            storages[storage_index], offset, shape
        )
        for shape, dtype, storage_index, offset in layouts
    ]
