# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.checkpoint._nested_dict import (
    flatten_state_dict,
    unflatten_state_dict,
)

from torchstore.logging import LatencyTracker

DELIM = "/"
MAPPING = "MAPPING"

logger = logging.getLogger(__name__)


@dataclass
class _DirectRDMACache:
    """Per-client cache for direct RDMA weight sync state."""

    transports: dict = field(default_factory=dict)


_rdma_cache: dict[int, _DirectRDMACache] = {}


def _get_rdma_cache(store) -> _DirectRDMACache:
    """Get or create the RDMA cache for a given client."""
    key = id(store)
    if key not in _rdma_cache:
        _rdma_cache[key] = _DirectRDMACache()
    return _rdma_cache[key]


async def put_state_dict(
    store,
    state_dict,
    key,
    direct_rdma=False,
    transfer_dtype=None,
):
    """Store a state dict in TorchStore.

    When ``direct_rdma=False`` (default), each parameter is stored in a
    StorageVolume via the normal two-hop transport path.

    When ``direct_rdma=True``, RDMA handles are registered against the
    caller's GPU memory and only the handles (not tensor data) are stored
    in TorchStore.  On the first call the handles are registered; on
    subsequent calls only non-contiguous staging buffers are refreshed.
    ``state_dict`` may be ``None`` on subsequent calls to skip the
    (potentially expensive) ``model.state_dict()`` construction.
    ``torch.distributed`` must be initialised before using this mode.


    Args:
        transfer_dtype: If set, cast floating-point weights to this dtype for
            transfer. Allows the source to keep higher-precision master weights
            while transferring in a lower precision (e.g. bfloat16).
    """
    # Coarse weight-sync phase timing, logged at INFO (no TORCHSTORE_LOG_LEVEL
    # =DEBUG / init_logging() needed) so callers can attribute latency to the
    # major phases and compare the direct-RDMA vs CPU-staged transports.
    direct_transport = (
        store.strategy.get_direct_transport_type() if direct_rdma else None
    )
    direct_path = direct_transport.name if direct_transport is not None else None
    tracker = LatencyTracker(
        f"put_state_dict[{key}]/{direct_path if direct_rdma else 'cpu_staged'}",
        level=logging.INFO,
    )

    if direct_rdma:
        await _put_state_dict_direct_rdma(
            store,
            state_dict,
            key,
            transfer_dtype,
            direct_transport,
        )
        # No throughput here: the direct-RDMA put only registers handles (and on
        # repeat calls refreshes staging buffers). The bytes move on the
        # generator's get via one-sided RDMA read, so there is no bulk transfer
        # in put to report a GB/s for.
        tracker.track_e2e()
        return

    flattened_state_dict, mapping = flatten_state_dict(state_dict)
    tracker.track_step("flatten")
    # TODO: when transfer_dtype actually casts (source dtype != transfer_dtype),
    # this allocates a full auxiliary copy of the floating weights. Fold the cast
    # into the per-tensor staging copy (allocate the staging buffer at
    # transfer_dtype and copy_ into it) so peak extra memory is one tensor, not
    # the whole state dict.
    flattened_state_dict = _cast_floating_tensors(flattened_state_dict, transfer_dtype)
    tracker.track_step("cast")
    nbytes = _flattened_nbytes(flattened_state_dict)

    # Batch all tensor entries, then put the mapping separately.
    # The mapping is stored last so it acts as a commit marker for the state dict.
    entries = {f"{key}{DELIM}{k}": v for k, v in flattened_state_dict.items()}
    await store.put_batch(entries)
    tracker.track_step("put_batch", nbytes=nbytes)
    await store.put(f"{key}{DELIM}{MAPPING}", mapping)
    tracker.track_step("put_mapping")
    tracker.track_e2e(nbytes=nbytes)


async def get_state_dict(
    store,
    key,
    user_state_dict: dict | None = None,
    strict=True,
    direct_rdma=False,
):
    """Retrieve a state dict from TorchStore.

    When ``direct_rdma=False`` (default), each parameter is fetched from
    a StorageVolume via the normal two-hop transport path.

    When ``direct_rdma=True``, RDMA handles are fetched from TorchStore
    (first call only, cached afterwards) and weights are pulled directly
    from the source's GPU memory via one-sided RDMA reads.
    ``user_state_dict`` must be provided in this mode so the destination
    tensors are available for in-place writes.
    """
    direct_transport = (
        store.strategy.get_direct_transport_type() if direct_rdma else None
    )
    direct_path = direct_transport.name if direct_transport is not None else None
    tracker = LatencyTracker(
        f"get_state_dict[{key}]/{direct_path if direct_rdma else 'cpu_staged'}",
        level=logging.INFO,
    )

    if direct_rdma:
        assert (
            user_state_dict is not None
        ), "user_state_dict is required for direct_rdma mode"
        await _get_state_dict_direct_rdma(store, key, user_state_dict, direct_transport)
        # The direct-RDMA get is the actual GPU-to-GPU transfer, so report its
        # throughput (comparable to the CPU-staged get_batch GB/s).
        tracker.track_e2e(nbytes=_state_dict_nbytes(user_state_dict))
        return user_state_dict

    try:
        # Since the mapping is the last thing we write out, it also gaurantees the state dict is not pending
        fetched_mapping = await store.get(f"{key}{DELIM}{MAPPING}")
    except Exception as e:
        raise RuntimeError(
            f"Mapping is missing from the store. This most likely means there is no matching 'push' call for this key: {key=}"
        ) from e
    tracker.track_step("get_mapping")

    user_flattened_state_dict, user_mapping = (
        flatten_state_dict(user_state_dict)
        if user_state_dict is not None
        else ({}, None)
    )
    if strict and user_mapping is not None:
        assert user_mapping == fetched_mapping

    get_id = lambda fk: f"{key}{DELIM}{fk}"
    flattened_keys = list(fetched_mapping.keys())

    get_batch_dict = {}
    for fk in flattened_keys:
        t = user_flattened_state_dict.get(fk, None)
        # inplace can only be a tensor, so skip non-tensor values
        if t is not None and not isinstance(t, torch.Tensor):
            t = None
            logger.warning(f"non-tensor value found for in-place: {fk}")
        get_batch_dict[get_id(fk)] = t

    results = await store.get_batch(get_batch_dict)
    nbytes = _flattened_nbytes(results)
    tracker.track_step("get_batch", nbytes=nbytes)
    fetched_state_dict = {fk: results[get_id(fk)] for fk in flattened_keys}

    out = unflatten_state_dict(fetched_state_dict, fetched_mapping)
    tracker.track_step("unflatten")
    tracker.track_e2e(nbytes=nbytes)
    return out


def _cast_floating_tensors(flattened_state_dict, dtype):
    if dtype is None:
        return flattened_state_dict
    return {
        key: value.to(dtype)
        if (
            isinstance(value, torch.Tensor)
            and value.is_floating_point()
            and value.dtype != dtype
        )
        else value
        for key, value in flattened_state_dict.items()
    }


def _flattened_nbytes(flattened_state_dict) -> int:
    """Total byte size of the tensor values in an already-flattened state dict."""
    return sum(
        t.numel() * t.element_size()
        for t in flattened_state_dict.values()
        if isinstance(t, torch.Tensor)
    )


def _state_dict_nbytes(state_dict) -> int:
    """Total tensor byte size of a (possibly nested) state dict."""
    sd, _ = flatten_state_dict(state_dict)
    return _flattened_nbytes(sd)


def _state_dict_size(state_dict):
    """Returns the size of the state dict in MBs"""
    return _state_dict_nbytes(state_dict) // (1024 * 1024)


# ---------------------------------------------------------------------------
# Direct RDMA helpers (used when direct_rdma=True)
# ---------------------------------------------------------------------------


async def _put_state_dict_direct_rdma(
    store, state_dict, key, transfer_dtype, transport_type
):
    """Register or refresh RDMA handles and publish via TorchStore.

    First call for a given key: registers RDMA handles for each param,
    stores handles in TorchStore as objects.  ``state_dict`` must be
    provided on this first call.

    Subsequent calls: refreshes staging buffers for non-contiguous params.
    ``state_dict`` may be ``None`` to skip building it when only a refresh
    is needed.
    """
    cache = _get_rdma_cache(store)
    transport = _get_or_create_direct_transport(
        cache, transport_type, store, key
    )
    await transport.put(
        state_dict,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        transfer_dtype=transfer_dtype,
    )


async def _get_state_dict_direct_rdma(
    store, key, user_state_dict, transport_type
):
    """Fetch RDMA handles and pull weights via direct RDMA reads.

    First call for a given key: fetches handles from TorchStore, caches them.
    Transfer plan is built on first pull and cached by DirectWeightSyncDest.
    All RDMA reads are issued concurrently for maximum throughput.
    """
    cache = _get_rdma_cache(store)
    transport = _get_or_create_direct_transport(
        cache, transport_type, store, key
    )
    await transport.get(user_state_dict)


def _get_or_create_direct_transport(cache, transport_type, store, key):
    from torchstore.direct_transport import create_direct_transport

    cache_key = (key, transport_type)
    transport = cache.transports.get(cache_key)
    if transport is None:
        transport = create_direct_transport(transport_type, store, key)
        cache.transports[cache_key] = transport
    return transport
