# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NIXL transport for StorageVolume tensor transfers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import threading
import time
import uuid
import weakref
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from functools import cache
from importlib.util import find_spec
from itertools import product
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
from typing_extensions import override

from torchstore.transport.buffers import TransportBuffer, TransportCache
from torchstore.transport.types import Request

if TYPE_CHECKING:
    from torchstore.strategy import StorageVolumeRef
    from torchstore.transport.buffers import TransportContext

logger = logging.getLogger(__name__)

ENV_TORCHSTORE_NIXL_BACKEND = os.environ.get("TORCHSTORE_NIXL_BACKEND", "UCX")
ENV_TORCHSTORE_NIXL_ENABLED = os.environ.get("TORCHSTORE_NIXL_ENABLED", "0") == "1"
ENV_TORCHSTORE_NIXL_TIMEOUT_S = int(os.environ.get("TORCHSTORE_NIXL_TIMEOUT_S", "60"))
# Maximum size of each of the two buffers that stage non-contiguous tensors for a get.
ENV_TORCHSTORE_NIXL_STAGING_BUFFER_BYTES = int(
    os.environ.get("TORCHSTORE_NIXL_STAGING_BUFFER_BYTES", str(32 * 1024 * 1024))
)
# Staged pieces start at this byte alignment so any dtype can view the buffer.
_STAGING_ALIGNMENT = 64


@dataclass(frozen=True)
class _NicCounterSnapshot:
    timestamp_ns: int
    counters: dict[str, int]


@cache
def _nic_counter_paths() -> dict[str, tuple[Path, int]]:
    """Resolve the IB counters for the UCX net device used by this process."""
    device_spec = os.environ.get("UCX_NET_DEVICES", "").split(",", 1)[0]
    if not device_spec:
        return {}
    net_device, _, port = device_spec.partition(":")
    infiniband_dir = Path("/sys/class/net") / net_device / "device/infiniband"
    try:
        ib_device = next(infiniband_dir.iterdir())
    except (FileNotFoundError, StopIteration):
        return {}

    port_dir = ib_device / "ports" / (port or "1")
    paths: dict[str, tuple[Path, int]] = {}
    standard = {
        "tx_bytes": ("port_xmit_data", 4),
        "rx_bytes": ("port_rcv_data", 4),
        "tx_packets": ("port_xmit_packets", 1),
        "rx_packets": ("port_rcv_packets", 1),
        "tx_wait": ("port_xmit_wait", 1),
        "tx_discards": ("port_xmit_discards", 1),
        "rx_errors": ("port_rcv_errors", 1),
    }
    for name, (filename, scale) in standard.items():
        path = port_dir / "counters" / filename
        if path.is_file():
            paths[name] = (path, scale)

    hardware = {
        "cnp_sent": "np_cnp_sent",
        "cnp_handled": "rp_cnp_handled",
        "cnp_ignored": "rp_cnp_ignored",
        "packet_seq_err": "packet_seq_err",
        "retry_exceeded": "retry_exceeded",
        "out_of_buffer": "out_of_buffer",
        "req_cqe_error": "req_cqe_error",
        "resp_cqe_error": "resp_cqe_error",
    }
    for name, filename in hardware.items():
        path = port_dir / "hw_counters" / filename
        if path.is_file():
            paths[name] = (path, 1)
    return paths


class _NicCounterSampler:
    """Sample shared node-level NIC counters while traced transfers are active."""

    _INTERVAL_SECONDS = 0.1

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._active: dict[tuple[str, str], int] = {}
        self._samples: list[_NicCounterSnapshot] = []
        self._thread: threading.Thread | None = None

    def _read(self) -> _NicCounterSnapshot | None:
        paths = _nic_counter_paths()
        if not paths:
            return None
        counters: dict[str, int] = {}
        try:
            for name, (path, scale) in paths.items():
                counters[name] = int(path.read_text()) * scale
        except (OSError, ValueError):
            return None
        return _NicCounterSnapshot(time.time_ns(), counters)

    def _append_sample(self) -> _NicCounterSnapshot | None:
        sample = self._read()
        if sample is not None:
            with self._lock:
                self._samples.append(sample)
        return sample

    def _run(self) -> None:
        while True:
            self._wake.wait()
            while True:
                with self._lock:
                    active = bool(self._active)
                    if not active:
                        self._wake.clear()
                        break
                self._append_sample()
                time.sleep(self._INTERVAL_SECONDS)

    def begin(self, request_id: str, role: str) -> bool:
        if not _nic_counter_paths():
            return False
        key = (request_id, role)
        sample = self._read()
        if sample is None:
            return False
        with self._lock:
            self._active[key] = sample.timestamp_ns
            self._samples.append(sample)
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run,
                    name="torchstore-nic-counters",
                    daemon=True,
                )
                self._thread.start()
            self._wake.set()
        return True

    def mark(self) -> _NicCounterSnapshot | None:
        return self._append_sample()

    def end(
        self,
        request_id: str,
        role: str,
        baseline: _NicCounterSnapshot | None = None,
    ) -> dict[str, int]:
        key = (request_id, role)
        completion = self._read()
        with self._lock:
            started_ns = self._active.pop(key, None)
            if started_ns is None or completion is None:
                return {}
            self._samples.append(completion)
            start = baseline
            if start is None:
                start = next(
                    (
                        sample
                        for sample in self._samples
                        if sample.timestamp_ns >= started_ns
                    ),
                    completion,
                )
            samples = [
                sample
                for sample in self._samples
                if start.timestamp_ns <= sample.timestamp_ns <= completion.timestamp_ns
            ]
            if not self._active:
                self._samples.clear()

        def delta(name: str) -> int:
            return max(
                0,
                completion.counters.get(name, 0) - start.counters.get(name, 0),
            )

        def activity_ns(counter: str) -> tuple[int, int, int]:
            previous = start.counters.get(counter, 0)
            first = 0
            last = 0
            for sample in samples:
                current = sample.counters.get(counter, previous)
                if current > previous:
                    first = first or sample.timestamp_ns
                    last = sample.timestamp_ns
                previous = current
            if not first:
                return -1, -1, -1
            return (
                first - start.timestamp_ns,
                max(0, last - first),
                max(0, completion.timestamp_ns - last),
            )

        tx_first, tx_active, tx_tail = activity_ns("tx_bytes")
        rx_first, rx_active, rx_tail = activity_ns("rx_bytes")
        return {
            "window_ns": completion.timestamp_ns - start.timestamp_ns,
            "tx_bytes": delta("tx_bytes"),
            "rx_bytes": delta("rx_bytes"),
            "tx_packets": delta("tx_packets"),
            "rx_packets": delta("rx_packets"),
            "tx_first_ns": tx_first,
            "tx_active_ns": tx_active,
            "tx_tail_ns": tx_tail,
            "rx_first_ns": rx_first,
            "rx_active_ns": rx_active,
            "rx_tail_ns": rx_tail,
            "tx_wait": delta("tx_wait"),
            "cnp_sent": delta("cnp_sent"),
            "cnp_handled": delta("cnp_handled"),
            "cnp_ignored": delta("cnp_ignored"),
            "packet_seq_err": delta("packet_seq_err"),
            "retry_exceeded": delta("retry_exceeded"),
            "tx_discards": delta("tx_discards"),
            "rx_errors": delta("rx_errors"),
            "out_of_buffer": delta("out_of_buffer"),
            "req_cqe_error": delta("req_cqe_error"),
            "resp_cqe_error": delta("resp_cqe_error"),
        }


_NIC_COUNTER_SAMPLER = _NicCounterSampler()


class _NixlTransferStatus(str, Enum):
    """Status strings returned by NIXL's Python transfer API."""

    PROCESSING = "PROC"
    DONE = "DONE"


class _NixlTransferOperation(str, Enum):
    """Operation strings accepted by NIXL's Python transfer API."""

    READ = "READ"
    WRITE = "WRITE"


@cache
def _load_nixl_api() -> tuple[Any, Any]:
    """Import NIXL only when constructing its transport."""
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError as error:
        raise RuntimeError("NIXL is not installed") from error
    return nixl_agent, nixl_agent_config


@cache
def nixl_available() -> bool:
    """Return whether the optional NIXL transport was explicitly enabled."""
    return ENV_TORCHSTORE_NIXL_ENABLED and find_spec("nixl") is not None


def _aligned(nbytes: int) -> int:
    return -(-nbytes // _STAGING_ALIGNMENT) * _STAGING_ALIGNMENT


async def _wait_terminal(tasks: list[asyncio.Task[Any]]) -> None:
    """Wait until every task is done, even if cancelled meanwhile.

    Returning early could free storage while DMA may still use it.
    """
    cancelled = False
    while True:
        try:
            await asyncio.wait(tasks)
            break
        except asyncio.CancelledError:
            cancelled = True
    for task in tasks:
        if not task.cancelled():
            task.exception()
    if cancelled:
        raise asyncio.CancelledError


def _staging_pieces(tensor: torch.Tensor, max_bytes: int):
    """Split a tensor into row-major pieces of at most max_bytes.

    Each piece is a view covering a contiguous byte range of the tensor's
    contiguous layout, so it maps to one destination descriptor.
    """
    if tensor.dim() == 0 or tensor.nbytes <= max_bytes:
        yield tensor
        return
    row_bytes = tensor.nbytes // tensor.shape[0]
    if row_bytes > max_bytes:
        for row in tensor:
            yield from _staging_pieces(row, max_bytes)
        return
    rows = max_bytes // row_bytes
    for start in range(0, tensor.shape[0], rows):
        yield tensor[start : start + rows]


def _contiguous_segments(tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if tensor.is_contiguous():
        return (tensor,)

    num_split_dims = 0
    segment = tensor
    while not segment.is_contiguous():
        num_split_dims += 1
        segment = tensor[(slice(0, 1),) * num_split_dims]

    return tuple(
        tensor[tuple(slice(index, index + 1) for index in indices)]
        for indices in product(
            *(range(tensor.shape[dim]) for dim in range(num_split_dims))
        )
    )


def _align_descriptors(agent, local_descs, remote_descs):
    local_sizes = [local_descs[index][1] for index in range(local_descs.descCount())]
    remote_sizes = [
        remote_descs[index][1] for index in range(remote_descs.descCount())
    ]
    if local_sizes == remote_sizes:
        return local_descs, remote_descs
    assert sum(local_sizes) == sum(remote_sizes)

    local_regions = []
    remote_regions = []
    local_index = remote_index = 0
    local_offset = remote_offset = 0
    while local_index < len(local_sizes) and remote_index < len(remote_sizes):
        size = min(
            local_sizes[local_index] - local_offset,
            remote_sizes[remote_index] - remote_offset,
        )
        local_addr, _, local_device = local_descs[local_index]
        remote_addr, _, remote_device = remote_descs[remote_index]
        local_regions.append((local_addr + local_offset, size, local_device))
        remote_regions.append((remote_addr + remote_offset, size, remote_device))
        local_offset += size
        remote_offset += size
        if local_offset == local_sizes[local_index]:
            local_index += 1
            local_offset = 0
        if remote_offset == remote_sizes[remote_index]:
            remote_index += 1
            remote_offset = 0

    memory_types = {agent.nixl_mems[name]: name for name in ("DRAM", "VRAM")}
    return (
        agent.get_xfer_descs(local_regions, memory_types[local_descs.getType()]),
        agent.get_xfer_descs(remote_regions, memory_types[remote_descs.getType()]),
    )


def _storage_key(tensor: torch.Tensor) -> tuple[int, int]:
    storage = tensor.untyped_storage()
    return storage.data_ptr(), storage.nbytes()


_BACKEND_PARAMS = {
    "UCX": {
        # CUDA_COPY_REG_WHOLE_ALLOC: UCX otherwise registers the cuMem allocation
        # holding a CUDA buffer's start. With expandable segments that is one 20 MiB
        # chunk, so a buffer no longer than a chunk that crosses a chunk boundary is
        # only partly registered; registration fails, or RDMA past it fails with a
        # local access violation that leaves the transfer stuck. Each registration
        # covers its whole storage.
        # RCACHE_ENABLE: UCX otherwise keeps deregistered memory registered, which
        # pins CUDA memory torch has since freed so the device cannot reuse it.
        # IB_MEM_REG_OVERHEAD, IB_MEM_REG_GROWTH: UCX picks each transfer's protocol
        # by its estimated time. With RCACHE_ENABLE=n, it adds a memory registration
        # to every zero-copy estimate, so it copies host memory through bounce
        # buffers instead. NIXL registers each buffer up front and passes its
        # handle with each transfer, so no transfer registers memory. Pricing
        # registration at 0 lets UCX pick zero-copy.
        "engine_config": (
            "CUDA_COPY_REG_WHOLE_ALLOC=off,RCACHE_ENABLE=n,"
            "IB_MEM_REG_OVERHEAD=0,IB_MEM_REG_GROWTH=0"
        ),
    },
}


@dataclass
class _Registration:
    """Cached NIXL descriptors tied to the lifetime of their backing tensor storage."""

    descriptors: Any
    storage_ref: weakref.ReferenceType[Any]


@dataclass
class _PreparedTransfer:
    agent_cache: NixlAgentCache
    handles: tuple[Any, ...]
    in_use: bool = False


_TransferSignature = tuple[
    tuple[str, tuple[int, int], int, int, tuple[int, ...]], ...
]

_ClientTransferKey = tuple[
    _NixlTransferOperation, str, _TransferSignature
]
_PreparedTransferKey = tuple[str, _NixlTransferOperation, str]


class NixlAgentCache(TransportCache):
    """Process-local NIXL agent and reusable memory registrations."""

    def __init__(self) -> None:
        nixl_agent, nixl_agent_config = _load_nixl_api()

        self.backend = ENV_TORCHSTORE_NIXL_BACKEND
        name = (
            f"torchstore-{socket.gethostname()[:24]}-{os.getpid()}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        config = nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=False,
            backends=[],
            capture_telemetry=True,
        )
        self.agent = nixl_agent(name, config)
        self.agent.create_backend(self.backend, _BACKEND_PARAMS.get(self.backend, {}))
        self._registrations: dict[tuple[int, int], _Registration] = {}
        # Uncached registrations: storage key -> [descriptors, holders].
        self._leases: dict[tuple[int, int], list[Any]] = {}
        # Storage key -> cancelled gets whose storage volume may still write into it.
        self._abandoned: Counter[tuple[int, int]] = Counter()
        self._remote_agents: dict[str, bytes] = {}
        self._client_transfer_ids: dict[_ClientTransferKey, str] = {}
        self._prepared_transfers: dict[_PreparedTransferKey, _PreparedTransfer] = {}

    def register(self, tensor: torch.Tensor) -> Any:
        """Register a tensor's whole storage once for the lifetime of that storage.

        Transfers build descriptors from the tensor itself, and NIXL accepts any
        descriptor inside a registered region, so one registration covers every
        view into the storage. Resharded gets write many views into one
        destination parameter (5.6k views over 1.5k parameters for Qwen3.5-27B
        pulled from FSDP4), and each registration costs ~1.8 ms over UCX/RoCE.
        """
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        existing = self._registrations.get(key)
        if existing is not None:
            return existing.descriptors
        # Two registrations of the same range break this agent's metadata for
        # peers (add_remote_agent fails with NIXL_ERR_NOT_ALLOWED). Not cached:
        # the registration ends with the lease.
        lease = self._leases.get(key)
        if lease is not None:
            return lease[0]

        # Not kept: a tensor over the storage would keep the storage alive.
        region = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(storage)
        descriptors = self.agent.register_memory(region, backends=[self.backend])
        storage_ref = weakref.ref(storage, lambda _ref, _key=key: self._evict(_key))
        self._registrations[key] = _Registration(descriptors, storage_ref)
        return descriptors

    def acquire(self, tensor: torch.Tensor) -> tuple[int, int]:
        """Register a tensor's storage until every ``acquire`` is released.

        For CUDA get destinations: ``register`` drops registrations only when the
        storage dies, but FSDP frees and reallocates parameter storage in place
        with ``resize_``, which would leave a registration over freed memory.
        Concurrent buffers of one get share a registration rather than
        registering the storage again.
        """
        key = _storage_key(tensor)
        if self._abandoned[key]:
            raise RuntimeError(
                "NIXL get destination is still registered for a cancelled get "
                "the storage volume has not answered"
            )
        lease = self._leases.get(key)
        if lease is not None:
            lease[1] += 1
            return key
        # Two registrations of the same range break this agent's metadata for
        # peers, and a cached one could outlive this storage's memory.
        self._evict(key)
        region = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
            tensor.untyped_storage()
        )
        descriptors = self.agent.register_memory(region, backends=[self.backend])
        self._leases[key] = [descriptors, 1]
        return key

    def release(self, key: tuple[int, int]) -> None:
        lease = self._leases[key]
        lease[1] -= 1
        if lease[1] == 0:
            del self._leases[key]
            self.agent.deregister_memory(lease[0])

    def release_when_answered(
        self,
        keys: set[tuple[int, int]],
        call: asyncio.Future[Any],
        tensors: list[torch.Tensor],
    ) -> None:
        """Release a cancelled get's leases once its storage volume call finishes.

        Until then the volume may still write into them, so ``acquire`` refuses
        these storages instead of sharing a registration over memory FSDP may
        have reallocated, and ``tensors`` stay referenced.
        """
        self._abandoned.update(keys)

        def release(answer: asyncio.Future[Any]) -> None:
            del tensors[:]
            if answer.cancelled():
                # The volume may still be writing; leak the registrations.
                return
            answer.exception()
            self._abandoned.subtract(keys)
            self._abandoned = +self._abandoned
            # Releases every lease even if one fails.
            with ExitStack() as stack:
                for key in keys:
                    stack.callback(self.release, key)

        call.add_done_callback(release)

    def _evict(self, key: tuple[int, int]) -> None:
        registration = self._registrations.get(key)
        if registration is None:
            return
        self.agent.deregister_memory(registration.descriptors)
        self._registrations.pop(key, None)
        for cache_key in list(self._client_transfer_ids):
            if any(item[1] == key for item in cache_key[2]):
                self._client_transfer_ids.pop(cache_key)

    def add_remote_agent(self, name: str, metadata: bytes) -> str:
        """Load new remote metadata, replacing an older copy when needed."""
        previous_metadata = self._remote_agents.get(name)
        if previous_metadata == metadata:
            return name
        if previous_metadata is not None:
            for cache_key in list(self._prepared_transfers):
                if cache_key[0] == name:
                    self._release_prepared_transfer(cache_key)
            self.agent.remove_remote_agent(name)
        remote_name = self.agent.add_remote_agent(metadata)
        if isinstance(remote_name, bytes):
            remote_name = remote_name.decode()
        if remote_name != name:
            raise RuntimeError(
                f"NIXL metadata named agent {remote_name!r}, expected {name!r}"
            )
        self._remote_agents[name] = metadata
        return remote_name

    def _release_prepared_transfer(self, cache_key: _PreparedTransferKey) -> None:
        transfer = self._prepared_transfers.pop(cache_key, None)
        if transfer is None:
            return
        if transfer.in_use:
            self._prepared_transfers[cache_key] = transfer
            raise RuntimeError("cannot replace an active NIXL transfer")
        for handle in transfer.handles:
            transfer.agent_cache.agent.release_xfer_handle(handle)
        transfer.agent_cache.clear()

    @override
    def clear(self) -> None:
        for cache_key in list(self._prepared_transfers):
            self._release_prepared_transfer(cache_key)
        self._client_transfer_ids.clear()
        for key in list(self._registrations):
            self._evict(key)
        for remote_name in list(self._remote_agents):
            self.agent.remove_remote_agent(remote_name)
            self._remote_agents.pop(remote_name)


@dataclass
class NixlRequestContext:
    """Serializable state for one tensor or object in a request batch."""

    remote_descriptors: bytes | None = None
    # Client-local; stripped from RPC state because NIXL transfers its data directly.
    tensor: torch.Tensor | None = None
    shape: torch.Size | None = None
    dtype: torch.dtype | None = None
    is_object: bool = False
    objects: Any = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["tensor"] = None
        return state


class NixlTransportBuffer(TransportBuffer):
    """Move StorageVolume tensors with NIXL's UCX backend."""

    supports_batch_puts = True
    supports_batch_gets = True
    supports_noncontiguous_inplace_resharding = True

    def __init__(self, storage_volume_ref: StorageVolumeRef) -> None:
        super().__init__(storage_volume_ref)
        if not nixl_available():
            raise RuntimeError(
                "NIXL transport is unavailable. Install nixl and set "
                "TORCHSTORE_NIXL_ENABLED=1."
            )
        _load_nixl_api()
        self._client_metadata: bytes | None = None
        self._client_agent_name: str | None = None
        self._contexts: list[NixlRequestContext] = []
        # Storage keys of this request's CUDA destinations acquired from the cache.
        self._leases: set[tuple[int, int]] = set()
        self._operation: _NixlTransferOperation | None = None
        self._signature: _TransferSignature | None = None
        self._transfer_id: str | None = None
        self._reuse_prepared = False
        self._profile_prepared_layout_reused = False
        self._profile_client_prepare_seconds = 0.0
        self._profile_server_prepare_seconds = 0.0
        self._profile_completion_wait_seconds = 0.0
        self._profile_request_id = uuid.uuid4().hex[:12]
        self._profile_client_host = socket.gethostname()
        self._profile_server_host = ""
        self._profile_volume_id = ""
        self._profile_transfer_bytes = 0
        self._profile_timeline_ns: dict[str, int] = {}
        self._profile_request_pickle_ns = 0
        self._profile_request_unpickle_ns = 0
        self._profile_request_frame_bytes = 0
        self._profile_response_unpickle_ns = 0
        self._profile_response_frame_bytes = 0
        self._profile_requester_nic: dict[str, int] = {}
        self._profile_publisher_nic: dict[str, int] = {}
        self._profile_nixl_telemetry: list[dict[str, int | str]] = []

    @override
    def _profile_event(self, event: str, **metadata: Any) -> None:
        self._profile_timeline_ns[event] = time.time_ns()
        self._profile_server_host = metadata.get(
            "server_host", self._profile_server_host
        )
        self._profile_volume_id = metadata.get("volume_id", self._profile_volume_id)
        if event == "client_request_sent":
            _NIC_COUNTER_SAMPLER.begin(self._profile_request_id, "requester")
        elif event == "client_response_received":
            self._profile_requester_nic = _NIC_COUNTER_SAMPLER.end(
                self._profile_request_id,
                "requester",
            )

    def __getstate__(self) -> dict[str, Any]:
        event = (
            "server_response_pickle_started"
            if "server_request_received" in self._profile_timeline_ns
            else "client_request_pickle_started"
        )
        self._profile_event(event)
        state = super().__getstate__()
        state["_leases"] = set()
        if self._reuse_prepared:
            state["_contexts"] = []
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        event = (
            "client_response_unpickle_finished"
            if "server_request_received" in self._profile_timeline_ns
            else "server_request_unpickle_finished"
        )
        self._profile_event(event)

    def _client_cache(self) -> NixlAgentCache:
        return self.storage_volume_ref.transport_context.get(NixlAgentCache)

    def _register_client_tensor(
        self, tensor: torch.Tensor, *, cached: bool = True
    ) -> NixlRequestContext:
        context = NixlRequestContext(
            tensor=tensor,
            shape=tensor.shape,
            dtype=tensor.dtype,
        )
        # NIXL cannot create transfer descriptors for zero-byte tensors.
        if tensor.numel() == 0:
            return context

        cache = self._client_cache()
        if cached:
            cache.register(tensor)
        elif _storage_key(tensor) not in self._leases:
            self._leases.add(cache.acquire(tensor))
        segments = _contiguous_segments(tensor)
        descriptors = cache.agent.get_xfer_descs(
            segments if len(segments) > 1 else tensor
        )
        context.remote_descriptors = cache.agent.get_serialized_descs(descriptors)
        return context

    def _publish_client_metadata(self) -> None:
        if any(context.remote_descriptors is not None for context in self._contexts):
            cache = self._client_cache()
            self._client_agent_name = cache.agent.name
            self._client_metadata = cache.agent.get_agent_metadata()

    def _transfer_signature(
        self, requests: list[Request]
    ) -> _TransferSignature | None:
        signature = []
        has_data = False
        for request in requests:
            tensor = request.tensor_val
            if (
                request.is_object
                or request.requester_id is None
                or tensor is None
                or tensor.is_cuda
            ):
                return None
            if tensor.numel() != 0:
                has_data = True
            signature.append(
                (
                    request.key,
                    _storage_key(tensor),
                    tensor.data_ptr(),
                    tensor.nbytes,
                    tensor.stride(),
                )
            )
        return tuple(signature) if has_data else None

    def _prepare_transfer(
        self,
        requests: list[Request],
        operation: _NixlTransferOperation,
    ) -> bool:
        self._operation = operation
        self._signature = self._transfer_signature(requests)
        if self._signature is None:
            self._transfer_id = None
            self._reuse_prepared = False
            return False

        cache_key = (
            operation,
            self.storage_volume_ref.volume_id,
            self._signature,
        )
        self._transfer_id = self._client_cache()._client_transfer_ids.get(cache_key)
        self._reuse_prepared = self._transfer_id is not None
        if self._transfer_id is None:
            self._transfer_id = uuid.uuid4().hex
        return self._reuse_prepared

    @override
    async def _post_request_success(self) -> None:
        if (
            self._operation is not None
            and self._signature is not None
            and self._transfer_id is not None
        ):
            cache_key = (
                self._operation,
                self.storage_volume_ref.volume_id,
                self._signature,
            )
            self._client_cache()._client_transfer_ids[cache_key] = self._transfer_id

    @override
    async def _pre_put_hook(self, requests: list[Request]) -> None:
        """Allocate RDMA memory for put (transport already set up)."""
        if self._prepare_transfer(requests, _NixlTransferOperation.READ):
            self._contexts = []
            self._client_agent_name = self._client_cache().agent.name
            self._client_metadata = None
            return

        self._contexts = []
        for request in requests:
            if request.is_object:
                self._contexts.append(
                    NixlRequestContext(is_object=True, objects=request.objects)
                )
                continue

            tensor = request.tensor_val
            assert tensor is not None
            self._contexts.append(self._register_client_tensor(tensor))
        self._publish_client_metadata()

    @override
    async def _pre_get_hook(self, requests: list[Request]) -> None:
        """Fetch metadata if needed and allocate RDMA buffers."""
        self._profile_event("client_prepare_started")
        started = time.perf_counter()
        self._profile_transfer_bytes = sum(
            request.tensor_val.nbytes
            for request in requests
            if request.tensor_val is not None
        )
        if self._prepare_transfer(requests, _NixlTransferOperation.WRITE):
            self._contexts = []
            for request in requests:
                tensor = request.tensor_val
                assert tensor is not None
                self._contexts.append(
                    NixlRequestContext(
                        tensor=tensor,
                        shape=tensor.shape,
                        dtype=tensor.dtype,
                    )
                )
            self._client_agent_name = self._client_cache().agent.name
            self._client_metadata = None
            self._profile_client_prepare_seconds = time.perf_counter() - started
            self._profile_event("client_prepare_finished")
            return

        # 1. fetch metadata in a single batch, preserving order
        meta_requests = [req.meta_only() for req in requests if req.tensor_val is None]
        meta_results = (
            await self.storage_volume_ref.volume.get_meta.call_one(meta_requests)
            if meta_requests
            else []
        )
        meta_iterator = iter(meta_results)

        # 2. build contexts
        self._contexts = []
        for request in requests:
            tensor = request.tensor_val
            if tensor is None:
                meta = next(meta_iterator)
                if isinstance(meta, str) or meta is None:
                    self._contexts.append(NixlRequestContext(is_object=True))
                    continue
                shape, dtype = meta
                if request.tensor_slice is not None:
                    shape = request.tensor_slice.local_shape
                tensor = torch.empty(shape, dtype=dtype, device="cpu")

            self._contexts.append(
                self._register_client_tensor(tensor, cached=not tensor.is_cuda)
            )
        self._publish_client_metadata()
        self._profile_client_prepare_seconds = time.perf_counter() - started
        self._profile_event("client_prepare_finished")

    async def _transfer(
        self,
        ctx: TransportContext,
        operation: _NixlTransferOperation,
        transfers: list[tuple[Request, torch.Tensor, bytes | None]],
        *,
        agent_cache: NixlAgentCache | None = None,
        prepared_handles: tuple[Any, ...] | None = None,
        keep_handles: bool = False,
    ) -> tuple[Any, ...]:
        if not transfers and prepared_handles is None:
            return ()

        self._profile_event("server_transfer_prepare_started")
        if transfers:
            self._profile_transfer_bytes = sum(
                tensor.nbytes for _, tensor, _ in transfers
            )
        started = time.perf_counter()
        cache = agent_cache or ctx.get(NixlAgentCache)
        remote_agent = self._connect_client(cache) if prepared_handles is None else None
        self._profile_prepared_layout_reused = prepared_handles is not None
        transfer_groups: dict[tuple[Any, Any], tuple[Any, Any]] = {}
        handles = list(prepared_handles or ())
        dispatch_error: Exception | None = None
        completed = False
        completion_started = started
        nic_active = _NIC_COUNTER_SAMPLER.begin(self._profile_request_id, "publisher")
        nic_baseline: _NicCounterSnapshot | None = None
        try:
            for _, tensor, serialized_remote_descs in (
                transfers if prepared_handles is None else ()
            ):
                assert serialized_remote_descs is not None
                cache.register(tensor)
                segments = _contiguous_segments(tensor)
                tensor_local_descs = cache.agent.get_xfer_descs(
                    segments if len(segments) > 1 else tensor
                )
                tensor_remote_descs = cache.agent.deserialize_descs(
                    serialized_remote_descs
                )
                tensor_local_descs, tensor_remote_descs = _align_descriptors(
                    cache.agent, tensor_local_descs, tensor_remote_descs
                )

                # Each NIXL descriptor list must have one memory type per side.
                # Split mixed CPU/GPU batches while preserving cross-device pairs.
                group_key = (
                    tensor_local_descs.getType(),
                    tensor_remote_descs.getType(),
                )
                group = transfer_groups.get(group_key)
                if group is None:
                    transfer_groups[group_key] = (
                        tensor_local_descs,
                        tensor_remote_descs,
                    )
                else:
                    local_descs, remote_descs = group
                    for index in range(tensor_local_descs.descCount()):
                        local_descs.append(tensor_local_descs[index])
                    for index in range(tensor_remote_descs.descCount()):
                        remote_descs.append(tensor_remote_descs[index])

            try:
                if nic_active:
                    nic_baseline = _NIC_COUNTER_SAMPLER.mark()
                for handle in handles:
                    cache.agent.transfer(handle)
                for local_descs, remote_descs in transfer_groups.values():
                    handle = cache.agent.initialize_xfer(
                        operation.value,
                        local_descs,
                        remote_descs,
                        remote_agent,
                        backends=[cache.backend],
                    )
                    handles.append(handle)
                    cache.agent.transfer(handle)
                completion_started = time.perf_counter()
                self._profile_server_prepare_seconds = completion_started - started
                self._profile_event("server_transfer_submitted")
            except Exception as error:
                # Earlier handles may already be active. Drain them below before
                # propagating the dispatch error.
                dispatch_error = error

            deadline = time.monotonic() + ENV_TORCHSTORE_NIXL_TIMEOUT_S
            timed_out = False
            cancelled = False
            failed = False
            check_error: Exception | None = None

            # All handles are already in flight, so waiting for them one at a time
            # does not serialize the transfers. NIXL has no cancellation API, so a
            # timeout is reported only after active operations reach a terminal
            # state. Returning earlier could free registered tensor storage while
            # DMA is still using it.
            for handle in handles:
                while True:
                    try:
                        status = cache.agent.check_xfer_state(handle)
                    except Exception as error:
                        # A status-query failure does not establish that the DMA has
                        # stopped. Remember the error, but keep polling so registered
                        # storage and the transfer handle remain alive until terminal.
                        if check_error is None:
                            check_error = error
                        status = _NixlTransferStatus.PROCESSING
                    if status != _NixlTransferStatus.PROCESSING:
                        break
                    timed_out |= time.monotonic() >= deadline
                    try:
                        await asyncio.sleep(0.001)
                    except asyncio.CancelledError:
                        # Keep polling so the RPC retains both sides' tensor storage.
                        cancelled = True
                if status != _NixlTransferStatus.DONE:
                    failed = True
            self._profile_completion_wait_seconds = (
                time.perf_counter() - completion_started
            )
            self._profile_event("server_transfer_completed")
            self._profile_nixl_telemetry = []
            for handle in handles:
                try:
                    telemetry = cache.agent.get_xfer_telemetry(handle)
                    self._profile_nixl_telemetry.append(
                        {
                            "backend": cache.agent.query_xfer_backend(handle),
                            "start_time_us": int(telemetry.startTime),
                            "post_duration_us": int(telemetry.postDuration),
                            "xfer_duration_us": int(telemetry.xferDuration),
                            "total_bytes": int(telemetry.totalBytes),
                            "desc_count": int(telemetry.descCount),
                        }
                    )
                except Exception:
                    pass

            if cancelled:
                raise asyncio.CancelledError
            if dispatch_error is not None:
                raise dispatch_error
            if check_error is not None:
                raise check_error
            if failed:
                raise RuntimeError("NIXL transfer failed")
            if timed_out:
                raise TimeoutError("NIXL transfer timed out")
            completed = True
        except Exception as error:
            raise RuntimeError(
                f"NIXL {operation.value} failed for keys="
                f"{[request.key for request, _, _ in transfers]!r}"
            ) from error
        finally:
            if nic_active:
                self._profile_publisher_nic = _NIC_COUNTER_SAMPLER.end(
                    self._profile_request_id,
                    "publisher",
                    nic_baseline,
                )
            # Handles are released only after their transfers reach a terminal
            # state; NIXL release is assumed safe at that point.
            if not keep_handles or (prepared_handles is None and not completed):
                for handle in handles:
                    cache.agent.release_xfer_handle(handle)
        return tuple(handles)

    def _connect_client(self, cache: NixlAgentCache) -> str:
        if self._client_agent_name is None or self._client_metadata is None:
            raise RuntimeError("NIXL request is missing client agent metadata")
        return cache.add_remote_agent(self._client_agent_name, self._client_metadata)

    @override
    async def handle_put_request(
        self,
        ctx: TransportContext,
        entries: list[tuple[Request, Any]],
    ) -> list[Any]:
        """Called by storage volume. Read from client's source RdmaMemory (put)."""
        if not entries:
            await self._replay_prepared_transfer(ctx, _NixlTransferOperation.READ)
            return []

        results: list[Any] = []
        transfers: list[tuple[Request, torch.Tensor, bytes]] = []

        for (request, tensor), request_context in zip(
            entries, self._contexts, strict=True
        ):
            if request_context.is_object:
                results.append(request_context.objects)
                continue

            if tensor is None:
                tensor = torch.empty(
                    request_context.shape,
                    dtype=request_context.dtype,
                    device="cpu",
                )
            self._assert_valid_tensor(
                tensor, request_context.dtype, request_context.shape
            )
            # Empty tensors have no bytes to move or NIXL descriptors to transfer.
            if tensor.numel() != 0:
                assert request_context.remote_descriptors is not None
                transfers.append((request, tensor, request_context.remote_descriptors))
            results.append(tensor)
        if self._transfer_id is not None:
            await self._cache_prepared_transfer(
                ctx, _NixlTransferOperation.READ, transfers
            )
        else:
            await self._transfer(ctx, _NixlTransferOperation.READ, transfers)
        return results

    @override
    async def handle_get_request(
        self,
        ctx: TransportContext,
        entries: list[tuple[Request, Any]],
    ) -> None:
        """Called by storage volume. Write to client's dest RdmaMemory (get)."""
        if not entries:
            await self._replay_prepared_transfer(ctx, _NixlTransferOperation.WRITE)
            return

        transfers: list[tuple[Request, torch.Tensor, bytes]] = []
        staged: list[tuple[Request, torch.Tensor, bytes]] = []

        for (request, data), request_context in zip(
            entries, self._contexts, strict=True
        ):
            if not isinstance(data, torch.Tensor):
                request_context.is_object = True
                request_context.objects = data
                continue

            self._assert_valid_tensor(
                data,
                request_context.dtype,
                request_context.shape,
                must_be_contiguous=False,
            )
            # Empty tensors have no bytes to move or NIXL descriptors to transfer.
            if data.numel() == 0:
                continue
            assert request_context.remote_descriptors is not None
            if data.is_contiguous() or (
                self._transfer_id is not None and data.device.type == "cpu"
            ):
                transfers.append((request, data, request_context.remote_descriptors))
            elif data.device.type == "cpu":
                staged.append((request, data, request_context.remote_descriptors))
            else:
                transfers.append(
                    (request, data.contiguous(), request_context.remote_descriptors)
                )

        if self._transfer_id is not None and not staged:
            await self._cache_prepared_transfer(
                ctx, _NixlTransferOperation.WRITE, transfers
            )
            return

        self._signature = None
        self._transfer_id = None

        tasks = [
            asyncio.create_task(
                self._transfer(ctx, _NixlTransferOperation.WRITE, transfers)
            )
        ]
        try:
            # Start the direct transfer before staging.
            await asyncio.sleep(0)
            await self._write_staged(ctx, staged, tasks)
            await asyncio.gather(*tasks)
        finally:
            await _wait_terminal(tasks)

    async def _cache_prepared_transfer(
        self,
        ctx: TransportContext,
        operation: _NixlTransferOperation,
        transfers: list[tuple[Request, torch.Tensor, bytes]],
    ) -> None:
        assert self._transfer_id is not None
        assert self._client_agent_name is not None
        cache = ctx.get(NixlAgentCache)
        cache_key = (self._client_agent_name, operation, self._transfer_id)
        cache._release_prepared_transfer(cache_key)
        transfer_cache = NixlAgentCache()
        try:
            handles = await self._transfer(
                ctx,
                operation,
                transfers,
                agent_cache=transfer_cache,
                keep_handles=True,
            )
        except BaseException:
            transfer_cache.clear()
            raise
        cache._prepared_transfers[cache_key] = _PreparedTransfer(
            transfer_cache, handles
        )

    async def _replay_prepared_transfer(
        self,
        ctx: TransportContext,
        operation: _NixlTransferOperation,
    ) -> None:
        if self._transfer_id is None or self._client_agent_name is None:
            raise RuntimeError("NIXL request has no prepared transfer ID")
        prepared = ctx.get(NixlAgentCache)._prepared_transfers.get(
            (self._client_agent_name, operation, self._transfer_id)
        )
        if prepared is None:
            raise RuntimeError("NIXL prepared transfer is not registered")
        if prepared.in_use:
            raise RuntimeError("concurrent use of a registered NIXL transfer")

        prepared.in_use = True
        try:
            await self._transfer(
                ctx,
                operation,
                [],
                agent_cache=prepared.agent_cache,
                prepared_handles=prepared.handles,
                keep_handles=True,
            )
        finally:
            prepared.in_use = False
        self._contexts = []

    async def _write_staged(
        self,
        ctx: TransportContext,
        staged: list[tuple[Request, torch.Tensor, bytes]],
        tasks: list[asyncio.Task[Any]],
    ) -> None:
        """Write non-contiguous CPU tensors through two reused staging buffers.

        Copying every non-contiguous tensor with .contiguous() before the transfer
        doubles peak memory for a large batch. Instead, pack row ranges into one
        buffer while the other buffer's transfer is in flight.

        Started transfer tasks are appended to ``tasks``; the caller must wait for
        all of them to finish, even if this raises or is cancelled.
        """
        if not staged:
            return
        if ENV_TORCHSTORE_NIXL_STAGING_BUFFER_BYTES < _STAGING_ALIGNMENT:
            raise ValueError(
                "TORCHSTORE_NIXL_STAGING_BUFFER_BYTES must be at least "
                f"{_STAGING_ALIGNMENT}, got {ENV_TORCHSTORE_NIXL_STAGING_BUFFER_BYTES}"
            )

        cache = ctx.get(NixlAgentCache)
        memory_type_names = {
            cache.agent.nixl_mems[name]: name for name in ("DRAM", "VRAM")
        }
        size = min(
            ENV_TORCHSTORE_NIXL_STAGING_BUFFER_BYTES,
            sum(_aligned(tensor.nbytes) for _, tensor, _ in staged),
        )
        buffers: list[torch.Tensor | None] = [None, None]
        buffer_tasks: list[asyncio.Task[None] | None] = [None, None]
        index = 0
        offset = 0
        pending: list[tuple[Request, torch.Tensor, bytes | None]] = []

        def flush() -> None:
            task = asyncio.create_task(
                self._transfer(ctx, _NixlTransferOperation.WRITE, pending.copy())
            )
            tasks.append(task)
            buffer_tasks[index] = task
            pending.clear()

        for request, source, serialized_remote_descs in staged:
            remote_descs = cache.agent.deserialize_descs(serialized_remote_descs)
            assert remote_descs.descCount() == 1
            remote_addr, _, remote_device = remote_descs[0]
            remote_type = memory_type_names[remote_descs.getType()]
            remote_offset = 0
            for piece in _staging_pieces(source, size):
                start = _aligned(offset)
                if buffers[index] is None or start + piece.nbytes > size:
                    if pending:
                        flush()
                        # Let the transfer start before filling the other buffer.
                        await asyncio.sleep(0)
                        index ^= 1
                    if buffer_tasks[index] is not None:
                        await buffer_tasks[index]
                    if buffers[index] is None:
                        buffers[index] = torch.empty(
                            size, dtype=torch.uint8, device="cpu"
                        )
                    start = 0
                buffer = buffers[index]
                assert buffer is not None
                local = buffer[start : start + piece.nbytes]
                local.view(piece.dtype).view(piece.shape).copy_(piece)
                remote = cache.agent.get_xfer_descs(
                    [(remote_addr + remote_offset, piece.nbytes, remote_device)],
                    remote_type,
                )
                pending.append(
                    (request, local, cache.agent.get_serialized_descs(remote))
                )
                offset = start + piece.nbytes
                remote_offset += piece.nbytes
        flush()

    @override
    async def _call_volume_put(self, meta_requests: list[Request]) -> None:
        for request in meta_requests:
            request.objects = None
            if request.requester_id is not None:
                request.tensor_slice = None
        if self._reuse_prepared:
            meta_requests = []
        await super()._call_volume_put(meta_requests)

    @override
    async def _call_volume_get(self, meta_requests: list[Request]) -> TransportBuffer:
        for request in meta_requests:
            request.objects = None
            if request.requester_id is not None:
                request.tensor_slice = None
        if self._reuse_prepared:
            meta_requests = []
        if not self._leases:
            return await super()._call_volume_get(meta_requests)
        call = asyncio.ensure_future(super()._call_volume_get(meta_requests))
        try:
            return await asyncio.shield(call)
        except asyncio.CancelledError:
            await self._wait_for_volume(call)
            raise

    async def _wait_for_volume(self, call: asyncio.Future[TransportBuffer]) -> None:
        """Keep a cancelled get's destinations registered while the volume writes.

        Waits for the volume's answer so drop() deregisters them and the caller
        reuses them only afterwards. On a timeout or another cancel, hands the
        leases to the cache to release once the volume answers.
        """
        try:
            await asyncio.wait([call], timeout=ENV_TORCHSTORE_NIXL_TIMEOUT_S)
        finally:
            if not call.done():
                logger.warning(
                    "NIXL get cancelled while the storage volume may still write "
                    "into its destinations; they stay registered until it answers"
                )
                tensors = [
                    context.tensor
                    for context in self._contexts
                    if context.tensor is not None
                ]
                leases, self._leases = self._leases, set()
                self._client_cache().release_when_answered(leases, call, tensors)
            elif not call.cancelled():
                call.exception()

    @override
    async def _handle_storage_volume_response(
        self,
        requests: list[Request],
        transport_buffer: TransportBuffer,
    ) -> list[Any]:
        assert isinstance(transport_buffer, NixlTransportBuffer)
        if transport_buffer._transfer_id is None:
            self._signature = None
            self._transfer_id = None
        client_response_received = self._profile_timeline_ns.get(
            "client_response_received"
        )
        self._profile_timeline_ns.update(transport_buffer._profile_timeline_ns)
        if client_response_received is not None:
            self._profile_timeline_ns["client_response_received"] = (
                client_response_received
            )
        self._profile_server_host = transport_buffer._profile_server_host
        self._profile_volume_id = transport_buffer._profile_volume_id
        self._profile_transfer_bytes = transport_buffer._profile_transfer_bytes
        self._profile_server_prepare_seconds = (
            transport_buffer._profile_server_prepare_seconds
        )
        self._profile_completion_wait_seconds = (
            transport_buffer._profile_completion_wait_seconds
        )
        self._profile_request_unpickle_ns = (
            transport_buffer._profile_request_unpickle_ns
        )
        self._profile_response_unpickle_ns = (
            transport_buffer._profile_response_unpickle_ns
        )
        self._profile_response_frame_bytes = (
            transport_buffer._profile_response_frame_bytes
        )
        self._profile_publisher_nic = transport_buffer._profile_publisher_nic
        self._profile_nixl_telemetry = transport_buffer._profile_nixl_telemetry
        self._profile_prepared_layout_reused = (
            transport_buffer._profile_prepared_layout_reused
        )
        timeline = self._profile_timeline_ns
        logging.info(
            "NIXL_GET_TIMELINE request_id=%s volume_id=%s bytes=%d "
            "transfer_id=%s compact_request=%d prepared_layout_reused=%d "
            "client_host=%s server_host=%s client_prepare_started_ns=%d "
            "client_prepare_finished_ns=%d client_request_sent_ns=%d "
            "client_request_pickle_started_ns=%d "
            "server_request_unpickle_finished_ns=%d "
            "server_request_received_ns=%d server_data_ready_ns=%d "
            "server_transfer_prepare_started_ns=%d "
            "server_transfer_submitted_ns=%d server_transfer_completed_ns=%d "
            "server_response_sent_ns=%d server_response_pickle_started_ns=%d "
            "client_response_unpickle_finished_ns=%d "
            "client_response_received_ns=%d "
            "monarch_request_pickle_ns=%d monarch_request_unpickle_ns=%d "
            "monarch_request_frame_bytes=%d "
            "monarch_response_unpickle_ns=%d monarch_response_frame_bytes=%d "
            "requester_nic=%s publisher_nic=%s nixl_telemetry=%s",
            self._profile_request_id,
            self._profile_volume_id,
            self._profile_transfer_bytes,
            self._transfer_id or "-",
            self._reuse_prepared,
            self._profile_prepared_layout_reused,
            self._profile_client_host,
            self._profile_server_host,
            timeline.get("client_prepare_started", 0),
            timeline.get("client_prepare_finished", 0),
            timeline.get("client_request_sent", 0),
            timeline.get("client_request_pickle_started", 0),
            timeline.get("server_request_unpickle_finished", 0),
            timeline.get("server_request_received", 0),
            timeline.get("server_data_ready", 0),
            timeline.get("server_transfer_prepare_started", 0),
            timeline.get("server_transfer_submitted", 0),
            timeline.get("server_transfer_completed", 0),
            timeline.get("server_response_sent", 0),
            timeline.get("server_response_pickle_started", 0),
            timeline.get("client_response_unpickle_finished", 0),
            timeline.get("client_response_received", 0),
            self._profile_request_pickle_ns,
            self._profile_request_unpickle_ns,
            self._profile_request_frame_bytes,
            self._profile_response_unpickle_ns,
            self._profile_response_frame_bytes,
            json.dumps(self._profile_requester_nic, separators=(",", ":")),
            json.dumps(self._profile_publisher_nic, separators=(",", ":")),
            json.dumps(self._profile_nixl_telemetry, separators=(",", ":")),
        )
        if self._reuse_prepared:
            return [context.tensor for context in self._contexts]
        results: list[Any] = []
        for client_context, volume_context in zip(
            self._contexts, transport_buffer._contexts, strict=True
        ):
            results.append(
                volume_context.objects
                if volume_context.is_object
                else client_context.tensor
            )
        return results

    async def drop(self) -> None:
        self._client_metadata = None
        self._contexts = []
        self._operation = None
        self._signature = None
        self._transfer_id = None
        self._reuse_prepared = False
        leases, self._leases = self._leases, set()
        if not leases:
            return
        cache = self._client_cache()
        # Releases every lease even if one fails.
        with ExitStack() as stack:
            for key in leases:
                stack.callback(cache.release, key)


def _find_profiled_buffer(value: Any, depth: int = 0) -> NixlTransportBuffer | None:
    if isinstance(value, NixlTransportBuffer):
        return value
    if depth >= 4:
        return None
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        return None
    for child in children:
        buffer = _find_profiled_buffer(child, depth + 1)
        if buffer is not None:
            return buffer
    return None


def _monarch_rpc_direction(buffer: NixlTransportBuffer) -> str | None:
    timeline = buffer._profile_timeline_ns
    if "server_response_sent" in timeline:
        return "response"
    if "client_request_sent" in timeline:
        return "request"
    return None


def _install_monarch_rpc_profiling() -> None:
    """Measure Monarch codec work and exact serialized frame sizes."""
    try:
        from monarch._src.actor import actor_mesh
    except ImportError:
        return
    if getattr(actor_mesh, "_torchstore_rpc_profiling", False):
        return

    original_pickle = actor_mesh.pickle
    original_pickling_state = actor_mesh.PicklingState

    def profiled_pickle(value: Any, *args: Any, **kwargs: Any) -> Any:
        buffer = _find_profiled_buffer(value)
        direction = _monarch_rpc_direction(buffer) if buffer is not None else None
        started = time.perf_counter_ns()
        state = original_pickle(value, *args, **kwargs)
        finished = time.perf_counter_ns()
        if buffer is not None and direction is not None:
            duration_ns = finished - started
            frame_bytes = len(state.buffer())
            if direction == "request":
                buffer._profile_request_pickle_ns = duration_ns
                buffer._profile_request_frame_bytes = frame_bytes
            elif buffer._profile_transfer_bytes >= 1_000_000_000:
                logging.info(
                    "MONARCH_RESPONSE_PICKLE request_id=%s duration_ns=%d "
                    "frame_bytes=%d",
                    buffer._profile_request_id,
                    duration_ns,
                    frame_bytes,
                )
        return state

    class ProfiledPicklingState:
        def __init__(self, message: Any, *args: Any, **kwargs: Any) -> None:
            self._state = original_pickling_state(message, *args, **kwargs)
            self._frame_bytes = len(message)

        def unpickle(self) -> Any:
            started = time.perf_counter_ns()
            value = self._state.unpickle()
            finished = time.perf_counter_ns()
            buffer = _find_profiled_buffer(value)
            direction = _monarch_rpc_direction(buffer) if buffer is not None else None
            if buffer is not None and direction is not None:
                if direction == "request":
                    buffer._profile_request_unpickle_ns = finished - started
                else:
                    buffer._profile_response_unpickle_ns = finished - started
                    buffer._profile_response_frame_bytes = self._frame_bytes
            return value

        def __getattr__(self, name: str) -> Any:
            return getattr(self._state, name)

    actor_mesh.pickle = profiled_pickle
    actor_mesh.PicklingState = ProfiledPicklingState
    actor_mesh._torchstore_rpc_profiling = True


_install_monarch_rpc_profiling()
