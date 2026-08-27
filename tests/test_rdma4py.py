# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the rdma4py transport using an in-process verbs fake."""

from __future__ import annotations

import gc
import weakref
from enum import IntEnum, IntFlag
from types import SimpleNamespace

import pytest
import torch

import torchstore.transport.rdma4py as rdma4py
from torchstore.transport.buffers import TransportContext
from torchstore.transport.rdma4py import (
    Rdma4PyRequestContext,
    Rdma4PyTransportBuffer,
)
from torchstore.transport.types import Request


class FakeAccessFlags(IntFlag):
    LOCAL_WRITE = 1
    REMOTE_WRITE = 2
    REMOTE_READ = 4


class FakeRdmaAccess(IntFlag):
    NONE = 0
    READ = 1
    WRITE = 2


class FakeWROpcode(IntEnum):
    RDMA_WRITE = 0
    RDMA_READ = 4


class FakeSendFlags(IntFlag):
    SIGNALED = 2


class FakeQPInfo:
    def __init__(self, qp_num: int):
        self.qp_num = qp_num

    def to_bytes(self) -> bytes:
        return self.qp_num.to_bytes(4, "big")

    @classmethod
    def from_bytes(cls, data: bytes) -> "FakeQPInfo":
        return cls(int.from_bytes(data, "big"))


class FakeCompletion:
    def raise_for_status(self) -> None:
        pass


class FakeMR:
    _next_key = 1
    registry: dict[tuple[int, int], "FakeMR"] = {}

    def __init__(self, tensor: torch.Tensor):
        self._tensor_ref = weakref.ref(tensor)
        # Match real rdma4py registrations, which retain their owner until the
        # persistent cache explicitly releases it after installing a weakref.
        self._owner = tensor
        self.addr = tensor.data_ptr()
        self.rkey = FakeMR._next_key
        FakeMR._next_key += 1
        self.closed = False
        FakeMR.registry[(self.addr, self.rkey)] = self

    @property
    def tensor(self):
        tensor = self._tensor_ref()
        if tensor is None:
            raise RuntimeError("fake registered tensor was released")
        return tensor

    def sge(self, length=None, offset=0):
        if length is None:
            length = self.tensor.nbytes - offset
        return SimpleNamespace(owner=self, length=length, offset=offset)

    def release_owner(self):
        self._owner = None

    def close(self) -> None:
        self.closed = True
        FakeMR.registry.pop((self.addr, self.rkey), None)


def _byte_view(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 0:
        tensor = tensor.unsqueeze(0)
    return tensor.view(torch.uint8).flatten()


class FakeCQ:
    def __init__(self):
        self.completions: list[FakeCompletion] = []
        self.closed = False

    def poll(self, count: int):
        results = self.completions[:count]
        del self.completions[:count]
        return results

    def close(self) -> None:
        self.closed = True


class FakeQP:
    _next_num = 1

    def __init__(self, cq: FakeCQ):
        self.qp_num = FakeQP._next_num
        FakeQP._next_num += 1
        self.cq = cq
        self.closed = False

    def post_send(self, work_requests) -> None:
        for work_request in work_requests:
            local = work_request.sg_list[0]
            remote_address = work_request.remote_addr - local.offset
            remote = FakeMR.registry[(remote_address, work_request.rkey)]
            local_bytes = _byte_view(local.owner.tensor)
            remote_bytes = _byte_view(remote.tensor)
            local_slice = local_bytes[local.offset : local.offset + local.length]
            remote_slice = remote_bytes[local.offset : local.offset + local.length]
            if work_request.opcode == FakeWROpcode.RDMA_READ:
                local_slice.copy_(remote_slice)
            else:
                remote_slice.copy_(local_slice)
            self.cq.completions.append(FakeCompletion())

    def close(self) -> None:
        self.closed = True


class FakePD:
    def __init__(self, context):
        self.context = context
        self.closed = False

    def create_qp(self, init_attr):
        return FakeQP(init_attr.send_cq)

    def close(self) -> None:
        self.closed = True


class FakeContext:
    def __init__(self):
        self.closed = False

    def query_device(self):
        return SimpleNamespace(phys_port_cnt=1)

    def query_port(self, port):
        return SimpleNamespace(state=1, gid_tbl_len=1, active_mtu=4, lid=1)

    def query_gid(self, port, index):
        return SimpleNamespace(raw=b"\x01" * 16)

    def alloc_pd(self):
        return FakePD(self)

    def create_cq(self, cqe):
        return FakeCQ()

    def close(self) -> None:
        self.closed = True


class FakeDevice:
    name = "fake0"

    def open(self):
        return FakeContext()


class FakeQPInitAttr:
    def __init__(self, send_cq, recv_cq, **kwargs):
        self.send_cq = send_cq
        self.recv_cq = recv_cq


class FakeSendWR:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeTransferRequest:
    def __init__(self, local_mr, remote_addr, rkey, length, **kwargs):
        self.local_mr = local_mr
        self.remote_addr = remote_addr
        self.rkey = rkey
        self.length = length
        self.local_offset = kwargs.get("local_offset", 0)
        self.remote_offset = kwargs.get("remote_offset", 0)

    @classmethod
    def read(cls, local_mr, remote_memory, **kwargs):
        return cls(
            local_mr,
            remote_memory.address,
            remote_memory.rkey,
            kwargs.pop("length", remote_memory.nbytes),
            **kwargs,
        )

    @classmethod
    def write(cls, local_mr, remote_memory, **kwargs):
        return cls(
            local_mr,
            remote_memory.address,
            remote_memory.rkey,
            kwargs.pop("length", remote_memory.nbytes),
            **kwargs,
        )


class FakeScheduler:
    def __init__(self, context, pd, qps, cqs, chunk_size):
        self.context = context
        self.pd = pd
        self.qps = tuple(qps)
        self.cqs = tuple(cqs)
        self.chunk_size = chunk_size
        self.closed = False
        self.poisoned = False

    @classmethod
    def create(cls, context, pd, *, qp_count, queue_depth, chunk_size):
        cqs = [context.create_cq(queue_depth) for _ in range(qp_count)]
        qps = [pd.create_qp(FakeQPInitAttr(cq, cq)) for cq in cqs]
        return cls(context, pd, qps, cqs, chunk_size)

    def local_infos(self, port_attr, gid, *, port):
        return tuple(FakeQPInfo(qp.qp_num) for qp in self.qps)

    def connect(self, remote_infos, **kwargs):
        assert len(remote_infos) == len(self.qps)

    async def read_many_async(self, requests, *, timeout):
        self._transfer(requests, write=False)

    async def write_many_async(self, requests, *, timeout):
        self._transfer(requests, write=True)

    def _transfer(self, requests, *, write):
        for request in requests:
            remote = FakeMR.registry[(request.remote_addr, request.rkey)]
            local_bytes = _byte_view(request.local_mr.tensor)
            remote_bytes = _byte_view(remote.tensor)
            for offset in range(0, request.length, self.chunk_size):
                length = min(self.chunk_size, request.length - offset)
                local_start = request.local_offset + offset
                remote_start = request.remote_offset + offset
                local_slice = local_bytes[local_start : local_start + length]
                remote_slice = remote_bytes[remote_start : remote_start + length]
                if write:
                    remote_slice.copy_(local_slice)
                else:
                    local_slice.copy_(remote_slice)

    def close(self):
        if self.closed:
            return
        self.closed = True
        for qp in self.qps:
            qp.close()
        for cq in self.cqs:
            cq.close()


class FakeRdmaTransport:
    def __init__(self, factory, *, qp_count, queue_depth, chunk_size):
        self.factory = factory
        self.scheduler = FakeScheduler.create(
            factory.context,
            factory.pd,
            qp_count=qp_count,
            queue_depth=queue_depth,
            chunk_size=chunk_size,
        )
        port_attr = factory.context.query_port(factory.port)
        self.bind_info = tuple(
            info.to_bytes()
            for info in self.scheduler.local_infos(
                port_attr,
                factory.gid,
                port=factory.port,
            )
        )

    def bind(self):
        return self.bind_info[0] if len(self.bind_info) == 1 else self.bind_info

    @property
    def usable(self):
        return not self.scheduler.closed and not self.scheduler.poisoned

    def connect(self, remote_info, *, incoming_access):
        if isinstance(remote_info, bytes):
            remote_info = (remote_info,)
        self.scheduler.connect(
            tuple(FakeQPInfo.from_bytes(info) for info in remote_info),
            access=incoming_access,
        )

    async def read_many_async(self, requests, *, timeout, on_complete=None):
        await self.scheduler.read_many_async(requests, timeout=timeout)
        if on_complete is not None:
            on_complete()

    async def write_many_async(self, requests, *, timeout, on_complete=None):
        await self.scheduler.write_many_async(requests, timeout=timeout)
        if on_complete is not None:
            on_complete()

    def close(self):
        self.scheduler.close()


class FakeRdmaTransportFactory:
    def __init__(self):
        self.context = FakeContext()
        self.pd = self.context.alloc_pd()
        self.device_name = "fake0"
        self.port = 1
        self.gid_index = 0
        self.gid = self.context.query_gid(1, 0)
        self.transports = []

    @classmethod
    def open(cls, **kwargs):
        return cls()

    def create_transport(self, *, qp_count, queue_depth, chunk_size):
        transport = FakeRdmaTransport(
            self,
            qp_count=qp_count,
            queue_depth=queue_depth,
            chunk_size=chunk_size,
        )
        self.transports.append(transport)
        return transport

    def register_tensor(self, tensor, access):
        return FakeMR(tensor)

    def close(self):
        for transport in self.transports:
            transport.close()
        self.transports.clear()
        self.pd.close()
        self.context.close()


class FakeIB:
    AccessFlags = FakeAccessFlags
    RdmaAccess = FakeRdmaAccess
    WROpcode = FakeWROpcode
    SendFlags = FakeSendFlags
    QPType = SimpleNamespace(RC=2)
    PortState = SimpleNamespace(ACTIVE=1)
    QPInfo = FakeQPInfo
    QPInitAttr = FakeQPInitAttr
    SendWR = FakeSendWR
    RdmaReadRequest = FakeTransferRequest
    RdmaWriteRequest = FakeTransferRequest
    RdmaTransferRequest = FakeTransferRequest
    RdmaTransportFactory = FakeRdmaTransportFactory

    @staticmethod
    def release_tensor_owner(registration):
        registration.release_owner()

    @staticmethod
    def get_device_list():
        return [FakeDevice()]

    @staticmethod
    def local_qp_info(qp, port_attr, gid, port):
        return FakeQPInfo(qp.qp_num)

    @staticmethod
    def connect_rc(qp, remote, **kwargs):
        pass

    @staticmethod
    def reg_tensor(pd, tensor, access):
        return FakeMR(tensor)


class FakeEndpoint:
    def __init__(self, callback):
        self.callback = callback

    async def call(self, *args):
        return await self.callback(*args)

    async def call_one(self, *args):
        return await self.callback(*args)


class FakeVolume:
    def __init__(self):
        self.ctx = TransportContext()
        self.values = {}
        self.handshake_calls = 0
        self.handshake = FakeEndpoint(self._handshake)
        self.put = FakeEndpoint(self._put)
        self.get = FakeEndpoint(self._get)
        self.get_meta = FakeEndpoint(self._get_meta)

    async def _handshake(self, transport, requests):
        self.handshake_calls += 1
        entries = [(request, self.values.get(request.key)) for request in requests]
        return await transport.recv_handshake(self.ctx, entries)

    async def _put(self, transport, requests):
        entries = [(request, self.values.get(request.key)) for request in requests]
        values = await transport.handle_put_request(self.ctx, entries)
        for request, value in zip(requests, values, strict=True):
            self.values[request.key] = value

    async def _get(self, transport, requests):
        entries = [(request, self.values[request.key]) for request in requests]
        await transport.handle_get_request(self.ctx, entries)
        return transport

    async def _get_meta(self, requests):
        results = []
        for request in requests:
            value = self.values[request.key]
            results.append(
                (value.shape, value.dtype) if isinstance(value, torch.Tensor) else "obj"
            )
        return results


class FakeStorageVolumeRef:
    def __init__(self):
        self.volume_id = "volume"
        self.volume = FakeVolume()
        self.transport_context = TransportContext()


@pytest.fixture(autouse=True)
def fake_ibverbs(monkeypatch):
    FakeMR._next_key = 1
    FakeMR.registry = {}
    FakeQP._next_num = 1
    for name in (
        "TORCHSTORE_RDMA4PY_DEVICE",
        "TORCHSTORE_RDMA4PY_PORT",
        "TORCHSTORE_RDMA4PY_GID_INDEX",
        "TORCHSTORE_RDMA4PY_QUEUE_DEPTH",
        "TORCHSTORE_RDMA4PY_QP_COUNT",
        "TORCHSTORE_RDMA4PY_CHUNK_SIZE_BYTES",
        "TORCHSTORE_RDMA4PY_TIMEOUT_SECONDS",
        "TORCHSTORE_RDMA4PY_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(rdma4py, "ib", FakeIB)
    rdma4py.rdma4py_transport_available.cache_clear()
    yield
    rdma4py.rdma4py_transport_available.cache_clear()


@pytest.mark.asyncio
async def test_batched_put_and_get_moves_tensors_and_objects(monkeypatch):
    # Force multi-WR tensors and multiple queue-depth waves with tiny test data.
    monkeypatch.setattr(rdma4py, "_MAX_SGE_BYTES", 7)
    monkeypatch.setenv("TORCHSTORE_RDMA4PY_QUEUE_DEPTH", "2")
    monkeypatch.setenv("TORCHSTORE_RDMA4PY_QP_COUNT", "3")
    monkeypatch.setenv("TORCHSTORE_RDMA4PY_CHUNK_SIZE_BYTES", "7")
    ref = FakeStorageVolumeRef()
    source_a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    source_b = torch.arange(5, dtype=torch.int64)
    put_buffer = Rdma4PyTransportBuffer(ref)

    await put_buffer.put_to_storage_volume(
        [
            Request.from_tensor("a", source_a),
            Request.from_objects("object", {"answer": 42}),
            Request.from_tensor("b", source_b),
        ]
    )

    assert torch.equal(ref.volume.values["a"], source_a)
    assert ref.volume.values["object"] == {"answer": 42}
    assert torch.equal(ref.volume.values["b"], source_b)
    assert len(FakeMR.registry) == 4
    assert ref.volume.handshake_calls == 1
    client_endpoint = ref.transport_context.get(
        rdma4py.Rdma4PyConnectionCache
    ).get(put_buffer.peer_key)
    server_endpoint = ref.volume.ctx.get(rdma4py.Rdma4PyConnectionCache).get(
        put_buffer.peer_key
    )
    assert len(client_endpoint.scheduler.qps) == 3
    assert len(server_endpoint.scheduler.qps) == 3

    destination = torch.zeros_like(source_a)
    get_buffer = Rdma4PyTransportBuffer(ref)
    results = await get_buffer.get_from_storage_volume(
        [
            Request.from_tensor("a", destination),
            Request.from_any("object", None),
            Request.from_any("b", None),
        ]
    )

    assert results[0] is destination
    assert torch.equal(results[0], source_a)
    assert results[1] == {"answer": 42}
    assert torch.equal(results[2], source_b)
    assert ref.volume.handshake_calls == 1
    registrations_after_get = len(FakeMR.registry)

    await Rdma4PyTransportBuffer(ref).get_from_storage_volume(
        [Request.from_tensor("a", destination)]
    )

    assert ref.volume.handshake_calls == 1
    assert len(FakeMR.registry) == registrations_after_get
    ref.transport_context.clear()
    ref.volume.ctx.clear()
    assert FakeMR.registry == {}


@pytest.mark.asyncio
async def test_non_contiguous_get_uses_staging_tensor():
    ref = FakeStorageVolumeRef()
    stored = torch.arange(4, dtype=torch.float32)
    ref.volume.values["key"] = stored
    base = torch.zeros(4, 2)
    destination = base[:, 1]
    assert not destination.is_contiguous()

    buffer = Rdma4PyTransportBuffer(ref)
    results = await buffer.get_from_storage_volume(
        [Request.from_tensor("key", destination)]
    )

    assert results[0] is destination
    assert torch.equal(destination, stored)


@pytest.mark.asyncio
async def test_empty_tensor_skips_memory_registration():
    ref = FakeStorageVolumeRef()
    empty = torch.empty(0, dtype=torch.float32)
    buffer = Rdma4PyTransportBuffer(ref)

    await buffer.put_to_storage_volume([Request.from_tensor("empty", empty)])

    assert ref.volume.values["empty"].shape == torch.Size([0])
    assert FakeMR.registry == {}


def test_registration_cache_weakly_evicts_released_storage():
    cache = rdma4py.Rdma4PyConnectionCache()
    endpoint = SimpleNamespace(
        pd=object(),
        factory=SimpleNamespace(register_tensor=lambda tensor, access: FakeMR(tensor)),
    )
    tensor = torch.ones(8)

    registration = cache.get_or_register(
        endpoint, tensor, FakeRdmaAccess.NONE
    )
    assert not registration.closed
    assert cache.get_or_register(
        endpoint, tensor, FakeRdmaAccess.NONE
    ) is registration

    del tensor
    gc.collect()

    assert registration.closed
    assert cache._registrations == {}


def test_cuda_batch_helpers_run_once_per_unique_device(monkeypatch):
    tensors = [
        SimpleNamespace(is_cuda=True, device="cuda:0"),
        SimpleNamespace(is_cuda=True, device="cuda:0"),
        SimpleNamespace(is_cuda=True, device="cuda:1"),
        SimpleNamespace(is_cuda=False, device="cpu"),
    ]
    synchronized = []
    flushed = []

    class FakeStream:
        def __init__(self, device):
            self.device = device

        def synchronize(self):
            synchronized.append(self.device)

    class FakeDeviceContext:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            flushed.append(("enter", self.device))

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(torch.cuda, "current_stream", FakeStream)
    monkeypatch.setattr(torch.cuda, "device", FakeDeviceContext)
    monkeypatch.setattr(
        rdma4py,
        "_cuda_module",
        lambda: SimpleNamespace(
            flush_gpudirect_writes=lambda: flushed.append(("flush", None))
        ),
    )

    rdma4py._synchronize_cuda_sources(tensors)
    rdma4py._flush_cuda_destinations(tensors)

    assert set(synchronized) == {"cuda:0", "cuda:1"}
    assert len(synchronized) == 2
    assert sum(event == "flush" for event, _ in flushed) == 2


@pytest.mark.asyncio
async def test_registration_failure_aborts_server_connection(monkeypatch):
    ref = FakeStorageVolumeRef()
    buffer = Rdma4PyTransportBuffer(ref)

    def fail_registration(*args, **kwargs):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(
        FakeRdmaTransportFactory,
        "register_tensor",
        fail_registration,
    )
    with pytest.raises(RuntimeError, match="registration failed"):
        await buffer.put_to_storage_volume([Request.from_tensor("key", torch.ones(4))])

    cache = ref.volume.ctx.get(rdma4py.Rdma4PyConnectionCache)
    assert cache._connections == {}


def test_request_context_serialization_strips_process_local_state():
    tensor = torch.arange(4)
    context = Rdma4PyRequestContext(
        tensor=tensor,
        result_tensor=tensor,
        memory_region=object(),
        shape=tensor.shape,
        dtype=tensor.dtype,
    )

    state = context.__getstate__()

    assert state["tensor"] is None
    assert state["result_tensor"] is None
    assert state["memory_region"] is None
    assert state["shape"] == tensor.shape


def test_availability_requires_enabled_active_device(monkeypatch):
    assert rdma4py.rdma4py_transport_available()

    monkeypatch.setenv("TORCHSTORE_RDMA4PY_ENABLED", "0")
    rdma4py.rdma4py_transport_available.cache_clear()
    assert not rdma4py.rdma4py_transport_available()
