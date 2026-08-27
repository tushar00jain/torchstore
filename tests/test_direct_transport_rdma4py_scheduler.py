# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from torchstore.transport import rdma4py


class _FakeInfo:
    def __init__(self, value: bytes) -> None:
        self.value = value

    def to_bytes(self) -> bytes:
        return self.value


class _FakeScheduler:
    def __init__(self) -> None:
        self.queue_depth = 0
        self.connect_calls = []
        self.read_calls = []
        self.closed = False

    def local_infos(self, port_attr, gid, *, port):
        assert port_attr == "port-attr"
        assert gid == "gid"
        assert port == 1
        return (_FakeInfo(b"qp-0"), _FakeInfo(b"qp-1"))

    def connect(self, remote_infos, **kwargs) -> None:
        self.connect_calls.append((remote_infos, kwargs))

    async def read_many_async(self, requests, **kwargs):
        requests = tuple(requests)
        self.read_calls.append((requests, kwargs))
        kwargs["on_complete"]()
        await asyncio.sleep(0)

    def close(self) -> None:
        self.closed = True


class _FakeResource:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeContext(_FakeResource):
    def __init__(self) -> None:
        super().__init__()
        self.pd = _FakeResource()

    def alloc_pd(self):
        return self.pd

    def query_port(self, port):
        assert port == 1
        return "port-attr"


def test_rdma4py_scheduler_configuration_preserves_legacy_defaults(monkeypatch):
    monkeypatch.delenv("TORCHSTORE_RDMA4PY_QP_COUNT", raising=False)
    monkeypatch.delenv("TORCHSTORE_RDMA4PY_CHUNK_SIZE_BYTES", raising=False)

    assert rdma4py._qp_count() == 1
    assert rdma4py._read_chunk_size() == (1 << 32) - 1

    monkeypatch.setenv("TORCHSTORE_RDMA4PY_QP_COUNT", "0")
    with pytest.raises(ValueError, match="QP_COUNT"):
        rdma4py._qp_count()

    monkeypatch.setenv("TORCHSTORE_RDMA4PY_CHUNK_SIZE_BYTES", str(1 << 32))
    with pytest.raises(ValueError, match="CHUNK_SIZE_BYTES"):
        rdma4py._read_chunk_size()


def test_rdma4py_endpoint_exchanges_and_connects_every_qp(monkeypatch):
    context = _FakeContext()
    scheduler = _FakeScheduler()
    create_calls = []

    class _Transport:
        def __init__(self, **kwargs):
            create_calls.append(kwargs)
            scheduler.queue_depth = kwargs["queue_depth"]
            self.scheduler = scheduler
            self.bind_info = (b"qp-0", b"qp-1")

        def bind(self):
            return self.bind_info

        @property
        def usable(self):
            return not scheduler.closed

        def connect(self, remote_infos, *, incoming_access):
            scheduler.connect(
                tuple(f"decoded-{value.decode()}" for value in remote_infos),
                port=1,
                sgid_index=3,
                access=incoming_access,
            )

        def close(self):
            scheduler.close()

    class _Factory:
        def __init__(self):
            self.context = context
            self.pd = context.pd
            self.closed = False

        def create_transport(self, **kwargs):
            return _Transport(**kwargs)

        def close(self):
            self.closed = True
            self.pd.close()
            self.context.close()

    factory = _Factory()
    monkeypatch.setattr(rdma4py, "_open_configured_factory", lambda: factory)
    monkeypatch.setattr(rdma4py, "_qp_count", lambda: 2)
    monkeypatch.setattr(rdma4py, "_queue_depth", lambda: 17)
    monkeypatch.setattr(rdma4py, "_read_chunk_size", lambda: 4096)

    endpoint = rdma4py._Rdma4PyEndpoint(17, 2, 4096)

    assert endpoint.local_infos == (b"qp-0", b"qp-1")
    assert create_calls == [
        {"qp_count": 2, "queue_depth": 17, "chunk_size": 4096}
    ]
    endpoint.connect((b"remote-0", b"remote-1"), incoming_access=9)
    assert scheduler.connect_calls == [
        (
            ("decoded-remote-0", "decoded-remote-1"),
            {"port": 1, "sgid_index": 3, "access": 9},
        )
    ]

    endpoint.close()
    assert scheduler.closed
    assert context.pd.closed
    assert context.closed


class _FakeStorage:
    pass


class _FakeTensor:
    is_cuda = True
    device = "cuda:0"

    def __init__(self, address: int, nbytes: int) -> None:
        self._address = address
        self._nbytes = nbytes
        self._storage = _FakeStorage()

    def numel(self) -> int:
        return self._nbytes

    def element_size(self) -> int:
        return 1

    def data_ptr(self) -> int:
        return self._address

    def untyped_storage(self):
        return self._storage


class _FakeRegistration:
    def __init__(self, address: int) -> None:
        self.address = address
        self.closed = False

    def sge(self, length, *, offset=0):
        return (self.address, length, offset)

    def close(self) -> None:
        self.closed = True

    def release_owner(self) -> None:
        pass


@pytest.mark.asyncio
async def test_destination_coalesces_plan_reads_and_flushes_once(monkeypatch):
    scheduler = _FakeScheduler()
    endpoint = SimpleNamespace(
        pd=object(),
        factory=object(),
        scheduler=scheduler,
        local_infos=(b"local",),
        connect=lambda *_args, **_kwargs: None,
        close_scheduler=lambda: None,
        close=lambda: None,
    )

    async def read_many(requests, **kwargs):
        await scheduler.read_many_async(requests, **kwargs)

    endpoint.read_many = read_many
    registrations = []
    flushes = []

    monkeypatch.setattr(
        rdma4py, "_Rdma4PyEndpoint", lambda *_args: endpoint
    )
    monkeypatch.setattr(
        rdma4py,
        "_register_tensor",
        lambda _pd, tensor, _access: registrations.append(
            _FakeRegistration(tensor.data_ptr())
        )
        or registrations[-1],
    )
    monkeypatch.setattr(
        rdma4py, "_flush_cuda_destination", lambda _tensor: flushes.append("flush")
    )

    connection = rdma4py._Rdma4PyDirectPeer()
    first = _FakeTensor(0x1000, 16)
    second = _FakeTensor(0x2000, 32)
    empty = _FakeTensor(0, 0)
    await asyncio.gather(
        connection.read_into(
            rdma4py.RemoteMemory(address=0xA000, rkey=1, nbytes=16), first
        ),
        connection.read_into(
            rdma4py.RemoteMemory(address=0xB000, rkey=2, nbytes=32), second
        ),
        connection.read_into(
            rdma4py.RemoteMemory(address=0, rkey=0, nbytes=0), empty
        ),
    )

    assert len(scheduler.read_calls) == 1
    requests, options = scheduler.read_calls[0]
    assert [request.length for request in requests] == [16, 32]
    assert len(registrations) == 2
    assert flushes == ["flush"]

    await asyncio.gather(
        connection.read_into(
            rdma4py.RemoteMemory(address=0xA000, rkey=1, nbytes=16), first
        ),
        connection.read_into(
            rdma4py.RemoteMemory(address=0xB000, rkey=2, nbytes=32), second
        ),
    )

    assert len(scheduler.read_calls) == 2
    assert len(registrations) == 2
    assert flushes == ["flush", "flush"]

    connection.close()
    assert all(registration.closed for registration in registrations)


def test_rdma4py_zero_byte_source_tensor_needs_no_registration(monkeypatch):
    endpoint = SimpleNamespace(
        pd=object(),
        factory=object(),
        close_scheduler=lambda: None,
        close=lambda: None,
    )
    calls = []
    connection = rdma4py._Rdma4PyDirectPeer.__new__(
        rdma4py._Rdma4PyDirectPeer
    )
    connection._cache = rdma4py.Rdma4PyConnectionCache()
    connection._endpoint = endpoint
    connection._cache.put("direct", endpoint)
    connection._pending_reads = []
    connection._drain_task = None
    monkeypatch.setattr(
        rdma4py,
        "_register_tensor",
        lambda *_args, **_kwargs: calls.append("register"),
    )

    remote = connection.register(_FakeTensor(0, 0))

    assert remote == rdma4py.RemoteMemory(0, 0, 0)
    assert calls == []
    connection.close()


def test_rdma4py_connection_closes_qps_before_mrs_and_pd(monkeypatch):
    events = []
    endpoint = SimpleNamespace(
        pd=object(),
        factory=object(),
        scheduler=_FakeScheduler(),
        local_infos=(b"local",),
        close_scheduler=lambda: events.append("qps"),
        close=lambda: events.append("pd"),
    )
    registration = _FakeRegistration(0x1000)
    registration.close = lambda: events.append("mr")
    monkeypatch.setattr(
        rdma4py, "_Rdma4PyEndpoint", lambda *_args: endpoint
    )

    connection = rdma4py._Rdma4PyDirectPeer()
    registration_key = (id(endpoint.pd), 0x1000, 16, 0)
    connection._cache._registrations[registration_key] = registration
    connection.close()

    assert events == ["qps", "mr", "pd"]


def test_rdma4py_connect_failure_releases_endpoint(monkeypatch):
    events = []

    def fail_connect(*_args, **_kwargs):
        raise ValueError("QP handshake mismatch")

    endpoint = SimpleNamespace(
        pd=object(),
        factory=object(),
        scheduler=_FakeScheduler(),
        local_infos=(b"local",),
        connect=fail_connect,
        close_scheduler=lambda: events.append("qps"),
        close=lambda: events.append("pd"),
    )
    monkeypatch.setattr(
        rdma4py, "_Rdma4PyEndpoint", lambda *_args: endpoint
    )
    connection = rdma4py._Rdma4PyDirectPeer()

    with pytest.raises(ValueError, match="handshake mismatch"):
        connection.connect(b"remote")

    assert events == ["qps", "pd"]
