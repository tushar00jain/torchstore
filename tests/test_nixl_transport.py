# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Focused tests for the NIXL StorageVolume transport lifecycle."""

import pickle

import pytest
import torch

import torchstore.transport.nixl as nixl_transport
from torchstore.transport.buffers import TransportContext
from torchstore.transport.nixl import NixlAgentCache, NixlTransportBuffer
from torchstore.transport.types import Request


class FakeNixlConfig:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class FakeNixlAgent:
    tensors: dict[int, torch.Tensor] = {}
    instances: list["FakeNixlAgent"] = []

    def __init__(self, name, config) -> None:
        self.name = name
        self.config = config
        self.register_calls = 0
        self.released = []
        FakeNixlAgent.instances.append(self)

    def register_memory(self, tensor, *, backends):
        self.register_calls += 1
        descriptor = id(tensor)
        self.tensors[descriptor] = tensor
        return [descriptor]

    def deregister_memory(self, descriptors, *, backends):
        for descriptor in descriptors:
            self.tensors.pop(descriptor, None)

    def get_xfer_descs(self, tensor):
        return [id(tensor)]

    def get_serialized_descs(self, descriptors):
        return pickle.dumps(descriptors)

    def deserialize_descs(self, descriptors):
        return pickle.loads(descriptors)

    def get_agent_metadata(self):
        return self.name.encode()

    def add_remote_agent(self, metadata):
        return metadata.decode()

    def initialize_xfer(
        self,
        operation,
        local_descs,
        remote_descs,
        remote_agent,
        *,
        backends,
    ):
        return (operation, local_descs[0], remote_descs[0], remote_agent)

    def transfer(self, handle):
        operation, local_descriptor, remote_descriptor, _ = handle
        local = self.tensors[local_descriptor]
        remote = self.tensors[remote_descriptor]
        if operation == "READ":
            local.copy_(remote)
        else:
            remote.copy_(local)
        return "DONE"

    def check_xfer_state(self, handle):
        return "DONE"

    def release_xfer_handle(self, handle):
        self.released.append(handle)

    def remove_remote_agent(self, remote_name):
        return None


class MockEndpoint:
    def __init__(self, result) -> None:
        self.result = result

    async def call_one(self, *args, **kwargs):
        return self.result


class MockVolume:
    def __init__(self, metas=None) -> None:
        self.get_meta = MockEndpoint(metas or [])


class MockStorageVolumeRef:
    def __init__(self, metas=None) -> None:
        self.volume_id = "test-volume"
        self.volume = MockVolume(metas)
        self.transport_context = TransportContext()


@pytest.fixture(autouse=True)
def fake_nixl(monkeypatch):
    FakeNixlAgent.tensors = {}
    FakeNixlAgent.instances = []
    monkeypatch.setenv("TORCHSTORE_NIXL_ENABLED", "1")
    monkeypatch.setattr(nixl_transport, "_nixl_agent", FakeNixlAgent)
    monkeypatch.setattr(nixl_transport, "_nixl_agent_config", FakeNixlConfig)
    nixl_transport.nixl_available.cache_clear()
    yield
    nixl_transport.nixl_available.cache_clear()


@pytest.mark.asyncio
async def test_put_reads_client_tensor_into_storage_volume():
    ref = MockStorageVolumeRef()
    client = NixlTransportBuffer(ref)
    source = torch.arange(8, dtype=torch.float32)
    request = Request.from_tensor("weight", source)

    assert not client.requires_handshake([request])
    await client._pre_put_hook([request])
    assert client._client_metadata is not None
    server = pickle.loads(pickle.dumps(client))
    server_context = TransportContext()
    results = await server.handle_put_request(
        server_context, [(request.meta_only(), None)]
    )

    assert torch.equal(results[0], source)
    assert results[0].device.type == "cpu"
    assert len(FakeNixlAgent.instances[-1].released) == 1

    ref.transport_context.clear()
    server_context.clear()


@pytest.mark.asyncio
async def test_get_writes_storage_tensor_into_client_destination():
    ref = MockStorageVolumeRef()
    client = NixlTransportBuffer(ref)
    destination = torch.zeros(8)
    request = Request.from_tensor("weight", destination)

    await client._pre_get_hook([request])
    server = pickle.loads(pickle.dumps(client))
    stored = torch.arange(8, dtype=torch.float32)
    server_context = TransportContext()
    await server.handle_get_request(
        server_context, [(request.meta_only(), stored)]
    )
    results = await client._handle_storage_volume_response([request], server)

    assert results[0] is destination
    assert torch.equal(destination, stored)

    ref.transport_context.clear()
    server_context.clear()


@pytest.mark.asyncio
async def test_get_allocates_cpu_destination_from_metadata():
    ref = MockStorageVolumeRef(metas=[(torch.Size([4]), torch.float32)])
    client = NixlTransportBuffer(ref)
    request = Request.from_any("weight", None)

    await client._pre_get_hook([request])

    assert client._contexts[0].tensor is not None
    assert client._contexts[0].tensor.shape == (4,)
    assert client._contexts[0].tensor.device.type == "cpu"
    ref.transport_context.clear()


@pytest.mark.asyncio
async def test_objects_stay_on_rpc_path_without_creating_nixl_agent():
    ref = MockStorageVolumeRef()
    client = NixlTransportBuffer(ref)
    request = Request.from_objects("metadata", {"step": 3})

    await client._pre_put_hook([request])
    server = pickle.loads(pickle.dumps(client))
    server_context = TransportContext()
    results = await server.handle_put_request(
        server_context, [(request.meta_only(), None)]
    )

    assert results == [{"step": 3}]
    assert FakeNixlAgent.instances == []


def test_memory_registration_is_reused_for_same_tensor():
    cache = NixlAgentCache()
    tensor = torch.zeros(16)

    first = cache.register(tensor)
    second = cache.register(tensor)

    assert first is second
    assert cache.agent.register_calls == 1
    cache.clear()
