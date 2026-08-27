# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Correctness tests for direct_weight_sync.py.

Uses a MockRDMABuffer that copies source bytes into the dest byte view,
simulating what real RDMA does. No GPU or RDMA infrastructure needed.
"""

from types import SimpleNamespace

import pytest
import torch

import torchstore.direct_transport as direct_transport
import torchstore.state_dict_utils as state_dict_utils
import torchstore.strategy as strategy_module
from torchstore.direct_transport import DirectTransport, create_direct_transport
from torchstore.direct_weight_sync import (
    DirectWeightSyncDest,
    DirectWeightSyncSource,
    RDMAWeightHandle,
)
from torchstore.strategy import TorchStoreStrategy
from torchstore.transport import TransportType
from torchstore.transport.types import TensorSlice
from torchstore.utils import to_byte_view

pytestmark = pytest.mark.asyncio


class MockRDMABuffer:
    """Simulates Monarch RDMABuffer by copying source bytes into dest."""

    def __init__(self, source_bytes: torch.Tensor):
        self._source = source_bytes

    async def read_into(self, dest_byte_view: torch.Tensor):
        dest_byte_view.copy_(self._source)

    async def drop(self):
        pass


def _make_sharded_handles(
    original: torch.Tensor,
    num_shards: int,
    shard_dim: int,
) -> list[RDMAWeightHandle]:
    """Create mock handles simulating a tensor sharded across num_shards ranks."""
    handles = []
    shard_size = original.shape[shard_dim] // num_shards

    for rank in range(num_shards):
        idx = [slice(None)] * original.ndim
        idx[shard_dim] = slice(rank * shard_size, (rank + 1) * shard_size)
        shard_data = original[tuple(idx)].contiguous()

        offsets = [0] * original.ndim
        offsets[shard_dim] = rank * shard_size
        local_shape = list(original.shape)
        local_shape[shard_dim] = shard_size

        tensor_slice = TensorSlice(
            offsets=tuple(offsets),
            coordinates=(rank,),
            global_shape=tuple(original.shape),
            local_shape=tuple(local_shape),
            mesh_shape=(num_shards,),
        )
        buf = MockRDMABuffer(to_byte_view(shard_data))
        handles.append(
            RDMAWeightHandle(
                rdma_buffer=buf,
                tensor_slice=tensor_slice,
                source_rank=rank,
            )
        )
    return handles


def _make_replicated_handles(
    original: torch.Tensor,
    num_ranks: int,
) -> list[RDMAWeightHandle]:
    """Create mock handles simulating a replicated tensor across num_ranks."""
    handles = []
    for rank in range(num_ranks):
        data = original.contiguous()
        tensor_slice = TensorSlice(
            offsets=tuple(0 for _ in range(original.ndim)),
            coordinates=(rank,),
            global_shape=tuple(original.shape),
            local_shape=tuple(original.shape),
            mesh_shape=(num_ranks,),
        )
        buf = MockRDMABuffer(to_byte_view(data))
        handles.append(
            RDMAWeightHandle(
                rdma_buffer=buf,
                tensor_slice=tensor_slice,
                source_rank=rank,
            )
        )
    return handles


async def test_exact_match():
    """Source is full tensor, dest is full tensor → Case 1 (zero-copy)."""
    original = torch.arange(512 * 512, dtype=torch.float32).reshape(512, 512)
    handles = _make_sharded_handles(original, num_shards=1, shard_dim=0)

    dest = torch.zeros_like(original)
    sync = DirectWeightSyncDest()
    await sync.pull({"weight": handles}, {"weight": dest})

    assert torch.equal(dest, original)
    # Case 1: zero-copy, no recv_buffer or dest_tensor
    assert len(sync._plan) == 1
    assert sync._plan[0].dest_tensor is None
    assert sync._plan[0].recv_buffer is None


@pytest.mark.parametrize(
    "num_shards,shard_dim",
    [
        (2, 0),  # row sharding
        (4, 0),  # finer row sharding
        (2, 1),  # column sharding
    ],
)
async def test_resharding(num_shards, shard_dim):
    """Source is sharded, dest is full tensor → Case 3 (resharding)."""
    original = torch.arange(512 * 512, dtype=torch.float32).reshape(512, 512)
    handles = _make_sharded_handles(
        original, num_shards=num_shards, shard_dim=shard_dim
    )

    dest = torch.zeros_like(original)
    sync = DirectWeightSyncDest()
    await sync.pull({"weight": handles}, {"weight": dest})

    assert torch.equal(dest, original)
    assert len(sync._plan) == num_shards


async def test_replicated_dedup():
    """Replicated source (2 ranks, same data) → should only read once."""
    original = torch.arange(512 * 512, dtype=torch.float32).reshape(512, 512)
    handles = _make_replicated_handles(original, num_ranks=2)

    dest = torch.zeros_like(original)
    sync = DirectWeightSyncDest()
    await sync.pull({"weight": handles}, {"weight": dest})

    assert torch.equal(dest, original)
    # Dedup: only 1 op despite 2 source ranks
    assert len(sync._plan) == 1


async def test_multiple_params():
    """State dict with multiple params, each handled independently."""
    w1 = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    w2 = torch.arange(100, 200, dtype=torch.float32).reshape(10, 10)

    all_handles = {
        "layer.weight": _make_sharded_handles(w1, num_shards=2, shard_dim=0),
        "layer.bias": _make_sharded_handles(w2, num_shards=1, shard_dim=0),
    }
    dest_sd = {
        "layer.weight": torch.zeros_like(w1),
        "layer.bias": torch.zeros_like(w2),
    }

    sync = DirectWeightSyncDest()
    await sync.pull(all_handles, dest_sd)

    assert torch.equal(dest_sd["layer.weight"], w1)
    assert torch.equal(dest_sd["layer.bias"], w2)


async def test_refresh():
    """After modifying source and calling refresh(), dest sees updated values."""
    source = DirectWeightSyncSource()

    # Create a non-contiguous source tensor (simulates column sharding)
    backing = torch.arange(100, dtype=torch.float32).reshape(10, 10)
    source_tensor = backing[:, :5]  # non-contiguous view, shape (10, 5)
    assert not source_tensor.is_contiguous()

    # Manually set up staging (simulating what register() does)
    staging_buf = source_tensor.contiguous()
    source._staging["weight"] = (staging_buf, source_tensor)

    # Create mock handle pointing at the staging buffer
    mock_buf = MockRDMABuffer(to_byte_view(staging_buf))
    handle = RDMAWeightHandle(
        rdma_buffer=mock_buf,
        tensor_slice=TensorSlice(
            offsets=(0, 0),
            coordinates=(0,),
            global_shape=(10, 5),
            local_shape=(10, 5),
            mesh_shape=(1,),
        ),
        source_rank=0,
    )
    all_handles = {"weight": [handle]}

    # First pull: should get original values
    dest = torch.zeros(10, 5, dtype=torch.float32)
    sync = DirectWeightSyncDest()
    await sync.pull(all_handles, {"weight": dest})
    assert torch.equal(dest, source_tensor.contiguous())

    # Modify source (simulates optimizer.step())
    backing.fill_(99.0)
    assert not torch.equal(staging_buf, source_tensor.contiguous())

    # Refresh copies updated source into staging buffer
    source.refresh()
    assert torch.equal(staging_buf, source_tensor.contiguous())

    # Second pull: should get updated values
    dest2 = torch.zeros(10, 5, dtype=torch.float32)
    sync2 = DirectWeightSyncDest()
    await sync2.pull(all_handles, {"weight": dest2})
    assert torch.equal(dest2, torch.full((10, 5), 99.0))


async def test_transfer_dtype():
    """Source is float32, transfer_dtype=bfloat16 → staging casts, dest gets bfloat16."""
    source = DirectWeightSyncSource()

    # Float32 source tensor (simulates model param)
    original = torch.arange(100, dtype=torch.float32).reshape(10, 10)

    # Set up staging with bfloat16 cast (simulates register(transfer_dtype=bfloat16))
    # Mock RDMABuffer by patching — register() calls RDMABuffer internally,
    # so we need to test via the staging + mock handle approach instead.
    staging_buf = original.to(torch.bfloat16).contiguous()
    source._staging["weight"] = (staging_buf, original)

    # Create mock handle pointing at the bfloat16 staging buffer
    mock_buf = MockRDMABuffer(to_byte_view(staging_buf))
    handle = RDMAWeightHandle(
        rdma_buffer=mock_buf,
        tensor_slice=TensorSlice(
            offsets=(0, 0),
            coordinates=(0,),
            global_shape=(10, 10),
            local_shape=(10, 10),
            mesh_shape=(1,),
        ),
        source_rank=0,
    )

    # Pull into bfloat16 dest (matching transfer dtype)
    dest = torch.zeros(10, 10, dtype=torch.bfloat16)
    sync = DirectWeightSyncDest()
    await sync.pull({"weight": [handle]}, {"weight": dest})

    # Values should match (float32 → bfloat16 → bfloat16)
    expected = original.to(torch.bfloat16)
    assert torch.equal(dest, expected)

    # All params should be staged (due to transfer_dtype)
    assert "weight" in source._staging

    # Modify source (simulates optimizer.step() updating float32 param)
    original.fill_(42.0)

    # Refresh re-casts float32 → bfloat16 into staging buffer
    source.refresh()
    assert torch.equal(staging_buf, torch.full((10, 10), 42.0, dtype=torch.bfloat16))

    # Second pull sees updated values
    dest2 = torch.zeros(10, 10, dtype=torch.bfloat16)
    sync2 = DirectWeightSyncDest()
    await sync2.pull({"weight": [handle]}, {"weight": dest2})
    assert torch.equal(dest2, torch.full((10, 10), 42.0, dtype=torch.bfloat16))


class _FakeStore:
    def __init__(self):
        self.values = {}

    async def put(self, key, value):
        self.values[key] = value

    async def get(self, key):
        return self.values[key]

    async def exists(self, key):
        return key in self.values

    async def delete(self, key):
        self.values.pop(key, None)

    async def keys(self, prefix=None):
        return [key for key in self.values if prefix is None or key.startswith(prefix)]


class _FakeSourceConnection:
    connection_info = b"source"

    def __init__(self, _request):
        self.closed = False

    def register(self, tensor):
        return tensor

    def close(self):
        self.closed = True


class _FakeDestinationConnection:
    connection_info = b"destination"

    def __init__(self):
        self.connected = False
        self.closed = False

    def connect(self, connection_info):
        assert connection_info == b"source"
        self.connected = True

    async def read_into(self, remote_buffer, tensor):
        assert self.connected
        tensor.copy_(to_byte_view(remote_buffer))

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    "backend", [TransportType.Rdma4Py, TransportType.TorchComms]
)
async def test_selectable_direct_transport(monkeypatch, backend):
    """Both selectable transports use the direct source-to-destination plan."""
    if backend == TransportType.Rdma4Py:
        monkeypatch.setattr(
            "torchstore.transport.rdma4py.rdma4py_transport_available",
            lambda: True,
        )
    else:
        monkeypatch.setattr(
            "torchstore.transport.torchcomms.cache.torchcomms_rdma_available",
            lambda: True,
        )
    monkeypatch.setattr(direct_transport, "_require_cuda_tensor", lambda *_: None)
    monkeypatch.setattr(direct_transport, "_cuda_synchronize", lambda *_: None)

    store = _FakeStore()
    source_tensor = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    destination_tensor = torch.zeros_like(source_tensor)
    source = direct_transport.DirectTransportWeightSyncSource(
        backend,
        store,
        "model",
        lambda connection_info, _device: _FakeSourceConnection(connection_info),
    )
    destination = direct_transport.DirectTransportWeightSyncDest(
        backend,
        store,
        "model",
        lambda _device: _FakeDestinationConnection(),
    )

    await source.register({"weight": source_tensor}, rank=0, transfer_dtype=None)
    await destination.pull(num_ranks=1, user_state_dict={"weight": destination_tensor})
    assert torch.equal(destination_tensor, source_tensor)

    source_tensor.fill_(17)
    source.refresh()
    await destination.pull(num_ranks=1, user_state_dict={"weight": destination_tensor})
    assert torch.equal(destination_tensor, source_tensor)

    await destination.close()
    await source.close()


async def test_state_dict_helpers_use_strategy_transport(monkeypatch):
    calls = []

    async def fake_put(store, state_dict, key, transfer_dtype, backend):
        calls.append(("put", backend))

    async def fake_get(store, key, user_state_dict, backend):
        calls.append(("get", backend))

    monkeypatch.setattr(state_dict_utils, "_put_state_dict_direct_rdma", fake_put)
    monkeypatch.setattr(state_dict_utils, "_get_state_dict_direct_rdma", fake_get)

    store = SimpleNamespace(
        strategy=TorchStoreStrategy(TransportType.Rdma4Py)
    )
    state_dict = {"weight": torch.ones(2)}
    await state_dict_utils.put_state_dict(
        store,
        state_dict,
        "model",
        direct_rdma=True,
    )
    result = await state_dict_utils.get_state_dict(
        store,
        "model",
        state_dict,
        direct_rdma=True,
    )

    assert result is state_dict
    assert calls == [
        ("put", TransportType.Rdma4Py),
        ("get", TransportType.Rdma4Py),
    ]


async def test_strategy_resolves_normal_and_direct_transports_separately(monkeypatch):
    monkeypatch.setattr(
        strategy_module,
        "get_available_transport",
        lambda _storage_volume_ref: TransportType.Gloo,
    )
    monkeypatch.setattr(
        strategy_module,
        "get_available_direct_transport",
        lambda: TransportType.Rdma4Py,
    )

    strategy = TorchStoreStrategy()
    assert strategy.get_transport_type(object()) == TransportType.Gloo
    assert strategy.get_direct_transport_type() == TransportType.Rdma4Py


async def test_direct_transport_rejects_non_rdma_strategy():
    with pytest.raises(ValueError, match="does not support direct weight sync"):
        TorchStoreStrategy(TransportType.Gloo).get_direct_transport_type()


@pytest.mark.parametrize(
    "transport_type",
    [
        TransportType.MonarchRDMA,
        TransportType.Rdma4Py,
        TransportType.TorchComms,
    ],
)
async def test_direct_transport_factory(transport_type):
    transport = create_direct_transport(transport_type, _FakeStore(), "model")

    assert isinstance(transport, DirectTransport)
