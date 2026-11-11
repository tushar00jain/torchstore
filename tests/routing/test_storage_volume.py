# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchstore.routing.storage_volume import RoutingInMemoryStore
from torchstore.transport.types import Request, TensorSlice


def _slice(offset: int, size: int, total: int = 8) -> TensorSlice:
    return TensorSlice(
        offsets=(offset,),
        coordinates=(),
        global_shape=(total,),
        local_shape=(size,),
        mesh_shape=(),
    )


@pytest.mark.parametrize("shared_memory", [False, True])
def test_routed_layouts_share_identical_requester_segments(
    shared_memory: bool,
) -> None:
    store = RoutingInMemoryStore(shared_memory=shared_memory)
    store.install_layouts(
        {
            "weight": (
                torch.float32,
                {
                    "requester-a": _slice(0, 4),
                    "requester-b": _slice(0, 4),
                    "requester-c": _slice(4, 4),
                },
            )
        }
    )

    first = Request(key="weight", requester_id="requester-a")
    second = Request(key="weight", requester_id="requester-b")
    third = Request(key="weight", requester_id="requester-c")
    assert store._get_data(first).is_contiguous()
    assert store._get_data(first).is_shared() is shared_memory
    assert (
        store._get_data(first).untyped_storage().data_ptr()
        == store._get_data(second).untyped_storage().data_ptr()
    )
    assert (
        store._get_data(first).untyped_storage().data_ptr()
        != store._get_data(third).untyped_storage().data_ptr()
    )


@pytest.mark.asyncio
async def test_routed_put_reuses_requester_storage() -> None:
    store = RoutingInMemoryStore()
    tensor_slice = _slice(2, 4)
    store.install_layouts(
        {
            "weight": (
                torch.float32,
                {"requester": tensor_slice},
            )
        }
    )

    class DirectPutBuffer:
        def __init__(self, source: torch.Tensor) -> None:
            self.source = source
            self.destination = None

        async def recv_handshake(self, context, entries):
            [(request, destination)] = entries
            assert context is store.transport_context
            assert destination is not None
            return [destination]

        async def handle_put_request(self, context, entries):
            [(request, destination)] = entries
            assert context is store.transport_context
            destination.copy_(self.source)
            self.destination = destination
            return [destination]

    first = Request.from_any("weight", torch.arange(4.0), tensor_slice)
    first.requester_id = "requester"
    first_buffer = DirectPutBuffer(first.tensor_val)
    metadata = first.meta_only()
    [handshake_destination] = await store.handshake(first_buffer, [metadata])
    await store.put(first_buffer, [metadata])

    second = Request.from_any("weight", torch.arange(10.0, 14.0), tensor_slice)
    second.requester_id = "requester"
    second_buffer = DirectPutBuffer(second.tensor_val)
    await store.put(second_buffer, [second.meta_only()])

    assert first_buffer.destination is not None
    assert handshake_destination.data_ptr() == first_buffer.destination.data_ptr()
    assert second_buffer.destination.data_ptr() == first_buffer.destination.data_ptr()
    torch.testing.assert_close(second_buffer.destination, second.tensor_val)

    requester = Request(key="weight", requester_id="requester")
    torch.testing.assert_close(store._get_data(requester), second.tensor_val)
