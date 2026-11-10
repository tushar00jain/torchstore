# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch

from torchstore.routing.storage_volume import RoutingInMemoryStore
from torchstore.transport.buffers import TransportBuffer, TransportContext
from torchstore.transport.types import Request, TensorSlice


def _whole_slice() -> TensorSlice:
    return TensorSlice(
        offsets=(0,),
        coordinates=(),
        global_shape=(4,),
        local_shape=(4,),
        mesh_shape=(),
    )


@pytest.mark.asyncio
async def test_routed_put_reuses_its_final_shared_segment() -> None:
    store = RoutingInMemoryStore()
    tensor_slice = _whole_slice()

    class DirectPutBuffer:
        def __init__(self, source: torch.Tensor) -> None:
            self.source = source
            self.destination = None

        async def recv_handshake(self, context, entries):
            assert context is store.transport_context
            [(request, destination)] = entries
            assert request.stable_storage
            assert destination.dtype == torch.bfloat16
            return [destination]

        async def handle_put_request(self, context, entries):
            assert context is store.transport_context
            [(request, destination)] = entries
            assert request.requester_id == "requester"
            assert destination.dtype == torch.bfloat16
            destination.copy_(self.source)
            self.destination = destination
            return [destination]

    first = Request.from_any("weight", torch.arange(4.0), tensor_slice)
    first.requester_id = "requester"
    first.storage_layouts = {"requester": (tensor_slice, torch.bfloat16)}
    first_buffer = DirectPutBuffer(first.tensor_val)
    first_metadata = first.meta_only()
    [handshake_destination] = await store.handshake(first_buffer, [first_metadata])
    await store.put(first_buffer, [first_metadata])

    first_destination = first_buffer.destination
    assert first_destination is not None
    assert handshake_destination.data_ptr() == first_destination.data_ptr()
    torch.testing.assert_close(first_destination, first.tensor_val.to(torch.bfloat16))

    second = Request.from_any("weight", torch.arange(10.0, 14.0), tensor_slice)
    second.requester_id = "requester"
    second_buffer = DirectPutBuffer(second.tensor_val)
    await store.put(second_buffer, [second.meta_only()])

    assert second_buffer.destination.data_ptr() == first_destination.data_ptr()
    torch.testing.assert_close(
        second_buffer.destination, second.tensor_val.to(torch.bfloat16)
    )
    store.reset()


def test_routing_request_cache_omits_repeated_payload_and_slices() -> None:
    client_context = TransportContext()
    server_context = TransportContext()
    ref = SimpleNamespace(volume_id="volume", transport_context=client_context)
    tensor_request = Request.from_tensor_slice("weight", _whole_slice())
    tensor_request.requester_id = "requester"
    object_request = Request.from_objects("mapping", {"weight": ("weight",)})
    requests = [tensor_request, object_request]

    first = TransportBuffer(ref)
    first.enable_request_metadata_cache("requester")
    first_requests = first._prepare_storage_volume_requests("get", requests)
    assert first_requests[0].tensor_slice is None
    assert first_requests[1].is_object
    assert first_requests[1].objects is None
    first.resolve_cached_requests(server_context, "get", first_requests)
    first._publish_request_layout()

    second = TransportBuffer(ref)
    second.enable_request_metadata_cache("requester")
    compact_requests = second._prepare_storage_volume_requests("get", requests)
    assert compact_requests == []

    cached = second.resolve_cached_requests(server_context, "get", compact_requests)
    assert cached[0].tensor_slice is None
    assert cached[1].objects is None
    assert all(request.storage_layouts is None for request in cached)

    client_context.clear()
    server_context.clear()
