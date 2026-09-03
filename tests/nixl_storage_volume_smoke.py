# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-process GPU/IB smoke test for TorchStore's NIXL transport.

The first half of ranks act as TorchStore clients and own CUDA tensors. The
second half act as their StorageVolumes and own CPU copies. Only serialized
transport buffers use the Gloo control plane; tensor bytes move through
NIXL/UCX.
"""

import asyncio
import os
import pickle
import subprocess
import sys
import traceback

import torch
import torch.distributed as dist

from torchstore.transport.buffers import TransportContext
from torchstore.transport.nixl import NixlTransportBuffer
from torchstore.transport.shared_memory import allocate_shared_tensor
from torchstore.transport.types import Request


class ClientVolumeRef:
    def __init__(self, pair_index: int) -> None:
        self.volume_id = f"nixl-smoke-volume-{pair_index}"
        self.volume = None
        self.transport_context = TransportContext()


def broadcast_value(value, source_rank: int, group):
    payload = pickle.dumps(value) if dist.get_rank() == source_rank else None
    values = [payload]
    dist.broadcast_object_list(values, src=source_rank, group=group)
    return pickle.loads(values[0])


def configure_slurm_distributed_environment() -> None:
    """Translate Slurm's two task variables into torch.distributed env vars."""
    if "RANK" in os.environ or "SLURM_PROCID" not in os.environ:
        return
    hosts = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
        text=True,
    ).splitlines()
    os.environ.update(
        RANK=os.environ["SLURM_PROCID"],
        LOCAL_RANK=os.environ.get("SLURM_LOCALID", "0"),
        WORLD_SIZE=os.environ["SLURM_NTASKS"],
        MASTER_ADDR=hosts[0],
        MASTER_PORT=os.environ.get(
            "MASTER_PORT", str(20000 + int(os.environ["SLURM_JOB_ID"]) % 20000)
        ),
    )


async def run_test() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    world_size = dist.get_world_size()
    if world_size < 2 or world_size % 2:
        raise RuntimeError(f"expected an even world size, got {world_size}")

    rank = dist.get_rank()
    pair_count = world_size // 2
    pair_index = rank if rank < pair_count else rank - pair_count
    client_rank = pair_index
    server_rank = pair_index + pair_count
    pair_group = None
    for index in range(pair_count):
        group = dist.new_group([index, index + pair_count])
        if index == pair_index:
            pair_group = group
    assert pair_group is not None

    is_client = rank == client_rank
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.device_count() == 1:
        local_rank = 0
    torch.cuda.set_device(local_rank)
    server_context = TransportContext()
    tensor_count = int(os.environ.get("NIXL_SMOKE_TENSOR_COUNT", "1"))
    tensor_bytes = int(os.environ.get("NIXL_SMOKE_TENSOR_BYTES", str(16 * 1024 * 1024)))
    get_only = os.environ.get("NIXL_SMOKE_GET_ONLY", "0") == "1"
    shared_source = os.environ.get("NIXL_SMOKE_SHARED_SOURCE", "0") == "1"
    keys = [f"weight-{index}" for index in range(tensor_count)]
    stored = None

    if is_client:
        ref = ClientVolumeRef(pair_index)
        source = None
        if not get_only:
            source = [
                torch.full((tensor_bytes,), 17, dtype=torch.uint8, device="cuda")
                for _ in keys
            ]
            requests = [
                Request.from_tensor(key, tensor)
                for key, tensor in zip(keys, source)
            ]
            put_buffer = NixlTransportBuffer(ref)
        else:
            requests = [Request.from_tensor(key, torch.empty(0)) for key in keys]
            put_buffer = None
    else:
        requests = [Request.from_tensor(key, torch.empty(0)) for key in keys]
        put_buffer = None

    if get_only:
        if not is_client:
            if shared_source:
                stored = [
                    allocate_shared_tensor(torch.Size((tensor_bytes,)), torch.uint8)
                    for _ in keys
                ]
                for tensor in stored:
                    tensor.fill_(17)
            else:
                stored = [
                    torch.full((tensor_bytes,), 17, dtype=torch.uint8) for _ in keys
                ]
    else:
        if is_client:
            await put_buffer._pre_put_hook(requests)
        remote_put_buffer = broadcast_value(
            put_buffer, source_rank=client_rank, group=pair_group
        )
        if rank == server_rank:
            stored = await remote_put_buffer.handle_put_request(
                server_context,
                [(request.meta_only(), None) for request in requests],
            )
            if any(not torch.all(tensor == 17).item() for tensor in stored):
                raise RuntimeError("NIXL PUT produced incorrect StorageVolume bytes")
        dist.barrier(group=pair_group)

    # GET: each server writes its stored CPU tensors into its client's CUDA tensors.
    if is_client:
        destination = [
            torch.zeros((tensor_bytes,), dtype=torch.uint8, device="cuda")
            for _ in keys
        ]
        requests = [
            Request.from_tensor(key, tensor)
            for key, tensor in zip(keys, destination)
        ]
        get_buffer = NixlTransportBuffer(ref)
    else:
        get_buffer = None

    if is_client:
        await get_buffer._pre_get_hook(requests)
    remote_get_buffer = broadcast_value(
        get_buffer, source_rank=client_rank, group=pair_group
    )
    if rank == server_rank:
        assert stored is not None
        await remote_get_buffer.handle_get_request(
            server_context,
            [
                (request.meta_only(), tensor)
                for request, tensor in zip(requests, stored)
            ],
        )
    dist.barrier(group=pair_group)

    if is_client and any(
        not torch.all(tensor == 17).item() for tensor in destination
    ):
        raise RuntimeError("NIXL GET produced incorrect client CUDA bytes")

    if is_client:
        ref.transport_context.clear()
    else:
        server_context.clear()
    dist.barrier(group=pair_group)
    print(
        f"rank {rank}: TorchStore NIXL {'GET' if get_only else 'PUT and GET'} "
        f"passed ({tensor_count} tensors, {pair_count} pairs)",
        flush=True,
    )


def main() -> None:
    os.environ.setdefault("TORCHSTORE_NIXL_ENABLED", "1")
    configure_slurm_distributed_environment()
    dist.init_process_group("gloo")
    try:
        asyncio.run(run_test())
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
