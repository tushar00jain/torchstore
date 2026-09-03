# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Two-rank smoke test for the legacy TorchComms RDMA transport.

Run this on a GPU node with::

    torchrun --standalone --nproc-per-node=2 tests/torchcomms_rdma_smoke.py

The CUDA tensors are intentionally allocated before the transports. That is
the ordering used when TorchStore registers parameters from an already-loaded
model and exercises CTran's dynamic memory-registration path.
"""

import os
import sys
import traceback

import torch
import torch.distributed as dist


def _exit_without_native_teardown(status: int) -> None:
    """Avoid the slow CTran process-global destructor in this short-lived test."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    import torchcomms._transport as torchcomms_transport

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 2:
        raise RuntimeError(f"expected exactly 2 ranks, got {world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Preserve the real model/TorchStore ordering: parameters are allocated
    # before CTran is initialized and later registered dynamically.
    tensor = torch.full(
        (8 * 1024 * 1024,),
        17 if rank == 0 else 0,
        dtype=torch.uint8,
        device=device,
    )

    if not torchcomms_transport.RdmaTransport.supported():
        raise RuntimeError("TorchComms RdmaTransport is not supported")

    transport = torchcomms_transport.RdmaTransport(device)
    local_address = transport.bind()
    addresses = [None] * world_size
    dist.all_gather_object(addresses, local_address)

    peer = 1 - rank
    connect_result = transport.connect(addresses[peer])
    if connect_result != 0 or not transport.connected():
        raise RuntimeError(
            f"rank {rank}: connect failed with result {connect_result}"
        )

    memory = torchcomms_transport.RdmaMemory(tensor)
    remote_buffers = [None] * world_size
    dist.all_gather_object(remote_buffers, memory.to_remote_buffer())

    dist.barrier()
    if rank == 0:
        write_result = transport.write(memory.to_view(), remote_buffers[peer])
        if write_result != 0:
            raise RuntimeError(f"rank 0: write failed with result {write_result}")
    dist.barrier()

    if rank == 1 and not torch.all(tensor == 17).item():
        raise RuntimeError("rank 1: transferred tensor contents did not match")

    print(f"rank {rank}: TorchComms RDMA registration and write passed", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        _exit_without_native_teardown(1)
    _exit_without_native_teardown(0)
