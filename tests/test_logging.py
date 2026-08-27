# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

from torchstore.logging import LatencyTracker, collect_latencies, record_observation


def test_latency_collector_captures_steps_and_explicit_observations(monkeypatch):
    times = iter((10.0, 10.25, 10.5))
    monkeypatch.setattr("torchstore.logging.time.perf_counter", lambda: next(times))

    with collect_latencies() as collector:
        tracker = LatencyTracker("get_batch")
        tracker.track_step("fetch", nbytes=1_000_000_000)
        tracker.track_e2e(nbytes=1_000_000_000)
        record_observation("fetch/request_count", 4)

    assert collector.snapshot() == {
        "get_batch/fetch/seconds": [0.25],
        "get_batch/fetch/bytes": [1_000_000_000.0],
        "get_batch/fetch/throughput_gbps": [4.0],
        "get_batch/e2e/seconds": [0.5],
        "get_batch/e2e/bytes": [1_000_000_000.0],
        "get_batch/e2e/throughput_gbps": [2.0],
        "fetch/request_count": [4.0],
    }


def test_latency_collector_is_scoped():
    with collect_latencies() as collector:
        record_observation("inside", 1)

    record_observation("outside", 2)
    assert collector.snapshot() == {"inside": [1.0]}


def test_latency_collector_flows_into_async_child_tasks():
    async def run():
        with collect_latencies() as collector:
            await asyncio.gather(
                asyncio.create_task(_record("child", 1)),
                asyncio.create_task(_record("child", 2)),
            )
        return collector.snapshot()

    async def _record(key, value):
        record_observation(key, value)

    assert asyncio.run(run()) == {"child": [1.0, 2.0]}
