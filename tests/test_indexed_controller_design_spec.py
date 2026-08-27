# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import random

import pytest

import torchstore.controllers.indexed.directory as indexed_module
from torchstore.controller import Controller
from torchstore.controllers import (
    IndexedController,
    IndexedDirectoryBackend,
    ObjectType,
    StorageInfo,
    get_controller_class,
)
from torchstore.controllers.legacy import locate_slices_exhaustive
from torchstore.transport.types import Request, TensorSlice


def _slice(
    offsets,
    local_shape,
    global_shape,
    coordinates=(0,),
    mesh_shape=(1,),
):
    return TensorSlice(
        offsets=offsets,
        coordinates=coordinates,
        global_shape=global_shape,
        local_shape=local_shape,
        mesh_shape=mesh_shape,
    )


def _info(*tensor_slices):
    return StorageInfo(ObjectType.TENSOR_SLICE, set(tensor_slices))


def _request(key, tensor_slice):
    return Request.from_tensor_slice(key, tensor_slice)


def _normalized(plan):
    return (
        plan.key,
        plan.object_type,
        plan.requested_region,
        tuple(
            (
                offer.geometry.source_region,
                None
                if offer.geometry.intersection is None
                else (
                    offer.geometry.intersection.offsets,
                    offer.geometry.intersection.local_shape,
                    offer.geometry.intersection.global_shape,
                ),
                offer.geometry.storage_offsets,
                offer.geometry.destination_offsets,
                offer.geometry.lengths,
                offer.live_volumes,
                offer.pending,
            )
            for offer in plan.offers
        ),
    )


class _CountingRanked:
    def __init__(self, sources):
        self.sources = tuple(sources)
        self.iterations = 0

    def __iter__(self):
        for source in self.sources:
            self.iterations += 1
            yield source


@pytest.mark.parametrize(
    ("stored", "requested", "expected"),
    [
        ((2,), (2,), ((2,), (4,))),
        ((0,), (2,), ((2,), (2,))),
        ((6,), (0,), None),
    ],
)
def test_exact_partial_and_disjoint_overlap(stored, requested, expected):
    backend = IndexedDirectoryBackend()
    backend.add_source(
        "weight",
        "v0",
        0,
        _info(_slice(stored, (4,), (10,))),
    )

    plan = backend.locate_slices([_request("weight", _slice(requested, (4,), (10,)))])[
        "weight"
    ]

    if expected is None:
        assert plan.offers == ()
    else:
        intersection = plan.offers[0].geometry.intersection
        assert (intersection.offsets, intersection.local_shape) == expected


def test_multidimensional_overlap_prunes_false_candidates():
    backend = IndexedDirectoryBackend()
    stored = [
        _slice((0, 0), (4, 4), (8, 8)),
        _slice((0, 4), (4, 4), (8, 8)),
        _slice((4, 0), (4, 4), (8, 8)),
        _slice((4, 4), (4, 4), (8, 8)),
    ]
    for index, tensor_slice in enumerate(stored):
        backend.add_source("weight", f"v{index}", 0, _info(tensor_slice))

    plan = backend.locate_slices([_request("weight", _slice((2, 2), (4, 4), (8, 8)))])[
        "weight"
    ]

    assert {
        (offer.geometry.intersection.offsets, offer.geometry.intersection.local_shape)
        for offer in plan.offers
    } == {
        ((2, 2), (2, 2)),
        ((2, 4), (2, 2)),
        ((4, 2), (2, 2)),
        ((4, 4), (2, 2)),
    }


def test_scalar_tensor_slice_uses_the_indexed_scalar_bucket():
    backend = IndexedDirectoryBackend()
    scalar = _slice((), (), (), (), (1,))
    backend.add_source("scalar", "v0", 0, _info(scalar))

    plan = backend.locate_slices([_request("scalar", scalar)])["scalar"]

    assert len(plan.offers) == 1
    offer = plan.offers[0]
    assert offer.geometry.intersection.offsets == ()
    assert offer.geometry.lengths == ()
    assert offer.live_volumes == ("v0",)


def test_normalized_template_cache_keeps_global_shape_compatibility():
    backend = IndexedDirectoryBackend()
    backend.add_source("valid", "v0", 0, _info(_slice((0,), (8,), (8,))))
    backend.add_source("mismatch", "v0", 0, _info(_slice((0,), (16,), (16,))))
    requests = [
        _request("mismatch", _slice((0,), (8,), (8,))),
        _request("valid", _slice((0,), (8,), (8,))),
    ]

    assert backend.serving_union(requests) == frozenset({(0, "v0")})


def test_replicas_share_one_logical_offer():
    backend = IndexedDirectoryBackend()
    tensor_slice = _slice((0,), (8,), (8,))
    backend.add_source("weight", "v2", 0, _info(tensor_slice))
    backend.add_source("weight", "v1", 0, _info(tensor_slice))
    backend.add_source("weight", "vp", 7, _info(tensor_slice))

    plan = backend.locate_slices([_request("weight", tensor_slice)])["weight"]

    assert len(plan.offers) == 1
    assert plan.offers[0].live_volumes == ("v1", "v2")
    assert plan.offers[0].pending == ((7, "vp"),)


def test_pending_to_live_and_retirement_preserve_cached_geometry():
    controller = IndexedController()
    controller.is_initialized = True
    tensor_slice = _slice((0,), (8,), (8,))
    request = _request("weight", tensor_slice)

    pub = controller._notify_put_batch([request], "vp", pending=True)
    backend = controller.get_directory_backend()
    first = backend.locate_slices([request])["weight"]
    first_geometry = first.offers[0].geometry

    controller._notify_put_batch([request], "vp", pending=False)
    landed = backend.locate_slices([request])["weight"]
    assert landed.geometry_epoch == first.geometry_epoch
    assert landed.offers[0].geometry is first_geometry
    assert landed.offers[0].live_volumes == ("vp",)
    assert landed.offers[0].pending == ((pub, "vp"),)

    controller._notify_delete_batch(pub=pub)
    retired = backend.locate_slices([request])["weight"]
    assert retired.geometry_epoch == first.geometry_epoch
    assert retired.offers[0].geometry is first_geometry
    assert retired.offers[0].pending == ()

    controller._notify_delete("weight", "vp")
    with pytest.raises(KeyError):
        backend.locate_slices([request])


def test_new_region_invalidates_geometry_but_replica_does_not():
    backend = IndexedDirectoryBackend()
    left = _slice((0,), (4,), (8,), (0,), (2,))
    right = _slice((4,), (4,), (8,), (1,), (2,))
    request = _request("weight", left)
    backend.add_source("weight", "v0", 0, _info(left))

    first = backend.locate_slices([request])["weight"]
    first_geometry = first.offers[0].geometry

    backend.add_source("weight", "replica", 0, _info(left))
    replica = backend.locate_slices([request])["weight"]
    assert replica.geometry_epoch == first.geometry_epoch
    assert replica.offers[0].geometry is first_geometry

    backend.add_source("weight", "v1", 0, _info(right))
    changed = backend.locate_slices([request])["weight"]
    assert changed.geometry_epoch > first.geometry_epoch
    assert changed.offers[0].geometry is not first_geometry


def test_incremental_completeness_counts_replica_coordinates():
    backend = IndexedDirectoryBackend()
    shards = [
        _slice((0,), (4,), (10,), (0,), (3,)),
        _slice((4,), (3,), (10,), (1,), (3,)),
        _slice((7,), (3,), (10,), (2,), (3,)),
    ]
    backend.add_source("weight", "v0", 0, _info(shards[0]))
    backend.add_source("weight", "v2", 0, _info(shards[2]))
    assert not backend.is_fully_committed("weight")

    backend.add_source("weight", "v1", 0, _info(shards[1]))
    assert backend.is_fully_committed("weight")
    backend.add_source("weight", "replica", 0, _info(shards[1]))
    backend.remove_source("weight", "v1", 0)
    assert backend.is_fully_committed("weight")
    backend.remove_source("weight", "replica", 0)
    assert not backend.is_fully_committed("weight")


def test_completeness_does_not_mix_row_and_column_layouts():
    backend = IndexedDirectoryBackend()
    row0 = _slice((0, 0), (4, 8), (8, 8), (0,), (2,))
    row1 = _slice((4, 0), (4, 8), (8, 8), (1,), (2,))
    column1 = _slice((0, 4), (8, 4), (8, 8), (1,), (2,))

    backend.add_source("weight", "row0", 0, _info(row0))
    backend.add_source("weight", "column1", 0, _info(column1))
    assert not backend.is_fully_committed("weight")

    backend.add_source("weight", "row1", 0, _info(row1))
    assert backend.is_fully_committed("weight")


def test_incremental_completeness_matches_full_scan_after_mutations():
    rng = random.Random(404)
    backend = IndexedDirectoryBackend()
    candidates = {
        "row0": _slice((0, 0), (4, 8), (8, 8), (0,), (2,)),
        "row1": _slice((4, 0), (4, 8), (8, 8), (1,), (2,)),
        "column0": _slice((0, 0), (8, 4), (8, 8), (0,), (2,)),
        "column1": _slice((0, 4), (8, 4), (8, 8), (1,), (2,)),
    }
    live = {}

    def fully_committed_by_scan():
        layouts = {}
        for tensor_slice in live.values():
            sharded_dimensions = tuple(
                dim
                for dim, (local, global_) in enumerate(
                    zip(
                        tensor_slice.local_shape,
                        tensor_slice.global_shape,
                        strict=True,
                    )
                )
                if local != global_
            )
            layout = (
                tensor_slice.global_shape,
                tensor_slice.mesh_shape,
                sharded_dimensions,
            )
            layouts.setdefault(layout, set()).add(tensor_slice.coordinates)
        return any(
            coordinates == {(index,) for index in range(layout[1][0])}
            for layout, coordinates in layouts.items()
        )

    for _ in range(100):
        volume = rng.choice(tuple(candidates))
        if volume in live:
            backend.remove_source("weight", volume, 0)
            del live[volume]
        else:
            tensor_slice = candidates[volume]
            backend.add_source("weight", volume, 0, _info(tensor_slice))
            live[volume] = tensor_slice
        assert backend.is_fully_committed("weight") == fully_committed_by_scan()


def test_uneven_shards_and_multiple_slices_on_one_volume():
    backend = IndexedDirectoryBackend()
    shards = [
        _slice((0,), (4,), (10,), (0,), (3,)),
        _slice((4,), (3,), (10,), (1,), (3,)),
        _slice((7,), (3,), (10,), (2,), (3,)),
    ]
    backend.add_source("weight", "v0", 0, _info(shards[0], shards[1]))
    backend.add_source("weight", "v1", 0, _info(shards[2]))

    request = _request("weight", _slice((3,), (5,), (10,)))
    plan = backend.locate_slices([request])["weight"]

    assert [offer.geometry.lengths for offer in plan.offers] == [(1,), (3,), (1,)]
    assert backend.greedy_cover([request], [(0, "v0"), (0, "v1")]) == [
        (0, "v0"),
        (0, "v1"),
    ]


def test_one_source_can_hold_replica_coordinates_for_one_region():
    backend = IndexedDirectoryBackend()
    replicas = (
        _slice((0,), (8,), (8,), (0,), (2,)),
        _slice((0,), (8,), (8,), (1,), (2,)),
    )
    backend.add_source("weight", "v0", 0, _info(*replicas))

    plan = backend.locate_slices([_request("weight", replicas[0])])["weight"]
    assert len(plan.offers) == 1
    assert backend.is_fully_committed("weight")

    backend.remove_source("weight", "v0", 0)
    assert not backend.has_key("weight")


@pytest.mark.parametrize("object_type", [ObjectType.OBJECT, ObjectType.TENSOR])
def test_whole_values_use_one_offer(object_type):
    backend = IndexedDirectoryBackend()
    info = StorageInfo(object_type, {None})
    backend.add_source("value", "v1", 0, info)
    backend.add_source("value", "v0", 0, info)
    request = Request(key="value", is_object=object_type is ObjectType.OBJECT)

    plan = backend.locate_slices([request])["value"]

    assert len(plan.offers) == 1
    assert plan.offers[0].geometry.intersection is None
    assert plan.offers[0].live_volumes == ("v0", "v1")
    assert backend.greedy_cover([request], [(0, "v1"), (0, "v0")]) == [(0, "v1")]


def test_indexed_and_exhaustive_plans_match_generated_rectangles():
    rng = random.Random(93741)
    for case in range(80):
        rank = rng.choice((1, 2))
        global_shape = tuple(rng.randint(6, 18) for _ in range(rank))
        backend = IndexedDirectoryBackend()
        directory = {"weight": {}}
        for source_index in range(rng.randint(1, 12)):
            offsets = tuple(rng.randrange(size) for size in global_shape)
            local_shape = tuple(
                rng.randint(1, size - offset)
                for size, offset in zip(global_shape, offsets, strict=True)
            )
            tensor_slice = _slice(
                offsets,
                local_shape,
                global_shape,
                (source_index,),
                (20,),
            )
            volume = f"v{source_index}"
            pub = 0 if rng.random() < 0.7 else source_index + 1
            info = _info(tensor_slice)
            directory["weight"][volume] = {pub: info}
            backend.add_source("weight", volume, pub, info)
        request_offsets = tuple(rng.randrange(size) for size in global_shape)
        request_shape = tuple(
            rng.randint(1, size - offset)
            for size, offset in zip(global_shape, request_offsets, strict=True)
        )
        request = _request(
            "weight", _slice(request_offsets, request_shape, global_shape)
        )

        indexed = backend.locate_slices([request])["weight"]
        exhaustive = locate_slices_exhaustive(
            [request],
            directory,
            missing_ok=False,
            require_fully_committed=False,
            include_pending=True,
            prefer=None,
            completeness=lambda key, live: True,
        )["weight"]

        assert _normalized(indexed) == _normalized(exhaustive), case


def test_serving_and_cover_match_legacy_on_generated_cases():
    rng = random.Random(112358)
    for case in range(40):
        legacy = Controller()
        indexed = IndexedController()
        legacy.is_initialized = indexed.is_initialized = True
        ranked = []
        for source_index in range(10):
            start = rng.randrange(16)
            length = rng.randint(1, 16 - start)
            tensor_slice = _slice((start,), (length,), (16,), (source_index,), (10,))
            volume = f"v{source_index}"
            pending = rng.random() < 0.4
            legacy_pub = legacy._notify_put_batch(
                [_request("weight", tensor_slice)], volume, pending=pending
            )
            indexed_pub = indexed._notify_put_batch(
                [_request("weight", tensor_slice)], volume, pending=pending
            )
            assert legacy_pub == indexed_pub
            ranked.append((legacy_pub, volume))
        rng.shuffle(ranked)
        start = rng.randrange(16)
        request = _request(
            "weight", _slice((start,), (rng.randint(1, 16 - start),), (16,))
        )

        assert indexed.serving_union([request]) == legacy.serving_union([request]), case
        assert indexed.greedy_cover([request], ranked) == legacy.greedy_cover(
            [request], ranked
        ), case


def test_grouped_multi_key_coverage_matches_legacy():
    rng = random.Random(271828)
    for case in range(30):
        key_count = rng.randint(2, 8)
        source_count = rng.randint(2, 8)
        legacy = Controller()
        indexed = IndexedController()
        legacy.is_initialized = indexed.is_initialized = True
        ranked = []
        for source_index in range(source_count):
            volume = f"v{source_index}"
            batch = []
            for key_index in range(key_count):
                scale = key_index + 1
                global_size = source_count * 4 * scale
                if key_index % 2 == 0:
                    stored = _slice(
                        (0,),
                        (global_size,),
                        (global_size,),
                        (source_index,),
                        (source_count,),
                    )
                else:
                    shard_size = 4 * scale
                    stored = _slice(
                        (source_index * shard_size,),
                        (shard_size,),
                        (global_size,),
                        (source_index,),
                        (source_count,),
                    )
                batch.append(_request(f"weight.{key_index}", stored))
            pending = rng.random() < 0.4
            legacy_pub = legacy._notify_put_batch(batch, volume, pending=pending)
            indexed_pub = indexed._notify_put_batch(batch, volume, pending=pending)
            assert legacy_pub == indexed_pub
            ranked.append((legacy_pub, volume))
        rng.shuffle(ranked)
        requests = []
        for key_index in range(key_count):
            scale = key_index + 1
            global_size = source_count * 4 * scale
            requested = (
                _slice((scale,), (2 * scale,), (global_size,))
                if key_index % 2 == 0
                else _slice((0,), (global_size,), (global_size,))
            )
            requests.append(_request(f"weight.{key_index}", requested))

        assert indexed.serving_union(requests) == legacy.serving_union(requests), case
        assert indexed.greedy_cover(requests, ranked) == legacy.greedy_cover(
            requests, ranked
        ), case


def test_plan_order_is_independent_of_insertion_order():
    slices = [
        _slice((8,), (2,), (10,), (2,), (3,)),
        _slice((0,), (4,), (10,), (0,), (3,)),
        _slice((4,), (4,), (10,), (1,), (3,)),
    ]
    forward = IndexedDirectoryBackend()
    reverse = IndexedDirectoryBackend()
    for index, tensor_slice in enumerate(slices):
        forward.add_source("weight", f"v{index}", 0, _info(tensor_slice))
    for index in reversed(range(len(slices))):
        reverse.add_source("weight", f"v{index}", 0, _info(slices[index]))
    request = _request("weight", _slice((0,), (10,), (10,)))

    assert _normalized(forward.locate_slices([request])["weight"]) == _normalized(
        reverse.locate_slices([request])["weight"]
    )


def test_preferred_live_sources_are_ranked_without_dropping_fallbacks():
    backend = IndexedDirectoryBackend()
    tensor_slice = _slice((0,), (8,), (8,))
    for volume in ("v0", "v1", "v2"):
        backend.add_source("weight", volume, 0, _info(tensor_slice))

    plan = backend.locate_slices(
        [_request("weight", tensor_slice)], prefer=("v2", "missing", "v0")
    )["weight"]

    assert plan.offers[0].live_volumes == ("v2", "v0", "v1")


def test_interval_query_and_cache_avoid_full_region_scans(monkeypatch):
    backend = IndexedDirectoryBackend()
    for index in range(72):
        tensor_slice = _slice((index * 10,), (10,), (720,), (index,), (72,))
        backend.add_source("weight", f"v{index}", 0, _info(tensor_slice))
    request = _request("weight", _slice((195,), (5,), (720,)))
    calls = 0
    intersect = indexed_module.get_slice_intersection

    def counted_intersection(stored, wanted):
        nonlocal calls
        calls += 1
        return intersect(stored, wanted)

    monkeypatch.setattr(indexed_module, "get_slice_intersection", counted_intersection)

    first = backend.locate_slices([request])["weight"]
    second = backend.locate_slices([request])["weight"]

    assert len(first.offers) == 1
    assert second.offers[0].geometry is first.offers[0].geometry
    assert calls == 1


def test_replica_heavy_coverage_work_scales_with_sources_plus_keys(monkeypatch):
    key_count = 48
    source_count = 32
    backend = IndexedDirectoryBackend()
    requests = []
    for key_index in range(key_count):
        scale = key_index + 1
        global_size = 100 * scale
        key = f"weight.{key_index}"
        tensor_slice = _slice((0,), (global_size,), (global_size,), (0,), (1,))
        for source_index in range(source_count):
            backend.add_source(
                key,
                f"v{source_index}",
                0,
                _info(tensor_slice),
            )
        requests.append(
            _request(
                key,
                _slice(
                    (20 * scale,),
                    (10 * scale,),
                    (global_size,),
                ),
            )
        )
    ranked_sources = [(0, f"v{index}") for index in reversed(range(source_count))]
    calls = 0
    intersect = indexed_module.get_slice_intersection

    def counted_intersection(stored, wanted):
        nonlocal calls
        calls += 1
        return intersect(stored, wanted)

    monkeypatch.setattr(indexed_module, "get_slice_intersection", counted_intersection)

    union = backend.serving_union(requests)
    ranked = _CountingRanked(ranked_sources)
    cover = backend.greedy_cover(requests, ranked)

    assert union == frozenset(ranked_sources)
    assert cover == [ranked_sources[0]]
    assert ranked.iterations == source_count
    assert calls == 1
    assert len(backend._coverage_cache) == 1
    assert len(backend._template_intern) == 1


def test_shared_replica_binding_partitions_sources_once(monkeypatch):
    key_count = 32
    source_count = 20
    backend = IndexedDirectoryBackend()
    requests = []
    for key_index in range(key_count):
        key = f"weight.{key_index}"
        tensor_slice = _slice((0,), (64,), (64,))
        for source_index in range(source_count):
            backend.add_source(
                key,
                f"v{source_index}",
                0,
                _info(tensor_slice),
            )
        requests.append(_request(key, tensor_slice))
    calls = 0
    order = indexed_module.order_live_volumes

    def counted_order(volumes, prefer):
        nonlocal calls
        calls += 1
        return order(volumes, prefer)

    monkeypatch.setattr(indexed_module, "order_live_volumes", counted_order)

    plans = backend.locate_slices(requests)

    assert len(plans) == key_count
    assert all(
        len(plan.offers[0].live_volumes) == source_count for plan in plans.values()
    )
    assert calls == 1


def test_full_span_shard_coverage_reuses_one_structural_template(monkeypatch):
    key_count = 40
    source_count = 24
    backend = IndexedDirectoryBackend()
    requests = []
    for key_index in range(key_count):
        shard_size = key_index + 1
        global_size = source_count * shard_size
        key = f"weight.{key_index}"
        for source_index in range(source_count):
            backend.add_source(
                key,
                f"v{source_index}",
                0,
                _info(
                    _slice(
                        (source_index * shard_size,),
                        (shard_size,),
                        (global_size,),
                        (source_index,),
                        (source_count,),
                    )
                ),
            )
        requests.append(_request(key, _slice((0,), (global_size,), (global_size,))))
    ranked_sources = [(0, f"v{index}") for index in reversed(range(source_count))]
    calls = 0
    intersect = indexed_module.get_slice_intersection

    def counted_intersection(stored, wanted):
        nonlocal calls
        calls += 1
        return intersect(stored, wanted)

    monkeypatch.setattr(indexed_module, "get_slice_intersection", counted_intersection)

    union = backend.serving_union(requests)
    ranked = _CountingRanked(ranked_sources)
    cover = backend.greedy_cover(requests, ranked)

    assert union == frozenset(ranked_sources)
    assert cover == ranked_sources
    assert ranked.iterations == source_count
    assert calls == source_count
    assert len(backend._coverage_cache) == 1
    assert len(backend._template_intern) == 1


def test_controller_class_selection_keeps_legacy_explicit():
    assert get_controller_class("legacy") is Controller
    assert get_controller_class("indexed") is IndexedController


def test_indexed_controller_surface_uses_its_region_directory():
    controller = IndexedController()
    controller.is_initialized = True
    left = _slice((0,), (4,), (8,), (0,), (2,))
    right = _slice((4,), (4,), (8,), (1,), (2,))
    controller._notify_put_batch([_request("model.left", left)], "v0", pending=False)
    controller._notify_put_batch([_request("model.left", right)], "v1", pending=False)

    located = controller._locate(["model.left"])

    assert tuple(located["model.left"]) == ("v0", "v1")
    assert controller._keys("model") == ["model.left"]
    assert controller.get_keys_to_storage_volumes()["model.left"]["v0"][
        0
    ].tensor_slices == {left}

    asyncio.run(controller._teardown())
    assert controller._keys() == []
