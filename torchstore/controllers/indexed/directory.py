# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from torchstore.controller import ObjectType, StorageInfo
from torchstore.controllers.indexed.intervals import IntervalIndex
from torchstore.controllers.indexed.plans import (
    BoundPlanOffer,
    PlanOffer,
    Publication,
    RegionKey,
    SlicePlan,
    order_live_volumes,
)
from torchstore.storage_utils.trie import Trie
from torchstore.transport.types import Request, TensorSlice
from torchstore.utils import get_slice_intersection

_SourceSet = frozenset[Publication]
_NormalizedRegion = tuple[tuple[Fraction, ...], tuple[Fraction, ...]]
_TopologyItem = tuple[_NormalizedRegion | None, _SourceSet]
_Topology = frozenset[_TopologyItem]


def _normalized_region(region: RegionKey) -> _NormalizedRegion:
    offsets = tuple(
        Fraction(offset, size) if size else Fraction(0)
        for offset, size in zip(region.offsets, region.global_shape, strict=True)
    )
    lengths = tuple(
        Fraction(length, size) if size else Fraction(0)
        for length, size in zip(region.local_shape, region.global_shape, strict=True)
    )
    return offsets, lengths


class _SourceSetPool:
    def __init__(self) -> None:
        self.empty: _SourceSet = frozenset()
        self._sets: dict[_SourceSet, _SourceSet] = {self.empty: self.empty}
        self._adds: dict[tuple[_SourceSet, Publication], _SourceSet] = {}
        self._removes: dict[tuple[_SourceSet, Publication], _SourceSet] = {}

    def intern(self, sources: _SourceSet) -> _SourceSet:
        held = self._sets.get(sources)
        if held is not None:
            return held
        self._sets[sources] = sources
        return sources

    def add(self, sources: _SourceSet, source: Publication) -> _SourceSet:
        key = sources, source
        held = self._adds.get(key)
        if held is not None:
            return held
        updated = self.intern(sources | {source})
        self._adds[key] = updated
        return updated

    def remove(self, sources: _SourceSet, source: Publication) -> _SourceSet:
        key = sources, source
        held = self._removes.get(key)
        if held is not None:
            return held
        updated = self.intern(sources - {source})
        self._removes[key] = updated
        return updated


class _TopologyPool:
    def __init__(self) -> None:
        self.empty: _Topology = frozenset()
        self._states: dict[_Topology, _Topology] = {self.empty: self.empty}
        self._transitions: dict[
            tuple[_Topology, _TopologyItem | None, _TopologyItem | None],
            _Topology,
        ] = {}

    def replace(
        self,
        topology: _Topology,
        old: _TopologyItem | None,
        new: _TopologyItem | None,
    ) -> _Topology:
        key = topology, old, new
        held = self._transitions.get(key)
        if held is not None:
            return held
        values = set(topology)
        if old is not None:
            values.remove(old)
        if new is not None:
            values.add(new)
        candidate = frozenset(values)
        updated = self._states.setdefault(candidate, candidate)
        self._transitions[key] = updated
        return updated


@dataclass(frozen=True)
class _CoverageTemplate:
    pieces: tuple[_SourceSet, ...]
    whole: bool
    union_sources: _SourceSet


@dataclass
class _LayoutState:
    expected: int
    coordinates: Counter[tuple] = field(default_factory=Counter)
    invalid: int = 0

    def add(self, tensor_slice: TensorSlice) -> None:
        coordinate = tensor_slice.coordinates
        mesh_shape = tensor_slice.mesh_shape
        if (
            coordinate is None
            or mesh_shape is None
            or len(coordinate) != len(mesh_shape)
            or any(
                value < 0 or value >= bound
                for value, bound in zip(coordinate, mesh_shape)
            )
        ):
            self.invalid += 1
            return
        self.coordinates[tuple(coordinate)] += 1

    def remove(self, tensor_slice: TensorSlice) -> None:
        coordinate = tensor_slice.coordinates
        mesh_shape = tensor_slice.mesh_shape
        if (
            coordinate is None
            or mesh_shape is None
            or len(coordinate) != len(mesh_shape)
            or any(
                value < 0 or value >= bound
                for value, bound in zip(coordinate, mesh_shape)
            )
        ):
            self.invalid -= 1
            return
        coordinate = tuple(coordinate)
        self.coordinates[coordinate] -= 1
        if self.coordinates[coordinate] == 0:
            del self.coordinates[coordinate]

    @property
    def complete(self) -> bool:
        return self.invalid == 0 and len(self.coordinates) == self.expected


@dataclass
class _RegionEntry:
    sources: dict[Publication, set[TensorSlice]] = field(default_factory=dict)
    source_ids: _SourceSet = field(default_factory=frozenset)

    def add(self, source: Publication, tensor_slice: TensorSlice) -> bool:
        slices = self.sources.setdefault(source, set())
        before = len(slices)
        slices.add(tensor_slice)
        return len(slices) != before

    def remove(self, source: Publication) -> bool:
        return self.sources.pop(source, None) is not None


@dataclass
class _CachedGeometry:
    epoch: int
    offers: tuple[PlanOffer, ...]


@dataclass
class _KeyEntry:
    object_type: ObjectType
    source_pool: _SourceSetPool
    topology_pool: _TopologyPool
    geometry_epoch: int = 0
    availability: int = 0
    interval_index: IntervalIndex = field(default_factory=IntervalIndex)
    regions: dict[RegionKey, _RegionEntry] = field(default_factory=dict)
    whole_source_ids: _SourceSet = field(default_factory=frozenset)
    source_infos: dict[Publication, StorageInfo] = field(default_factory=dict)
    global_shapes: Counter[tuple] = field(default_factory=Counter)
    layouts: dict[tuple[tuple, tuple, tuple[int, ...]], _LayoutState] = field(
        default_factory=dict
    )
    cache: dict[RegionKey | None, _CachedGeometry] = field(default_factory=dict)
    topology: _Topology = field(default_factory=frozenset)

    @property
    def has_sources(self) -> bool:
        return bool(self.source_infos)

    @property
    def has_live_sources(self) -> bool:
        return any(pub == 0 for pub, _ in self.source_infos)

    @property
    def fully_committed(self) -> bool:
        if self.object_type is not ObjectType.TENSOR_SLICE:
            return any(pub == 0 for pub, _ in self.whole_source_ids)
        return any(state.complete for state in self.layouts.values())

    def _geometry_changed(self) -> None:
        self.geometry_epoch += 1
        self.cache.clear()

    def _replace_topology(
        self,
        region: RegionKey | None,
        old_sources: _SourceSet,
        new_sources: _SourceSet,
    ) -> None:
        normalized = None if region is None else _normalized_region(region)
        old = None if not old_sources else (normalized, old_sources)
        new = None if not new_sources else (normalized, new_sources)
        self.topology = self.topology_pool.replace(self.topology, old, new)

    def _layout_key(
        self, tensor_slice: TensorSlice
    ) -> tuple[tuple, tuple, tuple[int, ...]]:
        global_shape = tuple(tensor_slice.global_shape)
        local_shape = tuple(tensor_slice.local_shape)
        mesh_shape = (
            () if tensor_slice.mesh_shape is None else tuple(tensor_slice.mesh_shape)
        )
        sharded_dimensions = tuple(
            dim
            for dim, (local, global_) in enumerate(
                zip(local_shape, global_shape, strict=True)
            )
            if local != global_
        )
        return global_shape, mesh_shape, sharded_dimensions

    def _add_live_layout(self, tensor_slice: TensorSlice) -> None:
        layout_key = self._layout_key(tensor_slice)
        state = self.layouts.get(layout_key)
        if state is None:
            state = _LayoutState(expected=math.prod(layout_key[1]))
            self.layouts[layout_key] = state
        state.add(tensor_slice)

    def _remove_live_layout(self, tensor_slice: TensorSlice) -> None:
        layout_key = self._layout_key(tensor_slice)
        state = self.layouts[layout_key]
        state.remove(tensor_slice)
        if not state.coordinates and state.invalid == 0:
            del self.layouts[layout_key]

    def add_source(self, source: Publication, info: StorageInfo) -> None:
        assert self.object_type is info.object_type
        held = self.source_infos.get(source)
        if held is None:
            held = StorageInfo(info.object_type)
            self.source_infos[source] = held
        new_slices = info.tensor_slices - held.tensor_slices
        if not new_slices:
            return
        held.tensor_slices.update(new_slices)
        self.availability += 1
        if info.object_type is not ObjectType.TENSOR_SLICE:
            old_sources = self.whole_source_ids
            self.whole_source_ids = self.source_pool.add(old_sources, source)
            self._replace_topology(None, old_sources, self.whole_source_ids)
            was_empty = not old_sources
            if was_empty:
                self._geometry_changed()
            return
        for tensor_slice in new_slices:
            assert tensor_slice is not None
            region = RegionKey.from_slice(tensor_slice)
            entry = self.regions.get(region)
            if entry is None:
                entry = _RegionEntry()
                self.regions[region] = entry
                self.global_shapes[region.global_shape] += 1
                self.interval_index.add(region)
                self._geometry_changed()
            old_sources = entry.source_ids
            entry.add(source, tensor_slice)
            if source not in old_sources:
                entry.source_ids = self.source_pool.add(old_sources, source)
                self._replace_topology(region, old_sources, entry.source_ids)
            if source[0] == 0:
                self._add_live_layout(tensor_slice)

    def remove_source(self, source: Publication) -> None:
        held = self.source_infos.pop(source, None)
        if held is None:
            return
        self.availability += 1
        if held.object_type is not ObjectType.TENSOR_SLICE:
            old_sources = self.whole_source_ids
            self.whole_source_ids = self.source_pool.remove(old_sources, source)
            self._replace_topology(None, old_sources, self.whole_source_ids)
            if not self.whole_source_ids:
                self._geometry_changed()
            return
        regions = set()
        for tensor_slice in held.tensor_slices:
            assert tensor_slice is not None
            regions.add(RegionKey.from_slice(tensor_slice))
            if source[0] == 0:
                self._remove_live_layout(tensor_slice)
        for region in regions:
            entry = self.regions[region]
            old_sources = entry.source_ids
            entry.remove(source)
            if source in old_sources:
                entry.source_ids = self.source_pool.remove(old_sources, source)
                self._replace_topology(region, old_sources, entry.source_ids)
            if not entry.sources:
                del self.regions[region]
                self.global_shapes[region.global_shape] -= 1
                if self.global_shapes[region.global_shape] == 0:
                    del self.global_shapes[region.global_shape]
                self.interval_index.remove(region)
                self._geometry_changed()

    def _build_geometry(self, requested: RegionKey | None) -> tuple[PlanOffer, ...]:
        if self.object_type is not ObjectType.TENSOR_SLICE:
            return (PlanOffer(None, None, None, None, None),)
        candidates = (
            self.interval_index.regions()
            if requested is None
            else self.interval_index.overlap_candidates(requested)
        )
        requested_slice = None if requested is None else requested.as_tensor_slice()
        offers = []
        for region in candidates:
            if requested is not None and region.global_shape != requested.global_shape:
                continue
            intersection = region.as_tensor_slice()
            if requested_slice is not None:
                intersection = get_slice_intersection(intersection, requested_slice)
                if intersection is None:
                    continue
                destination_offsets = tuple(
                    offset - base
                    for offset, base in zip(
                        intersection.offsets, requested.offsets, strict=True
                    )
                )
            else:
                destination_offsets = intersection.offsets
            intersection_region = RegionKey.from_slice(intersection)
            offers.append(
                PlanOffer(
                    source_region=region,
                    intersection=intersection_region,
                    storage_offsets=tuple(
                        offset - base
                        for offset, base in zip(
                            intersection.offsets, region.offsets, strict=True
                        )
                    ),
                    destination_offsets=destination_offsets,
                    lengths=intersection_region.local_shape,
                )
            )
        return tuple(sorted(offers, key=lambda offer: offer.source_region))

    def geometry(self, requested: RegionKey | None) -> tuple[PlanOffer, ...]:
        cached = self.cache.get(requested)
        if cached is not None and cached.epoch == self.geometry_epoch:
            return cached.offers
        offers = self._build_geometry(requested)
        self.cache[requested] = _CachedGeometry(self.geometry_epoch, offers)
        return offers

    def plan(
        self,
        key: str,
        request: Request,
        bind_sources,
        prefer: Sequence[str] | None = None,
    ) -> SlicePlan:
        requested = (
            None
            if request.tensor_slice is None
            else RegionKey.from_slice(request.tensor_slice)
        )
        bound = []
        for offer in self.geometry(requested):
            sources = (
                self.whole_source_ids
                if offer.source_region is None
                else self.regions[offer.source_region].source_ids
            )
            live, pending = bind_sources(
                sources, None if prefer is None else tuple(prefer)
            )
            bound.append(BoundPlanOffer(offer, live, pending))
        return SlicePlan(
            key=key,
            object_type=self.object_type,
            requested_region=requested,
            geometry_epoch=self.geometry_epoch,
            availability=self.availability,
            offers=tuple(bound),
        )


class IndexedDirectoryBackend:
    name = "indexed"
    indexed = True

    def __init__(self) -> None:
        self._keys: Trie = Trie()
        self._source_pool = _SourceSetPool()
        self._topology_pool = _TopologyPool()
        self._bindings: dict[
            tuple[_SourceSet, tuple[str, ...] | None],
            tuple[tuple[str, ...], tuple[Publication, ...]],
        ] = {}
        self._coverage_cache: dict[
            tuple[_Topology, _NormalizedRegion | None, bool], _CoverageTemplate
        ] = {}
        self._template_intern: dict[
            tuple[tuple[_SourceSet, ...], bool], _CoverageTemplate
        ] = {}

    def reset(self) -> None:
        self.__init__()

    def _bind_sources(
        self,
        sources: _SourceSet,
        prefer: tuple[str, ...] | None,
    ) -> tuple[tuple[str, ...], tuple[Publication, ...]]:
        key = sources, prefer
        held = self._bindings.get(key)
        if held is not None:
            return held
        live = order_live_volumes(
            {volume for pub, volume in sources if pub == 0}, prefer
        )
        pending = tuple(sorted(source for source in sources if source[0] != 0))
        result = live, pending
        self._bindings[key] = result
        return result

    def has_key(self, key: str) -> bool:
        return key in self._keys

    def has_source(self, key: str, volume: str, pub: int) -> bool:
        entry = self._keys.get(key)
        return entry is not None and (pub, volume) in entry.source_infos

    def live_volume_map(self, key: str) -> dict[str, StorageInfo]:
        entry = self._keys.get(key)
        if entry is None:
            return {}
        return {
            volume: StorageInfo(info.object_type, set(info.tensor_slices))
            for (pub, volume), info in entry.source_infos.items()
            if pub == 0
        }

    def live_keys(self, prefix: str | None = None) -> list[str]:
        candidates = (
            list(self._keys.keys())
            if prefix is None
            else self._keys.keys().filter_by_prefix(prefix)
        )
        return [key for key in candidates if self._keys[key].has_live_sources]

    def snapshot(self) -> dict[str, dict[str, dict[int, StorageInfo]]]:
        result = {}
        for key, entry in self._keys.items():
            volumes: dict[str, dict[int, StorageInfo]] = {}
            for (pub, volume), info in entry.source_infos.items():
                volumes.setdefault(volume, {})[pub] = StorageInfo(
                    info.object_type, set(info.tensor_slices)
                )
            result[key] = volumes
        return result

    def add_source(
        self,
        key: str,
        volume: str,
        pub: int,
        info: StorageInfo,
    ) -> None:
        entry = self._keys.get(key)
        if entry is None:
            entry = _KeyEntry(
                info.object_type,
                self._source_pool,
                self._topology_pool,
                topology=self._topology_pool.empty,
            )
            self._keys[key] = entry
        entry.add_source((pub, volume), info)

    def remove_source(self, key: str, volume: str, pub: int) -> None:
        entry = self._keys.get(key)
        if entry is None:
            return
        entry.remove_source((pub, volume))
        if not entry.has_sources:
            del self._keys[key]

    def is_fully_committed(self, key: str) -> bool:
        entry = self._keys.get(key)
        return entry is not None and entry.fully_committed

    def locate_slices(
        self,
        requests: Sequence[Request],
        *,
        missing_ok: bool = False,
        require_fully_committed: bool = False,
        include_pending: bool = True,
        prefer: Sequence[str] | None = None,
    ) -> dict[str, SlicePlan]:
        plans = {}
        for request in requests:
            entry = self._keys.get(request.key)
            present = entry is not None and (
                entry.has_sources if include_pending else entry.has_live_sources
            )
            if not present:
                if missing_ok:
                    continue
                raise KeyError(
                    f"Unable to locate {request.key} in any storage volumes."
                )
            assert entry is not None
            if require_fully_committed and not entry.fully_committed:
                raise KeyError(
                    f"DTensor '{request.key}' is only partially committed. "
                    "Not all shards have been stored yet. Please ensure all ranks "
                    "complete their put() operations."
                )
            plan = entry.plan(request.key, request, self._bind_sources, prefer)
            if not include_pending:
                plan = SlicePlan(
                    key=plan.key,
                    object_type=plan.object_type,
                    requested_region=plan.requested_region,
                    geometry_epoch=plan.geometry_epoch,
                    availability=plan.availability,
                    offers=tuple(
                        BoundPlanOffer(offer.geometry, offer.live_volumes, ())
                        for offer in plan.offers
                        if offer.live_volumes
                    ),
                )
            plans[request.key] = plan
        return plans

    def _offer_sources(self, entry: _KeyEntry, offer: PlanOffer) -> _SourceSet:
        if offer.source_region is None:
            return entry.whole_source_ids
        return entry.regions[offer.source_region].source_ids

    def _intern_sources(
        self,
        groups: Iterable[_SourceSet],
    ) -> _SourceSet:
        values: set[Publication] = set()
        for group in groups:
            for source in group:
                values.add(source)
        return self._source_pool.intern(frozenset(values))

    def _coverage_template(
        self,
        request: Request,
    ) -> _CoverageTemplate | None:
        entry = self._keys.get(request.key)
        if entry is None:
            return None
        requested = (
            None
            if request.tensor_slice is None
            else RegionKey.from_slice(request.tensor_slice)
        )
        normalized = None if requested is None else _normalized_region(requested)
        shape_matches = (
            entry.object_type is not ObjectType.TENSOR_SLICE
            or requested is None
            or requested.global_shape in entry.global_shapes
        )
        cache_key = entry.topology, normalized, shape_matches
        cached = self._coverage_cache.get(cache_key)
        if cached is not None:
            return cached

        by_region: dict[_NormalizedRegion | None, list[_SourceSet]] = {}
        offers = () if not shape_matches else entry.geometry(requested)
        for offer in offers:
            intersection = offer.intersection
            region = None if intersection is None else _normalized_region(intersection)
            by_region.setdefault(region, []).append(self._offer_sources(entry, offer))
        ordered = sorted(
            by_region,
            key=lambda region: (region is not None, region),
        )
        pieces = tuple(
            groups[0] if len(groups) == 1 else self._intern_sources(groups)
            for region in ordered
            for groups in (by_region[region],)
        )
        union_sources = (
            self._source_pool.empty
            if not pieces
            else pieces[0]
            if len(pieces) == 1
            else self._intern_sources(pieces)
        )
        whole = bool(ordered) and ordered[0] is None
        signature = pieces, whole
        template = self._template_intern.get(signature)
        if template is None:
            template = _CoverageTemplate(pieces, whole, union_sources)
            self._template_intern[signature] = template
        self._coverage_cache[cache_key] = template
        return template

    def _exact_coverage(
        self,
        request: Request,
    ) -> tuple[tuple[RegionKey | None, _SourceSet], ...]:
        entry = self._keys.get(request.key)
        if entry is None:
            return ()
        requested = (
            None
            if request.tensor_slice is None
            else RegionKey.from_slice(request.tensor_slice)
        )
        grouped: dict[RegionKey | None, list[_SourceSet]] = {}
        for offer in entry.geometry(requested):
            grouped.setdefault(offer.intersection, []).append(
                self._offer_sources(entry, offer)
            )
        ordered = sorted(
            grouped,
            key=lambda region: (region is not None, region),
        )
        return tuple(
            (
                region,
                groups[0] if len(groups) == 1 else self._intern_sources(groups),
            )
            for region in ordered
            for groups in (grouped[region],)
        )

    def serving_union(self, requests: Sequence[Request]) -> frozenset[Publication]:
        sources: set[Publication] = set()
        seen: set[_CoverageTemplate] = set()
        for request in requests:
            template = self._coverage_template(request)
            if template is None or template in seen:
                continue
            seen.add(template)
            for source in template.union_sources:
                sources.add(source)
        return frozenset(sources)

    def greedy_cover(
        self,
        requests: Sequence[Request],
        ranked: Iterable[Publication],
    ) -> list[Publication]:
        rank: dict[Publication, int] = {}
        for source in ranked:
            rank.setdefault(source, len(rank))

        best_cache: dict[_SourceSet, Publication | None] = {}

        def best_source(sources: _SourceSet) -> Publication | None:
            if sources in best_cache:
                return best_cache[sources]
            best = None
            best_rank = len(rank)
            for source in sources:
                position = rank.get(source)
                if position is not None and position < best_rank:
                    best = source
                    best_rank = position
            best_cache[sources] = best
            return best

        def selected_sources(groups: Iterable[_SourceSet]) -> tuple[Publication, ...]:
            selected = {
                source
                for sources in groups
                if (source := best_source(sources)) is not None
            }
            return tuple(sorted(selected, key=rank.__getitem__))

        chosen: dict[Publication, None] = {}
        repeated = Counter(request.key for request in requests)
        seen_regions: dict[str, set[RegionKey | None]] = {}
        selected_templates: dict[_CoverageTemplate, tuple[Publication, ...]] = {}
        for request in requests:
            if repeated[request.key] > 1:
                seen = seen_regions.setdefault(request.key, set())
                exact = self._exact_coverage(request)
                fresh = [sources for region, sources in exact if region not in seen]
                seen.update(region for region, _ in exact)
                selected = selected_sources(fresh)
            else:
                template = self._coverage_template(request)
                if template is None:
                    continue
                selected = selected_templates.get(template)
                if selected is None:
                    selected = selected_sources(template.pieces)
                    selected_templates[template] = selected
                else:
                    continue
            for source in selected:
                chosen.setdefault(source, None)
        return list(chosen)
