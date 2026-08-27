# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import auto, Enum
from itertools import product

from monarch.actor import Actor, endpoint

from torchstore.storage_utils.trie import Trie
from torchstore.storage_volume import StorageVolume
from torchstore.strategy import ControllerStorageVolumes, TorchStoreStrategy
from torchstore.transport.types import Request, TensorSlice


# TODO: move this into request as a field
class ObjectType(Enum):
    OBJECT = auto()
    TENSOR = auto()
    TENSOR_SLICE = auto()

    @classmethod
    def from_request(cls, request: Request) -> "ObjectType":
        if request.is_object:
            return cls.OBJECT
        elif request.tensor_slice is not None:
            return cls.TENSOR_SLICE
        else:
            return cls.TENSOR


@dataclass
class StorageInfo:
    object_type: ObjectType
    tensor_slices: set[TensorSlice | None] = field(default_factory=set)

    def update(self, other_storage_info: "StorageInfo"):
        assert (
            self.object_type == other_storage_info.object_type
        ), "Particularly dangerous to change storage type of an existing key, are you sure? Raise an issue if so."

        self.tensor_slices.update(other_storage_info.tensor_slices)


_ShapeEntry = tuple[str, ObjectType, frozenset[TensorSlice | None]]
_Shape = tuple[_ShapeEntry, ...]


@dataclass
class _Publication:
    volume: str
    keys: frozenset[str]
    shape: _Shape


# Publication identity: the store's own name for "a pub id on a volume".
# pub_id == 0: the volume holds a live entry (already landed sentinel)
# pub_id > 0: an outstanding publication on the volume
Publication = tuple[int, str]


def _live_view(
    volume_map: dict[str, dict[int, StorageInfo]],
) -> dict[str, StorageInfo]:
    return {volume: slot[0] for volume, slot in volume_map.items() if 0 in slot}


class Controller(Actor):
    def __init__(
        self,
    ) -> None:
        self.keys_to_storage_volumes: Trie = Trie()
        self.is_initialized: bool = False
        self.strategy: TorchStoreStrategy | None = None
        self.storage_volumes: StorageVolume | None = None
        self.num_storage_volumes: int | None = None
        self.strategy: TorchStoreStrategy | None = None
        self._next_pub = 1
        self._publications: dict[int, _Publication] = {}
        self._shape_pubs: dict[_Shape, set[int]] = {}

    def assert_initialized(self) -> None:
        assert (
            self.is_initialized
        ), "Please call torchstore.initialize before attempting to use store."

    def _is_dtensor_fully_committed(
        self, key: str, volume_map: dict[str, StorageInfo]
    ) -> bool:
        """
        Check if all shards of a DTensor have been committed.

        For a DTensor to be fully committed, we need all coordinates in the mesh
        to have been stored. The mesh_shape tells us the total number of shards,
        and coordinates tell us which shards we have.

        Args:
            key (str): The key to check.
            volume_map (Dict[str, StorageInfo]): Mapping from storage volume IDs to StorageInfo.

        Returns:
            bool: True if fully committed, False if partial.
        """
        # Collect all tensor slices across all storage volumes
        all_slices = set()
        mesh_shape = None

        if not volume_map:
            return False
        for storage_info in volume_map.values():
            if storage_info.object_type != ObjectType.TENSOR_SLICE:
                return True  # Not a DTensor, so it's "fully committed"

            for tensor_slice in storage_info.tensor_slices:
                all_slices.add(tensor_slice.coordinates)
                if mesh_shape is None:
                    mesh_shape = tensor_slice.mesh_shape
                else:
                    assert (
                        mesh_shape == tensor_slice.mesh_shape
                    ), "Inconsistent mesh shapes in stored slices"

        # Generate all expected coordinates for the mesh
        expected_coords = set(product(*(range(s) for s in mesh_shape)))

        # Check if we have all coordinates
        return all_slices == expected_coords

    @endpoint
    async def init(
        self,
        strategy: TorchStoreStrategy,
        num_storage_volumes: int,
        storage_volumes: StorageVolume,
    ) -> None:
        if self.is_initialized:
            raise RuntimeError("TorchStore is already initialized")

        if isinstance(strategy, ControllerStorageVolumes):
            warnings.warn(
                "ControllerStorageVolumes is deprecated and will be removed in a future "
                "release. It spawns a singleton storage volume on the controller, which "
                "may become a bottleneck. Use LocalRankStrategy for better scalability.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.strategy = strategy
        self.storage_volumes = storage_volumes
        self.num_storage_volumes = num_storage_volumes

        await self.strategy.set_storage_volumes(self.storage_volumes)
        self.is_initialized = True

    @endpoint
    async def get_controller_strategy(self) -> TorchStoreStrategy:
        self.assert_initialized()
        assert self.strategy is not None, "Strategy is not set"
        return self.strategy

    @endpoint
    async def locate_volumes(
        self,
        keys: list[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
        prefer: Sequence[str] | None = None,
    ) -> dict[str, dict[str, StorageInfo]]:
        """Locate storage volumes containing shards of the specified keys.

        Returns {<key> -> {<storage_volume_id> -> StorageInfo}} where each key maps to
        the storage volumes holding shards of its data.

        For example, if a key holds a DTensor with 3 shards, the returned map will look like:
        {
            "<dtensor_fqn>": {
                "<storage_volume_id>": StorageInfo.tensor_slice=set([
                    "<tensor_slice>",
                    "<tensor_slice>",
                    "<tensor_slice>",
                ]),
                ...
            },
            ...
        }

        Args:
            keys (list[str]): The keys to locate in storage volumes.
            missing_ok (bool): If True, omit missing keys instead of raising.
            require_fully_committed (bool): If True, reject partially committed
                DTensor entries.

        Returns:
            Dict[str, Dict[str, StorageInfo]]: Mapping from each key to a mapping from
                storage volume IDs to StorageInfo objects containing metadata about
                the stored data shards.

        Raises:
            KeyError: If any key is not found in any storage volumes and
                missing_ok is False, or if a key is a DTensor that is only
                partially committed and require_fully_committed is True.
        """
        return self._locate(keys, missing_ok, require_fully_committed, prefer=prefer)

    def _locate(
        self,
        keys: Sequence[str],
        missing_ok: bool = False,
        require_fully_committed: bool = True,
        *,
        prefer: Sequence[str] | None = None,
    ) -> dict[str, dict[str, StorageInfo]]:
        self.assert_initialized()
        result = {}
        for key in keys:
            if key not in self.keys_to_storage_volumes:
                if missing_ok:
                    continue
                raise KeyError(f"Unable to locate {key} in any storage volumes.")
            volume_map = _live_view(self.keys_to_storage_volumes[key])
            if not volume_map:
                if missing_ok:
                    continue
                raise KeyError(f"Unable to locate {key} in any storage volumes.")
            if require_fully_committed and not self._is_dtensor_fully_committed(
                key, volume_map
            ):
                raise KeyError(
                    f"DTensor '{key}' is only partially committed. "
                    f"Not all shards have been stored yet. "
                    f"Please ensure all ranks complete their put() operations."
                )
            result[key] = self._prefer(volume_map, prefer)
        return result

    @endpoint
    async def notify_put_batch(
        self,
        requests: list[Request],
        storage_volume_id: str,
        *,
        pending: bool = True,
    ) -> int:
        """Notify the controller that data has been stored in a storage volume.

        This should called after a successful put operation to
        maintain the distributed storage index.

        Args:
            requests: List of Requests (meta-only, no tensor data).
            storage_volume_id: ID of the storage volume where the data was stored.
        """
        return self._notify_put_batch(requests, storage_volume_id, pending=pending)

    def _notify_put_batch(
        self,
        requests: Sequence[Request],
        storage_volume_id: str,
        *,
        pending: bool = True,
    ) -> int:
        self.assert_initialized()
        pub = self._allocate_publication() if pending else 0
        grouped = self._group_requests(requests)
        if pending:
            for key, info in grouped.items():
                self._slot(key, storage_volume_id)[pub] = info
            shape = self._shape(grouped)
            shape_pubs = self._shape_pubs.get(shape)
            if shape_pubs is None:
                shape_pubs = set()
                self._shape_pubs[shape] = shape_pubs
            else:
                shape = next(
                    indexed for indexed in self._shape_pubs if indexed == shape
                )
            grouped.clear()
            self._publications[pub] = _Publication(
                storage_volume_id,
                frozenset(entry[0] for entry in shape),
                shape,
            )
            shape_pubs.add(pub)
        else:
            for key, info in grouped.items():
                self._put_live_info(key, info, storage_volume_id)
        return pub

    def _notify_put(
        self,
        request: Request,
        storage_volume_id: str,
        *,
        pending: bool = True,
    ) -> int:
        assert (
            request.tensor_val is None
        ), "request should not contain tensor data, as this will significantly increase e2e latency"
        return self._notify_put_batch(
            [request], storage_volume_id, pending=pending
        )

    def _put_live_info(
        self,
        key: str,
        info: StorageInfo,
        storage_volume_id: str,
    ) -> None:
        slot = self._slot(key, storage_volume_id)
        held = slot.get(0)
        if held is None:
            slot[0] = info
        else:
            held.update(info)

    @endpoint
    async def teardown(self) -> None:
        self.is_initialized = False
        self.keys_to_storage_volumes = Trie()
        self._next_pub = 1
        self._publications.clear()
        self._shape_pubs.clear()
        self.strategy = None
        # StorageVolume in ControllerStrategy can be reused because it was spawned with get_or_spawn_controller.
        # So we have to reset it, otherwise new TensorSlice values for the same key will get piled up in the set.
        if self.storage_volumes is not None:
            await self.storage_volumes.reset.call()
        self.storage_volumes = None
        self.num_storage_volumes = None

    @endpoint
    async def keys(self, prefix=None) -> list[str]:
        return self._keys(prefix)

    def _keys(self, prefix=None) -> list[str]:
        candidates = (
            list(self.keys_to_storage_volumes.keys())
            if prefix is None
            else self.keys_to_storage_volumes.keys().filter_by_prefix(prefix)
        )
        return [
            key
            for key in candidates
            if _live_view(self.keys_to_storage_volumes[key])
        ]

    @endpoint
    async def notify_delete(self, key: str, storage_volume_id: str) -> None:
        """
        Notify the controller that deletion of data is initiated in a storage volume.

        This should called after a successful delete operation to
        maintain the distributed storage index.
        """
        self.assert_initialized()
        self._notify_delete(key, storage_volume_id)

    def _notify_delete(
        self,
        key: str,
        storage_volume_id: str,
        missing_ok: bool = False,
    ) -> None:
        """Remove one key-to-volume mapping from the controller index."""
        if key not in self.keys_to_storage_volumes:
            if missing_ok:
                return
            raise KeyError(f"Unable to locate {key} in any storage volumes.")
        volume_map = self.keys_to_storage_volumes[key]
        slot = volume_map.get(storage_volume_id)
        if slot is None or 0 not in slot:
            if missing_ok:
                return
            raise KeyError(
                f"Unable to locate {key} in storage volume {storage_volume_id}."
            )
        del slot[0]
        self._cleanup_slot(key, storage_volume_id)

    @endpoint
    async def notify_delete_batch(
        self,
        volume_to_keys: dict[str, list[str]] | None = None,
        *,
        pub: int | None = None,
    ) -> None:
        """Notify the controller about an idempotent batch delete."""
        self.assert_initialized()
        self._notify_delete_batch(volume_to_keys, pub=pub)

    def _notify_delete_batch(
        self,
        volume_to_keys: dict[str, list[str]] | None = None,
        *,
        pub: int | None = None,
    ) -> None:
        if pub is not None:
            self._retire(pub)
            return
        if volume_to_keys is None:
            return
        for storage_volume_id, keys in volume_to_keys.items():
            for key in keys:
                self._notify_delete(key, storage_volume_id, missing_ok=True)

    def serving_union(
        self, requests: Sequence[Request]
    ) -> frozenset[Publication]:
        """Live and pending sources overlapping any requested region."""
        from torchstore import coverage

        wanted = self._group_requests(requests)
        sources: set[Publication] = set()
        for key, request_info in wanted.items():
            for volume, slot in self.keys_to_storage_volumes.get(key, {}).items():
                info = slot.get(0)
                if info is not None and coverage._overlaps(request_info, info):
                    sources.add((0, volume))
        for shape, shape_pubs in self._shape_pubs.items():
            if self._same_shape(shape, wanted):
                for pub in shape_pubs:
                    sources.add((pub, self._publications[pub].volume))
                continue
            for entry_key, object_type, tensor_slices in shape:
                request_info = wanted.get(entry_key)
                if request_info is None:
                    continue
                if not coverage._overlaps(
                    request_info, StorageInfo(object_type, set(tensor_slices))
                ):
                    continue
                for pub in shape_pubs:
                    sources.add((pub, self._publications[pub].volume))
                break
        return frozenset(sources)

    def greedy_cover(
        self,
        requests: Sequence[Request],
        ranked: Iterable[Publication],
    ) -> list[Publication]:
        """Per-key/per-slice cover over ranked live and pending sources."""
        from torchstore import coverage

        source_maps: dict[str, dict[Publication, StorageInfo]] = {
            request.key: {} for request in requests
        }
        wanted = self._group_requests(requests)
        remaining = dict.fromkeys(source_maps)
        for source in ranked:
            pub, volume = source
            publication = None if pub == 0 else self._publications.get(pub)
            if publication is not None and publication.volume != volume:
                publication = None
            for key in tuple(remaining):
                slot = self.keys_to_storage_volumes.get(key, {}).get(volume, {})
                info = slot.get(pub) if pub == 0 or publication is not None else None
                if info is None or not coverage._overlaps(wanted[key], info):
                    continue
                source_maps[key][source] = info
                if info.object_type is not ObjectType.TENSOR_SLICE:
                    del remaining[key]
            if not remaining:
                break
        return coverage.cover(requests, source_maps)

    def _allocate_publication(self) -> int:
        # Publication 0 is the terminal sentinel for a live source.
        pub = self._next_pub
        self._next_pub += 1
        return pub

    @staticmethod
    def _group_requests(requests: Sequence[Request]) -> dict[str, StorageInfo]:
        grouped: dict[str, StorageInfo] = {}
        for request in requests:
            assert request.tensor_val is None, (
                "request should not contain tensor data, as this will significantly "
                "increase e2e latency"
            )
            info = StorageInfo(
                ObjectType.from_request(request), {request.tensor_slice}
            )
            held = grouped.get(request.key)
            if held is None:
                grouped[request.key] = info
            else:
                held.update(info)
        return grouped

    @staticmethod
    def _shape(rows: Mapping[str, StorageInfo]) -> _Shape:
        return tuple(
            (key, info.object_type, frozenset(info.tensor_slices))
            for key, info in rows.items()
        )

    @staticmethod
    def _same_shape(shape: _Shape, rows: Mapping[str, StorageInfo]) -> bool:
        if len(shape) != len(rows):
            return False
        return all(
            key == row_key
            and object_type is info.object_type
            and tensor_slices == info.tensor_slices
            for (key, object_type, tensor_slices), (row_key, info) in zip(
                shape, rows.items()
            )
        )

    def _slot(self, key: str, volume: str) -> dict[int, StorageInfo]:
        if key not in self.keys_to_storage_volumes:
            self.keys_to_storage_volumes[key] = {}
        return self.keys_to_storage_volumes[key].setdefault(volume, {})

    def _cleanup_slot(self, key: str, volume: str) -> None:
        volume_map = self.keys_to_storage_volumes.get(key)
        if volume_map is None:
            return
        slot = volume_map.get(volume)
        if slot:
            return
        volume_map.pop(volume, None)
        if not volume_map:
            del self.keys_to_storage_volumes[key]

    def _retire(self, pub: int) -> None:
        publication = self._publications.get(pub)
        if publication is None:
            return
        for key in publication.keys:
            slot = self.keys_to_storage_volumes.get(key, {}).get(
                publication.volume
            )
            if slot is not None:
                slot.pop(pub, None)
                self._cleanup_slot(key, publication.volume)
        pubs = self._shape_pubs[publication.shape]
        pubs.discard(pub)
        if not pubs:
            del self._shape_pubs[publication.shape]
        del self._publications[pub]

    @staticmethod
    def _prefer(
        volumes: dict[str, StorageInfo], prefer: Sequence[str] | None
    ) -> dict[str, StorageInfo]:
        if prefer is None:
            return volumes
        ranked = {volume: volumes[volume] for volume in prefer if volume in volumes}
        return ranked if ranked else volumes

    def get_keys_to_storage_volumes(
        self,
    ) -> Mapping[str, dict[str, dict[int, StorageInfo]]]:
        return self.keys_to_storage_volumes
