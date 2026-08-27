# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from monarch.actor import Actor, endpoint

from torchstore.controller import ObjectType, Publication, StorageInfo
from torchstore.controllers.indexed.directory import IndexedDirectoryBackend
from torchstore.controllers.indexed.plans import SlicePlan
from torchstore.storage_volume import StorageVolume
from torchstore.strategy import ControllerStorageVolumes, TorchStoreStrategy
from torchstore.transport.types import Request


@dataclass(frozen=True)
class _Publication:
    volume: str
    keys: frozenset[str]


class IndexedController(Actor):
    """Controller backed by a logical-region directory."""

    def __init__(self) -> None:
        self.is_initialized = False
        self.strategy: TorchStoreStrategy | None = None
        self.storage_volumes: StorageVolume | None = None
        self.num_storage_volumes: int | None = None
        self._directory = IndexedDirectoryBackend()
        self._next_pub = 1
        self._publications: dict[int, _Publication] = {}

    def assert_initialized(self) -> None:
        assert self.is_initialized, (
            "Please call torchstore.initialize before attempting to use store."
        )

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
        await strategy.set_storage_volumes(storage_volumes)
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
            volumes = self._directory.live_volume_map(key)
            if not volumes:
                if missing_ok:
                    continue
                raise KeyError(f"Unable to locate {key} in any storage volumes.")
            if require_fully_committed and not self._directory.is_fully_committed(key):
                raise KeyError(
                    f"DTensor '{key}' is only partially committed. "
                    f"Not all shards have been stored yet. "
                    f"Please ensure all ranks complete their put() operations."
                )
            result[key] = self._prefer(volumes, prefer)
        return result

    @endpoint
    async def locate_slices(
        self,
        requests: list[Request],
        missing_ok: bool = False,
        require_fully_committed: bool = False,
        include_pending: bool = True,
        prefer: Sequence[str] | None = None,
    ) -> dict[str, SlicePlan]:
        """Return logical overlap plans with current source availability."""
        self.assert_initialized()
        return self._directory.locate_slices(
            requests,
            missing_ok=missing_ok,
            require_fully_committed=require_fully_committed,
            include_pending=include_pending,
            prefer=prefer,
        )

    @endpoint
    async def notify_put_batch(
        self,
        requests: list[Request],
        storage_volume_id: str,
        *,
        pending: bool = True,
    ) -> int:
        return self._notify_put_batch(requests, storage_volume_id, pending=pending)

    def _notify_put_batch(
        self,
        requests: Sequence[Request],
        storage_volume_id: str,
        *,
        pending: bool = True,
    ) -> int:
        self.assert_initialized()
        grouped = self._group_requests(requests)
        pub = self._allocate_publication() if pending else 0
        for key, info in grouped.items():
            self._directory.add_source(key, storage_volume_id, pub, info)
        if pending:
            self._publications[pub] = _Publication(
                storage_volume_id, frozenset(grouped)
            )
        return pub

    def _notify_put(
        self,
        request: Request,
        storage_volume_id: str,
        *,
        pending: bool = True,
    ) -> int:
        assert request.tensor_val is None, (
            "request should not contain tensor data, as this will significantly increase e2e latency"
        )
        return self._notify_put_batch([request], storage_volume_id, pending=pending)

    @endpoint
    async def notify_delete(self, key: str, storage_volume_id: str) -> None:
        self.assert_initialized()
        self._notify_delete(key, storage_volume_id)

    def _notify_delete(
        self,
        key: str,
        storage_volume_id: str,
        missing_ok: bool = False,
    ) -> None:
        if not self._directory.has_source(key, storage_volume_id, 0):
            if missing_ok:
                return
            if not self._directory.has_key(key):
                raise KeyError(f"Unable to locate {key} in any storage volumes.")
            raise KeyError(
                f"Unable to locate {key} in storage volume {storage_volume_id}."
            )
        self._directory.remove_source(key, storage_volume_id, 0)

    @endpoint
    async def notify_delete_batch(
        self,
        volume_to_keys: dict[str, list[str]] | None = None,
        *,
        pub: int | None = None,
    ) -> None:
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

    def serving_union(self, requests: Sequence[Request]) -> frozenset[Publication]:
        return self._directory.serving_union(requests)

    def greedy_cover(
        self,
        requests: Sequence[Request],
        ranked: Iterable[Publication],
    ) -> list[Publication]:
        return self._directory.greedy_cover(requests, ranked)

    @endpoint
    async def keys(self, prefix=None) -> list[str]:
        return self._keys(prefix)

    def _keys(self, prefix=None) -> list[str]:
        return self._directory.live_keys(prefix)

    @endpoint
    async def teardown(self) -> None:
        await self._teardown()

    async def _teardown(self) -> None:
        self.is_initialized = False
        self._directory.reset()
        self._next_pub = 1
        self._publications.clear()
        self.strategy = None
        if self.storage_volumes is not None:
            await self.storage_volumes.reset.call()
        self.storage_volumes = None
        self.num_storage_volumes = None

    def _allocate_publication(self) -> int:
        pub = self._next_pub
        self._next_pub += 1
        return pub

    def _retire(self, pub: int) -> None:
        publication = self._publications.pop(pub, None)
        if publication is None:
            return
        for key in publication.keys:
            self._directory.remove_source(key, publication.volume, pub)

    @staticmethod
    def _group_requests(requests: Sequence[Request]) -> dict[str, StorageInfo]:
        grouped: dict[str, StorageInfo] = {}
        for request in requests:
            assert request.tensor_val is None, (
                "request should not contain tensor data, as this will significantly "
                "increase e2e latency"
            )
            info = StorageInfo(ObjectType.from_request(request), {request.tensor_slice})
            held = grouped.get(request.key)
            if held is None:
                grouped[request.key] = info
            else:
                held.update(info)
        return grouped

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
        return self._directory.snapshot()

    def get_directory_backend(self) -> IndexedDirectoryBackend:
        return self._directory
