# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exchanges publisher and requester layouts for precomputed routing."""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

from monarch.actor import (  # type: ignore[import-untyped]
    Actor,
    concurrent_endpoint,
    endpoint,
)

from torchstore.strategy import TorchStoreStrategy
from ._model import KeyRegistration, Registrations

__all__ = ["RoutingCoordinator"]

_LAYOUT_REGISTRATION_TIMEOUT_S = 5 * 60


@dataclass
class _Layouts:
    """Publisher and requester registrations for one state-dict namespace."""

    publishers: Registrations = field(default_factory=dict)
    # Coordinate idempotent publisher registration with requester reads:
    # - The first call from a publisher stores its layout.
    # - Later calls from that publisher leave the stored layout unchanged.
    # - Once all publishers have registered, notify every requester; subsequent
    #   requesters satisfy the condition immediately and receive the same layouts.
    requester_layouts: Registrations = field(default_factory=dict)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)


class RoutingCoordinator(Actor):
    """Exchanges fixed layouts between publishers and requesters."""

    def __init__(self) -> None:
        self._publishers: frozenset[str] | None = None
        self._strategy: TorchStoreStrategy | None = None
        self._layouts: dict[str, _Layouts] = {}

    @endpoint
    async def init(
        self,
        publishers: Collection[str],
        strategy: TorchStoreStrategy,
    ) -> None:
        self._publishers = frozenset(publishers)
        self._strategy = strategy
        self._layouts = {}

    @endpoint
    async def strategy(self) -> TorchStoreStrategy:
        """The strategy resolving every participating volume mesh."""
        if self._strategy is None:
            raise RuntimeError("RoutingCoordinator.init has not run")
        return self._strategy

    @concurrent_endpoint
    async def register_layout(
        self,
        rank: str,
        key: str,
        registrations: Mapping[str, KeyRegistration],
    ) -> None:
        """Record one publisher's layout without waiting for other ranks."""
        if self._publishers is None:
            raise RuntimeError("RoutingCoordinator.init has not run")
        if rank not in self._publishers:
            raise KeyError(f"{rank!r} is not a publisher rank")
        layouts = self._layouts.setdefault(key, _Layouts())
        async with layouts.condition:
            if rank not in layouts.publishers:
                layouts.publishers[rank] = dict(registrations)
                if layouts.publishers.keys() == self._publishers:
                    layouts.condition.notify_all()

    @concurrent_endpoint
    async def get_layouts(
        self,
        key: str,
    ) -> Registrations:
        """Wait for and return every publisher layout for ``key``."""
        if self._publishers is None:
            raise RuntimeError("RoutingCoordinator.init has not run")
        layouts = self._layouts.setdefault(key, _Layouts())
        async with layouts.condition:
            try:
                await asyncio.wait_for(
                    layouts.condition.wait_for(
                        lambda: layouts.publishers.keys() == self._publishers
                    ),
                    timeout=_LAYOUT_REGISTRATION_TIMEOUT_S,
                )
            except asyncio.TimeoutError as error:
                missing = self._publishers - layouts.publishers.keys()
                raise TimeoutError(
                    f"timed out after {_LAYOUT_REGISTRATION_TIMEOUT_S} seconds "
                    f"waiting for publisher layouts for {key!r}; "
                    f"missing publishers: {sorted(missing)}"
                ) from error
        return layouts.publishers

    @concurrent_endpoint
    async def register_requester_layout(
        self,
        rank: str,
        key: str,
        registrations: Mapping[str, KeyRegistration],
    ) -> None:
        """Record one requester's layout without waiting for other ranks."""
        if self._publishers is None:
            raise RuntimeError("RoutingCoordinator.init has not run")
        layouts = self._layouts.setdefault(key, _Layouts())
        async with layouts.condition:
            if rank not in layouts.requester_layouts:
                layouts.requester_layouts[rank] = dict(registrations)
                layouts.condition.notify_all()

    @concurrent_endpoint
    async def get_requester_layouts(
        self,
        key: str,
    ) -> Registrations:
        """Wait for and return every requester layout for ``key``."""
        if self._publishers is None:
            raise RuntimeError("RoutingCoordinator.init has not run")
        layouts = self._layouts.setdefault(key, _Layouts())
        async with layouts.condition:
            try:
                await asyncio.wait_for(
                    layouts.condition.wait_for(
                        lambda: bool(layouts.requester_layouts)
                    ),
                    timeout=_LAYOUT_REGISTRATION_TIMEOUT_S,
                )
            except asyncio.TimeoutError as error:
                raise TimeoutError(
                    f"timed out after {_LAYOUT_REGISTRATION_TIMEOUT_S} seconds "
                    f"waiting for requester layouts for {key!r}"
                ) from error
        return layouts.requester_layouts
