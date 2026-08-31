# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from hashlib import blake2b

from torchstore.controllers.indexed.plans import RegionKey

_IntervalKey = tuple[int, int, RegionKey]


@dataclass
class _Node:
    key: _IntervalKey
    priority: int
    max_end: int
    left: "_Node | None" = None
    right: "_Node | None" = None


def _priority(key: _IntervalKey) -> int:
    digest = blake2b(repr(key).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _max_end(node: _Node | None) -> int:
    return -1 if node is None else node.max_end


def _refresh(node: _Node) -> None:
    node.max_end = max(node.key[1], _max_end(node.left), _max_end(node.right))


def _rotate_left(node: _Node) -> _Node:
    child = node.right
    assert child is not None
    node.right = child.left
    child.left = node
    _refresh(node)
    _refresh(child)
    return child


def _rotate_right(node: _Node) -> _Node:
    child = node.left
    assert child is not None
    node.left = child.right
    child.right = node
    _refresh(node)
    _refresh(child)
    return child


def _insert(node: _Node | None, key: _IntervalKey) -> _Node:
    if node is None:
        return _Node(key=key, priority=_priority(key), max_end=key[1])
    if key == node.key:
        return node
    if key < node.key:
        node.left = _insert(node.left, key)
        if node.left.priority < node.priority:
            node = _rotate_right(node)
    else:
        node.right = _insert(node.right, key)
        if node.right.priority < node.priority:
            node = _rotate_left(node)
    _refresh(node)
    return node


def _delete(node: _Node | None, key: _IntervalKey) -> _Node | None:
    if node is None:
        return None
    if key < node.key:
        node.left = _delete(node.left, key)
    elif key > node.key:
        node.right = _delete(node.right, key)
    elif node.left is None:
        return node.right
    elif node.right is None:
        return node.left
    elif node.left.priority < node.right.priority:
        node = _rotate_right(node)
        node.right = _delete(node.right, key)
    else:
        node = _rotate_left(node)
        node.left = _delete(node.left, key)
    _refresh(node)
    return node


def _query(
    node: _Node | None,
    start: int,
    end: int,
    matches: list[RegionKey],
) -> None:
    if node is None or node.max_end <= start:
        return
    if node.left is not None and node.left.max_end > start:
        _query(node.left, start, end, matches)
    node_start, node_end, region = node.key
    if node_start < end and node_end > start:
        matches.append(region)
    if node_start < end:
        _query(node.right, start, end, matches)


class IntervalIndex:
    """Mutable interval trees for every global shape and dimension."""

    def __init__(self) -> None:
        self._roots: dict[tuple[tuple, int], _Node | None] = {}
        self._regions: set[RegionKey] = set()
        self._scalars: dict[tuple, set[RegionKey]] = {}

    def add(self, region: RegionKey) -> None:
        if region in self._regions:
            return
        if not region.global_shape:
            self._scalars.setdefault(region.global_shape, set()).add(region)
        for dim, (offset, length) in enumerate(
            zip(region.offsets, region.local_shape, strict=True)
        ):
            slot = (region.global_shape, dim)
            key = (offset, offset + length, region)
            self._roots[slot] = _insert(self._roots.get(slot), key)
        self._regions.add(region)

    def remove(self, region: RegionKey) -> None:
        if region not in self._regions:
            return
        if not region.global_shape:
            scalars = self._scalars[region.global_shape]
            scalars.remove(region)
            if not scalars:
                del self._scalars[region.global_shape]
        for dim, (offset, length) in enumerate(
            zip(region.offsets, region.local_shape, strict=True)
        ):
            slot = (region.global_shape, dim)
            key = (offset, offset + length, region)
            root = _delete(self._roots.get(slot), key)
            if root is None:
                self._roots.pop(slot, None)
            else:
                self._roots[slot] = root
        self._regions.remove(region)

    def overlap_candidates(self, requested: RegionKey) -> list[RegionKey]:
        if not requested.local_shape:
            return sorted(self._scalars.get(requested.global_shape, ()))
        dim = max(
            range(len(requested.local_shape)), key=requested.local_shape.__getitem__
        )
        start = requested.offsets[dim]
        matches: list[RegionKey] = []
        _query(
            self._roots.get((requested.global_shape, dim)),
            start,
            start + requested.local_shape[dim],
            matches,
        )
        return matches

    def regions(self) -> tuple[RegionKey, ...]:
        return tuple(sorted(self._regions))

    def clear(self) -> None:
        self._roots.clear()
        self._regions.clear()
        self._scalars.clear()
