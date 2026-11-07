# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio

import pytest
import torch
import torchstore.api as api
from torchstore.state_dict_utils import (
    _state_dict_storage_metadata,
    get_state_dict,
    put_state_dict,
)


def test_state_dict_metadata() -> None:
    """Describe nested tensors with their storage keys and wire dtypes."""
    slices, dtypes, mapping = _state_dict_storage_metadata(
        {
            "layer": {"weight": torch.empty(3, 5)},
            "running": torch.empty(2, dtype=torch.float32),
        },
        "model",
        transfer_dtype=torch.bfloat16,
        preserve_dtype_keys=frozenset({"running"}),
    )

    weight = slices["model/layer.weight"]
    assert weight.offsets == (0, 0)
    assert weight.global_shape == (3, 5)
    assert weight.local_shape == (3, 5)
    assert dtypes == {
        "model/layer.weight": torch.bfloat16,
        "model/running": torch.float32,
    }
    assert mapping == {"layer.weight": ("layer", "weight"), "running": ("running",)}


def test_state_dict_metadata_rejects_objects() -> None:
    """Reject values that cannot participate in tensor routing."""
    with pytest.raises(TypeError, match="only supports tensor values.*'epoch'.*int"):
        _state_dict_storage_metadata({"epoch": 1}, "model")


def test_put_waits_for_requester_layouts_and_defers_dtype_conversion() -> None:
    """Routing stores receive native tensors after requester layouts are ready."""

    class Client:
        stages_state_dict_dtypes = True

        def __init__(self) -> None:
            self.calls = []
            self.entries = None

        async def register_layout(
            self,
            state_dict,
            key,
            *,
            transfer_dtype=None,
            preserve_dtype_keys=frozenset(),
        ):
            self.calls.append(("register_layout", transfer_dtype, preserve_dtype_keys))

        async def get_requester_layouts(self, key):
            self.calls.append(("get_requester_layouts", key))

        async def put_batch(self, entries):
            self.calls.append(("put_batch", tuple(entries)))
            self.entries = entries

        async def put(self, key, value):
            self.calls.append(("put", key))

    client = Client()
    weight = torch.arange(4, dtype=torch.float32)

    asyncio.run(
        put_state_dict(
            client,
            {"weight": weight},
            "model",
            transfer_dtype=torch.bfloat16,
            preserve_dtype_keys=frozenset({"running"}),
        )
    )

    assert client.calls[:2] == [
        ("register_layout", None, frozenset({"running"})),
        ("get_requester_layouts", "model"),
    ]
    assert client.entries["model/weight"] is weight


def test_put_keeps_client_side_dtype_conversion_for_standard_stores() -> None:
    """Standard stores retain the existing transfer-dtype behavior."""

    class Client:
        stages_state_dict_dtypes = False

        def __init__(self) -> None:
            self.entries = None

        async def register_layout(
            self,
            state_dict,
            key,
            *,
            transfer_dtype=None,
            preserve_dtype_keys=frozenset(),
        ):
            pass

        async def get_requester_layouts(self, key):
            pass

        async def put_batch(self, entries):
            self.entries = entries

        async def put(self, key, value):
            pass

    client = Client()
    asyncio.run(
        put_state_dict(
            client,
            {
                "weight": torch.arange(4, dtype=torch.float32),
                "running": torch.arange(2, dtype=torch.float32),
            },
            "model",
            transfer_dtype=torch.bfloat16,
            preserve_dtype_keys=frozenset({"running"}),
        )
    )

    assert client.entries["model/weight"].dtype == torch.bfloat16
    assert client.entries["model/running"].dtype == torch.float32


def test_prepare_state_dict_only_registers_requester_layout(monkeypatch) -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = []

        async def register_requester_layout(
            self,
            state_dict,
            key,
        ):
            self.calls.append((state_dict, key))

    client = Client()

    async def get_client(store_name):
        assert store_name == "weights"
        return client

    monkeypatch.setattr(api, "client", get_client)
    state_dict = {"weight": torch.empty(2)}

    asyncio.run(
        api.prepare_state_dict(
            "model",
            state_dict,
            store_name="weights",
        )
    )

    assert client.calls == [(state_dict, "model")]


def test_api_put_state_dict_forwards_preserved_dtype_keys(monkeypatch) -> None:
    client = object()
    calls = []

    async def get_client(store_name):
        assert store_name == "weights"
        return client

    async def put(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(api, "client", get_client)
    monkeypatch.setattr(api.torchstore.state_dict_utils, "put_state_dict", put)
    state_dict = {"weight": torch.empty(2)}
    preserve_dtype_keys = frozenset({"weight"})

    asyncio.run(
        api.put_state_dict(
            state_dict,
            "model",
            store_name="weights",
            transfer_dtype=torch.bfloat16,
            preserve_dtype_keys=preserve_dtype_keys,
        )
    )

    assert calls == [
        {
            "store": client,
            "state_dict": state_dict,
            "key": "model",
            "direct_rdma": False,
            "transfer_dtype": torch.bfloat16,
            "preserve_dtype_keys": preserve_dtype_keys,
        }
    ]


def test_strict_state_dict_mismatch() -> None:
    """Explain that strict reads cannot use a rank-local routed layout."""

    class Client:
        async def get_layouts(self, state_dict, key, *, strict=True):
            pass

        async def get(self, key):
            return {"weight": ("weight",)}

    with pytest.raises(AssertionError, match="routing does not support strict=True"):
        asyncio.run(
            get_state_dict(
                Client(),
                "model",
                user_state_dict={"other": torch.empty(1)},
            )
        )
