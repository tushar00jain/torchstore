from torchstore.storage_utils.trie import Trie


def test_filter_by_prefix_accepts_slash_delimited_keys() -> None:
    trie = Trie()
    request = "model/direct/rdma4py/rank_0/requests/client-a"
    response = "model/direct/rdma4py/rank_0/responses/client-a"
    trie[request] = 1
    trie[response] = 2

    assert trie.keys().filter_by_prefix(
        "model/direct/rdma4py/rank_0/requests/"
    ) == [request]


def test_filter_by_prefix_preserves_dot_delimited_behavior() -> None:
    trie = Trie({"abc": 1, "abc.xyz": 2, "xyz": 3})

    assert set(trie.keys().filter_by_prefix("abc")) == {"abc", "abc.xyz"}
