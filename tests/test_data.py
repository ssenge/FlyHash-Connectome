"""Checks against the built connectome graph. Skipped until `flypath build`."""

from __future__ import annotations

import numpy as np
import pytest

from flypath import config, data

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def graph():
    cfg = config.load()
    if not (cfg.graph_dir / "nodes.parquet").exists():
        pytest.skip("no built graph; run: python -m flypath build")
    return data.load_graph(cfg)


def test_node_count_in_range(graph):
    assert 160_000 < graph.n < 170_000
    assert graph.n == graph.stats["n_neurons"]


def test_body_ids_sorted_unique(graph):
    assert np.all(np.diff(graph.body_ids) > 0)


def test_edges_in_range_and_no_self_loops(graph):
    assert graph.indices.min() >= 0
    assert graph.indices.max() < graph.n
    rows = np.repeat(np.arange(graph.n), np.diff(graph.indptr))
    assert not (rows == graph.indices).any()
    assert graph.weight.min() >= graph.stats["min_weight"]


def test_olfactory_classes_are_present(graph):
    """The two populations the experiment needs."""
    n = graph.nodes
    assert (n["class"] == "ALPN").sum() > 500           # projection neurons
    assert (n["class"] == "Kenyon_Cell").sum() > 3000   # mushroom body


def test_both_hemispheres_are_annotated(graph):
    n = graph.nodes[graph.nodes["class"] == "Kenyon_Cell"]
    assert set(n["somaSide"].dropna().unique()) >= {"L", "R"}
    assert n["somaSide"].value_counts().min() > 1500


def test_selector_rejects_unknown_keys(graph):
    with pytest.raises(ValueError):
        graph.select({"nope": 1})
    with pytest.raises(ValueError):
        graph.select({"type": "definitely-not-a-cell-type"})
