"""Download the MaleCNS v1.0 bulk files and build the connectome graph."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import scipy.sparse as sp

from .config import Config

NODE_COLS = [
    "idx", "bodyId", "type", "instance", "superclass", "class",
    "somaSide", "nt", "sign", "x", "y", "z",
]


# --------------------------------------------------------------------------- download

def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {dest.name}")

    last = [-1]

    def hook(blocks: int, bs: int, total: int) -> None:
        if total <= 0:
            return
        pct = int(min(100.0, 100.0 * blocks * bs / total))
        if pct > last[0]:                     # one line per percent, not per block
            last[0] = pct
            print(f"\r    {pct:3d}%  of {total / 1e6:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, part, reporthook=hook)
    print()
    part.rename(dest)


def fetch_raw(cfg: Config) -> dict[str, Path]:
    base = cfg["data"]["base_url"].rstrip("/")
    out = {}
    for key, name in cfg["data"]["files"].items():
        dest = cfg.raw_dir / name
        _download(f"{base}/{name}", dest)
        out[key] = dest
    return out


# --------------------------------------------------------------------------- nodes

def build_nodes(cfg: Config, paths: dict[str, Path]) -> pd.DataFrame:
    ann = pd.read_feather(
        paths["annotations"],
        columns=["bodyId", "type", "instance", "superclass", "class",
                 "somaSide", "status", "somaLocation"],
    )
    keep = ann["status"] == cfg["build"]["status"]
    nodes = ann[keep].drop(columns=["status"]).reset_index(drop=True)
    n = len(nodes)
    print(f"  traced neurons: {n:,}")
    if not 160_000 < n < 170_000:
        raise SystemExit(
            f"expected 160k-170k traced neurons, got {n:,}. "
            "The dataset or the status filter changed; check config build.status."
        )

    nt = pd.read_feather(paths["neurotransmitters"], columns=["body", "consensus_nt"])
    nt = nt.drop_duplicates(subset="body")
    nodes = nodes.merge(nt, left_on="bodyId", right_on="body", how="left")
    nodes = nodes.drop(columns=["body"])
    nodes["nt"] = nodes["consensus_nt"].fillna("unclear")
    nodes = nodes.drop(columns=["consensus_nt"])

    signs = cfg["signs"]
    unknown = sorted(set(nodes["nt"]) - set(signs))
    if unknown:
        raise SystemExit(f"neurotransmitters missing from config signs: {unknown}")
    nodes["sign"] = nodes["nt"].map(signs).astype(np.int8)

    # somaLocation is a list [x, y, z] or null.
    loc = nodes.pop("somaLocation")
    has = loc.notna()
    xyz = np.full((len(nodes), 3), np.nan, np.float64)
    if has.any():
        xyz[has.values] = np.stack(loc[has].to_numpy())
    nodes["x"], nodes["y"], nodes["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    nodes = nodes.sort_values("bodyId", kind="stable").reset_index(drop=True)
    nodes.insert(0, "idx", np.arange(len(nodes), dtype=np.int32))
    return nodes[NODE_COLS]


# --------------------------------------------------------------------------- edges

def _map_bodies(body: np.ndarray, sorted_ids: np.ndarray) -> np.ndarray:
    """Body IDs to row index; -1 where the ID is not a kept neuron."""
    pos = np.searchsorted(sorted_ids, body)
    pos_clipped = np.minimum(pos, len(sorted_ids) - 1)
    ok = sorted_ids[pos_clipped] == body
    return np.where(ok, pos_clipped, -1).astype(np.int64)


def build_edges(cfg: Config, weights_path: Path, nodes: pd.DataFrame):
    """Stream the 1 GB weights file, keep strong edges between kept neurons."""
    min_w = int(cfg["build"]["min_weight"])
    ids = nodes["bodyId"].to_numpy()
    assert np.all(np.diff(ids) > 0), "bodyId must be sorted and unique"

    reader = ipc.open_file(pa.memory_map(str(weights_path), "r"))
    nb = reader.num_record_batches
    print(f"  weights file: {nb} record batches")

    pres, posts, ws = [], [], []
    sorted_desc = True          # does the file stay sorted by weight, descending?
    prev_min = None
    scanned = 0
    for b in range(nb):
        batch = reader.get_batch(b)
        w = batch.column("weight").to_numpy(zero_copy_only=False)
        if w.size:
            if np.any(np.diff(w) > 0) or (prev_min is not None and w.max() > prev_min):
                sorted_desc = False
            prev_min = w.min()
        scanned += 1
        sel = w >= min_w
        if sel.any():
            pre = batch.column("body_pre").to_numpy(zero_copy_only=False)[sel]
            post = batch.column("body_post").to_numpy(zero_copy_only=False)[sel]
            wi = w[sel]
            a, c = _map_bodies(pre, ids), _map_bodies(post, ids)
            ok = (a >= 0) & (c >= 0) & (a != c)
            if ok.any():
                pres.append(a[ok]); posts.append(c[ok]); ws.append(wi[ok])
        if sorted_desc and w.size and w.max() < min_w:
            print(f"  file is weight-sorted; stopped after {scanned}/{nb} batches")
            break
        if b % 200 == 0:
            print(f"\r    batch {b}/{nb}", end="", flush=True)
    else:
        if not sorted_desc:
            print("\n  file was not weight-sorted; scanned all batches")
    print()

    if not sorted_desc:
        print("  note: weight ordering assumption did not hold (full scan used)")

    pre = np.concatenate(pres) if pres else np.empty(0, np.int64)
    post = np.concatenate(posts) if posts else np.empty(0, np.int64)
    w = np.concatenate(ws).astype(np.int32) if ws else np.empty(0, np.int32)
    print(f"  edges with >= {min_w} synapses between traced neurons: {len(w):,}")

    n = len(nodes)
    g = sp.coo_matrix((w, (pre, post)), shape=(n, n), dtype=np.int32).tocsr()
    g.sum_duplicates()
    return g


# --------------------------------------------------------------------------- io

def build(cfg: Config) -> None:
    print("fetching raw data")
    paths = fetch_raw(cfg)
    print("building nodes")
    nodes = build_nodes(cfg, paths)
    print("building edges")
    g = build_edges(cfg, paths["weights"], nodes)

    out = cfg.graph_dir
    out.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(out / "nodes.parquet", index=False)
    np.savez_compressed(
        out / "edges.npz",
        indptr=g.indptr.astype(np.int64),
        indices=g.indices.astype(np.int32),
        weight=g.data.astype(np.int32),
    )

    outdeg = np.diff(g.indptr)
    ncomp, labels = sp.csgraph.connected_components(g, directed=True, connection="weak")
    sizes = np.bincount(labels)
    stats = {
        "n_neurons": int(len(nodes)),
        "n_edges": int(g.nnz),
        "n_synapses": int(g.data.sum()),
        "min_weight": int(cfg["build"]["min_weight"]),
        "mean_out_degree": float(outdeg.mean()),
        "max_out_degree": int(outdeg.max()),
        "n_excitatory": int((nodes["sign"] == 1).sum()),
        "n_inhibitory": int((nodes["sign"] == -1).sum()),
        "n_with_soma": int(nodes["x"].notna().sum()),
        "n_types": int(nodes["type"].nunique()),
        "nt_counts": {k: int(v) for k, v in nodes["nt"].value_counts().items()},
        "weakly_connected_components": int(ncomp),
        "largest_component": int(sizes.max()),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {out}/nodes.parquet, edges.npz, stats.json")


class Graph:
    """The built graph, loaded once."""

    def __init__(self, nodes: pd.DataFrame, indptr, indices, weight, stats: dict):
        self.nodes = nodes
        self.indptr = indptr
        self.indices = indices
        self.weight = weight
        self.stats = stats
        self.n = len(nodes)
        self.sign = nodes["sign"].to_numpy().astype(np.int8)
        self.body_ids = nodes["bodyId"].to_numpy()

    # ----------------------------------------------------------------- selectors
    def select(self, sel: dict | None) -> np.ndarray:
        """Resolve a selector dict (SPEC §7.1) to node indices."""
        if not sel:
            return np.empty(0, np.int64)
        allowed = {"bodyId", "type", "type_regex", "class", "superclass",
                   "side", "max_n", "seed"}
        bad = set(sel) - allowed
        if bad:
            raise ValueError(f"unknown selector keys: {sorted(bad)}")

        nodes = self.nodes
        mask = np.ones(self.n, bool)
        if "bodyId" in sel:
            want = np.atleast_1d(np.asarray(sel["bodyId"], dtype=np.int64))
            mask &= np.isin(self.body_ids, want)
        if "type" in sel:
            mask &= (nodes["type"] == sel["type"]).to_numpy()
        if "type_regex" in sel:
            mask &= nodes["type"].fillna("").str.match(sel["type_regex"]).to_numpy()
        if "class" in sel:
            mask &= (nodes["class"] == sel["class"]).to_numpy()
        if "superclass" in sel:
            mask &= (nodes["superclass"] == sel["superclass"]).to_numpy()
        if "side" in sel:
            mask &= (nodes["somaSide"] == sel["side"]).to_numpy()

        idx = np.flatnonzero(mask).astype(np.int64)
        if not idx.size:
            raise ValueError(f"selector matched no neurons: {sel}")
        if "max_n" in sel and len(idx) > int(sel["max_n"]):
            rng = np.random.default_rng(int(sel.get("seed", 0)))
            idx = np.sort(rng.choice(idx, int(sel["max_n"]), replace=False))
        return idx

    def describe(self, i: int) -> dict:
        r = self.nodes.iloc[int(i)]
        return {
            "idx": int(i),
            "bodyId": int(r["bodyId"]),
            "type": None if pd.isna(r["type"]) else str(r["type"]),
            "instance": None if pd.isna(r["instance"]) else str(r["instance"]),
            "superclass": None if pd.isna(r["superclass"]) else str(r["superclass"]),
            "nt": str(r["nt"]),
            "sign": int(r["sign"]),
        }


def load_graph(cfg: Config) -> Graph:
    d = cfg.graph_dir
    if not (d / "nodes.parquet").exists():
        raise SystemExit(f"no graph in {d}. Run: python -m flypath build")
    nodes = pd.read_parquet(d / "nodes.parquet")
    z = np.load(d / "edges.npz")
    stats = json.loads((d / "stats.json").read_text())
    return Graph(nodes, z["indptr"], z["indices"], z["weight"], stats)
