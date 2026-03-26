#!/usr/bin/env python3

# https://graphsinspace.net
# https://tigraphs.pmf.uns.ac.rs

from __future__ import annotations

import argparse
import itertools
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined, assignment]


def _peek_variant(argv: list[str]) -> str:
    if "--variant" in argv:
        i = argv.index("--variant")
        if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            return argv[i + 1].lower()
    return "lid"


def _setup_typing_for_lid() -> None:
    import typing as _typing

    try:
        import typing_extensions as _typing_extensions
    except ModuleNotFoundError:
        _typing_extensions = _typing  # type: ignore[assignment]

    if not hasattr(_typing_extensions, "TypeIs"):
        if hasattr(_typing, "TypeIs"):
            _typing_extensions.TypeIs = _typing.TypeIs  # type: ignore[attr-defined]
        else:
            _typing_extensions.TypeIs = _typing.TypeGuard  # type: ignore[attr-defined]


if _peek_variant(sys.argv) == "lid":
    _setup_typing_for_lid()

REPO_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "graspe" / "src" / "graspe"
if not REPO_ROOT.is_dir():
    raise SystemExit(
        f"Nedostaje klon graspe: {REPO_ROOT}\n"
        "git clone https://github.com/graphsinspace/graspe.git third_party/graspe"
    )

if "dgl" not in sys.modules:
    _dgl = types.ModuleType("dgl")

    class _DGLGraph:
        pass

    _dgl.DGLGraph = _DGLGraph
    _dgl.from_networkx = lambda *a, **k: None
    sys.modules["dgl"] = _dgl

sys.path.insert(0, str(REPO_ROOT))

from common.graph import Graph as GraspeGraph  # noqa: E402

_DATA = Path(__file__).resolve().parents[1] / "data"
DEFAULT_CSV = _DATA / "loto7hh_4586_k24.csv"
DEFAULT_COMBOS = _DATA / "kombinacijeH_39C7.csv"
SEED = 39

# LID (variant lid)
TUNED_W2V_EPOCHS = 18
TUNED_WALK_LENGTH_LID = 14
TUNED_NUM_WALKS_LID = 56
TUNED_ALPHA = 0.85
TUNED_TOP_NODES_LID = 15
TUNED_CLAMP_WL_MAX = 48
TUNED_CLAMP_NW_MAX = 240
TUNED_W_GRAPH_LID = 0.42
TUNED_W_EMB_LID = 0.58

_HA_PATCHED = False
HA_CLASSES: dict[str, type] = {}


def _apply_gensim_word2vec_compat() -> None:
    import gensim
    from gensim.models import Word2Vec

    import embeddings.embedding_node2vec as n2vm

    def embed_patched(self) -> None:
        n2vm.Embedding.embed(self)
        self.preprocess_transition_probs()
        walks = self.simulate_walks()
        walks = [list(map(str, w)) for w in walks]
        major = int(gensim.__version__.split(".")[0])
        workers = max(1, int(self._workers))
        if major >= 4:
            model = Word2Vec(
                walks,
                vector_size=self._d,
                min_count=0,
                sg=1,
                workers=workers,
                seed=int(self._seed),
                epochs=TUNED_W2V_EPOCHS,
            )
            wv = model.wv
        else:
            model = Word2Vec(
                walks,
                size=self._d,
                min_count=0,
                sg=1,
                workers=workers,
                seed=int(self._seed),
            )
            wv = model.wv
        self._embedding = {}
        for node in self._g.nodes():
            nid = str(node[0])
            self._embedding[node[0]] = np.asarray(wv[nid], dtype=np.float64)

    n2vm.Node2VecEmbeddingBase.embed = embed_patched


_apply_gensim_word2vec_compat()


def _ensure_ha_patch() -> None:
    global _HA_PATCHED, HA_CLASSES
    if _HA_PATCHED:
        return
    from embeddings.embedding_ha_node2vec import (
        HANode2VecNumWalksEmbedding,
        HANode2VecNumWalksHubsLessEmbedding,
        HANode2VecNumWalksHubsLessLogEmbedding,
        HANode2VecNumWalksHubsMoreEmbedding,
        HANode2VecNumWalksHubsMoreLogEmbedding,
    )

    _orig = HANode2VecNumWalksEmbedding.distribute_walks

    def distribute_walks(values: dict, num_walks_total: int) -> dict:
        values_sum = sum(values.values())
        if values_sum <= 0:
            nodes = sorted(values.keys())
            if not nodes:
                return {}
            k = len(nodes)
            base = max(1, num_walks_total // k)
            out = {node: base for node in nodes}
            leftover = num_walks_total - sum(out.values())
            idx = 0
            while leftover > 0:
                out[nodes[idx % k]] += 1
                leftover -= 1
                idx += 1
            return out
        return _orig(values, num_walks_total)

    HANode2VecNumWalksEmbedding.distribute_walks = staticmethod(distribute_walks)

    HA_CLASSES.update(
        {
            "HubsMore": HANode2VecNumWalksHubsMoreEmbedding,
            "HubsLess": HANode2VecNumWalksHubsLessEmbedding,
            "HubsMoreLog": HANode2VecNumWalksHubsMoreLogEmbedding,
            "HubsLessLog": HANode2VecNumWalksHubsLessLogEmbedding,
        }
    )
    _HA_PATCHED = True


def load_draws(csv_path: Path):
    df = pd.read_csv(csv_path, encoding="utf-8")
    cols = [f"Num{i}" for i in range(1, 8)]
    if all(c in df.columns for c in cols):
        use = cols
    else:
        use = list(df.columns[:7])
    draws = []
    for _, row in df.iterrows():
        draws.append(sorted(int(row[c]) for c in use))
    return draws


def dynamic_pair_weights(draws: list[list[int]], decay: float) -> dict[tuple[int, int], float]:
    T = len(draws)
    acc: dict[tuple[int, int], float] = {}
    for t, nums in enumerate(draws):
        w = float(decay) ** (T - 1 - t)
        for u, v in itertools.combinations(nums, 2):
            a, b = (u, v) if u < v else (v, u)
            acc[(a, b)] = acc.get((a, b), 0.0) + w
    return acc


def build_graspe_graph(pair_w: dict[tuple[int, int], float]) -> GraspeGraph:
    G = GraspeGraph()
    for i in range(1, 40):
        G.add_node(i)
    for (u, v), w in pair_w.items():
        if w <= 0:
            continue
        G.add_edge(u, v, weight=w)
        G.add_edge(v, u, weight=w)
    return G


def run_native_node2vec(
    G: GraspeGraph, dim: int, walk_length: int, num_walks: int, seed: int
) -> dict[int, np.ndarray]:
    from embeddings.embedding_node2vec import Node2VecEmbeddingNative

    emb = Node2VecEmbeddingNative(
        G,
        d=dim,
        p=1.0,
        q=1.0,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=1,
        seed=seed,
    )
    emb.embed()
    return {n: emb[n].copy() for n in range(1, 40)}


def run_ha_node2vec(
    G: GraspeGraph,
    ha_name: str,
    dim: int,
    walk_length: int,
    num_walks: int,
    seed: int,
) -> dict[int, np.ndarray]:
    _ensure_ha_patch()
    cls = HA_CLASSES[ha_name]
    emb = cls(
        G,
        d=dim,
        p=1.0,
        q=1.0,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=1,
        seed=seed,
    )
    emb.embed()
    return {n: emb[n].copy() for n in range(1, 40)}


def clamp_lid_params(emb) -> None:
    for n in list(emb.nw_dict.keys()):
        emb.nw_dict[n] = max(1, min(TUNED_CLAMP_NW_MAX, int(emb.nw_dict[n])))
        emb.wl_dict[n] = max(1, min(TUNED_CLAMP_WL_MAX, int(emb.wl_dict[n])))


def run_lid_node2vec(
    G: GraspeGraph,
    lid_name: str,
    dim: int,
    walk_length: int,
    num_walks: int,
    alpha: float,
    seed: int,
) -> dict[int, np.ndarray]:
    from embeddings.embedding_lid_node2vec import (
        LIDNode2VecElasticW,
        LIDNode2VecElasticWPQ,
    )

    lid_map = {"ElasticW": LIDNode2VecElasticW, "ElasticWPQ": LIDNode2VecElasticWPQ}
    cls = lid_map[lid_name]
    emb = cls(
        G,
        d=dim,
        p=1.0,
        q=1.0,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=1,
        seed=seed,
        alpha=alpha,
    )
    clamp_lid_params(emb)
    emb.embed()
    return {n: emb[n].copy() for n in range(1, 40)}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pair_scores(
    pair_w: dict[tuple[int, int], float],
    vectors: dict[int, np.ndarray],
    w_graph: float,
    w_emb: float,
) -> dict[tuple[int, int], float]:
    if not pair_w:
        return {}
    mx = max(pair_w.values()) or 1.0
    scores = {}
    for (u, v), w in pair_w.items():
        gnorm = w / mx
        c = cosine(vectors[u], vectors[v])
        scores[(u, v)] = w_graph * gnorm + w_emb * c
    return scores


def best_combo_from_scores(
    scores: dict[tuple[int, int], float],
    top_nodes: int,
) -> tuple[int, ...]:
    strength = {n: 0.0 for n in range(1, 40)}
    for (u, v), s in scores.items():
        strength[u] += s
        strength[v] += s
    ranked = sorted(range(1, 40), key=lambda x: (-strength[x], x))[:top_nodes]
    best: tuple[int, ...] | None = None
    best_val = -1e18
    for combo in itertools.combinations(sorted(ranked), 7):
        sv = 0.0
        for u, v in itertools.combinations(combo, 2):
            a, b = (u, v) if u < v else (v, u)
            sv += scores.get((a, b), 0.0)
        if sv > best_val or (sv == best_val and best is not None and combo < best):
            best_val = sv
            best = combo
    assert best is not None
    return best


def main():
    ap = argparse.ArgumentParser(
        description="GRASP loto grupa A: Node2Vec — graspe1_loto_node2vec.py"
    )
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--combos", type=Path, default=DEFAULT_COMBOS)
    ap.add_argument(
        "--variant",
        type=str,
        default="lid",
        choices=("native", "ha", "lid"),
        help="native=_1, ha=_2, lid=_3",
    )
    ap.add_argument("--decay", type=float, default=0.999)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--walk-length", type=int, default=None)
    ap.add_argument("--num-walks", type=int, default=None)
    ap.add_argument("--top-nodes", type=int, default=None)
    ap.add_argument(
        "--ha",
        type=str,
        default="HubsMore",
        choices=("HubsMore", "HubsLess", "HubsMoreLog", "HubsLessLog"),
        help="Samo za --variant ha",
    )
    ap.add_argument(
        "--lid",
        type=str,
        default="ElasticW",
        choices=("ElasticW", "ElasticWPQ"),
        help="Samo za --variant lid",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=TUNED_ALPHA,
        help="LFM alpha za variant lid",
    )
    args = ap.parse_args()

    wl = args.walk_length
    nw = args.num_walks
    tn = args.top_nodes
    if args.variant == "lid":
        if wl is None:
            wl = TUNED_WALK_LENGTH_LID
        if nw is None:
            nw = TUNED_NUM_WALKS_LID
        if tn is None:
            tn = TUNED_TOP_NODES_LID
        wg, we = TUNED_W_GRAPH_LID, TUNED_W_EMB_LID
    else:
        if wl is None:
            wl = 10
        if nw is None:
            nw = 50
        if tn is None:
            tn = 14
        wg, we = 0.45, 0.55

    np.random.seed(SEED)

    draws = load_draws(args.csv)
    pair_w = dynamic_pair_weights(draws, args.decay)
    G = build_graspe_graph(pair_w)

    print(f"CSV izvučenih: {args.csv.resolve()}")
    print(f"CSV svih komb.: {args.combos.resolve()}  (postoji: {args.combos.is_file()})")
    print(f"Izvlačenja: {len(draws)} | parova: {len(pair_w)} | decay={args.decay}")
    print(f"graspe: {REPO_ROOT}")
    print(f"variant={args.variant}")

    if args.variant == "native":
        print("Embedding: Node2VecEmbeddingNative")
        vectors = run_native_node2vec(G, args.dim, wl, nw, SEED)
        tag = "Node2Vec native"
    elif args.variant == "ha":
        _ensure_ha_patch()
        print(f"Embedding: HA_N2V {args.ha}")
        try:
            import gensim as _gs

            print(f"gensim: {_gs.__version__}")
        except ImportError:
            pass
        vectors = run_ha_node2vec(G, args.ha, args.dim, wl, nw, SEED)
        tag = f"HA_N2V {args.ha}"
    else:
        print(f"Embedding: LID {args.lid} | alpha={args.alpha}")
        try:
            import gensim as _gs
            import torch as _torch

            print(f"gensim: {_gs.__version__} | torch: {_torch.__version__}")
        except ImportError as e:
            print(f"Upozorenje import: {e}")
        vectors = run_lid_node2vec(
            G, args.lid, args.dim, wl, nw, args.alpha, SEED
        )
        tag = f"LID {args.lid}"

    scores = pair_scores(pair_w, vectors, wg, we)
    combo = best_combo_from_scores(scores, top_nodes=tn)

    print()
    print(f"Predikcija (dinamički graf + {tag}):")
    print(list(combo))
    print()


if __name__ == "__main__":
    main()




########################################################




"""
python3 graspe1_loto_node2vec.py

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
variant=lid
Embedding: LID ElasticW | alpha=0.85
gensim: 4.4.0 | torch: 2.8.0

Predikcija (dinamički graf + LID ElasticW):
[7, 8, x, y, z, 34, 37]
"""




"""
python3 graspe1_loto_node2vec.py --variant lid

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
variant=lid
Embedding: LID ElasticW | alpha=0.85
gensim: 4.4.0 | torch: 2.8.0

Predikcija (dinamički graf + LID ElasticW):
[8, 10, x, y, z, 34, 39]
"""




"""
python3 graspe1_loto_node2vec.py --variant lid --lid ElasticWPQ --alpha 0.85

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
variant=lid
Embedding: LID ElasticWPQ | alpha=0.85
gensim: 4.4.0 | torch: 2.8.0

Predikcija (dinamički graf + LID ElasticWPQ):
[7, 8, x, y, z, 34, 37]
"""




"""
python3 graspe1_loto_node2vec.py --variant native

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
variant=native
Embedding: Node2VecEmbeddingNative

Predikcija (dinamički graf + Node2Vec native):
[8, 10, x, y, z, 34, 39]
"""




"""
python3 graspe1_loto_node2vec.py --variant ha   

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
variant=ha
Embedding: HA_N2V HubsMore
gensim: 4.4.0

Predikcija (dinamički graf + HA_N2V HubsMore):
[8, 10, x, y, z, 34, 37]
"""




"""
python3 graspe1_loto_node2vec.py --variant ha --ha HubsMore

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
variant=ha
Embedding: HA_N2V HubsMore
gensim: 4.4.0

Predikcija (dinamički graf + HA_N2V HubsMore):
[8, 10, x, y, z, 34, 39]
"""




"""
python3 graspe1_loto_node2vec.py --variant ha --ha HubsLess

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
variant=ha
Embedding: HA_N2V HubsLess
gensim: 4.4.0

Predikcija (dinamički graf + HA_N2V HubsLess):
[8, 10, x, y, z, 34, 39]
"""




"""
python3 graspe1_loto_node2vec.py --variant ha --ha HubsMoreLog

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
variant=ha
Embedding: HA_N2V HubsMoreLog
gensim: 4.4.0

Predikcija (dinamički graf + HA_N2V HubsMoreLog):
[8, 10, x, y, z, 34, 39]
"""




"""
python3 graspe1_loto_node2vec.py --variant ha --ha HubsLessLog

Izvlačenja: 4586 | parova: 741 | decay=0.999
graspe: /Users/4c/Desktop/GHQ/third_party/graspe/src/graspe
variant=ha
Embedding: HA_N2V HubsLessLog
gensim: 4.4.0

Predikcija (dinamički graf + HA_N2V HubsLessLog):
[7, 8, x, y, z, 34, 37]
"""




########################################################
