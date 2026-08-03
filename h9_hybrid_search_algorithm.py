import numpy as np

from model.model_graph import graph
from model.model_bm25 import bm25


def hybrid_search_nodes(
    query_text: str,
    query_vector: np.ndarray,
    top_k: int = 5,
    alpha: float = 0.7
):
    """
    Hybrid Retrieval.

    Parameters
    ----------
    query_text : str
        Câu hỏi sau khi đã được tiền xử lý.

    query_vector : np.ndarray
        Embedding của câu hỏi.

    top_k : int
        Số lượng Seed Nodes cần lấy.

    alpha : float
        Trọng số của Dense Retrieval.

    Returns
    -------
    list[dict]
        Danh sách Seed Nodes.
    """

    G = graph.graph

    # ==================================================
    # BM25 Retrieval
    # ==================================================

    bm25_scores = bm25.search(query_text)
    node_ids = bm25.node_ids

    # ==================================================
    # Dense Retrieval (Cosine Similarity)
    # ==================================================

    dense_matrix = np.array([
        G.nodes[node_id]["embedding"]
        for node_id in node_ids
    ])

    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        query_norm = 1e-10

    matrix_norm = np.linalg.norm(
        dense_matrix,
        axis=1
    )

    matrix_norm[matrix_norm == 0] = 1e-10

    dense_scores = np.dot(
        dense_matrix,
        query_vector
    ) / (matrix_norm * query_norm)

    # ==================================================
    # Normalize BM25
    # ==================================================

    bm25_min = bm25_scores.min()
    bm25_max = bm25_scores.max()

    if bm25_max > bm25_min:

        bm25_scores = (
            bm25_scores - bm25_min
        ) / (
            bm25_max - bm25_min
        )

    else:

        bm25_scores = np.zeros_like(bm25_scores)

    # ==================================================
    # Normalize Dense
    # ==================================================

    dense_min = dense_scores.min()
    dense_max = dense_scores.max()

    if dense_max > dense_min:

        dense_scores = (
            dense_scores - dense_min
        ) / (
            dense_max - dense_min
        )

    else:

        dense_scores = np.zeros_like(dense_scores)

    # ==================================================
    # Hybrid Score
    # ==================================================

    hybrid_scores = (
        alpha * dense_scores
        + (1 - alpha) * bm25_scores
    )

    # ==================================================
    # Ranking
    # ==================================================

    ranking = np.argsort(
        hybrid_scores
    )[::-1][:top_k]

    # ==================================================
    # Output Seed Nodes
    # ==================================================

    seed_nodes = []

    for idx in ranking:

        seed_nodes.append({

            "node_id": node_ids[idx],

            "hybrid_score": float(hybrid_scores[idx]),

            "dense_score": float(dense_scores[idx]),

            "bm25_score": float(bm25_scores[idx])

        })

    return seed_nodes