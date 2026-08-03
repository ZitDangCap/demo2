import pickle
import networkx as nx
from rank_bm25 import BM25Okapi
from model.model_graph import graph




def build_bm25_index(
    G: nx.MultiDiGraph,
    output_path: str = "h4_output_embeddings/bm25_index.pkl"
):
    """
    Xây dựng BM25 Index từ toàn bộ node trong Graph.

    Chỉ chạy một lần sau khi Graph đã hoàn thiện.
    """

    corpus = []
    node_ids = []

    for node_id, attrs in G.nodes(data=True):

        text = attrs.get("text") or attrs.get("content")

        if text:
            corpus.append(text.lower().split())
            node_ids.append(node_id)

    print(f"[BM25] Building index from {len(corpus)} documents...")

    if not corpus:
        raise ValueError("BM25 corpus is empty. Check that graph nodes contain text/content fields.")

    bm25 = BM25Okapi(corpus)

    with open(output_path, "wb") as f:
        pickle.dump(
    {
        "bm25": bm25,
        "node_ids": node_ids
    },
    f
)

    print(f"[BM25] Saved index to {output_path}")

if __name__ == "__main__":

    G = graph.graph

    build_bm25_index(G)