import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np

NODES_PATH = Path("h3_output_node&edge") / "nodes.json"
EDGES_PATH = Path("h3_output_node&edge") / "edges.json"
EMBEDDINGS_PATH = Path("h4_output_embeddings") / "embeddings.npz"
GRAPH_PATH = Path("h4_output_embeddings") / "graph.gpickle"


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_embeddings(path: Path):
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    ids = data["ids"].tolist()
    vectors = data["vectors"]
    return {str(id_): vectors[i] for i, id_ in enumerate(ids)}


def build_graph(nodes_data, edges_data, embeddings_map=None):
    G = nx.MultiDiGraph()

    for node in nodes_data:
        node_id = node["id"]
        attrs = {k: v for k, v in node.items() if k != "id"}
        G.add_node(node_id, **attrs)

    if embeddings_map is not None:
        for node_id, vector in embeddings_map.items():
            if node_id in G:
                G.nodes[node_id]["embedding"] = vector

    for edge in edges_data:
        source = edge.get("source")
        target = edge.get("target")
        edge_type = edge.get("type")
        attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
        key = edge_type if edge_type is not None else None
        G.add_edge(source, target, key=key, **attrs)

    return G


def summarize_graph(G):
    node_types = {}
    for _, attrs in G.nodes(data=True):
        node_types[attrs.get("type", "unknown")] = node_types.get(attrs.get("type", "unknown"), 0) + 1

    edge_types = {}
    for _, _, attrs in G.edges(data=True):
        edge_types[attrs.get("type", "unknown")] = edge_types.get(attrs.get("type", "unknown"), 0) + 1

    print("Graph summary:")
    print(f" - nodes: {G.number_of_nodes()}")
    for name, count in sorted(node_types.items()):
        print(f"   - {name}: {count}")
    print(f" - edges: {G.number_of_edges()}")
    for name, count in sorted(edge_types.items()):
        print(f"   - {name}: {count}")

    embeddings_count = sum(1 for _, attrs in G.nodes(data=True) if "embedding" in attrs)
    print(f" - nodes with embeddings: {embeddings_count}")


def save_graph_pickle(G, path: Path):
    """Lưu đồ thị NetworkX bằng thư viện pickle chuẩn."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph_pickle(path: Path):
    """Hàm bổ sung: Đọc lại đồ thị từ file pickle khi làm RAG."""
    with path.open("rb") as f:
        return pickle.load(f)


def main():
    print("Loading graph data...")
    nodes_data = load_json(NODES_PATH)
    edges_data = load_json(EDGES_PATH)
    embeddings_map = load_embeddings(EMBEDDINGS_PATH)

    print("Building NetworkX graph...")
    G = build_graph(nodes_data, edges_data, embeddings_map=embeddings_map)

    print(f"Saving graph to {GRAPH_PATH}...")
    save_graph_pickle(G, GRAPH_PATH)

    summarize_graph(G)
    print(f"🎉 Graph saved successfully: {GRAPH_PATH}")


if __name__ == "__main__":
    main()