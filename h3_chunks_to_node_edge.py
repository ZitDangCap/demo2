import json
import os
from pathlib import Path


def main():
    base = Path(__file__).parent
    src = base / "h2_schunk.json"
    outdir = base / "h3_output_node_edge"
    outdir.mkdir(exist_ok=True)

    with src.open(encoding="utf-8") as f:
        data = json.load(f)

    nodes = []
    edges = []
    parent_nodes = {}
    chunk_nodes = []

    def make_section_id(path_tuple):
        return "section:" + "|".join(path_tuple)

    def make_chunk_id(path, chunk_id):
        return "chunk:" + "|".join(path) + ":%d" % chunk_id

    # create nodes for sections and chunks
    for item in data:
        path = item.get("path", [])
        cid = item.get("chunk_id")
        text = item.get("clean_text", "")

        # ensure section nodes for each level
        for i in range(1, len(path) + 1):
            prefix = tuple(path[:i])
            if prefix not in parent_nodes:
                node_id = make_section_id(prefix)
                node = {
                    "id": node_id,
                    "type": "section",
                    "title": prefix[-1],
                    "path": list(prefix),
                }
                parent_nodes[prefix] = node_id
                nodes.append(node)

        # create chunk node
        chunk_id = make_chunk_id(path, cid)
        path_label = " | ".join(path)
        embedding_text = f"[{path_label}] {text}" if path_label else text
        chunk_node = {
            "id": chunk_id,
            "type": "chunk",
            "path": path,
            "chunk_id": cid,
            "text": text,
            "embedding_text": embedding_text,
        }
        nodes.append(chunk_node)
        chunk_nodes.append((tuple(path), cid, chunk_id))

        # link deepest section -> chunk
        parent_id = parent_nodes.get(tuple(path))
        if parent_id:
            edges.append({"source": parent_id, "target": chunk_id, "type": "has_chunk"})

    # create parent-child edges between sections
    for prefix in list(parent_nodes.keys()):
        if len(prefix) <= 1:
            continue
        parent = parent_nodes[tuple(prefix[:-1])]
        child = parent_nodes[tuple(prefix)]
        edges.append({"source": parent, "target": child, "type": "has_child"})

    # create sequential edges for chunks in same path
    from collections import defaultdict

    grouped = defaultdict(list)
    for path_tuple, cid, nodeid in chunk_nodes:
        grouped[path_tuple].append((cid, nodeid))

    for path_tuple, items in grouped.items():
        items.sort(key=lambda x: x[0])
        for a, b in zip(items, items[1:]):
            edges.append({"source": a[1], "target": b[1], "type": "next"})

    # write outputs
    (outdir / "nodes.json").write_text(json.dumps(nodes, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "edges.json").write_text(json.dumps(edges, ensure_ascii=False, indent=2), encoding="utf-8")

    # write embedding-ready CSV (id,text)
    import csv

    with (outdir / "embeddings.csv").open("w", encoding="utf-8", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["id", "text", "embedding_text"])
        for n in nodes:
            if n.get("type") == "chunk":
                writer.writerow([n["id"], n.get("text", ""), n.get("embedding_text", "")])

    print("Wrote:")
    print(" -", outdir / "nodes.json")
    print(" -", outdir / "edges.json")
    print(" -", outdir / "embeddings.csv")


if __name__ == "__main__":
    main()
