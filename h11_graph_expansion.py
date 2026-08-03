import networkx as nx


def expand_subgraph_raw(
    G: nx.MultiDiGraph,
    seed_nodes: list,
    max_hops: int = 3,
    max_neighbors: int = 10,
    max_nodes: int = 100,
):
    """Expand graph từ seed nodes có giới hạn, đảm bảo KHÔNG trùng node hay

    edge.
    """

    # ==========================
    # 1. Chuẩn hóa Seed Nodes (Lọc trùng seed ngay từ đầu)
    # ==========================
    seed_map = {}
    for item in seed_nodes:
        nid = item.get("node_id")
        if nid and nid not in seed_map:
            seed_map[nid] = item

    visited = set(seed_map.keys())
    frontier = set(seed_map.keys())

    collected_edges = []
    edge_seen = set()

    # ==========================
    # 2. BFS Duyệt Đồ Thị
    # ==========================
    for hop in range(max_hops):
        if len(visited) >= max_nodes or not frontier:
            break

        next_frontier = set()

        for node in frontier:
            if len(visited) >= max_nodes:
                break

            if not G.has_node(node):
                continue

            # Lấy danh sách hàng xóm (Cả tiến lẫn lùi nếu là Directed Graph)
            neighbors = set(G.successors(node))
            if G.is_directed():
                neighbors |= set(G.predecessors(node))

            # Giới hạn số lượng hàng xóm cho mỗi node
            neighbors = list(neighbors)[:max_neighbors]

            for nbr in neighbors:
                if len(visited) >= max_nodes:
                    break

                # Đánh dấu node mới
                if nbr not in visited:
                    visited.add(nbr)
                    next_frontier.add(nbr)

                # ------------------
                # Trích xuất Edges (Xử lý chuẩn cho MultiDiGraph)
                # ------------------
                # Kiểm tra chiều node -> nbr
                edges_forward = G.get_edge_data(node, nbr) or {}
                for k, e in edges_forward.items():
                    rel = (
                        e.get("relation")
                        or e.get("label")
                        or e.get("type")
                        or "CONNECTED"
                    )
                    edge_key = (node, nbr, rel)
                    if edge_key not in edge_seen:
                        edge_seen.add(edge_key)
                        collected_edges.append(
                            {"source": node, "target": nbr, "relation": rel}
                        )

                # Kiểm tra chiều ngược nbr -> node (nếu có)
                edges_backward = G.get_edge_data(nbr, node) or {}
                for k, e in edges_backward.items():
                    rel = (
                        e.get("relation")
                        or e.get("label")
                        or e.get("type")
                        or "CONNECTED"
                    )
                    edge_key = (nbr, node, rel)
                    if edge_key not in edge_seen:
                        edge_seen.add(edge_key)
                        collected_edges.append(
                            {"source": nbr, "target": node, "relation": rel}
                        )

        frontier = next_frontier

    # ==========================
    # 3. Tạo Candidate Nodes (Đảm bảo ID duy nhất 100%)
    # ==========================
    candidate_nodes = []
    processed_node_ids = set()  # Khóa bảo vệ chống trùng cuối cùng

    for node_id in visited:
        if node_id in processed_node_ids:
            continue

        if not G.has_node(node_id):
            continue

        attr = G.nodes[node_id]

        node_info = {
            "node_id": node_id,
            "text": (
                attr.get("text")
                or attr.get("content")
                or attr.get("page_content")
                or ""
            ).strip(),
            "title": attr.get("title", ""),
            "node_type": attr.get("type") or attr.get("node_type") or "",
            "is_seed": node_id in seed_map,
        }

        # Bổ sung điểm số nếu là Seed Node
        if node_id in seed_map:
            node_info["retrieval_score"] = seed_map[node_id].get(
                "hybrid_score", seed_map[node_id].get("score", 0)
            )

        candidate_nodes.append(node_info)
        processed_node_ids.add(node_id)

    return {
        "candidate_nodes": candidate_nodes,
        "edges": collected_edges,
        "total_nodes_found": len(candidate_nodes),
    }