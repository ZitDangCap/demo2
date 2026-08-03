def process_and_reorder_for_generator(
    judge_output: dict, 
    graph_obj
) -> dict:
    """
    1. Phân nhóm theo quyết định của Agent D (Judge)
    2. Sắp xếp lại chunk_order CHỈ TRONG NỘI BỘ MỖI NHÓM
    """
    
    # --------------------------------------------------
    # 1. Tách danh sách node id theo nhóm do Agent D phân loại
    # (Giả sử Agent D trả về 2 danh sách: core_nodes và supplementary_nodes)
    # --------------------------------------------------
    core_ids = judge_output.get("core_nodes", [])
    supp_ids = judge_output.get("supplementary_nodes", [])

    # --------------------------------------------------
    # Hàm phụ: Lấy thông tin chi tiết từ Đồ thị & Sắp xếp theo chunk_order
    # --------------------------------------------------
    def fetch_and_sort_nodes(node_ids: list) -> list:
        nodes_info = []
        for nid in node_ids:
            if graph_obj.has_node(nid):
                attr = graph_obj.nodes[nid]
                nodes_info.append({
                    "node_id": nid,
                    "title": attr.get("title", ""),
                    "text": attr.get("text", attr.get("content", "")).strip(),
                    "doc_id": attr.get("doc_id", ""),
                    # Lấy chunk_order để sort (mặc định là 0 nếu không tìm thấy)
                    "chunk_order": attr.get("chunk_order", 0)
                })
        
        # Sắp xếp nội bộ nhóm theo doc_id rồi đến chunk_order
        sorted_nodes = sorted(
            nodes_info, 
            key=lambda x: (x["doc_id"], x["chunk_order"])
        )
        return sorted_nodes

    # --------------------------------------------------
    # 2. Thực thi sắp xếp nội bộ từng nhóm
    # --------------------------------------------------
    sorted_core_nodes = fetch_and_sort_nodes(core_ids)
    sorted_supp_nodes = fetch_and_sort_nodes(supp_ids)

    # Trả về kết quả đã được đóng gói chuẩn chỉnh
    return {
        "core_nodes": sorted_core_nodes,          # Đã nằm ở ĐẦU và đúng thứ tự C1 -> C3
        "supplementary_nodes": sorted_supp_nodes # Đã nằm ở SAU và đúng thứ tự C2 -> C4
    }