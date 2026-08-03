import pickle
from pathlib import Path
from pyvis.network import Network

GRAPH_PATH = Path("h4_output_embeddings") / "graph.gpickle"

# 1. Load đồ thị NetworkX đã lưu
with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)

# 2. Khởi tạo mạng lưới Pyvis (có thanh công cụ & giao diện trực quan)
net = Network(height="750px", width="100%", notebook=False, directed=True)

# 3. Chuyển từ NetworkX sang Pyvis
# (Lưu ý: Ta xóa thuộc tính 'embedding' trước khi vẽ vì Pyvis không vẽ được mảng vector)
G_vis = G.copy()
for n in G_vis.nodes():
    G_vis.nodes[n].pop("embedding", None)

net.from_nx(G_vis)

# 4. Thêm hiệu ứng kéo thả mượt mà (Physics)
net.show_buttons(filter_=['physics'])

# 5. Xuất ra file HTML và tự động mở
output_html = "h7_graph_demo.html"
net.write_html(output_html)
print(f" Đã tạo file demo thành công! Mở file '{output_html}' bằng trình duyệt để xem.")