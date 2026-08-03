# Cách vận hành dự án

## Lấy chunks đầu vào là chunks demo của khoa phạm 
schunks.json
-> Lý do là có sự phân tầng nhiều cấp giữa các chunks
## Chuyển đổi các chunks về node và edge kết hợp gán nhãn dữ liệu tạo ra trường embedding_text có chứa text và ngữ cảnh.
chunks_to_node&edge.py
-> Ta được file output chứa nodes.json , edges.json và embeddings.csv
## 


