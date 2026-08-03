import csv
import json
from pathlib import Path
import numpy as np
from model.model_embedder import embedder
from tqdm import tqdm

# ==========================================
# CẤU HÌNH CƠ BẢN
# ==========================================
INPUT_CSV = "h3_output_node_edge/embeddings.csv"      # Đường dẫn file CSV đầu vào 
OUTPUT_DIR = "h4_output_embeddings"   # Thư mục lưu kết quả
BATCH_SIZE = 32                    # Chỉnh thành 64 nếu máy mạnh, 16/32 nếu máy vừa
DEVICE = "cpu"                     # Đổi thành "cuda" nếu có card NVIDIA

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Chuẩn hóa L2 để tính Cosine Similarity nhanh bằng Tích vô hướng (Dot Product)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

def main():
    input_path = Path(INPUT_CSV.strip())
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Không tìm thấy file: {input_path.resolve()}")
        return

    # 1. Đọc dữ liệu từ CSV
    print(f"Đang đọc dữ liệu từ {INPUT_CSV.strip()}...")
    rows = []
    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        print("File CSV rỗng!")
        return

    # Lấy danh sách ID và đoạn Text dùng để embed (ưu tiên cột embedding_text)
    ids = [r.get("id") for r in rows]
    texts = [r.get("embedding_text") or r.get("text") or "" for r in rows]

    # 3. Tiến hành Embedding
    print(f" Bắt đầu Embedding {len(texts)} đoạn văn bản...")
    all_vectors = []
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Đang xử lý"):
        batch_texts = texts[i : i + BATCH_SIZE]
        # Encode batch
        vecs = embedder.encode_batch(batch_texts)
        all_vectors.append(vecs)

    # Gộp tất cả vector thành 1 ma trận numpy
    vectors = np.vstack(all_vectors)

    # 4. L2 Normalize
    vectors = l2_normalize(vectors)

    # 5. Lưu ra các định dạng file
    # File 1: Matrix Vector (.npz) - Dùng nạp vào Neo4j / DB sau này
    npz_path = out_dir / "embeddings.npz"
    np.savez_compressed(npz_path, ids=np.array(ids), vectors=vectors.astype(np.float32))
    print(f"\n Đã lưu file Vector dạng nén: {npz_path}")

    # File 2: Metadata + Vector Info (.jsonl) - Dễ đọc / debug
    jsonl_path = out_dir / "embeddings.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for r, vec in zip(rows, vectors):
            data = {
                "id": r.get("id"),
                "text": r.get("text"),
                "embedding_text": r.get("embedding_text"),
                "vector_dim": int(vec.shape[0]),
                # "vector": vec.tolist() # Bỏ comment nếu muốn lưu cả mảng vector vào file text JSONL
            }
            jf.write(json.dumps(data, ensure_ascii=False) + "\n")
    print(f" Đã lưu file Metadata chi tiết: {jsonl_path}")
    print("\n HOÀN THÀNH!")

if __name__ == "__main__":
    main()