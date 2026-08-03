from sentence_transformers import SentenceTransformer


class EmbedderModel:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        Khởi tạo và nạp mô hình Embedding vào RAM.
        """
        print(f"[Embedder] Đang nạp mô hình: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("[Embedder] Nạp mô hình thành công!")

    def encode_text(self, text):
        return self.model.encode(text, convert_to_numpy=True)

    def encode_batch(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )


# Chỉ tạo duy nhất 1 object
embedder = EmbedderModel()