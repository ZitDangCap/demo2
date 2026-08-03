import pickle


class BM25Model:

    def __init__(
        self,
        bm25_path="h4_output_embeddings/bm25_index.pkl"
    ):

        print("[BM25] Loading index...")

        with open(bm25_path, "rb") as f:
            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.node_ids = data["node_ids"]

        print("[BM25] Loaded successfully.")

    def search(self, query_text):

        tokens = query_text.lower().split()

        return self.bm25.get_scores(tokens)


# Singleton
bm25 = BM25Model()