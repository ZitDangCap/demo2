import pickle


class GraphModel:

    def __init__(
        self,
        graph_path="h4_output_embeddings/graph.gpickle"
    ):

        print("[Graph] Loading graph...")

        with open(graph_path, "rb") as f:
            self.graph = pickle.load(f)

        print(
            f"[Graph] Loaded {self.graph.number_of_nodes()} nodes."
        )


graph = GraphModel()