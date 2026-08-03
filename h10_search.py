from h8_query_processor import QueryProcessor
from h9_hybrid_search_algorithm import hybrid_search_nodes


def main():

    print("=" * 60)
    print("Hybrid Retrieval Demo")
    print("=" * 60)

    # ==========================================
    # Query Processor
    # ==========================================

    qp = QueryProcessor()

    # ==========================================
    # Query Loop
    # ==========================================

    while True:

        try:

            question = input(
                "\nNhập câu hỏi (exit để thoát): "
            ).strip()

            if question.lower() in [
                "exit",
                "quit",
                "q"
            ]:
                print("\nĐã thoát chương trình.")
                break

            if question == "":
                print("Câu hỏi không được để trống.")
                continue

            # --------------------------------------
            # Query Processing
            # --------------------------------------

            processed = qp.process(question)

            # --------------------------------------
            # Hybrid Search
            # --------------------------------------

            seed_nodes = hybrid_search_nodes(

                query_text=processed["query_text"],

                query_vector=processed["query_vector"],

                top_k=5,

                alpha=0.7

            )

            # --------------------------------------
            # Result
            # --------------------------------------

            print("\n")
            print("=" * 60)
            print(f"TOP {len(seed_nodes)} SEED NODES")
            print("=" * 60)

            for idx, node in enumerate(seed_nodes, start=1):

                print(f"\n[{idx}] {node['node_id']}")

                print(
                    f"Hybrid : {node['hybrid_score']:.4f}"
                )

                print(
                    f"Dense  : {node['dense_score']:.4f}"
                )

                print(
                    f"BM25   : {node['bm25_score']:.4f}"
                )

        except KeyboardInterrupt:

            print("\nĐã dừng chương trình.")
            break


if __name__ == "__main__":

    main()