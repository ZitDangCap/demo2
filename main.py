import json
import os
import numpy as np

# Import các module trong dự án của bạn
from model.model_graph import graph
from h8_query_processor import QueryProcessor
from h9_hybrid_search_algorithm import hybrid_search_nodes
from h11_graph_expansion import expand_subgraph_raw

# Import 3 Agents
from h13_agents.literal_agent import LiteralAgent
from h13_agents.semantic_agent import SemanticAgent
from h13_agents.exception_agent import ExceptionAgent

from h14_consensus import consensus




# ======================================================
# OUTPUT FOLDER
# ======================================================

OUTPUT_DIR = "h12_output_multi_qwen"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_json(filename: str, data):
    """
    Lưu kết quả JSON vào thư mục output.
    Nếu thư mục chưa tồn tại sẽ tự tạo.
    """

    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


def run_pipeline(question: str):

    print("=" * 60)
    print(f"CÂU HỎI: {question}")
    print("=" * 60)

    # --------------------------------------------------
    # BƯỚC 1: Query Processing
    # --------------------------------------------------
    print("\n[1/4] Đang xử lý câu hỏi & tạo Embedding...")

    qp = QueryProcessor()

    processed_query = qp.process(question)

    # --------------------------------------------------
    # BƯỚC 2: Hybrid Search
    # --------------------------------------------------
    print("\n[2/4] Đang thực hiện Hybrid Search lấy Seed Nodes...")

    seed_nodes = hybrid_search_nodes(
        query_text=processed_query["query_text"],
        query_vector=processed_query["query_vector"],
        top_k=5,
        alpha=0.7
    )

    print(f"-> Tìm thấy {len(seed_nodes)} Seed Nodes ban đầu.")

    # --------------------------------------------------
    # BƯỚC 3: Graph Expansion
    # --------------------------------------------------
    print("\n[3/4] Đang mở rộng đồ thị (Sub-graph Expansion)...")

    graph_result = expand_subgraph_raw(
        G=graph.graph,
        seed_nodes=seed_nodes,
        max_hops=1,
        max_neighbors=10,
        max_nodes=100
    )

    print(f"-> Tổng số Candidates Nodes sau expansion: {graph_result['total_nodes_found']}")
    print(f"-> Tổng số Edges tìm thấy: {len(graph_result['edges'])}")

    # --------------------------------------------------
    # BƯỚC 4: 3 AGENTS
    # --------------------------------------------------
    print("\n[4/4] Đang thực thi 3 Agents đánh giá...")

    literal_agent = LiteralAgent()
    semantic_agent = SemanticAgent()
    exception_agent = ExceptionAgent()

    print("  - Running Literal Agent...")
    literal_result = literal_agent.run(question, graph_result)
    save_json("literal.json", literal_result)

    print("  - Running Semantic Agent...")
    semantic_result = semantic_agent.run(question, graph_result)
    save_json("semantic.json", semantic_result)

    print("  - Running Exception Agent...")
    exception_result = exception_agent.run(question, graph_result)
    save_json("exception.json", exception_result)

    print(f"\n-> Đã lưu kết quả vào thư mục: {OUTPUT_DIR}")

    # --------------------------------------------------
    # BƯỚC 5: HIỂN THỊ KẾT QUẢ
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ TỪ 3 AGENTS")
    print("=" * 60)

    print("\n--- LITERAL AGENT RESULT ---")
    print(json.dumps(literal_result, indent=2, ensure_ascii=False))

    print("\n--- SEMANTIC AGENT RESULT ---")
    print(json.dumps(semantic_result, indent=2, ensure_ascii=False))

    print("\n--- EXCEPTION AGENT RESULT ---")
    print(json.dumps(exception_result, indent=2, ensure_ascii=False))

    print("\n[5/5] Đang tính Consensus Score...")

    consensus_result = consensus.compute()

    print("-> Consensus hoàn tất.")

    return {
        "literal": literal_result,
        "semantic": semantic_result,
        "exception": exception_result,
        "consensus": consensus_result,
        "graph_result": graph_result
    }


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    while True:

        try:

            user_question = input(
                "\nNhập câu hỏi (nhập 'exit' để dừng): "
            ).strip()

            if user_question.lower() in ["exit", "quit", "q"]:
                print("Đã dừng chương trình.")
                break

            if not user_question:
                continue

            run_pipeline(user_question)

        except KeyboardInterrupt:
            print("\nĐã dừng chương trình.")
            break

        except Exception as e:
            print(f"\n[LỖI]: {e}")