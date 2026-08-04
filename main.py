import json
import os
import numpy as np


from model.model_graph import graph
from h8_query_processor import QueryProcessor
from h9_hybrid_search_algorithm import hybrid_search_nodes
from h11_graph_expansion import expand_subgraph_raw
from concurrent.futures import ThreadPoolExecutor


from h13_agents.literal_agent import LiteralAgent
from h13_agents.semantic_agent import SemanticAgent
from h13_agents.exception_agent import ExceptionAgent
from h13_agents.judge_agent import judge
from h13_agents.generator_agent import generator

from h14_consensus import consensus
from h15_filter import filter_consensus

literal_agent = LiteralAgent()
semantic_agent = SemanticAgent()
exception_agent = ExceptionAgent()




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
        top_k=3,
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

    

    print("  - Running Literal / Semantic / Exception Agent...")

    with ThreadPoolExecutor(max_workers=3) as executor:

        future_literal = executor.submit(
            literal_agent.run,
            question,
            graph_result
        )

        future_semantic = executor.submit(
            semantic_agent.run,
            question,
            graph_result
        )

        future_exception = executor.submit(
            exception_agent.run,
            question,
            graph_result
        )

        # Chờ cả 3 Agent hoàn thành
        literal_result = future_literal.result()
        semantic_result = future_semantic.result()
        exception_result = future_exception.result()

    # Lưu file sau khi cả 3 Agent chạy xong
    save_json("literal.json", literal_result)
    save_json("semantic.json", semantic_result)
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

    print("\n Đang tính Consensus Score...")

    consensus_result = consensus.compute()

    print("-> Consensus hoàn tất.")

    print("\n Đang lọc Candidate Nodes...")

    filtered_nodes = filter_consensus.compute()

    print(f"-> Giữ lại {len(filtered_nodes)} Candidate Nodes.")

    print("\n Judge đang chọn Context...")

    judge_result = judge.run(question)

    print("-> Judge hoàn tất.")

    print("\nGenerator đang sinh câu trả lời...")

    generator_result = generator.run(question)

    print("-> Generator hoàn tất.")

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