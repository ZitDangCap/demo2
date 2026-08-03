import json
from pathlib import Path

from model.model_graph import graph
from model.model_llm import llm


class JudgeAgent:

    def __init__(
        self,
        input_file="h12_output_multi_qwen/filtered_consensus.json",
        output_file="h12_output_multi_qwen/judge_context.json",
        temperature=0.0
    ):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.temperature = temperature

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def get_system_prompt(self):

        return """
Bạn là Judge Agent trong hệ thống Legal Graph RAG.

Bạn sẽ nhận:

- Câu hỏi.
- Danh sách Candidate Nodes.
- Consensus Score của từng node. Hãy tham khảo điểm này để củng cố đánh giá của bạn.

Nhiệm vụ:

1. Chọn những node cần thiết nhất.
2. Loại node dư thừa.
3. Loại node cha nếu node con đã chứa đầy đủ nội dung.
4. Giữ lại ngoại lệ nếu cần.

KHÔNG trả lời câu hỏi.

Output DUY NHẤT là JSON.

[
    {
    "node_id":"...",
    }
]
"""

    def build_context(
        self,
        question: str,
        candidates: list
    ):

        context = f"""
QUESTION:

{question}

========================
CANDIDATE NODES
========================

"""

        for idx, item in enumerate(candidates, start=1):

            node = graph.graph.nodes[item["node_id"]]

            context += f"""
----------------------------

Node [{idx}]

Node ID:
{item["node_id"]}

Consensus Score:
{item["consensus_score"]}

Title:
{node.get("title","")}

Type:
{node.get("node_type","")}

Content:
{node.get("text","")}

"""

        return context

    def run(
        self,
        question: str
    ):

        with open(
            self.input_file,
            "r",
            encoding="utf-8"
        ) as f:

            candidates = json.load(f)

        prompt = self.build_context(
            question,
            candidates
        )

        response = llm.generate(
            system_prompt=self.get_system_prompt(),
            user_prompt=prompt,
            temperature=self.temperature
        )

        result = json.loads(response)

        with open(
            self.output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False
            )

        return result


judge = JudgeAgent()