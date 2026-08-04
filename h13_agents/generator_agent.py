import json
from pathlib import Path

from model.model_graph import graph
from model.model_llm import llm


class GeneratorAgent:

    def __init__(
        self,
        input_file="h12_output_multi_qwen/judge_context.json",
        output_file="h12_output_multi_qwen/final_answer.json",
        temperature=0.2
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
Bạn là Generator trong hệ thống Legal Graph RAG.

Bạn là trợ lý pháp lý tiếng Việt.

LUÔN LUÔN trả lời bằng TIẾNG VIỆT.

Bạn sẽ nhận:

- Một câu hỏi.
- Danh sách các đoạn văn bản pháp luật đã được Judge lựa chọn.

NHIỆM VỤ

1. Chỉ sử dụng thông tin trong các đoạn văn bản được cung cấp.
2. Không sử dụng kiến thức ngoài.
3. Không tự suy diễn hoặc bịa thêm thông tin.
4. Tổng hợp thành một câu trả lời đầy đủ, mạch lạc.
5. Nếu dữ liệu không đủ để trả lời thì phải nói rõ.

OUTPUT

Chỉ trả về DUY NHẤT một JSON hợp lệ.

{
    "answer": "..."
}
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
LEGAL CONTEXT
========================

"""

        for idx, item in enumerate(candidates, start=1):

            node = graph.graph.nodes[item["node_id"]]

            context += f"""
----------------------------

Node [{idx}]

Node ID:
{item["node_id"]}

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

        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(result["answer"])

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


generator = GeneratorAgent()