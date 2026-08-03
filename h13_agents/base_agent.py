import json
import re
from model.model_llm import llm


class BaseAgent:
    """
    Base class cho tất cả Agent.
    Chỉ xử lý các phần dùng chung.
    """

    def __init__(
        self,
        name: str,
        temperature: float = 0.1
    ):
        self.name = name
        self.temperature = temperature

    def get_system_prompt(self) -> str:
        """
        Mỗi Agent sẽ tự định nghĩa Prompt.
        """
        raise NotImplementedError

    def build_context(self, graph_result: dict) -> str:
        """
        Dựng Context đầy đủ bao gồm Danh sách Node và Danh sách Edge (Quan hệ)
        """
        candidate_nodes = graph_result.get("candidate_nodes", [])
        edges = graph_result.get("edges", [])

        # --------------------------------------------------
        # 1. Format Nodes Context
        # --------------------------------------------------
        nodes_context = "=== DANH SÁCH CANDIDATE NODES ===\n"
        for idx, node in enumerate(candidate_nodes, start=1):
            nodes_context += f"""
--- Node [{idx}] ---
Node ID: {node.get("node_id", "")}
Title  : {node.get("title", "")}
Type   : {node.get("node_type", "")}
Is Seed: {node.get("is_seed", False)}
Content:
{node.get("text", "").strip()}
"""

        # --------------------------------------------------
        # 2. Format Edges Context (Mối quan hệ giữa các Node)
        # --------------------------------------------------
        edges_context = "\n=== DANH SÁCH CÁC MỐI QUAN HỆ (EDGES) ===\n"
        if edges:
            for idx, edge in enumerate(edges, start=1):
                source = edge.get("source", "")
                target = edge.get("target", "")
                relation = edge.get("relation", "CONNECTED")
                edges_context += f"[{idx}] {source} --[{relation}]--> {target}\n"
        else:
            edges_context += "Không có thông tin mối quan hệ.\n"

        return f"{nodes_context}\n{edges_context}"

    def build_user_prompt(
        self,
        question: str,
        graph_result: dict
    ) -> str:

        context = self.build_context(graph_result)

        return f"""
QUESTION (CÂU HỎI NGƯỜI DÙNG):
{question}

GRAPH DATA (DỮ LIỆU ĐỒ THỊ BẠN CẦN ĐÁNH GIÁ):
{context}
"""
    def _clean_json_string(self, text: str) -> str:
        """Bóc tách chính xác mảng JSON hoặc đối tượng JSON đầu tiên,

        loại bỏ toàn bộ text thừa surrounding.
        """
        text = text.strip()

        # 1. Bỏ markdown block (```json ... ```) nếu có
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        # 2. Tìm mảng JSON [...]
        match_array = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
        if match_array:
            return match_array.group(0)

        # 3. Tìm đối tượng JSON {...}
        match_object = re.search(r"\{.*\}", text, re.DOTALL)
        if match_object:
            return match_object.group(0)

        return text

    def run(
        self,
        question: str,
        graph_result: dict
    ) -> list:

        response = llm.generate(
            system_prompt=self.get_system_prompt(),
            user_prompt=self.build_user_prompt(
                question,
                graph_result
            ),
            temperature=self.temperature
        )

        cleaned_response = self._clean_json_string(response)

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            print(f"[{self.name}] Lỗi decode JSON từ LLM output:\n{response}")
            raise e