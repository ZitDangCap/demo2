import json
import os


class ConsensusScorer:

    def __init__(
        self,
        input_dir="h12_output_multi_qwen",
        output_file="consensus.json",
        literal_weight=0.4,
        semantic_weight=0.4,
        exception_weight=0.2
    ):

        self.input_dir = input_dir
        self.output_file = output_file

        self.literal_weight = literal_weight
        self.semantic_weight = semantic_weight
        self.exception_weight = exception_weight

    def _load_json(self, filename):

        with open(
            os.path.join(self.input_dir, filename),
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    def compute(self):

        literal = self._load_json("literal.json")
        semantic = self._load_json("semantic.json")
        exception = self._load_json("exception.json")

        literal = {x["node_id"]: x["score"] for x in literal}
        semantic = {x["node_id"]: x["score"] for x in semantic}
        exception = {x["node_id"]: x["score"] for x in exception}

        node_ids = (
            set(literal.keys())
            | set(semantic.keys())
            | set(exception.keys())
        )

        results = []

        for node_id in node_ids:

            score = (
                literal.get(node_id, 0) * self.literal_weight
                + semantic.get(node_id, 0) * self.semantic_weight
                + exception.get(node_id, 0) * self.exception_weight
            )

            results.append({
                "node_id": node_id,
                "consensus_score": round(score, 2)
            })

        results.sort(
            key=lambda x: x["consensus_score"],
            reverse=True
        )

        with open(
            os.path.join(self.input_dir, self.output_file),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                results,
                f,
                ensure_ascii=False,
                indent=2
            )

        return results


consensus = ConsensusScorer()