import json
from pathlib import Path


class ConsensusFilter:
    """
    Lọc các node có điểm gần với node tốt nhất.
    """

    def __init__(
        self,
        input_file="h12_output_multi/consensus.json",
        output_file="h12_output_multi/filtered_consensus.json",
        delta=2.0,
        top_k=8
    ):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)

        self.delta = delta
        self.top_k = top_k

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def compute(self):

        with open(
            self.input_file,
            "r",
            encoding="utf-8"
        ) as f:
            consensus = json.load(f)

        if len(consensus) == 0:

            with open(
                self.output_file,
                "w",
                encoding="utf-8"
            ) as f:
                json.dump([], f, indent=2, ensure_ascii=False)

            return []

        max_score = max(
            item["consensus_score"]
            for item in consensus
        )

        filtered = [
            item
            for item in consensus
            if item["consensus_score"] >= max_score - self.delta
        ]

        filtered.sort(
            key=lambda x: x["consensus_score"],
            reverse=True
        )

        filtered = filtered[:self.top_k]

        with open(
            self.output_file,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                filtered,
                f,
                indent=2,
                ensure_ascii=False
            )

        return filtered


filter_consensus = ConsensusFilter()