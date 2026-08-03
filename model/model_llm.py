from email.mime import text

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

class LLMModel:
    """Mô hình LLM chỉ được khởi tạo một lần."""

    def __init__(
        self,
        model_name: str = "qwen3:4b",
        host: str = "http://localhost:11434"
    ):
        print(f"[LLM] Connecting to Ollama ({host})...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model_name = model_name

        print(f"[LLM] Using model: {model_name}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2
    ) -> str:
        """
        Sinh câu trả lời từ LLM.
        """

        messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": user_prompt
    }
]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            temperature=temperature,
            max_new_tokens=512
        )
        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        return answer
# Singleton
llm = LLMModel()