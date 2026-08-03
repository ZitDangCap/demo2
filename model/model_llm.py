import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMModel:
    """
    Quản lý mô hình LLM.
    Model chỉ được load đúng một lần.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    ):
        print(f"[LLM] Loading model: {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )

        self.model_name = model_name

        print(f"[LLM] Using model: {model_name}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_new_tokens: int = 256
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

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        # Greedy decoding nếu temperature = 0
        if temperature <= 0:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs
            )

        # Chỉ lấy phần model sinh ra
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return answer


# Singleton
llm = LLMModel()