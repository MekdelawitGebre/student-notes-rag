from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LLM:
    def __init__(self, model_name: str, device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)  # Move model to CPU/GPU

    def generate(self, prompt: str, max_tokens: int = 256):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False  # deterministic
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
