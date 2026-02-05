from transformers import pipeline

class LLM:
    def __init__(self, model_name: str):
        self.pipe = pipeline("text2text-generation", model=model_name)

    def generate(self, prompt: str, max_tokens: int):
        output = self.pipe(prompt, max_length=max_tokens)
        return output[0]["generated_text"]
