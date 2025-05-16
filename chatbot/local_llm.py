from transformers import pipeline

model_name = "google/flan-t5-base"
generator = pipeline('text2text-generation', model=model_name, device="cpu")

def chat_with_local_llm(prompt: str) -> str:
    results = generator(prompt, max_length=200, num_return_sequences=1, do_sample=True, top_k=10, top_p=0.9)
    return results[0]['generated_text']