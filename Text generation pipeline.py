from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt="The AI agent quickly analyzed the data stream, and its core decision was to"

output=generator (
    prompt, 
    max_length=50, 
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
 )

print("Promt")
print(prompt)
print("\nOutput")
print(output[0]['generated_text'])
