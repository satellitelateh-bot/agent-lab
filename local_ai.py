import ollama

question = "What is an AI agent? Answer in one sentence."

response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": question}],
)

print(response["message"]["content"])
