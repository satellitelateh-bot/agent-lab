import anthropic
import ollama
from dotenv import load_dotenv

load_dotenv()
claude = anthropic.Anthropic()


def has_thai(text):
    return any("\u0e00" <= ch <= "\u0e7f" for ch in text)


def choose_model(question):
    if has_thai(question):
        return "claude"
    if len(question) > 200:
        return "claude"
    return "local"


def ask_local(question):
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": question}],
    )
    return response["message"]["content"]


def ask_claude(question):
    response = claude.messages.create(
        model="claude-haiku-5-5",
        max_tokens=500,
        messages=[{"role": "user", "content": question}],
    )
    return next(b.text for b in response.content if b.type == "text")


questions = [
    "What is the capital of Japan?",
    "อธิบายว่า API คืออะไร แบบสั้นๆ",
    "Give me 3 tips to learn Python.",
    "ร้านกาแฟเล็กๆ ควรใช้ AI ช่วยอะไรได้บ้าง",
]

stats = {"local": 0, "claude": 0}

for q in questions:
    model = choose_model(q)
    stats[model] += 1
    answer = ask_local(q) if model == "local" else ask_claude(q)
    print(f"[{model.upper()}] {q}")
    print(answer[:200])
    print("-" * 40)

print("Routing summary:", stats)
