import anthropic
import ollama
from dotenv import load_dotenv

load_dotenv()
claude = anthropic.Anthropic()


def ask_claude(system, prompt):
    response = claude.messages.create(
        model="claude-haiku-5-5",
        max_tokens=600,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return next(b.text for b in response.content if b.type == "text")


def ask_local(system, prompt):
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]


def planner(task):
    text = ask_claude(
        "You are a planner. Split the task into exactly 3 short steps. "
        "Reply with one step per line, no numbering, no extra text.",
        task,
    )
    lines = [line.strip("-* ").strip() for line in text.splitlines()]
    return [line for line in lines if line][:3]


def worker(step):
    return ask_local("You are a helpful worker. Do this step in under 80 words.", step)


def reviewer(task, results):
    report = "\n\n".join(f"STEP: {s}\nRESULT: {r}" for s, r in results)
    return ask_claude(
        "You are a strict reviewer. Check the work against the task. "
        "Give a score out of 10 and one line of feedback per step.",
        f"TASK: {task}\n\n{report}",
    )


task = "Help a small coffee shop in Phuket start using AI: plan it, then write one Instagram caption in English."

print("PLANNER (Claude)")
steps = planner(task)
for s in steps:
    print(" -", s)

results = []
for i, step in enumerate(steps, 1):
    print(f"\nWORKER {i} (llama)")
    result = worker(step)
    print(result[:300])
    results.append((step, result))

print("\nREVIEWER (Claude)")
print(reviewer(task, results))
