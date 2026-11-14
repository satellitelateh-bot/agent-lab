import re
import anthropic
import ollama
from dotenv import load_dotenv

load_dotenv()
claude = anthropic.Anthropic()
calls = {"claude": 0, "local": 0}

PASS_SCORE = 7
MAX_TRIES = 3


def ask_claude(system, prompt):
    calls["claude"] += 1
    response = claude.messages.create(
        model="claude-haiku-5-5",
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return next((b.text for b in response.content if b.type == "text"), "(no answer)")


def ask_local(system, prompt):
    calls["local"] += 1
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
        "You are a planner. Only plan, never do the work yourself. "
        "Split the task into exactly 3 short steps that a text-only AI can complete. "
        "Reply with one step per line, no numbering, no extra text.",
        task,
    )
    lines = [line.strip("-* ").strip() for line in text.splitlines()]
    return [line for line in lines if line][:3]


def worker(task, step, context, feedback, use_claude):
    system = (
        f"You are a helpful worker. The overall goal is: {task}\n"
        "Stay specific to that goal. Do only the step you are given, in under 100 words. Text only."
    )
    prompt = f"Work done so far:\n{context or 'nothing yet'}\n\nYour step: {step}"
    if feedback:
        prompt += f"\n\nYour last attempt was rejected. Feedback: {feedback}\nFix it."
    if use_claude:
        return ask_claude(system, prompt)
    return ask_local(system, prompt)


def review_step(task, step, result):
    text = ask_claude(
        "You are a strict reviewer. Judge ONLY this step, not the whole task. Reply in exactly two lines:\n"
        "SCORE: <number 1-10>\nFEEDBACK: <one short sentence>",
        f"TASK: {task}\nSTEP: {step}\nRESULT: {result}",
    )
    match = re.search(r"SCORE:\s*(\d+)", text)
    score = int(match.group(1)) if match else 0
    feedback = text.split("FEEDBACK:")[-1].strip()
    return score, feedback


task = "Help Bean There Cafe, a small coffee shop in Phuket Old Town (2 staff, about 80 customers a day, half are tourists, AI budget 1,000 THB a month, posts on Instagram and LINE) start using AI: plan it, then write one Instagram caption in English."

print("PLANNER (Claude)")
steps = planner(task)
for s in steps:
    print(" -", s)

final = []
context = ""
for i, step in enumerate(steps, 1):
    feedback = ""
    for attempt in range(1, MAX_TRIES + 1):
        use_claude = attempt == MAX_TRIES
        who = "Claude" if use_claude else "llama"
        result = worker(task, step, context, feedback, use_claude)
        score, feedback = review_step(task, step, result)
        print(f"\nSTEP {i} | try {attempt} ({who}) | score {score}/10")
        print("  feedback:", feedback)
        if score >= PASS_SCORE:
            break
    final.append((step, result, score, who))
    context += f"\nStep {i}: {result}\n"

print("\n===== FINAL WORK =====")
for i, (step, result, score, who) in enumerate(final, 1):
    print(f"\nSTEP {i} ({score}/10, by {who}): {step}")
    print(result)

print("\nAI calls:", calls)
