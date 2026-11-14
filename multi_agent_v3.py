import re
import anthropic
import ollama
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
claude = anthropic.Anthropic()
groq = Groq()
calls = {"local": 0, "groq": 0, "claude": 0}

PASS_SCORE = 7
GROQ_MODEL = "openai/gpt-oss-120b"
TIERS = ["local", "groq", "claude"]


def ask_claude(system, prompt):
    calls["claude"] += 1
    response = claude.messages.create(
        model="claude-haiku-5-5",
        max_tokens=2000,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return next((b.text for b in response.content if b.type == "text"), "(no answer)")


def ask_groq(system, prompt):
    calls["groq"] += 1
    response = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


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


ASK = {"local": ask_local, "groq": ask_groq, "claude": ask_claude}


def planner(task):
    text = ask_claude(
        "You are a planner. Only plan, never do the work yourself. "
        "Split the task into exactly 3 short steps that a text-only AI can complete. "
        "Reply with one step per line, no numbering, no extra text.",
        task,
    )
    lines = [line.strip("-* ").strip() for line in text.splitlines()]
    return [line for line in lines if line][:3]


def worker(tier, task, step, context, feedback):
    system = (
        f"You are a helpful worker. The overall goal is: {task}\n"
        "Stay specific to that goal. Do only the step you are given, in under 120 words. Text only."
    )
    prompt = f"Work done so far:\n{context or 'nothing yet'}\n\nYour step: {step}"
    if feedback:
        prompt += f"\n\nYour last attempt was rejected. Feedback: {feedback}\nFix it."
    return ASK[tier](system, prompt)


def review_step(task, step, result):
    text = ask_claude(
        "You are a strict reviewer. Judge ONLY this step, not the whole task. "
        "Reply in exactly two lines:\nSCORE: <number 1-10>\nFEEDBACK: <one short sentence>",
        f"OVERALL GOAL: {task}\nSTEP TO JUDGE: {step}\nRESULT: {result}",
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
    for tier in TIERS:
        result = worker(tier, task, step, context, feedback)
        score, feedback = review_step(task, step, result)
        print(f"\nSTEP {i} | {tier} | score {score}/10")
        print("  feedback:", feedback)
        if score >= PASS_SCORE:
            break
    final.append((step, result, score, tier))
    context += f"\nStep {i}: {result}\n"

print("\n===== FINAL WORK =====")
for i, (step, result, score, tier) in enumerate(final, 1):
    print(f"\nSTEP {i} ({score}/10, by {tier}): {step}")
    print(result)

print("\nWorker tier that finished each step:", [t for _, _, _, t in final])
print("AI calls:", calls)
