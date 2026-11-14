import re
import time
import anthropic
import ollama
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
claude = anthropic.Anthropic()
groq = Groq()

PASS_SCORE = 7
CLAUDE_MODEL = "claude-haiku-5-5"
GROQ_MODEL = "openai/gpt-oss-120b"
LOCAL_MODEL = "llama3.2"
TIERS = ["local", "groq", "claude"]
calls = {"local": 0, "groq": 0, "claude": 0}


def ask_claude(system, prompt):
    calls["claude"] += 1
    response = claude.messages.create(
        model=CLAUDE_MODEL,
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
    return response.choices[0].message.content or "(no answer)"


def ask_local(system, prompt):
    calls["local"] += 1
    response = ollama.chat(
        model=LOCAL_MODEL,
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
        "Use only the facts given. Never invent prices, addresses or details. "
        "Do only the step you are given, in under 150 words. Text only."
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


def run_task(task, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    for key in calls:
        calls[key] = 0
    start = time.time()

    steps = planner(task)
    say("PLANNER:", *steps, sep="\n - ")

    final = []
    context = ""
    for i, step in enumerate(steps, 1):
        feedback = ""
        for tier in TIERS:
            result = worker(tier, task, step, context, feedback)
            score, feedback = review_step(task, step, result)
            say(f"STEP {i} | {tier} | {score}/10 | {feedback}")
            if score >= PASS_SCORE:
                break
        final.append({"step": step, "result": result, "score": score, "tier": tier})
        context += f"\nStep {i}: {result}\n"

    return {
        "steps": final,
        "scores": [s["score"] for s in final],
        "tiers": [s["tier"] for s in final],
        "calls": dict(calls),
        "seconds": round(time.time() - start, 1),
        "final_text": "\n\n".join(s["result"] for s in final),
    }
