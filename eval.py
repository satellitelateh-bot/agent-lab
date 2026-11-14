import csv
import json
import os
import re
import sys
from datetime import datetime
from agent_core import run_task, groq, GROQ_MODEL


def judge(task, work):
    response = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": (
                "You are an independent judge. Score how well the work completes the task "
                "for the real business described: specific, accurate to the facts given, "
                "nothing invented, ready to use. Reply in exactly two lines:\n"
                "SCORE: <number 1-10>\nREASON: <one short sentence>"
            )},
            {"role": "user", "content": f"TASK: {task}\n\nWORK:\n{work}"},
        ],
    )
    text = response.choices[0].message.content or ""
    match = re.search(r"SCORE:\s*(\d+)", text)
    score = int(match.group(1)) if match else 0
    return score, text.split("REASON:")[-1].strip()


version = sys.argv[1] if len(sys.argv) > 1 else "unnamed"
with open("eval_set.json", encoding="utf-8") as f:
    cases = json.load(f)

os.makedirs("results", exist_ok=True)
rows = []

for case in cases:
    print(f"\n=== {case['id']} ({case['lang']}) ===")
    try:
        r = run_task(case["task"], verbose=False)
    except Exception as error:
        print("  FAILED:", error)
        rows.append({"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "version": version,
                     "case": case["id"], "reviewer_avg": 0, "judge": 0, "tiers": "error",
                     "claude_calls": 0, "groq_calls": 0, "local_calls": 0, "seconds": 0})
        continue

    judge_score, reason = judge(case["task"], r["final_text"])
    avg = round(sum(r["scores"]) / len(r["scores"]), 1) if r["scores"] else 0
    print(f"  reviewer avg {avg} | judge {judge_score} | tiers {r['tiers']} | {r['seconds']}s")
    print("  judge says:", reason)

    with open(f"results/{version}_{case['id']}.md", "w", encoding="utf-8") as f:
        f.write(f"# {case['id']} ({version})\n\nTASK: {case['task']}\n\n{r['final_text']}\n")

    rows.append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "version": version,
        "case": case["id"],
        "reviewer_avg": avg,
        "judge": judge_score,
        "tiers": "/".join(r["tiers"]),
        "claude_calls": r["calls"]["claude"],
        "groq_calls": r["calls"]["groq"],
        "local_calls": r["calls"]["local"],
        "seconds": r["seconds"],
    })

path = "results/history.csv"
new_file = not os.path.exists(path)
with open(path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    if new_file:
        writer.writeheader()
    writer.writerows(rows)

done = [row for row in rows if row["tiers"] != "error"]
all_tiers = "/".join(row["tiers"] for row in done).split("/") if done else []
free_share = round(100 * sum(t != "claude" for t in all_tiers) / len(all_tiers)) if all_tiers else 0

print(f"\n===== SUMMARY: {version} =====")
print("cases passed run:   ", f"{len(done)}/{len(rows)}")
print("avg reviewer score: ", round(sum(r["reviewer_avg"] for r in done) / max(len(done), 1), 1))
print("avg judge score:    ", round(sum(r["judge"] for r in done) / max(len(done), 1), 1))
print("steps done free:    ", f"{free_share}%")
print("total Claude calls: ", sum(r["claude_calls"] for r in done))
print("saved to:           ", path)
