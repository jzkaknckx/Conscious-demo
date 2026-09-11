import glob
import sqlite3

DB = "/home/p/.codex/state_5.sqlite"
FILES = [
    "/home/p/.codex/sessions/2026/07/12/rollout-2026-07-12T19-50-12-019f5629-de89-7010-b748-50fc733211c1.jsonl",
    "/home/p/.codex/sessions/2026/07/16/rollout-2026-07-16T20-24-09-019f6ae2-6407-7580-a76a-e51665675649.jsonl",
    "/home/p/.codex/sessions/2026/07/17/rollout-2026-07-17T15-56-20-019f6f13-8f59-7160-b706-73642397483b.jsonl",
    "/home/p/.codex/sessions/2026/07/26/rollout-2026-07-26T16-23-08-019f9d85-55aa-7781-9352-a6518ccf04b1.jsonl",
    "/home/p/.codex/sessions/2026/09/07/rollout-2026-09-07T20-20-51-01a07bd0-68fa-7fa0-b275-f958945d3a00.jsonl",
    "/home/p/.codex/sessions/2026/09/09/rollout-2026-09-09T21-46-45-01a0866b-c734-76e2-a0dc-c547c5498c44.jsonl",
]
NEW = '"model_provider":"openai"'
OLD = '"model_provider":"deepseek"'


def update_sqlite():
    con = sqlite3.connect(DB, timeout=30)
    con.execute("PRAGMA busy_timeout=30000")
    cur = con.execute(
        "UPDATE threads SET model_provider='deepseek' WHERE model_provider='openai'"
    )
    n = cur.rowcount
    con.commit()
    con.close()
    print(f"threads updated: {n}")


def update_rollouts():
    for path in FILES:
        with open(path, "r", encoding="utf-8") as fh:
            data = fh.read()
        count = data.count(OLD)
        data = data.replace(OLD, NEW)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(data)
        print(f"{path.split('/')[-1]}: {count} replaced")


if __name__ == "__main__":
    update_sqlite()
    update_rollouts()