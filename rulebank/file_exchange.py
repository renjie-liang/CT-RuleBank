"""The answering side of llm.FileCaller, run by a Claude Code subagent (one subagent per request, fresh context each time).
    python -m rulebank.file_exchange next <exchange> --wait 540     claims the oldest open request; prints 'REQUEST <path of its .md>', 'WAIT' or 'DONE'
    python -m rulebank.file_exchange answer <exchange> <name>        checks drafts/<name>.json against the request's schema; moves it to answers/ or prints the error
Uses only the standard library and jsonschema, so it runs on the login node."""
import argparse
import json
import os
import sys
import time

import jsonschema

MODEL = "claude-sonnet-5-5 (Claude Code subagent)"
STALE = 600            # the learner beats every few seconds while it waits; older means it has finished or died
RECLAIM = 1800         # a claim without an answer after this long is open again (its subagent died)


def open_requests(ex):
    req = os.path.join(ex, "requests")
    names = [f[:-5] for f in os.listdir(req) if f.endswith(".json") and not f.startswith(".")]
    return sorted((n for n in names if not os.path.exists(os.path.join(ex, "answers", n + ".json"))), key=lambda n: os.path.getmtime(os.path.join(req, n + ".json")))


def claim(ex, n):
    c = os.path.join(ex, "claimed", n)
    if os.path.exists(c) and time.time() - os.path.getmtime(c) > RECLAIM:
        os.remove(c)
    try:
        os.close(os.open(c, os.O_CREAT | os.O_EXCL))
        return True
    except FileExistsError:
        return False


def next_request(ex, wait):
    t0 = time.time()
    while time.time() - t0 < wait:
        for n in open_requests(ex):
            if claim(ex, n):
                return f"REQUEST {os.path.join(ex, 'requests', n + '.md')}"
        beat = os.path.join(ex, "HEARTBEAT")                                   # missing: the learner has not started yet
        if os.path.exists(os.path.join(ex, "DONE")) or (os.path.exists(beat) and time.time() - os.path.getmtime(beat) > STALE):
            return "DONE"
        time.sleep(3)
    return "WAIT"


def answer(ex, n):
    schema = json.load(open(os.path.join(ex, "requests", n + ".json")))["schema"]
    a = json.load(open(os.path.join(ex, "drafts", n + ".json")))
    jsonschema.validate(a, schema)
    tmp = os.path.join(ex, "answers", f".{n}.tmp")
    json.dump({"answer": a, "model": MODEL}, open(tmp, "w"))
    os.replace(tmp, os.path.join(ex, "answers", n + ".json"))
    return f"ANSWERED {n}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["next", "answer"])
    ap.add_argument("exchange")
    ap.add_argument("name", nargs="?")
    ap.add_argument("--wait", type=int, default=540)
    a = ap.parse_args()
    if a.cmd == "next":
        print(next_request(a.exchange, a.wait))
    else:
        try:
            print(answer(a.exchange, a.name))
        except (jsonschema.ValidationError, json.JSONDecodeError) as e:         # shown to the subagent so it can fix its JSON
            print(f"INVALID {type(e).__name__}: {str(e)[:1500]}")
            sys.exit(1)
