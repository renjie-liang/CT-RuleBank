"""The local LLM (gpt-oss-120b behind vLLM, bound to localhost) with a replayable JSONL trace, and a replay caller answering from a trace."""
import json
import os
import time


def json_schema(name, schema):
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}


class Caller:
    def __init__(self, base_url, trace, llm):
        self.llm = llm
        import openai
        self.client = openai.OpenAI(base_url=base_url, api_key="local", timeout=1800)
        self.done = {}
        if os.path.exists(trace):
            for r in map(json.loads, open(trace)):
                self.done[r["key"]] = r
        self.fh = open(trace, "a")
        self.failed = trace.replace(".jsonl", ".failed.jsonl")

    def try_call(self, key, system, prompt, schema, temperature):
        """the record, or None when the reply was cut off, has no answer or is not valid JSON (appended to <trace>.failed.jsonl, not cached)"""
        if key in self.done:
            return self.done[key]
        t0 = time.time()
        r = self.client.chat.completions.create(model=self.llm.model, temperature=temperature, max_tokens=self.llm.max_tokens, reasoning_effort=self.llm.reasoning_effort,
                                                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                                                response_format=schema, **({} if self.llm.seed is None else {"seed": self.llm.seed}))
        m = r.choices[0]
        try:
            answer = json.loads(m.message.content) if m.finish_reason == "stop" and m.message.content else None
        except json.JSONDecodeError:
            answer = None
        if answer is None:
            with open(self.failed, "a") as f:
                f.write(json.dumps({"key": key, "finish_reason": m.finish_reason, "prompt": prompt, "content": m.message.content,
                                    "reasoning": m.message.reasoning, "usage": r.usage.model_dump(), "seconds": time.time() - t0}) + "\n")
            return None
        rec = {"key": key, "prompt": prompt, "answer": answer, "reasoning": m.message.reasoning, "usage": r.usage.model_dump(), "seconds": time.time() - t0}
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()
        self.done[key] = rec
        return rec


class ReplayCaller:
    """answers from a recorded trace: each prompt must equal the recorded one byte for byte, and every recorded call must be used once"""
    def __init__(self, trace, folds):
        self.rec = {}
        for r in map(json.loads, open(trace)):
            if r["key"].split("|")[1] in folds:
                assert r["key"] not in self.rec, f"duplicate key {r['key']} in {trace}"
                self.rec[r["key"]] = r
        self.used = set()

    def try_call(self, key, system, prompt, schema, temperature):
        assert key not in self.used, f"{key} asked twice"
        assert self.rec[key]["prompt"] == prompt, f"prompt differs from the recorded one for {key}"
        self.used.add(key)
        return self.rec[key]

    def check_all_used(self):
        assert not set(self.rec) - self.used, sorted(set(self.rec) - self.used)


class FileCaller:
    """An LLM outside this process (Claude through Claude Code subagents, no API key) reached through files. A call writes
    <exchange>/requests/<name>.md (what the model reads: SYSTEM, USER, the answer's JSON schema) and <name>.json (the same, machine-readable), then waits for
    <exchange>/answers/<name>.json, written by `python -m rulebank.file_exchange answer` after a schema check. The trace has the same records as Caller's,
    so ReplayCaller replays it. Temperature and reasoning effort are set by whoever answers, not here (recorded in the trace as model)."""
    POLL = 5

    def __init__(self, exchange, trace, llm):
        import threading
        self.llm, self.ex = llm, exchange
        for sub in ("requests", "answers", "drafts", "claimed"):
            os.makedirs(os.path.join(exchange, sub), exist_ok=True)
        self.done = {}
        if os.path.exists(trace):
            for r in map(json.loads, open(trace)):
                self.done[r["key"]] = r
        self.fh = open(trace, "a")
        self.lock = threading.Lock()
        self.beat()

    def beat(self):
        """the answering side stops when this file is older than file_exchange.STALE seconds and no request is open"""
        open(os.path.join(self.ex, "HEARTBEAT"), "w").write(str(time.time()))

    @staticmethod
    def name(key):
        return key.replace("|", "__")

    def try_call(self, key, system, prompt, schema, temperature):
        if key in self.done:
            return self.done[key]
        n, t0 = self.name(key), time.time()
        req = {"key": key, "system": system, "prompt": prompt, "schema": schema["json_schema"]["schema"]}
        for ext, text in ((".json", json.dumps(req)),
                          (".md", f"=== SYSTEM ===\n{system}\n\n=== USER ===\n{prompt}\n\n=== ANSWER: one JSON object valid under this schema ===\n{json.dumps(req['schema'])}\n")):
            tmp = os.path.join(self.ex, "requests", f".{n}{ext}.tmp")
            open(tmp, "w").write(text)
            os.replace(tmp, os.path.join(self.ex, "requests", n + ext))       # the .json last: a worker claims a request by its .json
        ans = os.path.join(self.ex, "answers", n + ".json")
        while not os.path.exists(ans):
            time.sleep(self.POLL)
            self.beat()
        a = json.load(open(ans))
        rec = {"key": key, "prompt": prompt, "answer": a["answer"], "reasoning": None, "model": a["model"], "seconds": time.time() - t0}
        with self.lock:
            self.fh.write(json.dumps(rec) + "\n")
            self.fh.flush()
            self.done[key] = rec
        return rec

    def finish(self):
        open(os.path.join(self.ex, "DONE"), "w").write(str(time.time()))
