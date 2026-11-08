"""Agent loop with verified citations.

The model replies with one JSON object per turn (works with small local models
that lack native tool calling):
  {"thought": "...", "tool": "search_code", "args": {"query": "..."}}
  {"final": "answer with citations like path/to/file.py:12-40"}
`run_agent` is a generator of events so UIs can stream every step.
"""
from __future__ import annotations

import json
import re

from .config import SETTINGS
from .index import CodeIndex
from .tools import SPECS, ToolBox

SYSTEM = f"""You are CodeCompass, a code-understanding agent for the repository the user indexed.
Answer questions by investigating the code with tools, never from guesswork.

{SPECS}

Reply with ONE JSON object and nothing else:
  {{"thought": "short reasoning", "tool": "<tool name>", "args": {{...}}}}
or, when you have enough evidence:
  {{"final": "your answer"}}

Rules:
- Start with search_code, then read the most relevant files before answering.
- Cite evidence as path:start-end (for example auth/login.py:12-48). Only cite lines you have read or seen in results.
- If the code does not contain the answer, say so in the final answer."""

CITE = re.compile(r"([\w./\\-]+\.\w{1,5}):(\d+)(?:\s*[-–]\s*(\d+))?")


def extract_json(text: str):
    text = re.sub(r"```(?:json)?", "", text)
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
            if isinstance(obj, dict):
                return obj
        except ValueError:
            continue
    return None


def verify_citations(answer: str, box: ToolBox) -> list:
    out, seen_keys = [], set()
    for m in CITE.finditer(answer):
        path = m.group(1).replace("\\", "/")
        s = int(m.group(2))
        e = int(m.group(3) or s)
        key = (path, s, e)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        n = box.index.nlines.get(path)
        exists = n is not None and 1 <= s <= n and e >= s
        seen = any(p == path and s <= b and a <= e for p, a, b in box.seen)
        out.append({"cite": f"{path}:{s}-{e}", "exists": bool(exists), "seen_by_agent": seen,
                    "verified": bool(exists and seen)})
    return out


def run_agent(question: str, index: CodeIndex, llm, max_steps: int = None, history=None):
    max_steps = max_steps or SETTINGS.max_steps
    box = ToolBox(index)
    msgs = [{"role": "system", "content": SYSTEM}]
    for q, a in (history or [])[-4:]:  # short memory of earlier turns, in the JSON protocol
        msgs += [{"role": "user", "content": q},
                 {"role": "assistant", "content": json.dumps({"final": a[:1500]})}]
    msgs.append({"role": "user", "content": question})
    bad = 0
    for step in range(1, max_steps + 1):
        reply = llm.chat(msgs)
        obj = extract_json(reply)
        if obj is None:
            bad += 1
            if bad > 2:
                break
            msgs += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": "Reply with a single valid JSON object."}]
            continue
        if "final" in obj:
            ans = str(obj["final"])
            yield {"type": "answer", "text": ans, "citations": verify_citations(ans, box), "steps": step}
            return
        tool, args = obj.get("tool", ""), obj.get("args", {})
        obs = box.call(tool, args if isinstance(args, dict) else {})
        yield {"type": "step", "step": step, "thought": obj.get("thought", ""), "tool": tool,
               "args": args, "observation": obs[:600]}
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": "Observation:\n" + obs[:3000]}]
    # budget exhausted: force a best-effort answer from what was gathered
    msgs.append({"role": "user", "content": 'Give your best final answer now as {"final": "..."}.'})
    try:
        obj = extract_json(llm.chat(msgs)) or {}
    except Exception as e:
        obj = {"final": f"Agent stopped: {e}"}
    ans = str(obj.get("final", "I could not find a confident answer within the step budget."))
    yield {"type": "answer", "text": ans, "citations": verify_citations(ans, box), "steps": max_steps}


def ask(question: str, index: CodeIndex, llm, max_steps: int = None, history=None) -> dict:
    last = None
    for ev in run_agent(question, index, llm, max_steps, history):
        last = ev
    return last
