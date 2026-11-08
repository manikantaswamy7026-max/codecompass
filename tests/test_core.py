import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from codecompass.agent import ask, extract_json, verify_citations  # noqa: E402
from codecompass.chunker import chunk_file  # noqa: E402
from codecompass.index import CodeIndex, tokenize  # noqa: E402
from codecompass.retrieve import search  # noqa: E402
from codecompass.tools import ToolBox  # noqa: E402

AUTH = '''import hashlib


def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


class Auth:
    """Login handling."""

    def login(self, user, pw):
        if hash_password(pw) == user.hash:
            return make_token(user)


def make_token(user):
    return "tok-" + user.name
'''
ROUTES = '''from auth import Auth


def post_login(req):
    return Auth().login(req.user, req.pw)
'''


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.i = replies, 0

    def chat(self, messages, temperature=0.1):
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return r

    def embed(self, texts):
        raise RuntimeError("no embeddings")


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        for n, s in (("auth.py", AUTH), ("routes.py", ROUTES)):
            with open(os.path.join(self.d, n), "w") as fh:
                fh.write(s)
        self.idx = CodeIndex(self.d)
        self.stats = self.idx.build()

    def tearDown(self):
        shutil.rmtree(self.d)


class ChunkerTests(unittest.TestCase):
    def test_python_chunks_are_symbols(self):
        names = {c.name: c for c in chunk_file("auth.py", AUTH)}
        self.assertIn("hash_password", names)
        self.assertIn("Auth.login", names)
        self.assertIn("Auth", names)
        self.assertIn("login", names["Auth.login"].text)
        self.assertIn("hash_password", names["Auth.login"].calls)

    def test_line_ranges_match_text(self):
        for c in chunk_file("auth.py", AUTH):
            self.assertEqual(len(c.text.splitlines()), c.end - c.start + 1)

    def test_generic_and_big_split(self):
        js = "function a() {\n return 1\n}\n\nclass B {\n}\n"
        self.assertEqual([c.name for c in chunk_file("x.js", js)], ["a", "B"])
        big = "x = 1\n" * 300
        self.assertTrue(all(c.end - c.start + 1 <= 120 for c in chunk_file("big.md", big)))

    def test_tokenize_splits_identifiers(self):
        t = tokenize("getUserName hash_password")
        for w in ("getusername", "get", "user", "name", "hash", "password"):
            self.assertIn(w, t)


class IndexSearchTests(Base):
    def test_search_finds_definition(self):
        hits = search(self.idx, "how is the password hashed", k=3, mode="bm25")
        self.assertEqual(hits[0].chunk.name, "hash_password")

    def test_graph_expansion_adds_callers_or_callees(self):
        hits = search(self.idx, "login", k=6, mode="hybrid+graph")
        self.assertTrue(any(h.via != "search" for h in hits))

    def test_incremental_reindex(self):
        again = self.idx.build()
        self.assertEqual(again["reindexed_files"], 0)
        with open(os.path.join(self.d, "routes.py"), "a") as fh:
            fh.write("\n# changed\n")
        self.assertEqual(self.idx.build()["reindexed_files"], 1)

    def test_save_load_roundtrip(self):
        self.idx.save()
        loaded = CodeIndex.load_or_new(self.d)
        self.assertEqual(len(loaded.chunks), len(self.idx.chunks))


class ToolAndAgentTests(Base):
    def test_read_file_blocks_path_escape(self):
        box = ToolBox(self.idx)
        self.assertIn("not found", box.call("read_file", {"path": "../../etc/passwd"}).lower())

    def test_find_references_and_definition(self):
        box = ToolBox(self.idx)
        self.assertIn("Auth.login", box.call("find_references", {"symbol": "hash_password"}))
        self.assertIn("auth.py", box.call("go_to_definition", {"symbol": "make_token"}))

    def test_extract_json_handles_fences_and_noise(self):
        self.assertEqual(extract_json('Sure!\n```json\n{"final": "x"}\n```'), {"final": "x"})
        self.assertIsNone(extract_json("no json here"))

    def test_agent_loop_and_verified_citations(self):
        replies = [
            json.dumps({"thought": "look", "tool": "search_code", "args": {"query": "login"}}),
            json.dumps({"thought": "read", "tool": "read_file", "args": {"path": "auth.py", "start": 1, "end": 20}}),
            json.dumps({"final": "Login is in auth.py:11-14. Also see ghost.py:5-9 and auth.py:900."}),
        ]
        res = ask("where is login?", self.idx, FakeLLM(replies))
        self.assertEqual(res["type"], "answer")
        by = {c["cite"]: c for c in res["citations"]}
        self.assertTrue(by["auth.py:11-14"]["verified"])
        self.assertFalse(by["ghost.py:5-9"]["exists"])
        self.assertFalse(by["auth.py:900-900"]["exists"])

    def test_agent_recovers_from_bad_json_and_budget(self):
        res = ask("q", self.idx, FakeLLM(["not json", "still not", "nope", '{"final": "gave up"}']), max_steps=2)
        self.assertEqual(res["type"], "answer")

    def test_unseen_citation_is_flagged(self):
        box = ToolBox(self.idx)
        cites = verify_citations("see auth.py:1-5", box)
        self.assertTrue(cites[0]["exists"] and not cites[0]["verified"])


if __name__ == "__main__":
    unittest.main()
