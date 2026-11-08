import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from test_core import Base, FakeLLM  # noqa: E402
from codecompass import commands  # noqa: E402
from codecompass.agent import run_agent  # noqa: E402
from codecompass.cli import main  # noqa: E402


class CommandTests(Base):
    def test_stats(self):
        out = commands.stats(self.idx)
        self.assertIn("2 files", out)
        self.assertIn("python", out)

    def test_find_def_and_callers(self):
        self.assertIn("hash_password", commands.find_def(self.idx, "hash_password"))
        self.assertIn("Auth.login", commands.callers(self.idx, "hash_password"))
        self.assertIn("No definition", commands.find_def(self.idx, "nope"))
        self.assertIn("Nothing", commands.callers(self.idx, "nope"))

    def test_prompts_mention_target(self):
        self.assertIn("auth.py", commands.explain_prompt("auth.py"))
        self.assertIn("hash_password", commands.impact_prompt("hash_password"))
        self.assertIn("8 files", commands.onboard_prompt())

    def test_cli_stats_without_ollama(self):
        main(["stats", self.d])  # must not need a model

    def test_history_is_sent_to_the_model(self):
        seen = {}

        class Spy(FakeLLM):
            def chat(self, messages, temperature=0.1):
                seen["msgs"] = messages
                return json.dumps({"final": "ok"})

        list(run_agent("and who calls it?", self.idx, Spy([]), history=[("where is login?", "in auth.py")]))
        roles = [m["role"] for m in seen["msgs"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        self.assertIn("in auth.py", seen["msgs"][2]["content"])


if __name__ == "__main__":
    unittest.main()
