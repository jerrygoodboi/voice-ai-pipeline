"""
Unit tests for SessionManager in-memory conversation history.
"""

import unittest
from src.llm.session import SessionManager


class TestSessionManager(unittest.TestCase):
    def test_add_and_retrieve_history(self) -> None:
        sm = SessionManager(max_tokens=1000)
        sm.add_turn("session-1", "user", "Hello assistant")
        sm.add_turn("session-1", "model", "Hello user, how can I help?")

        history = sm.get_history("session-1")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "Hello assistant")
        self.assertEqual(history[1]["role"], "model")
        self.assertEqual(history[1]["content"], "Hello user, how can I help?")

    def test_session_isolation(self) -> None:
        sm = SessionManager(max_tokens=1000)
        sm.add_turn("session-a", "user", "Message for A")
        sm.add_turn("session-b", "user", "Message for B")

        hist_a = sm.get_history("session-a")
        hist_b = sm.get_history("session-b")

        self.assertEqual(len(hist_a), 1)
        self.assertEqual(hist_a[0]["content"], "Message for A")
        self.assertEqual(len(hist_b), 1)
        self.assertEqual(hist_b[0]["content"], "Message for B")

    def test_rolling_budget_pruning(self) -> None:
        # Small budget: 20 tokens -> 80 characters
        sm = SessionManager(max_tokens=20)
        sm.add_turn("sess", "user", "Old message 1 " + "a" * 30)
        sm.add_turn("sess", "model", "Old message 2 " + "b" * 30)
        sm.add_turn("sess", "user", "New message 3 " + "c" * 30)

        history = sm.get_history("sess")
        # Oldest message should have been pruned
        self.assertTrue(len(history) <= 2)
        self.assertEqual(history[-1]["content"], "New message 3 " + "c" * 30)

    def test_reset_session(self) -> None:
        sm = SessionManager(max_tokens=1000)
        sm.add_turn("sess", "user", "Test prompt")
        self.assertEqual(len(sm.get_history("sess")), 1)

        sm.reset_session("sess")
        self.assertEqual(len(sm.get_history("sess")), 0)


if __name__ == "__main__":
    unittest.main()
