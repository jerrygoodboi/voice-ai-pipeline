"""
In-memory session history manager for multi-turn Voice AI conversations.
All data is stored in Python process memory and resets completely when the server is killed or restarted.
Implements a rolling token budget to prevent latency degradation.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages in-memory conversation histories keyed by session ID.
    Enforces a rolling token limit (~4 characters per token).
    """

    def __init__(self, max_tokens: int = 1500) -> None:
        self.max_tokens = max_tokens
        # Character budget approximation: ~4 characters per token
        self.max_chars = max_tokens * 4
        self._sessions: dict[str, list[dict[str, str]]] = {}

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Return the current conversation history for a given session."""
        return list(self._sessions.get(session_id, []))

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """
        Append a conversation turn ('user' or 'model') and prune if exceeding budget.
        """
        if not session_id or not content.strip():
            return

        normalized_role = "user" if role.lower() == "user" else "model"
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self._sessions[session_id].append({
            "role": normalized_role,
            "content": content.strip()
        })
        self._prune(session_id)

    def _prune(self, session_id: str) -> None:
        """
        Prune oldest turns from the history if total character length exceeds max_chars.
        Pruning removes turns from the start while preserving conversational coherence.
        """
        history = self._sessions.get(session_id, [])
        total_chars = sum(len(turn.get("content", "")) for turn in history)

        while total_chars > self.max_chars and len(history) > 2:
            removed = history.pop(0)
            total_chars -= len(removed.get("content", ""))
            logger.debug("[Session] Pruned turn (%s) to stay within token budget.", removed.get("role"))

    def reset_session(self, session_id: str) -> bool:
        """Clear the history for a specific session ID."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("[Session] Reset history for session '%s'.", session_id)
            return True
        return False

    def clear_all(self) -> None:
        """Clear all active sessions in memory."""
        self._sessions.clear()
        logger.info("[Session] All session histories cleared.")
