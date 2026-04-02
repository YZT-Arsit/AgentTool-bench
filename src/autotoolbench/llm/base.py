from __future__ import annotations

from typing import Any, Dict, List


class LLMBase:
    model_name: str = "unknown"

    def generate(self, messages: List[Dict[str, Any]], schema_name: str = "text") -> str:
        """Generate a response based on a list of messages.
        Each message is a dict with 'role' and 'content'.
        """
        raise NotImplementedError
