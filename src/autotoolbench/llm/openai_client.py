from __future__ import annotations

import json
import os
from typing import Any, Dict, List
from urllib import request

from .base import LLMBase


class OpenAIClient(LLMBase):
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model
        self.temperature = temperature
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required when --llm openai is enabled")

    def generate(self, messages: List[Dict[str, Any]], schema_name: str = "text") -> str:
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": messages,
            "response_format": {"type": "json_object"} if schema_name in {"plan", "action", "reflection"} else None,
        }
        body = json.dumps({key: value for key, value in payload.items() if value is not None}).encode("utf-8")
        req = request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
