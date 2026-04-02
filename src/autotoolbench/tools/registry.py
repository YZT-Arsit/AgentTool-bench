from __future__ import annotations

from typing import Dict

from .base import Tool

_registry: Dict[str, Tool] = {}

def register(tool: Tool):
    _registry[tool.name] = tool


def get(name: str) -> Tool:
    return _registry[name]


def all_tools() -> Dict[str, Tool]:
    return dict(_registry)


def tool_names() -> list[str]:
    return list(_registry.keys())
