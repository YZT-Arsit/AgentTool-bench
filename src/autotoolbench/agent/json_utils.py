from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Tuple, Type

from pydantic import BaseModel, ValidationError

from .schema import ActionPayload, PlanPayload, ReflectionPayload


def extract_json(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    fence = re.search(r"```json\s*(.*?)```", stripped, re.IGNORECASE | re.DOTALL)
    candidate = fence.group(1).strip() if fence else stripped
    if not candidate:
        raise ValueError("empty response")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    start_positions = [idx for idx in (candidate.find("{"), candidate.find("[")) if idx >= 0]
    if not start_positions:
        raise ValueError("no json object found")
    start = min(start_positions)
    stack: list[str] = []
    in_string = False
    escaped = False
    for idx, char in enumerate(candidate[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                break
            opener = stack.pop()
            if (opener, char) not in {("{", "}"), ("[", "]")}:
                raise ValueError("malformed json delimiters")
            if not stack:
                snippet = candidate[start : idx + 1]
                return json.loads(snippet)
    raise ValueError("unable to extract json")


def repair_json(text: str) -> str:
    repaired = text.strip()
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    repaired = repaired.replace("True", "true").replace("False", "false")
    return repaired


def _validate(model_cls: Type[BaseModel], payload: Dict[str, Any]) -> Tuple[bool, list[str], BaseModel | None]:
    try:
        parsed = model_cls.model_validate(payload)
        return True, [], parsed
    except ValidationError as exc:
        return False, [err["msg"] for err in exc.errors()], None


def validate_plan(payload: Dict[str, Any]) -> Tuple[bool, list[str], PlanPayload | None]:
    return _validate(PlanPayload, payload)


def validate_action(payload: Dict[str, Any]) -> Tuple[bool, list[str], ActionPayload | None]:
    return _validate(ActionPayload, payload)


def validate_reflection(payload: Dict[str, Any]) -> Tuple[bool, list[str], ReflectionPayload | None]:
    return _validate(ReflectionPayload, payload)


def bounded_retry(
    call_llm_fn: Callable[[], str],
    validator: Callable[[Dict[str, Any]], Tuple[bool, list[str], BaseModel | None]],
    max_retries: int = 2,
) -> Dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    parse_errors = 0
    validation_errors: list[str] = []
    for attempt in range(max_retries + 1):
        raw = call_llm_fn()
        try:
            payload = extract_json(raw)
        except Exception:
            parse_errors += 1
            repaired = repair_json(raw)
            try:
                payload = extract_json(repaired)
            except Exception as exc:
                attempts.append({"raw": raw, "parsed": None, "errors": [str(exc)]})
                continue
        ok, errors, parsed = validator(payload)
        attempts.append({"raw": raw, "parsed": payload, "errors": errors})
        if ok and parsed is not None:
            return {
                "ok": True,
                "raw": raw,
                "payload": parsed.model_dump(),
                "validation_errors": [],
                "attempts": attempts,
                "parse_failures": parse_errors,
                "fallback_used": False,
            }
        validation_errors = errors
    return {
        "ok": False,
        "raw": attempts[-1]["raw"] if attempts else "",
        "payload": attempts[-1]["parsed"] if attempts else None,
        "validation_errors": validation_errors,
        "attempts": attempts,
        "parse_failures": parse_errors,
        "fallback_used": True,
    }
