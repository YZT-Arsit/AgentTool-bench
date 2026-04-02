from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Tuple

from ..env.tasks import Task
from ..tools.base import Tool
from ..tools.registry import get, tool_names
from .budget import BudgetController, estimate_token_count
from .json_utils import bounded_retry, validate_action
from .safety import inspect_action
from .schema import ActionCandidate, ActionPayload, ActionScore, MemoryEntry, Trajectory

MEMORY_REF_PATTERN = re.compile(r"^\$(?:mem|memory):([A-Za-z0-9_\-\.]+)$")


class Executor:
    def __init__(self, llm=None, budget_ctrl: BudgetController | None = None, disable_memory: bool = False):
        self.llm = llm
        self.budget_ctrl = budget_ctrl or BudgetController(max_calls=9999, max_steps=9999, max_time=9999, max_tokens=10**9)
        self.disable_memory = disable_memory
        self.last_output: Any = None
        self.last_trace: Dict[str, Any] = {}
        self.memory: Dict[str, MemoryEntry] = {}

    def reset(self) -> None:
        self.last_output = None
        self.last_trace = {}
        self.memory = {}

    def memory_snapshot(self) -> Dict[str, Dict[str, Any]]:
        if self.disable_memory:
            return {}
        return {key: entry.model_dump() for key, entry in self.memory.items()}

    def _materialize_args(self, tool_name: str, args: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        materialized, referenced_keys = self._resolve_memory_refs(args)
        if tool_name == "file_write" and not materialized.get("content") and self.last_output is not None:
            if isinstance(self.last_output, str):
                materialized["content"] = self.last_output
            else:
                materialized["content"] = json.dumps(self.last_output, indent=2, ensure_ascii=False)
        return materialized, referenced_keys

    def _resolve_memory_refs(self, value: Any) -> Tuple[Any, List[str]]:
        if isinstance(value, str):
            match = MEMORY_REF_PATTERN.match(value.strip())
            if match:
                key = match.group(1)
                if self.disable_memory:
                    raise KeyError(f"memory_key_not_found:{key}")
                entry = self.memory.get(key)
                if entry is None:
                    raise KeyError(f"memory_key_not_found:{key}")
                return entry.value, [key]
            return value, []
        if isinstance(value, list):
            resolved_items: List[Any] = []
            referenced_keys: List[str] = []
            for item in value:
                resolved, item_refs = self._resolve_memory_refs(item)
                resolved_items.append(resolved)
                referenced_keys.extend(item_refs)
            return resolved_items, referenced_keys
        if isinstance(value, dict):
            resolved_dict: Dict[str, Any] = {}
            referenced_keys: List[str] = []
            for key, item in value.items():
                resolved, item_refs = self._resolve_memory_refs(item)
                resolved_dict[key] = resolved
                referenced_keys.extend(item_refs)
            return resolved_dict, referenced_keys
        return value, []

    def _infer_value_type(self, tool_name: str, output: Any) -> str:
        if tool_name == "file_read":
            return "file_text"
        if tool_name == "log_search":
            return "log_lines"
        if tool_name == "sql_query":
            return "sql_result"
        if tool_name == "doc_search":
            return "retrieval_results"
        if tool_name == "file_write":
            return "text"
        if isinstance(output, str):
            return "text"
        if isinstance(output, list):
            if output and all(isinstance(item, dict) and {"source", "chunk", "score", "rank"} <= set(item.keys()) for item in output):
                return "retrieval_results"
            if output and all(isinstance(item, dict) and {"line", "text"} <= set(item.keys()) for item in output):
                return "log_lines"
            if output and all(isinstance(item, dict) for item in output):
                return "rows"
            return "json"
        if isinstance(output, dict):
            return "json"
        return "unknown"

    def _build_memory_entry(self, *, key: str, value: Any, tool_name: str, source_step_id: str | None) -> MemoryEntry:
        return MemoryEntry(
            key=key,
            value=value,
            value_type=self._infer_value_type(tool_name, value),
            source_step_id=source_step_id,
            source_tool=tool_name,
        )

    def _update_memory(self, step: Dict[str, Any], action: Dict[str, Any], tool_name: str, output: Any) -> Dict[str, Any]:
        if self.disable_memory:
            return {}
        save_key = action.get("save_as") or step.get("save_as")
        source_step_id = step.get("step_id")
        updates: Dict[str, Any] = {}
        if source_step_id:
            entry = self._build_memory_entry(key=source_step_id, value=output, tool_name=tool_name, source_step_id=source_step_id)
            self.memory[source_step_id] = entry
            updates[source_step_id] = entry.model_dump()
        if save_key:
            entry = self._build_memory_entry(key=save_key, value=output, tool_name=tool_name, source_step_id=source_step_id)
            self.memory[save_key] = entry
            updates[save_key] = entry.model_dump()
        last_output_entry = self._build_memory_entry(key="last_output", value=output, tool_name=tool_name, source_step_id=source_step_id)
        self.memory["last_output"] = last_output_entry
        updates["last_output"] = last_output_entry.model_dump()
        return updates

    def _coerce_action_candidate(self, candidate_id: str, source: str, action: Dict[str, Any]) -> Dict[str, Any]:
        payload = ActionPayload.model_validate(
            {
                "tool": action.get("tool") or "noop",
                "args": action.get("args", {}),
                "rationale": action.get("rationale"),
                "save_as": action.get("save_as"),
            }
        )
        return ActionCandidate(candidate_id=candidate_id, source=source, **payload.model_dump()).model_dump()

    def _step_record_kwargs(self, step: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "branch_group": step.get("branch_group"),
            "branch_id": step.get("branch_id"),
            "merge_point": bool(step.get("merge_into")),
            "merge_summary": dict(step.get("merge_summary", {}) or {}),
        }

    def _build_memory_adjusted_candidate(self, step: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any] | None:
        merged_args = dict(action.get("args", {}))
        changed = False
        for key, value in step.get("args", {}).items():
            if key not in merged_args or merged_args[key] in (None, "", [], {}):
                merged_args[key] = value
                changed = True
        if (
            action.get("tool") == "file_write"
            and not merged_args.get("content")
            and (("last_output" in self.memory and not self.disable_memory) or self.last_output is not None)
        ):
            merged_args["content"] = "$memory:last_output" if not self.disable_memory and "last_output" in self.memory else self.last_output
            changed = True
        if not changed:
            return None
        return {
            "tool": action.get("tool") or step.get("tool") or "noop",
            "args": merged_args,
            "rationale": "memory_adjusted",
            "save_as": action.get("save_as") or step.get("save_as"),
        }

    def _collect_memory_reference_info(self, value: Any, root_key: str | None = None) -> List[Tuple[str, str]]:
        refs: List[Tuple[str, str]] = []
        if isinstance(value, str):
            match = MEMORY_REF_PATTERN.match(value.strip())
            if match:
                refs.append((root_key or "", match.group(1)))
            return refs
        if isinstance(value, list):
            for item in value:
                refs.extend(self._collect_memory_reference_info(item, root_key=root_key))
            return refs
        if isinstance(value, dict):
            for key, item in value.items():
                refs.extend(self._collect_memory_reference_info(item, root_key=root_key or key))
            return refs
        return refs

    def _matches_constraint_type(self, value: Any, expected: str) -> bool:
        if expected in {"any", ""}:
            return True
        if expected == "string":
            return isinstance(value, str)
        if expected == "array":
            return isinstance(value, list)
        if expected == "object":
            return isinstance(value, dict)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "boolean":
            return isinstance(value, bool)
        return True

    def _normalize_tool_args(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(args)
        if tool_name == "file_write" and "content" in normalized and not isinstance(normalized["content"], str):
            if isinstance(normalized["content"], (dict, list)):
                normalized["content"] = json.dumps(normalized["content"], indent=2, ensure_ascii=False)
            elif normalized["content"] is None:
                normalized["content"] = ""
        return normalized

    def _validate_tool_args(
        self,
        tool: Tool,
        tool_name: str,
        args: Dict[str, Any],
        raw_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized_args = self._normalize_tool_args(tool_name, args)
        required_keys = set(tool.input_schema.get("required", []))
        required_keys.update(name for name, constraint in tool.argument_constraints.items() if constraint.required)
        for key in sorted(required_keys):
            if key not in normalized_args:
                return {
                    "ok": False,
                    "label": "ARGUMENT_CONSTRAINT_VIOLATION",
                    "error_code": "missing_required_argument",
                    "message": f"Missing required argument '{key}' for tool {tool_name}",
                    "details": {"tool": tool_name, "argument": key},
                    "args": normalized_args,
                }

        for arg_name, memory_key in self._collect_memory_reference_info(raw_args):
            entry = self.memory.get(memory_key)
            if entry is None:
                continue
            allowed_types = tool.allowed_memory_types.get(arg_name, [])
            if allowed_types and entry.value_type not in allowed_types:
                return {
                    "ok": False,
                    "label": "MEMORY_TYPE_MISMATCH",
                    "error_code": "memory_type_mismatch",
                    "message": f"Memory slot '{memory_key}' of type {entry.value_type} is not compatible with argument '{arg_name}' for tool {tool_name}",
                    "details": {
                        "tool": tool_name,
                        "argument": arg_name,
                        "memory_key": memory_key,
                        "memory_type": entry.value_type,
                        "allowed_memory_types": allowed_types,
                    },
                    "args": normalized_args,
                }

        for key, constraint in tool.argument_constraints.items():
            if key not in normalized_args:
                continue
            value = normalized_args.get(key)
            if not self._matches_constraint_type(value, constraint.type):
                return {
                    "ok": False,
                    "label": "ARGUMENT_CONSTRAINT_VIOLATION",
                    "error_code": "argument_type_mismatch",
                    "message": f"Argument '{key}' for tool {tool_name} must be {constraint.type}",
                    "details": {"tool": tool_name, "argument": key, "expected_type": constraint.type, "actual_type": type(value).__name__},
                    "args": normalized_args,
                }
            if constraint.non_empty and value in ("", [], {}):
                return {
                    "ok": False,
                    "label": "ARGUMENT_CONSTRAINT_VIOLATION",
                    "error_code": "empty_argument",
                    "message": f"Argument '{key}' for tool {tool_name} cannot be empty",
                    "details": {"tool": tool_name, "argument": key},
                    "args": normalized_args,
                }
            if constraint.min_items is not None and isinstance(value, list) and len(value) < constraint.min_items:
                return {
                    "ok": False,
                    "label": "ARGUMENT_CONSTRAINT_VIOLATION",
                    "error_code": "too_few_items",
                    "message": f"Argument '{key}' for tool {tool_name} requires at least {constraint.min_items} items",
                    "details": {"tool": tool_name, "argument": key, "min_items": constraint.min_items, "actual_items": len(value)},
                    "args": normalized_args,
                }
            if constraint.pattern and isinstance(value, str):
                try:
                    if not re.search(constraint.pattern, value):
                        return {
                            "ok": False,
                            "label": "ARGUMENT_CONSTRAINT_VIOLATION",
                            "error_code": "pattern_mismatch",
                            "message": f"Argument '{key}' for tool {tool_name} does not satisfy its required pattern",
                            "details": {"tool": tool_name, "argument": key, "pattern": constraint.pattern},
                            "args": normalized_args,
                        }
                except re.error:
                    pass
            if constraint.enum and value not in constraint.enum:
                return {
                    "ok": False,
                    "label": "ARGUMENT_CONSTRAINT_VIOLATION",
                    "error_code": "enum_mismatch",
                    "message": f"Argument '{key}' for tool {tool_name} must be one of {constraint.enum}",
                    "details": {"tool": tool_name, "argument": key, "allowed": constraint.enum, "actual": value},
                    "args": normalized_args,
                }

        if tool_name == "file_write" and not isinstance(normalized_args.get("content", ""), str):
            return {
                "ok": False,
                "label": "TOOL_IO_MISMATCH",
                "error_code": "downstream_input_incompatible",
                "message": "file_write content must be serializable to text",
                "details": {"tool": tool_name, "argument": "content", "actual_type": type(normalized_args.get('content')).__name__},
                "args": normalized_args,
            }

        return {"ok": True, "args": normalized_args, "label": "", "error_code": "", "message": "", "details": {}}

    def _collect_candidates(
        self,
        step: Dict[str, Any],
        model_action: Dict[str, Any],
        action_override: Dict[str, Any] | None,
    ) -> List[Dict[str, Any]]:
        raw_candidates: List[Tuple[str, Dict[str, Any]]] = []
        if action_override is not None:
            raw_candidates.append(("patch_candidate", action_override))
        raw_candidates.append(("model_suggested", model_action))
        memory_adjusted = self._build_memory_adjusted_candidate(step, model_action)
        if memory_adjusted is not None:
            raw_candidates.append(("memory_adjusted", memory_adjusted))
        raw_candidates.append(
            (
                "plan_hint",
                {
                    "tool": step.get("tool") or "noop",
                    "args": step.get("args", {}),
                    "rationale": "plan_hint",
                    "save_as": step.get("save_as"),
                },
            )
        )
        raw_candidates.append(("fallback", self._fallback_action(step)))

        seen: set[str] = set()
        candidates: List[Dict[str, Any]] = []
        for index, (source, action) in enumerate(raw_candidates, start=1):
            candidate = self._coerce_action_candidate(f"C{index}", source, action)
            signature = json.dumps(
                {
                    "tool": candidate["tool"],
                    "args": candidate["args"],
                    "save_as": candidate.get("save_as"),
                },
                sort_keys=True,
                ensure_ascii=False,
                default=repr,
            )
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(candidate)
            if len(candidates) == 4:
                break
        return candidates

    def _score_candidate(
        self,
        task: Task,
        step: Dict[str, Any],
        budget: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        tool_name = candidate["tool"] or "noop"
        reasons: List[str] = []
        materialized_args = dict(candidate.get("args", {}))
        referenced_keys: List[str] = []
        missing_memory_key: str | None = None
        tool_exists = True
        tool = None
        tool_risk_level = "high"
        validation = {"ok": True, "args": materialized_args, "label": "", "error_code": "", "message": "", "details": {}}
        try:
            materialized_args, referenced_keys = self._materialize_args(tool_name, candidate.get("args", {}))
        except KeyError as exc:
            missing_memory_key = str(exc).split("memory_key_not_found:", 1)[-1].strip("'")
            reasons.append(f"references missing memory key '{missing_memory_key}'")
        try:
            tool = get(tool_name)
            tool_risk_level = getattr(tool, "risk_level", "low")
            if missing_memory_key is None:
                validation = self._validate_tool_args(tool, tool_name, materialized_args, candidate.get("args", {}))
                materialized_args = dict(validation.get("args", materialized_args))
        except KeyError:
            tool_exists = False
            reasons.append("tool is not registered")

        success_likelihood = 0.35
        compatibility = 0.2
        risk_level = 0.25

        if tool_exists:
            success_likelihood += 0.2
            compatibility += 0.15
            reasons.append("registered tool")
        if tool_name == (step.get("tool") or "noop"):
            success_likelihood += 0.2
            compatibility += 0.35
            risk_level -= 0.1
            reasons.append("matches planned tool")
        else:
            risk_level += 0.2
            reasons.append("deviates from planned tool")

        if candidate["source"] in {"patch_candidate", "memory_adjusted"}:
            success_likelihood += 0.08
            compatibility += 0.1
            reasons.append(f"uses {candidate['source']} refinement")
        if candidate["source"] == "fallback":
            risk_level += 0.1
            reasons.append("fallback candidate is conservative but lower confidence")
        if tool_risk_level == "medium":
            risk_level += 0.05
            reasons.append("tool carries medium execution risk")
        elif tool_risk_level == "high":
            risk_level += 0.12
            reasons.append("tool carries high execution risk")

        if candidate.get("save_as") == step.get("save_as") and candidate.get("save_as"):
            compatibility += 0.05
            reasons.append("preserves expected memory slot")

        expected_args = step.get("args", {})
        if expected_args:
            overlap = sum(1 for key in expected_args if key in materialized_args and materialized_args.get(key) not in (None, "", [], {}))
            compatibility += min(0.2, 0.1 * overlap)
            if overlap:
                reasons.append(f"covers {overlap} planned args")
        elif materialized_args:
            compatibility += 0.05

        if tool_name == "noop" and (step.get("tool") or "noop") != "noop":
            success_likelihood -= 0.35
            risk_level += 0.3
            reasons.append("noop is unlikely to satisfy a real tool step")

        if missing_memory_key is not None:
            success_likelihood -= 0.45
            compatibility -= 0.2
            risk_level += 0.45
        elif referenced_keys:
            success_likelihood += 0.05
            compatibility += 0.05
            reasons.append(f"reuses memory: {', '.join(sorted(set(referenced_keys)))}")

        if not tool_exists:
            success_likelihood -= 0.4
            compatibility -= 0.3
            risk_level += 0.45
        elif not validation.get("ok", True):
            success_likelihood -= 0.45
            compatibility -= 0.15
            risk_level += 0.4
            reasons.append(f"{validation['label'].lower()}: {validation['message']}")

        if not materialized_args and expected_args:
            success_likelihood -= 0.15
            risk_level += 0.1
            reasons.append("missing planned arguments")

        estimated_cost = {"calls": 1.0, "steps": 1.0, "time": 1.0, "tokens": 500.0}
        if tool_exists and missing_memory_key is None and tool is not None:
            estimated_cost = self.budget_ctrl.normalize_cost(tool.estimate_cost(materialized_args))
        safety = inspect_action(tool_name, materialized_args, tool_risk_level=tool_risk_level)
        if safety["safety_decision"] == "warned":
            risk_level += 0.1
            reasons.append(f"safety warning: {safety['safety_reason']}")
        elif safety["safety_decision"] == "blocked":
            success_likelihood -= 0.5
            compatibility -= 0.1
            risk_level += 0.5
            reasons.append(f"safety blocked: {safety['safety_reason']}")
        raw_cost = (
            float(estimated_cost.get("calls", 0.0))
            + 0.5 * float(estimated_cost.get("steps", 0.0))
            + 2.0 * float(estimated_cost.get("time", 0.0))
            + float(estimated_cost.get("tokens", 0.0)) / 400.0
        )
        normalized_budget_cost = min(2.0, raw_cost / 5.0)
        if tool_exists and missing_memory_key is None and tool is not None and not self.budget_ctrl.can_afford_tool(budget, estimated_cost):
            normalized_budget_cost = min(2.0, normalized_budget_cost + 0.7)
            risk_level += 0.35
            reasons.append("estimated cost is close to or beyond remaining budget")

        success_likelihood = max(0.0, min(1.0, success_likelihood))
        compatibility = max(0.0, min(1.0, compatibility))
        risk_level = max(0.0, min(1.0, risk_level))
        total_score = round(
            (0.45 * success_likelihood)
            + (0.3 * compatibility)
            - (0.15 * normalized_budget_cost)
            - (0.1 * risk_level),
            4,
        )

        score = ActionScore(
            candidate_id=candidate["candidate_id"],
            estimated_success_likelihood=round(success_likelihood, 4),
            estimated_budget_cost=round(normalized_budget_cost, 4),
            estimated_cost_breakdown={key: round(float(value), 4) for key, value in estimated_cost.items()},
            risk_level=round(risk_level, 4),
            tool_compatibility=round(compatibility, 4),
            total_score=total_score,
            reasons=[f"tool_risk={tool_risk_level}"] + reasons,
        ).model_dump()
        execution = {
            "tool": tool_name,
            "args": materialized_args,
            "referenced_memory_keys": sorted(set(referenced_keys)),
            "missing_memory_key": missing_memory_key,
            "estimated_cost": estimated_cost,
            "safety": safety,
            "validation": validation,
        }
        return score, execution

    def _rank_candidates(
        self,
        task: Task,
        step: Dict[str, Any],
        budget: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        scored: List[Dict[str, Any]] = []
        for candidate in candidates:
            score, execution = self._score_candidate(task, step, budget, candidate)
            scored.append({"candidate": candidate, "score": score, "execution": execution})
        ranked = sorted(scored, key=lambda item: (item["score"]["total_score"], item["score"]["estimated_success_likelihood"]), reverse=True)
        chosen = ranked[0]
        score = chosen["score"]
        selection_reason = (
            f"Selected {chosen['candidate']['source']} because it balanced higher success likelihood "
            f"({score['estimated_success_likelihood']:.2f}) and compatibility ({score['tool_compatibility']:.2f}) "
            f"against lower cost ({score['estimated_budget_cost']:.2f}) and risk ({score['risk_level']:.2f})."
        )
        return {
            "candidate_actions": [item["candidate"] for item in ranked],
            "action_scores": [item["score"] for item in ranked],
            "chosen_action": {
                **chosen["candidate"],
                "args": chosen["execution"]["args"],
            },
            "selection_reason": selection_reason,
            "execution": chosen["execution"],
        }

    def decide_action(
        self,
        task: Task,
        step: Dict[str, Any],
        budget: Dict[str, Any],
        step_index: int,
        budget_mode: str,
        scenario: str,
        last_obs: Any = None,
    ) -> Dict[str, Any]:
        context = {
            "task_id": task.task_id,
            "task_instruction": task.instruction,
            "task_type": task.task_type,
            "budget_mode": budget_mode,
            "scenario": scenario,
            "step_index": step_index,
            "plan_step": {
                "step_id": step.get("step_id", f"S{step_index + 1}"),
                "subgoal": step.get("subgoal"),
                "tool": step.get("tool") or "noop",
                "args_hint": step.get("args", {}),
                "success_criteria": step.get("success_criteria", []),
                "save_as": step.get("save_as"),
            },
            "last_output": self.last_output,
            "last_obs": last_obs,
            "memory": self.memory_snapshot(),
            "budget": budget,
            "available_tools": tool_names(),
        }
        messages = self._messages(context)
        result = bounded_retry(lambda: self.llm.generate(messages, schema_name="action"), validate_action, max_retries=2)
        llm_meta = getattr(self.llm, "last_generation_metadata", {})
        estimated_tokens = self._estimate_trace_tokens(messages, result["attempts"])
        if result["ok"]:
            payload = ActionPayload.model_validate(result["payload"]).model_dump()
            self.last_trace = {
                "llm_raw_text": result["raw"],
                "parsed_json": payload,
                "validation_errors": [],
                "fallback_used": False,
                "fallback_reason": "",
                "parse_failures": result["parse_failures"],
                "attempts": result["attempts"],
                "estimated_tokens": estimated_tokens,
                "referenced_memory_keys": [],
                "injection_metadata": llm_meta.get("injection"),
            }
            return payload

        action = self._fallback_action(step)
        self.last_trace = {
            "llm_raw_text": result["raw"],
            "parsed_json": result["payload"],
            "validation_errors": result["validation_errors"],
            "fallback_used": True,
            "fallback_reason": "action_json_invalid",
            "parse_failures": result["parse_failures"],
            "attempts": result["attempts"],
            "estimated_tokens": estimated_tokens,
            "referenced_memory_keys": [],
            "injection_metadata": llm_meta.get("injection"),
        }
        return action

    def execute_step(
        self,
        task: Task,
        step: Dict[str, Any],
        budget: Dict[str, Any],
        step_index: int,
        budget_mode: str,
        scenario: str,
        last_obs: Any = None,
        action_override: Dict[str, Any] | None = None,
    ) -> Trajectory:
        traj = Trajectory(task_id=task.task_id)
        model_action = self.decide_action(task, step, budget, step_index, budget_mode, scenario, last_obs=last_obs)
        metadata = dict(self.last_trace)
        candidates = self._collect_candidates(step, model_action, action_override)
        ranking = self._rank_candidates(task, step, budget, candidates)
        action = ranking["chosen_action"]
        tool_name = action.get("tool") or "noop"
        args = ranking["execution"]["args"]
        referenced_keys = ranking["execution"]["referenced_memory_keys"]
        safety = ranking["execution"]["safety"]

        metadata["candidate_actions"] = ranking["candidate_actions"]
        metadata["chosen_action"] = ranking["chosen_action"]
        metadata["action_scores"] = ranking["action_scores"]
        metadata["selection_reason"] = ranking["selection_reason"]
        metadata["tool_risk_level"] = safety["tool_risk_level"]
        metadata["action_allowed"] = safety["action_allowed"]
        metadata["safety_decision"] = safety["safety_decision"]
        metadata["safety_reason"] = safety["safety_reason"]
        metadata["safety_level"] = safety["safety_level"]
        metadata["action_json"] = {
            "tool": tool_name,
            "args": args,
            "rationale": action.get("rationale"),
            "save_as": action.get("save_as") or step.get("save_as"),
        }
        metadata["branch_group"] = step.get("branch_group")
        metadata["branch_id"] = step.get("branch_id")
        if step.get("merge_into"):
            metadata["merge_into"] = step.get("merge_into")
            metadata["merge_summary"] = dict(step.get("merge_summary", {}) or {})
        self.last_trace = {**metadata}

        missing_key = ranking["execution"]["missing_memory_key"]
        if missing_key is not None:
            metadata["failure_label"] = "BAD_TOOL_ARGS"
            metadata["memory_before"] = self.memory_snapshot()
            metadata["memory_after"] = self.memory_snapshot()
            metadata["referenced_memory_keys"] = [missing_key]
            metadata["memory_resolution_error"] = f"Missing memory key: {missing_key}"
            traj.add_step(
                subgoal=step.get("subgoal"),
                tool=tool_name,
                input=action.get("args", {}),
                error=f"missing_memory_key:{missing_key}",
                candidate_actions=ranking["candidate_actions"],
                chosen_action=ranking["chosen_action"],
                action_scores=ranking["action_scores"],
                selection_reason=ranking["selection_reason"],
                budget=self.budget_ctrl.snapshot(budget),
                metadata=metadata,
                tool_risk_level=safety["tool_risk_level"],
                action_allowed=safety["action_allowed"],
                safety_decision=safety["safety_decision"],
                safety_reason=safety["safety_reason"],
                safety_level=safety["safety_level"],
                **self._step_record_kwargs(step),
            )
            return traj
        metadata["memory_before"] = self.memory_snapshot()
        metadata["referenced_memory_keys"] = sorted(set(referenced_keys))

        try:
            tool = get(tool_name)
        except KeyError:
            metadata["failure_label"] = "TOOL_NOT_FOUND"
            metadata["memory_after"] = self.memory_snapshot()
            traj.add_step(
                subgoal=step.get("subgoal"),
                tool=tool_name,
                input=args,
                error="tool_not_found",
                candidate_actions=ranking["candidate_actions"],
                chosen_action=ranking["chosen_action"],
                action_scores=ranking["action_scores"],
                selection_reason=ranking["selection_reason"],
                budget=self.budget_ctrl.snapshot(budget),
                metadata=metadata,
                tool_risk_level=safety["tool_risk_level"],
                action_allowed=safety["action_allowed"],
                safety_decision=safety["safety_decision"],
                safety_reason=safety["safety_reason"],
                safety_level=safety["safety_level"],
                **self._step_record_kwargs(step),
            )
            return traj

        validation = ranking["execution"].get("validation", {"ok": True, "args": args})
        args = dict(validation.get("args", args))
        metadata["action_json"]["args"] = args
        if not validation.get("ok", True):
            metadata["failure_label"] = validation["label"]
            metadata["argument_validation"] = {
                "label": validation["label"],
                "error_code": validation["error_code"],
                "message": validation["message"],
                "details": validation["details"],
            }
            metadata["memory_after"] = self.memory_snapshot()
            traj.add_step(
                subgoal=step.get("subgoal"),
                tool=tool_name,
                input=args,
                error=validation["error_code"],
                candidate_actions=ranking["candidate_actions"],
                chosen_action=ranking["chosen_action"],
                action_scores=ranking["action_scores"],
                selection_reason=ranking["selection_reason"],
                tool_risk_level=safety["tool_risk_level"],
                action_allowed=safety["action_allowed"],
                safety_decision=safety["safety_decision"],
                safety_reason=safety["safety_reason"],
                safety_level=safety["safety_level"],
                budget=self.budget_ctrl.snapshot(budget),
                metadata=metadata,
                **self._step_record_kwargs(step),
            )
            return traj

        if not safety["action_allowed"]:
            metadata["failure_label"] = "TOOL_EXECUTION_FAILED"
            metadata["memory_after"] = self.memory_snapshot()
            traj.add_step(
                subgoal=step.get("subgoal"),
                tool=tool_name,
                input=args,
                error="safety_blocked",
                candidate_actions=ranking["candidate_actions"],
                chosen_action=ranking["chosen_action"],
                action_scores=ranking["action_scores"],
                selection_reason=ranking["selection_reason"],
                budget=self.budget_ctrl.snapshot(budget),
                metadata=metadata,
                tool_risk_level=safety["tool_risk_level"],
                action_allowed=safety["action_allowed"],
                safety_decision=safety["safety_decision"],
                safety_reason=safety["safety_reason"],
                safety_level=safety["safety_level"],
                **self._step_record_kwargs(step),
            )
            return traj

        cost = self.budget_ctrl.normalize_cost(tool.estimate_cost(args))
        if not self.budget_ctrl.can_afford_tool(budget, cost):
            metadata["failure_label"] = "BUDGET_EXHAUSTED"
            metadata["memory_after"] = self.memory_snapshot()
            traj.add_step(
                subgoal=step.get("subgoal"),
                tool=tool_name,
                input=args,
                error="budget_exhausted",
                candidate_actions=ranking["candidate_actions"],
                chosen_action=ranking["chosen_action"],
                action_scores=ranking["action_scores"],
                selection_reason=ranking["selection_reason"],
                budget=self.budget_ctrl.snapshot(budget),
                cost=cost,
                metadata=metadata,
                tool_risk_level=safety["tool_risk_level"],
                action_allowed=safety["action_allowed"],
                safety_decision=safety["safety_decision"],
                safety_reason=safety["safety_reason"],
                safety_level=safety["safety_level"],
                **self._step_record_kwargs(step),
            )
            return traj

        self.budget_ctrl.reserve_tool_estimate(budget, cost)
        started = time.perf_counter()
        res = tool.run(args)
        runtime_seconds = time.perf_counter() - started
        self.budget_ctrl.record_tool_call(budget, runtime_seconds)

        memory_delta: Dict[str, Any] = {}
        if res.ok:
            self.last_output = res.output
            memory_delta = self._update_memory(step, action, tool_name, res.output)
            if res.output in (None, "", [], {}):
                metadata["failure_label"] = "EMPTY_RESULT"
        elif res.error:
            metadata["failure_label"] = "TOOL_EXECUTION_FAILED"

        metadata["memory_after"] = self.memory_snapshot()
        metadata["tool_metadata"] = res.metadata
        traj.add_step(
            subgoal=step.get("subgoal"),
            tool=tool_name,
            input=args,
            output=res.output,
            error=res.error,
            candidate_actions=ranking["candidate_actions"],
            chosen_action=ranking["chosen_action"],
            action_scores=ranking["action_scores"],
            selection_reason=ranking["selection_reason"],
            tool_risk_level=safety["tool_risk_level"],
            action_allowed=safety["action_allowed"],
            safety_decision=safety["safety_decision"],
            safety_reason=safety["safety_reason"],
            safety_level=safety["safety_level"],
            budget=self.budget_ctrl.snapshot(budget),
            cost={**cost, "runtime": runtime_seconds},
            memory_delta=memory_delta,
            metadata=metadata,
            **self._step_record_kwargs(step),
        )
        traj.memory = dict(self.memory)
        traj.success = False
        return traj

    def execute(self, task: Task, plan: List[Dict[str, Any]], budget: Dict[str, Any], step_offset: int = 0) -> Trajectory:
        traj = Trajectory(task_id=task.task_id)
        last_obs: Any = None
        for idx, step in enumerate(plan, start=step_offset):
            step_traj = self.execute_step(task, step, budget, idx, "default", "legacy", last_obs=last_obs)
            traj.steps.extend(step_traj.steps)
            traj.memory = dict(self.memory)
            if step_traj.steps:
                last_obs = step_traj.steps[-1].output
        traj.success = False
        return traj

    def _messages(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are the executor for AutoToolBench. Return JSON only. "
                    'Schema: {"tool":"sql_query","args":{},"rationale":"...","save_as":"optional_name"}'
                ),
            },
            {"role": "user", "content": json.dumps(context, ensure_ascii=False, default=repr)},
        ]

    def _fallback_action(self, step: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = step.get("tool") or "noop"
        try:
            args, _ = self._materialize_args(tool_name, step.get("args", {}))
        except KeyError:
            args = dict(step.get("args", {}))
        return {
            "tool": tool_name,
            "args": args,
            "rationale": "fallback",
            "save_as": step.get("save_as"),
        }

    def _estimate_trace_tokens(self, messages: List[Dict[str, str]], attempts: List[Dict[str, Any]]) -> int:
        prompt_tokens = sum(estimate_token_count(message.get("content")) for message in messages)
        completion_tokens = sum(estimate_token_count(attempt.get("raw")) for attempt in attempts)
        return prompt_tokens + completion_tokens
