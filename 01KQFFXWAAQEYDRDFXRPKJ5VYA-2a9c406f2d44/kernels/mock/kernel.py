from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from kernel_sdk import Kernel, KernelDispatcher, KernelRuntimeContext, KernelSubtreeResult, ResolvedComponentNode

_JSON_SCHEMA_KEYS = {
    "$defs",
    "$ref",
    "additionalProperties",
    "allOf",
    "anyOf",
    "const",
    "default",
    "enum",
    "items",
    "oneOf",
    "patternProperties",
    "prefixItems",
    "properties",
    "required",
    "type",
}


def _node_id(node: ResolvedComponentNode) -> str:
    if node.local_metadata and node.local_metadata.node_id:
        return node.local_metadata.node_id
    return node.component_id


def _mock_output(output_schema: Mapping[str, Any] | None) -> dict[str, Any]:
    if not output_schema:
        return {}

    if _looks_like_json_schema(output_schema):
        value = _mock_value(output_schema)
        return value if isinstance(value, dict) else {"value": value}

    return {key: _mock_value(value, field_name=key) for key, value in output_schema.items()}


def _looks_like_json_schema(schema: Mapping[str, Any]) -> bool:
    return any(key in schema for key in _JSON_SCHEMA_KEYS)


def _mock_value(schema: Any, *, field_name: str | None = None) -> Any:
    if isinstance(schema, Mapping):
        if "default" in schema:
            return schema["default"]
        if "const" in schema:
            return schema["const"]
        enum_values = schema.get("enum")
        if isinstance(enum_values, Sequence) and not isinstance(enum_values, str) and enum_values:
            return enum_values[0]
        for union_key in ("oneOf", "anyOf"):
            options = schema.get(union_key)
            if isinstance(options, list) and options:
                return _mock_value(options[0], field_name=field_name)
        all_of = schema.get("allOf")
        if isinstance(all_of, list) and all_of:
            merged: dict[str, Any] = {}
            for option in all_of:
                value = _mock_value(option, field_name=field_name)
                if isinstance(value, dict):
                    merged.update(value)
            return merged

        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            schema_type = next((item for item in schema_type if item != "null"), schema_type[0] if schema_type else None)

        if schema_type == "object" or "properties" in schema:
            properties = schema.get("properties")
            if isinstance(properties, Mapping):
                return {key: _mock_value(value, field_name=key) for key, value in properties.items()}
            return {}
        if schema_type == "array":
            min_items = schema.get("minItems")
            if isinstance(min_items, int) and min_items > 0:
                item_schema = schema.get("items", {})
                return [_mock_value(item_schema) for _ in range(min_items)]
            return []
        if schema_type == "string":
            return _mock_string(field_name)
        if schema_type in {"integer", "number"}:
            minimum = schema.get("minimum")
            if isinstance(minimum, int | float):
                return minimum
            return 0
        if schema_type == "boolean":
            return False
        if schema_type == "null":
            return None

        if _looks_like_json_schema(schema):
            return {}
        return {key: _mock_value(value, field_name=key) for key, value in schema.items()}

    if isinstance(schema, str):
        normalized = schema.strip()
        if "|" in normalized:
            first_option = normalized.split("|", 1)[0].strip()
            return _coerce_named_value(first_option, field_name)
        return _coerce_named_value(normalized, field_name)
    if isinstance(schema, list):
        return []
    return schema


def _coerce_named_value(value: str, field_name: str | None) -> Any:
    normalized = value.strip()
    lowered = normalized.lower()
    if lowered == "string":
        return _mock_string(field_name)
    if lowered in {"number", "integer"}:
        return 0
    if lowered == "boolean":
        return False
    if lowered in {"object", "dict"}:
        return {}
    if lowered in {"array", "list"}:
        return []
    if lowered == "null":
        return None
    return normalized


def _mock_string(field_name: str | None) -> str:
    if field_name:
        return f"mock_{field_name}"
    return "mock"


class MockKernel(Kernel):
    kind: ClassVar[str] = "mock"

    async def execute(
        self,
        node: ResolvedComponentNode,
        runtime_context: KernelRuntimeContext,
        runtime_sources: Mapping[str, Any],
        runtime: KernelDispatcher,
    ) -> KernelSubtreeResult:
        result = self._result_for_node(node, runtime_context)
        runtime_context.store_node_result(result.node_id or result.component_id, result)
        return result

    def _result_for_node(
        self,
        node: ResolvedComponentNode,
        runtime_context: KernelRuntimeContext,
    ) -> KernelSubtreeResult:
        output = _mock_output(node.output_schema)
        result = KernelSubtreeResult(
            component_id=node.component_id,
            node_id=_node_id(node),
            kernel_kind=self.kind,
            output=output,
            reasoning="Mock result generated from the component output schema.",
            evidence=[],
            control={"decision": output["decision"]} if "decision" in output else {},
            children=[self._result_for_node(child, runtime_context) for child in node.children],
        )
        runtime_context.store_node_result(result.node_id or result.component_id, result)
        return result
