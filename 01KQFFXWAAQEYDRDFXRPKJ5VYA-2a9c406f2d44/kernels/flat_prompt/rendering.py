from __future__ import annotations

import json
from typing import Any

from kernel_sdk import ResolvedComponentNode


def render_component_tree_markdown(node: ResolvedComponentNode) -> str:
    lines: list[str] = []
    _render_node(node, lines, depth=0, path="root")
    return "\n".join(lines).strip()


def _render_node(node: ResolvedComponentNode, lines: list[str], *, depth: int, path: str) -> None:
    heading = "#" * (depth + 1)
    child_heading = "#" * (depth + 2)

    local_metadata = getattr(node, "local_metadata", None)
    label = getattr(local_metadata, "label", None) or getattr(node, "name", None) or getattr(node, "component_id", "unknown")

    instructions = (
        getattr(local_metadata, "override_instructions", None)
        or getattr(node, "instructions", None)
        or getattr(local_metadata, "instructions", None)
        or "No instructions provided."
    )

    children = getattr(node, "children", None) or []
    purpose = getattr(node, "purpose", None) or "No purpose provided."

    lines.extend(
        [
            f"{heading} Component {path}: {label}",
            f"- depth: {depth}",
            f"- component_id: `{getattr(node, 'component_id', 'unknown')}`",
            f"- direct_child_count: {len(children)}",
            f"- purpose: {purpose}",
        ]
    )

    if children:
        lines.append("- direct_children_in_order:")
        for index, child in enumerate(children, start=1):
            child_local_metadata = getattr(child, "local_metadata", None)
            child_label = (
                getattr(child_local_metadata, "label", None)
                or getattr(child, "name", None)
                or getattr(child, "component_id", "unknown")
            )
            lines.append(f"  {index}. `{getattr(child, 'component_id', 'unknown')}` -- {child_label}")
    else:
        lines.append("- direct_children_in_order: []")

    lines.extend(["", f"{child_heading} Instructions", instructions])

    output_schema = getattr(node, "output_schema", None)
    if output_schema:
        lines.extend(["", f"{child_heading} Output Contract"])
        lines.append("This component's `output` object must match this contract:")
        lines.append(f"```json\n{json.dumps(output_schema, indent=2, sort_keys=True)}\n```")

    combinator = getattr(node, "combinator", None)
    if combinator:
        lines.extend(["", f"{child_heading} Child Aggregation / Combinator"])
        lines.extend(_render_combinator(combinator))

    for index, child in enumerate(children, start=1):
        lines.append("")
        _render_node(child, lines, depth=depth + 1, path=f"{path}.{index}")


def _render_combinator(combinator: Any) -> list[str]:
    kind = getattr(combinator, "kind", None)
    short_circuit = bool(getattr(combinator, "short_circuit", False))

    if kind == "builtin":
        operator = getattr(combinator, "operator", "")
        if operator == "all_true":
            meaning = "All direct child results must support the parent outcome."
        elif operator == "any_true":
            meaning = "At least one direct child result may support the parent outcome."
        else:
            meaning = "Apply this built-in operator to the direct child outputs."
        return [
            "- type: built-in",
            f"- operator: `{operator}`",
            f"- short_circuit: {str(short_circuit).lower()}",
            f"- meaning: {meaning}",
            "- scope: combine only this component's direct child outputs; do not aggregate grandchildren except through their parent child result.",
        ]

    if kind == "prompt":
        instructions = getattr(combinator, "instructions", None) or "Use the component instructions to combine direct child outputs."
        return [
            "- type: prompt-defined",
            f"- short_circuit: {str(short_circuit).lower()}",
            "- scope: combine only this component's direct child outputs; do not aggregate grandchildren except through their parent child result.",
            "- instructions:",
            _indent(instructions, "  "),
        ]

    if hasattr(combinator, "model_dump"):
        return ["```json", json.dumps(combinator.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True), "```"]
    return [str(combinator)]


def _indent(text: str, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}" if line else prefix.rstrip() for line in text.splitlines())