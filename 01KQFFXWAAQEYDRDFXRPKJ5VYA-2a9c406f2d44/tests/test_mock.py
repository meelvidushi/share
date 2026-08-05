from __future__ import annotations

import pytest

from kernel_sdk import KernelSpec, LocalNodeMetadata, ResolvedComponentNode
from kernel_sdk.testing import FakeKernelDispatcher, make_runtime_context
from kernels.mock import MockKernel


@pytest.mark.asyncio
async def test_mock_kernel_returns_outputs_for_every_component() -> None:
    child = ResolvedComponentNode(
        component_id="component.child",
        major_version=1,
        minor_version=0,
        save_version=0,
        name="Child",
        purpose="Child purpose",
        output_schema={"status": "MET | NOT_MET | MISSING_INFORMATION", "notes": "string"},
        local_metadata=LocalNodeMetadata(node_id="node.child"),
    )
    root = ResolvedComponentNode(
        component_id="component.root",
        major_version=1,
        minor_version=0,
        save_version=0,
        name="Root",
        purpose="Root purpose",
        output_schema={"decision": "APPROVED | NEEDS_REVIEW"},
        kernel=KernelSpec(kind="mock"),
        local_metadata=LocalNodeMetadata(node_id="node.root"),
        children=[child],
    )
    context = make_runtime_context()

    result = await MockKernel().execute(root, context, {"payload": {}}, FakeKernelDispatcher())

    assert result.component_id == "component.root"
    assert result.node_id == "node.root"
    assert result.kernel_kind == "mock"
    assert result.output == {"decision": "APPROVED"}
    assert result.control == {"decision": "APPROVED"}
    assert result.children[0].component_id == "component.child"
    assert result.children[0].output == {"status": "MET", "notes": "mock_notes"}
    assert context.node_results["node.root"] == result
    assert context.node_results["node.child"] == result.children[0]


@pytest.mark.asyncio
async def test_mock_kernel_generates_values_from_json_schema() -> None:
    root = ResolvedComponentNode(
        component_id="component.root",
        major_version=1,
        minor_version=0,
        save_version=0,
        name="Root",
        purpose="Root purpose",
        output_schema={
            "type": "object",
            "properties": {
                "decision": {"enum": ["APPROVED", "NEEDS_REVIEW"]},
                "count": {"type": "integer", "minimum": 2},
                "details": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "flags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    },
                },
            },
        },
    )

    result = await MockKernel().execute(root, make_runtime_context(), {"payload": {}}, FakeKernelDispatcher())

    assert result.output == {
        "decision": "APPROVED",
        "count": 2,
        "details": {
            "summary": "mock_summary",
            "flags": ["mock"],
        },
    }
