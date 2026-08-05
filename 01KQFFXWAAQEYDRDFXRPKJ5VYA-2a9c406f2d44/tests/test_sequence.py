from __future__ import annotations

import pytest

from kernel_sdk import KernelSpec, KernelSubtreeResult, LocalNodeMetadata, ResolvedComponentNode
from kernel_sdk.testing import FakeKernelDispatcher, make_runtime_context
from kernels.sequence import SequenceKernel


def _node(component_id: str, *, node_id: str | None = None, always_run: bool = False) -> ResolvedComponentNode:
    return ResolvedComponentNode(
        component_id=component_id,
        major_version=1,
        minor_version=0,
        save_version=0,
        name=component_id,
        purpose="purpose",
        kernel=KernelSpec(kind="flat_prompt"),
        local_metadata=LocalNodeMetadata(node_id=node_id, always_run=always_run),
    )


@pytest.mark.asyncio
async def test_sequence_dispatches_children_and_returns_nested_result() -> None:
    child = _node("component.child", node_id="node.child")
    root = ResolvedComponentNode(
        component_id="component.root",
        major_version=1,
        minor_version=0,
        save_version=0,
        name="Root",
        purpose="purpose",
        kernel=KernelSpec(kind="sequence"),
        local_metadata=LocalNodeMetadata(node_id="node.root"),
        children=[child],
    )
    child_result = KernelSubtreeResult(
        component_id="component.child",
        node_id="node.child",
        kernel_kind="flat_prompt",
        output={"status": "MET"},
    )
    runtime = FakeKernelDispatcher({"component.child": child_result})
    context = make_runtime_context()

    result = await SequenceKernel().execute(root, context, {"payload": {}}, runtime)

    assert result.component_id == "component.root"
    assert result.node_id == "node.root"
    assert result.status == "completed"
    assert result.children == [child_result]
    assert runtime.executed_nodes == ["component.child"]
    assert context.node_results["node.root"] == result


@pytest.mark.asyncio
async def test_sequence_skips_after_failure_until_always_run_child() -> None:
    first = _node("component.first")
    second = _node("component.second")
    third = _node("component.third", always_run=True)
    root = ResolvedComponentNode(
        component_id="component.root",
        major_version=1,
        minor_version=0,
        save_version=0,
        name="Root",
        purpose="purpose",
        kernel=KernelSpec(kind="sequence"),
        children=[first, second, third],
    )
    runtime = FakeKernelDispatcher(
        {
            "component.first": KernelSubtreeResult(
                component_id="component.first",
                kernel_kind="flat_prompt",
                status="failed",
            ),
            "component.third": KernelSubtreeResult(
                component_id="component.third",
                kernel_kind="flat_prompt",
            ),
        }
    )

    result = await SequenceKernel().execute(root, make_runtime_context(), {"payload": {}}, runtime)

    assert result.status == "failed"
    assert [child.component_id for child in result.children] == [
        "component.first",
        "component.second",
        "component.third",
    ]
    assert result.children[1].status == "skipped"
    assert runtime.executed_nodes == ["component.first", "component.third"]
