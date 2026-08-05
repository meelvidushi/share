from __future__ import annotations

import json

import pytest

from kernel_sdk import (
    InputBinding,
    JsonCompletion,
    KernelSpec,
    LocalNodeMetadata,
    ResolvedComponentNode,
)
from kernel_sdk.testing import FakeKernelDispatcher, FakeLLMProvider, make_runtime_context
from kernels.flat_prompt import FlatPromptKernel


class _DispatcherWithLLMProvider(FakeKernelDispatcher):
    """Test double adding the `llm_provider(node)` factory Eagle's dispatcher exposes."""

    def __init__(self, llm_provider: FakeLLMProvider) -> None:
        super().__init__()
        self._llm_provider = llm_provider

    def llm_provider(self, node: ResolvedComponentNode) -> FakeLLMProvider:
        return self._llm_provider


@pytest.mark.asyncio
async def test_flat_prompt_renders_templates_calls_llm_and_returns_result_tree() -> None:
    child = ResolvedComponentNode(
        component_id="component.child",
        major_version=1,
        minor_version=0,
        save_version=0,
        name="Child",
        purpose="Child purpose",
        output_schema={"type": "object", "properties": {"status": {"type": "string"}}},
    )
    root = ResolvedComponentNode(
        component_id="component.root",
        major_version=1,
        minor_version=0,
        save_version=0,
        name="Root",
        purpose="Root purpose",
        output_schema={"type": "object", "properties": {"decision": {"type": "string"}}},
        kernel=KernelSpec(kind="flat_prompt"),
        local_metadata=LocalNodeMetadata(
            node_id="node.root",
            input_bindings=[
                InputBinding(
                    target_name="combined_documents",
                    source_kind="static",
                    static_value="combined docs",
                )
            ],
        ),
        children=[child],
    )
    completion = JsonCompletion(
        content=json.dumps(
            {
                "component_id": "component.root",
                "output": {"decision": "APPROVED"},
                "reasoning": "Root reasoning",
                "evidence": [{"text": "Root evidence", "line_reference": 42}],
                "sub_results": [
                    {
                        "component_id": "component.child",
                        "output": {"status": "MET"},
                        "reasoning": "Child reasoning",
                        "evidence": [{"text": "Child evidence", "lineReference": "43-44"}],
                    }
                ],
            }
        )
    )
    provider = FakeLLMProvider([completion])
    context = make_runtime_context()
    runtime = _DispatcherWithLLMProvider(provider)

    result = await FlatPromptKernel().execute(root, context, {"payload": {}}, runtime)

    assert result.component_id == "component.root"
    assert result.node_id == "node.root"
    assert result.output == {"decision": "APPROVED"}
    assert result.control == {"decision": "APPROVED"}
    assert result.evidence[0].line_reference == "42"
    assert result.children[0].component_id == "component.child"
    assert result.children[0].evidence[0].line_reference == "43-44"
    assert provider.requests[0].request_context["component_id"] == "component.root"
    assert "component.root" in provider.requests[0].system_prompt
    assert "combined docs" in provider.requests[0].user_prompt


@pytest.mark.asyncio
async def test_flat_prompt_validates_component_coverage() -> None:
    root = ResolvedComponentNode(
        component_id="component.root",
        major_version=1,
        minor_version=0,
        save_version=0,
        name="Root",
        purpose="Root purpose",
        kernel=KernelSpec(kind="flat_prompt"),
        local_metadata=LocalNodeMetadata(
            input_bindings=[
                InputBinding(
                    target_name="combined_documents",
                    source_kind="static",
                    static_value="combined docs",
                )
            ],
        ),
        children=[
            ResolvedComponentNode(
                component_id="component.child",
                major_version=1,
                minor_version=0,
                save_version=0,
                name="Child",
                purpose="Child purpose",
            )
        ],
    )
    provider = FakeLLMProvider([
        JsonCompletion(content=json.dumps({"component_id": "component.root", "output": {}}))
    ])

    with pytest.raises(Exception, match="omitted component ids"):
        await FlatPromptKernel().execute(
            root,
            make_runtime_context(),
            {"payload": {}},
            _DispatcherWithLLMProvider(provider),
        )
