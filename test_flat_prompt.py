from __future__ import annotations

import json

import pytest

from kernel_sdk import (
    CallSitePlacement,
    InputBinding,
    JsonCompletion,
    KernelSpec,
    LocalNodeMetadata,
    PromptRenderingError,
    ResolvedComponentNode,
    TemplateSelection,
)
from kernel_sdk.testing import (
    FakeKernelDispatcher,
    FakeLLMProvider,
    make_runtime_context,
)
from kernels.flat_prompt import FlatPromptKernel


def _dispatcher(provider: FakeLLMProvider | None = None) -> FakeKernelDispatcher:
    """Build a `FakeKernelDispatcher` with Eagle's `llm_provider(node)` factory shim attached."""
    dispatcher = FakeKernelDispatcher()
    if provider is not None:
        dispatcher.llm_provider = lambda node: provider
    return dispatcher


def _root(
    *,
    placements: list[CallSitePlacement] | None = None,
    children: list[ResolvedComponentNode] | None = None,
) -> ResolvedComponentNode:
    return ResolvedComponentNode(
        component_id="component.root",
        major_version=1,
        minor_version=0,
        save_version=0,
        name="Root",
        purpose="Root purpose",
        output_schema={
            "type": "object",
            "properties": {"decision": {"type": "string"}},
        },
        kernel=KernelSpec(kind="flat_prompt"),
        local_metadata=LocalNodeMetadata(
            node_id="node.root", call_site_placements=placements or []
        ),
        children=children or [],
    )


def _main_placement(
    *,
    user_template: TemplateSelection | None = None,
    system_template: TemplateSelection | None = None,
    bindings: list[InputBinding] | None = None,
) -> CallSitePlacement:
    return CallSitePlacement(
        call_site_id="main",
        user_template=user_template or TemplateSelection(library_id="standard_v1"),
        system_template=system_template or TemplateSelection(library_id="clinical_v1"),
        bindings=bindings or [],
    )


@pytest.mark.asyncio
async def test_flat_prompt_declares_main_call_site_with_libraries() -> None:
    spec = FlatPromptKernel.call_sites[0]
    assert spec.id == "main"
    assert {entry.id for entry in spec.user_template.library} == {
        "standard_v1",
        "chain_of_thought_v1",
    }
    assert {entry.id for entry in spec.system_template.library} == {
        "clinical_v1",
        "strict_json_v1",
    }


@pytest.mark.asyncio
async def test_flat_prompt_renders_clinical_system_and_standard_user_via_libraries() -> (
    None
):
    child = ResolvedComponentNode(
        component_id="component.child",
        major_version=1,
        minor_version=0,
        save_version=0,
        name="Child",
        purpose="Child purpose",
        output_schema={"type": "object", "properties": {"status": {"type": "string"}}},
    )
    placements = [
        _main_placement(
            bindings=[
                InputBinding(
                    target_name="combined_documents",
                    source_kind="host_binding",
                    source_path="combined_documents",
                ),
                InputBinding(
                    target_name="tool_results",
                    source_kind="host_binding",
                    source_path="tool_results",
                ),
            ]
        )
    ]
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
                        "evidence": [
                            {"text": "Child evidence", "lineReference": "43-44"}
                        ],
                    }
                ],
            }
        )
    )
    provider = FakeLLMProvider([completion])
    context = make_runtime_context()
    runtime_sources = {
        "host_bindings": {
            "combined_documents": "patient docs",
            "tool_results": {"lab": "normal"},
        },
    }

    result = await FlatPromptKernel().execute(
        _root(placements=placements, children=[child]),
        context,
        runtime_sources,
        _dispatcher(provider),
    )

    assert result.component_id == "component.root"
    assert result.node_id == "node.root"
    assert result.output == {"decision": "APPROVED"}
    assert result.control == {"decision": "APPROVED"}
    assert result.evidence[0].line_reference == "42"
    assert result.children[0].evidence[0].line_reference == "43-44"
    assert (
        "expert clinician at an insurance company" in provider.requests[0].system_prompt
    )
    assert "component.root" in provider.requests[0].system_prompt
    assert "patient docs" in provider.requests[0].user_prompt
    assert "Tool Results" in provider.requests[0].user_prompt
    assert context.call_site_results["main"] == {"decision": "APPROVED"}


@pytest.mark.asyncio
async def test_flat_prompt_appends_required_context_when_overrides_omit_bindings() -> None:
    placements = [
        _main_placement(
            user_template=TemplateSelection(override_body="Author guidance only."),
            system_template=TemplateSelection(override_body="System preamble only."),
            bindings=[
                InputBinding(
                    target_name="combined_documents",
                    source_kind="static",
                    static_value="doc body",
                )
            ],
        )
    ]
    provider = FakeLLMProvider(
        [
            JsonCompletion(
                content=json.dumps({"component_id": "component.root", "output": {}})
            )
        ]
    )

    await FlatPromptKernel().execute(
        _root(placements=placements),
        make_runtime_context(),
        {"host_bindings": {}},
        _dispatcher(provider),
    )

    assert "Author guidance only." in provider.requests[0].user_prompt
    assert "# Patient Documents\ndoc body" in provider.requests[0].user_prompt
    assert "System preamble only." in provider.requests[0].system_prompt
    assert "# Component Tree To Evaluate" in provider.requests[0].system_prompt
    assert "component.root" in provider.requests[0].system_prompt


@pytest.mark.asyncio
async def test_flat_prompt_includes_inline_child_user_body_in_component_tree() -> None:
    child = ResolvedComponentNode(
        component_id="component.child",
        major_version=1,
        minor_version=0,
        save_version=0,
        name="Child",
        purpose="Child purpose",
        kernel=KernelSpec(kind="flat_prompt"),
        local_metadata=LocalNodeMetadata(
            node_id="node.child",
            call_site_placements=[
                _main_placement(
                    user_template=TemplateSelection(
                        override_body="Child-specific clinical criteria."
                    ),
                    system_template=TemplateSelection(library_id="strict_json_v1"),
                )
            ],
        ),
        output_schema={"type": "object", "properties": {"status": {"type": "string"}}},
    )
    placements = [
        _main_placement(
            bindings=[
                InputBinding(
                    target_name="combined_documents",
                    source_kind="static",
                    static_value="doc body",
                )
            ]
        )
    ]
    provider = FakeLLMProvider(
        [
            JsonCompletion(
                content=json.dumps(
                    {
                        "component_id": "component.root",
                        "output": {},
                        "sub_results": [{"component_id": "component.child", "output": {"status": "MET"}}],
                    }
                )
            )
        ]
    )

    await FlatPromptKernel().execute(
        _root(placements=placements, children=[child]),
        make_runtime_context(),
        {"host_bindings": {}},
        _dispatcher(provider),
    )

    assert "Child-specific clinical criteria." in provider.requests[0].system_prompt
    assert "strict_json_v1" not in provider.requests[0].system_prompt


@pytest.mark.asyncio
async def test_flat_prompt_default_templates_render_when_placement_omits_library_id() -> (
    None
):
    placements = [
        CallSitePlacement(
            call_site_id="main",
            user_template=TemplateSelection(),
            system_template=TemplateSelection(),
            bindings=[
                InputBinding(
                    target_name="combined_documents",
                    source_kind="host_binding",
                    source_path="combined_documents",
                ),
            ],
        )
    ]
    provider = FakeLLMProvider(
        [
            JsonCompletion(
                content=json.dumps({"component_id": "component.root", "output": {}})
            )
        ]
    )
    runtime_sources = {"host_bindings": {"combined_documents": "doc body"}}

    await FlatPromptKernel().execute(
        _root(placements=placements),
        make_runtime_context(),
        runtime_sources,
        _dispatcher(provider),
    )

    assert "expert reasoning engine" in provider.requests[0].system_prompt
    assert "doc body" in provider.requests[0].user_prompt


@pytest.mark.asyncio
async def test_flat_prompt_chain_of_thought_user_library_renders() -> None:
    placements = [
        _main_placement(
            user_template=TemplateSelection(library_id="chain_of_thought_v1"),
            bindings=[
                InputBinding(
                    target_name="combined_documents",
                    source_kind="static",
                    static_value="doc body",
                )
            ],
        )
    ]
    provider = FakeLLMProvider(
        [
            JsonCompletion(
                content=json.dumps({"component_id": "component.root", "output": {}})
            )
        ]
    )
    await FlatPromptKernel().execute(
        _root(placements=placements),
        make_runtime_context(),
        {"host_bindings": {}},
        _dispatcher(provider),
    )
    assert "step by step" in provider.requests[0].user_prompt


@pytest.mark.asyncio
async def test_flat_prompt_override_body_replaces_library() -> None:
    placements = [
        _main_placement(
            system_template=TemplateSelection(
                override_body="OVERRIDE_SYSTEM_PROMPT for {{ bindings.component_tree }}"
            ),
            bindings=[
                InputBinding(
                    target_name="combined_documents",
                    source_kind="static",
                    static_value="doc body",
                )
            ],
        )
    ]
    provider = FakeLLMProvider(
        [
            JsonCompletion(
                content=json.dumps({"component_id": "component.root", "output": {}})
            )
        ]
    )
    await FlatPromptKernel().execute(
        _root(placements=placements),
        make_runtime_context(),
        {"host_bindings": {}},
        _dispatcher(provider),
    )
    assert "OVERRIDE_SYSTEM_PROMPT" in provider.requests[0].system_prompt
    assert "expert clinician" not in provider.requests[0].system_prompt


@pytest.mark.asyncio
async def test_flat_prompt_strict_override_raises_on_undeclared_name() -> None:
    placements = [
        _main_placement(
            user_template=TemplateSelection(
                override_body="{{ bindings.nonexistent_port }}"
            ),
            bindings=[
                InputBinding(
                    target_name="combined_documents",
                    source_kind="static",
                    static_value="doc body",
                )
            ],
        )
    ]
    with pytest.raises(PromptRenderingError):
        await FlatPromptKernel().execute(
            _root(placements=placements),
            make_runtime_context(),
            {"host_bindings": {}},
            _dispatcher(FakeLLMProvider()),
        )


@pytest.mark.asyncio
async def test_flat_prompt_validates_component_coverage() -> None:
    child = ResolvedComponentNode(
        component_id="component.child",
        major_version=1,
        minor_version=0,
        save_version=0,
        name="Child",
        purpose="Child purpose",
    )
    provider = FakeLLMProvider(
        [
            JsonCompletion(
                content=json.dumps({"component_id": "component.root", "output": {}})
            )
        ]
    )
    with pytest.raises(Exception, match="omitted component ids"):
        await FlatPromptKernel().execute(
            _root(children=[child]),
            make_runtime_context(),
            {"host_bindings": {"combined_documents": "doc"}},
            _dispatcher(provider),
        )


@pytest.mark.asyncio
async def test_flat_prompt_prepends_host_system_instructions_to_system_prompt() -> None:
    config = _config(system_template={"override_body": "Component system prompt."})
    config["system_instructions"] = "Cite evidence by line reference."
    provider = FakeLLMProvider(
        [
            JsonCompletion(
                content=json.dumps({"component_id": "component.root", "output": {}})
            )
        ]
    )

    await FlatPromptKernel().execute(
        _root(config=config),
        make_runtime_context(),
        {"host_bindings": {}},
        _dispatcher(provider),
    )

    system_prompt = provider.requests[0].system_prompt
    assert system_prompt.startswith("# Instructions\nCite evidence by line reference.\n\n")
    assert system_prompt.index("Cite evidence by line reference.") < system_prompt.index(
        "Component system prompt."
    )


@pytest.mark.asyncio
async def test_flat_prompt_system_prompt_unchanged_without_system_instructions() -> None:
    config = _config(system_template={"override_body": "Component system prompt."})
    provider = FakeLLMProvider(
        [
            JsonCompletion(
                content=json.dumps({"component_id": "component.root", "output": {}})
            )
        ]
    )

    await FlatPromptKernel().execute(
        _root(config=config),
        make_runtime_context(),
        {"host_bindings": {}},
        _dispatcher(provider),
    )

    system_prompt = provider.requests[0].system_prompt
    assert system_prompt.startswith("Component system prompt.")
