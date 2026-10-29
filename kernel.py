from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import ValidationError

from kernel_sdk import (
    ComponentEvidence,
    FlatPromptConfig,
    Kernel,
    KernelConfigSpec,
    KernelDispatcher,
    KernelExecutionError,
    KernelRuntimeContext,
    KernelSubtreeResult,
    KernelTemplateCatalog,
    KernelTemplateRenderingError,
    KernelValidationError,
    ResolvedComponentNode,
    TemplateSelection,
    read_package_text,
    render_package_template,
    render_template_body,
    resolve_bindings,
)

from .rendering import render_component_tree_markdown

_TEMPLATE_PACKAGE = __package__ or "kernels.flat_prompt"
_SYNTHETIC_COMPONENT_TREE = "component_tree"
# Host-injected `kernel.config` key carrying resolved Global System Instructions text. Hosts set it at
# execution time (it is never author-saved), so it is read from the raw config dict rather than
# `FlatPromptConfig`, which does not declare it.
_SYSTEM_INSTRUCTIONS_CONFIG_KEY = "system_instructions"

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "combined_documents": {"type": "string"},
        "tool_results": {"type": "object"},
        "parser_results": {"type": "object"},
        "routine_inputs": {"type": "object"},
        "previous_soap_notes": {"type": "object"},
        "policy_instructions": {"type": "string"},
        "additional_evidence": {"type": "string"},
        "component_tree": {
            "type": "string",
            "description": "Synthetic — rendered subtree markdown",
        },
    },
    "required": ["combined_documents", "component_tree"],
}

_USER_TEMPLATE_CATALOG = KernelTemplateCatalog(
    default="templates/user.jinja2",
    library=[],
)

_SYSTEM_TEMPLATE_CATALOG = KernelTemplateCatalog(
    default="templates/system.jinja2",
    library=[],
)

# JSONSchema authors' `kernel.config` is validated against. A flat_prompt component carries the
# selections for both render modes: `user_template`/`system_template` drive a boundary/root LLM call,
# while `inline_evaluation_body` is folded into the parent's prompt when the node realizes inline.
_TEMPLATE_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "library_id": {"type": "string"},
        "pinned_kernel_version": {"type": "string"},
        "override_body": {"type": "string"},
    },
    "additionalProperties": False,
}
_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "user_template": _TEMPLATE_SELECTION_SCHEMA,
        "system_template": _TEMPLATE_SELECTION_SCHEMA,
        "inline_evaluation_body": {"type": "string"},
    },
    "additionalProperties": False,
}

_CONFIG_SPEC = KernelConfigSpec(
    config_schema=_CONFIG_SCHEMA,
    input_schema=_INPUT_SCHEMA,
    output_schema={"$ref": "kernel_sdk_types#/KernelSubtreeResult"},
    user_template=_USER_TEMPLATE_CATALOG,
    system_template=_SYSTEM_TEMPLATE_CATALOG,
)


def _node_id(node: ResolvedComponentNode) -> str:
    """Return the placement-scoped node id, falling back to the component id."""
    if node.local_metadata and node.local_metadata.node_id:
        return node.local_metadata.node_id
    return node.component_id


def _kernel_kind(node: ResolvedComponentNode, default: str) -> str:
    """Return the kernel kind declared by the node, falling back to `default`."""
    return node.kernel.kind if node.kernel else default


def _evidence_from_payload(payload: Any) -> ComponentEvidence:
    """
    Coerce a raw LLM evidence record into a `ComponentEvidence`.

    Parameters
    ----------
    payload: Any
        Raw evidence record (mapping or already-validated model)
    """
    if not isinstance(payload, Mapping):
        return ComponentEvidence.model_validate(payload)
    evidence = dict(payload)
    if "line_reference" not in evidence and "lineReference" in evidence:
        evidence["line_reference"] = evidence.pop("lineReference")
    if evidence.get("line_reference") is not None and not isinstance(
        evidence["line_reference"], str
    ):
        line_reference = evidence["line_reference"]
        evidence["line_reference"] = (
            ",".join(str(item) for item in line_reference)
            if isinstance(line_reference, list)
            else str(line_reference)
        )
    return ComponentEvidence.model_validate(evidence)


_MISSING_COMPONENT_ID = "<missing component_id>"
# Keys of the result wrapper itself; anything else at the top level is a flattened output field.
_WRAPPER_KEYS = frozenset(
    {"component_id", "output", "reasoning", "evidence", "sub_results", "children"}
)
_OUTPUT_KEY_ALIASES = ("result", "outputs")


def _output_from_payload(payload: Mapping[str, Any]) -> Any:
    """
    Return the component output from `payload`, tolerating shapes some models return.

    GPT-6 Sol/Luna sometimes JSON-encode `output` as a string, name it `result`/`outputs`, or
    flatten its fields next to the wrapper keys. Anything else is returned as-is for the
    caller to reject.

    Parameters
    ----------
    payload: Mapping[str, Any]
        LLM JSON payload for a single component
    """
    output = payload.get("output")
    if isinstance(output, str):
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return output
    if output is not None:
        return output
    for alias in _OUTPUT_KEY_ALIASES:
        if isinstance(payload.get(alias), dict):
            return payload[alias]
    flattened = {key: value for key, value in payload.items() if key not in _WRAPPER_KEYS}
    return flattened or None


def _result_from_payload(
    payload: Mapping[str, Any], *, kernel_kind: str
) -> KernelSubtreeResult:
    """
    Recursively map a raw LLM JSON payload into a `KernelSubtreeResult`.

    Parameters
    ----------
    payload: Mapping[str, Any]
        LLM JSON payload for a single component
    kernel_kind: str
        Kernel kind to record on each constructed result
    """
    # `_align_component_ids` stamps the real id afterwards, so a missing one is not fatal here.
    component_id = payload.get("component_id")
    if not isinstance(component_id, str) or not component_id:
        component_id = _MISSING_COMPONENT_ID

    output = _output_from_payload(payload)
    if not isinstance(output, dict):
        raise KernelValidationError(
            f"LLM result for {component_id} is missing object output"
        )

    raw_evidence = payload.get("evidence") or []
    if not isinstance(raw_evidence, list):
        raise KernelValidationError(
            f"LLM result evidence for {component_id} must be a list"
        )

    raw_children = payload.get("sub_results") or payload.get("children") or []
    if not isinstance(raw_children, list):
        raise KernelValidationError(
            f"LLM result children for {component_id} must be a list"
        )

    try:
        evidence = [_evidence_from_payload(item) for item in raw_evidence]
    except ValidationError as exc:
        raise KernelValidationError(
            f"LLM result evidence for {component_id} is invalid"
        ) from exc

    children: list[KernelSubtreeResult] = []
    for child in raw_children:
        if not isinstance(child, Mapping):
            raise KernelValidationError(
                f"LLM result child for {component_id} must be an object"
            )
        children.append(_result_from_payload(child, kernel_kind=kernel_kind))

    return KernelSubtreeResult(
        component_id=component_id,
        kernel_kind=kernel_kind,
        output=output,
        reasoning=payload.get("reasoning")
        if isinstance(payload.get("reasoning"), str)
        else None,
        evidence=evidence,
        children=children,
    )


def _collect_component_ids(node: ResolvedComponentNode) -> set[str]:
    """Return the set of `component_id`s reachable from `node` (inclusive)."""
    return {
        node.component_id,
        *(
            component_id
            for child in node.children
            for component_id in _collect_component_ids(child)
        ),
    }


def _align_component_ids(
    node: ResolvedComponentNode, result: KernelSubtreeResult
) -> None:
    """
    Stamp `node`'s known component ids onto `result` instead of trusting the LLM to echo them.

    Some models (GPT-6 Sol/Luna) return the right structure with dropped or placeholder ids, which
    would otherwise fail coverage validation. Children whose ids already match are paired by id;
    otherwise they are paired by position, since the prompt requires rendered order. A child-count
    mismatch is left untouched so `_validate_component_coverage` still reports it.

    Parameters
    ----------
    node: ResolvedComponentNode
        Resolved subtree the LLM was asked to evaluate
    result: KernelSubtreeResult
        Result mapped from the LLM payload, updated in place
    """
    result.component_id = node.component_id
    if len(result.children) != len(node.children):
        return
    children_by_id = {child.component_id: child for child in node.children}
    returned_ids = [child.component_id for child in result.children]
    if sorted(returned_ids) == sorted(children_by_id):
        pairs = [(children_by_id[child.component_id], child) for child in result.children]
    else:
        pairs = list(zip(node.children, result.children))
    for child_node, child_result in pairs:
        _align_component_ids(child_node, child_result)


def _validate_component_coverage(
    node: ResolvedComponentNode, result: KernelSubtreeResult
) -> None:
    """Raise `KernelValidationError` when `result` omits or invents component ids relative to `node`."""

    def collect(result_node: KernelSubtreeResult) -> set[str]:
        return {
            result_node.component_id,
            *(
                component_id
                for child in result_node.children
                for component_id in collect(child)
            ),
        }

    expected = _collect_component_ids(node)
    actual = collect(result)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise KernelValidationError(
            f"LLM result omitted component ids: {sorted(missing)}"
        )
    if extra:
        raise KernelValidationError(
            f"LLM result included unknown component ids: {sorted(extra)}"
        )


def _resolve_llm_provider(runtime: Any, node: ResolvedComponentNode) -> Any:
    """
    Locate the per-node LLM provider Eagle wires onto the dispatcher.

    Eagle's `_ExternalKernelDispatcher` exposes a `llm_provider(node)` factory that returns a
    `kernel_sdk.llm.LLMProvider`-shaped object (see Eagle task #6). `runtime_sources` stays
    purely the declarative host-bindings channel.

    Parameters
    ----------
    runtime: Any
        Dispatcher passed into `Kernel.execute`
    node: ResolvedComponentNode
        Node being executed (forwarded so Eagle can scope the provider to the right system version)
    """
    factory = getattr(runtime, "llm_provider", None)
    if factory is None:
        raise KernelExecutionError(
            "FlatPromptKernel requires an LLM provider factory on the runtime dispatcher (runtime.llm_provider(node))"
        )
    provider = factory(node) if callable(factory) else factory
    if provider is None:
        raise KernelExecutionError(
            f"FlatPromptKernel runtime.llm_provider(node) returned None for component '{node.component_id}'"
        )
    return provider


def _flat_prompt_config(node: ResolvedComponentNode) -> FlatPromptConfig:
    """Parse the component-owned `kernel.config` into a `FlatPromptConfig` (tolerating absent config)."""
    config = node.kernel.config if node.kernel else None
    if config is None:
        return FlatPromptConfig()
    try:
        return FlatPromptConfig.model_validate(config)
    except ValidationError as exc:
        raise KernelValidationError(
            f"flat_prompt config is invalid for component '{node.component_id}'"
        ) from exc


def _system_instructions(node: ResolvedComponentNode) -> str | None:
    """Return host-injected system instructions from the raw `kernel.config`, if any."""
    config = node.kernel.config if node.kernel else None
    if not isinstance(config, Mapping):
        return None
    value = config.get(_SYSTEM_INSTRUCTIONS_CONFIG_KEY)
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _prepend_system_instructions(system_prompt: str, system_instructions: str | None) -> str:
    """Place shared system instructions ahead of the component's own system prompt."""
    if system_instructions is None:
        return system_prompt
    return f"# Instructions\n{system_instructions.strip()}\n\n{system_prompt}"


def _resolve_library_file(catalog: KernelTemplateCatalog, library_id: str) -> str:
    """Return the packaged file path for a declared `library_id`, raising when unknown."""
    match = next(
        (entry for entry in catalog.library if entry.id == library_id), None
    )
    if match is None:
        raise KernelTemplateRenderingError(
            f"Library template '{library_id}' is not declared by this kernel"
        )
    return match.file


def _template_source_body(
    selection: TemplateSelection | None,
    catalog: KernelTemplateCatalog,
) -> str:
    """Return the Jinja source body selected for a template role."""
    if selection and selection.override_body:
        return selection.override_body
    if selection and selection.library_id:
        return read_package_text(
            _TEMPLATE_PACKAGE,
            _resolve_library_file(catalog, selection.library_id),
        )
    return read_package_text(_TEMPLATE_PACKAGE, catalog.default)


def _template_references_binding(
    selection: TemplateSelection | None,
    catalog: KernelTemplateCatalog,
    binding_name: str,
) -> bool:
    """Return True when a selected template source references the input `binding_name` in Jinja."""
    return bool(
        re.search(
            r"\{[{%][^}]*\b" + re.escape(binding_name) + r"\b",
            _template_source_body(selection, catalog),
        )
    )


def _ensure_required_prompt_context(
    *,
    system_prompt: str,
    user_prompt: str,
    config: FlatPromptConfig,
    bindings: dict[str, Any],
) -> tuple[str, str]:
    """Append required flat_prompt context when author template bodies omit kernel-required bindings."""
    references_component_tree = _template_references_binding(
        config.system_template,
        _SYSTEM_TEMPLATE_CATALOG,
        _SYNTHETIC_COMPONENT_TREE,
    ) or _template_references_binding(
        config.user_template,
        _USER_TEMPLATE_CATALOG,
        _SYNTHETIC_COMPONENT_TREE,
    )
    references_documents = _template_references_binding(
        config.system_template,
        _SYSTEM_TEMPLATE_CATALOG,
        "combined_documents",
    ) or _template_references_binding(
        config.user_template,
        _USER_TEMPLATE_CATALOG,
        "combined_documents",
    )

    if not references_component_tree and bindings.get(_SYNTHETIC_COMPONENT_TREE):
        system_prompt = (
            f"{system_prompt.rstrip()}\n\n"
            "# Component Tree To Evaluate\n"
            f"{bindings[_SYNTHETIC_COMPONENT_TREE]}"
        )
    if not references_documents and bindings.get("combined_documents"):
        user_prompt = (
            f"{user_prompt.rstrip()}\n\n"
            "# Patient Documents\n"
            f"{bindings['combined_documents']}"
        )
    return system_prompt, user_prompt


def _render_role_template(
    selection: TemplateSelection | None,
    catalog: KernelTemplateCatalog,
    bindings: dict[str, Any],
) -> str:
    """
    Render a single role's prompt according to its config `TemplateSelection`.

    Selection precedence: explicit inline body → declared library entry → kernel-shipped default.
    Strict-mode is enforced by the SDK renderers; missing names raise `PromptRenderingError`.

    Parameters
    ----------
    selection: TemplateSelection | None
        Config-supplied selection (`None` falls through to the default file)
    catalog: KernelTemplateCatalog
        Default + library the kernel publishes for this role
    bindings: dict[str, Any]
        Resolved binding dict; each input is exposed to Jinja by its own name (e.g. `{{ name }}`)
    """
    params = dict(bindings)
    if selection and selection.override_body:
        return render_template_body(selection.override_body, params)
    if selection and selection.library_id:
        return render_package_template(
            _TEMPLATE_PACKAGE,
            _resolve_library_file(catalog, selection.library_id),
            params,
        )
    return render_package_template(_TEMPLATE_PACKAGE, catalog.default, params)


class FlatPromptKernel(Kernel):
    """
    Single-LLM-call kernel that evaluates a resolved component subtree.

    Templates and the inline evaluation body are component-owned, carried in `kernel.config`
    (`FlatPromptConfig`). The kernel resolves the node's `input_bindings`, injects the synthetic
    `bindings.component_tree` view of the subtree, renders the selected system + user templates
    under strict-undefined Jinja, calls the LLM in JSON mode, and recursively maps the response
    into a `KernelSubtreeResult` covering every component in the subtree.
    """

    kind: ClassVar[str] = "flat_prompt"
    config_spec: ClassVar[KernelConfigSpec | None] = _CONFIG_SPEC

    async def execute(
        self,
        node: ResolvedComponentNode,
        runtime_context: KernelRuntimeContext,
        runtime_sources: Mapping[str, Any],
        runtime: KernelDispatcher,
    ) -> KernelSubtreeResult:
        llm_provider = _resolve_llm_provider(runtime, node)
        config = _flat_prompt_config(node)
        declared_ports = list((_INPUT_SCHEMA.get("properties") or {}).keys())
        bindings: dict[str, Any] = dict.fromkeys(declared_ports)
        bindings.update(
            resolve_bindings(
                node.local_metadata.input_bindings if node.local_metadata else None,
                runtime_sources,
                runtime_context.node_results,
                runtime_context.call_site_results,
            )
        )
        if _SYNTHETIC_COMPONENT_TREE in declared_ports:
            bindings[_SYNTHETIC_COMPONENT_TREE] = render_component_tree_markdown(node)

        system_prompt = _render_role_template(
            config.system_template,
            _SYSTEM_TEMPLATE_CATALOG,
            bindings,
        )
        user_prompt = _render_role_template(
            config.user_template,
            _USER_TEMPLATE_CATALOG,
            bindings,
        )
        system_prompt, user_prompt = _ensure_required_prompt_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=config,
            bindings=bindings,
        )
        system_prompt = _prepend_system_instructions(system_prompt, _system_instructions(node))

        completion = await llm_provider.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            request_context={
                "component_id": node.component_id,
                "node_id": _node_id(node),
            },
        )

        try:
            payload = json.loads(completion.content)
        except json.JSONDecodeError as exc:
            raise KernelValidationError(
                "FlatPromptKernel received non-JSON LLM content"
            ) from exc
        if not isinstance(payload, Mapping):
            raise KernelValidationError(
                "FlatPromptKernel expected a JSON object result"
            )

        result = _result_from_payload(
            payload, kernel_kind=_kernel_kind(node, self.kind)
        )
        result.node_id = _node_id(node)
        result.control = (
            {"decision": result.output.get("decision")}
            if "decision" in result.output
            else {}
        )
        _align_component_ids(node, result)
        _validate_component_coverage(node, result)
        runtime_context.store_node_result(result.node_id or result.component_id, result)
        return result