from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from kernel_sdk import (
    ChildRealization,
    Kernel,
    KernelDispatcher,
    KernelRuntimeContext,
    KernelSubtreeResult,
    ResolvedComponentNode,
    evaluate_condition,
)


def _node_id(node: ResolvedComponentNode) -> str:
    if node.local_metadata and node.local_metadata.node_id:
        return node.local_metadata.node_id
    return node.component_id


def _kernel_kind(node: ResolvedComponentNode, default: str) -> str:
    return node.kernel.kind if node.kernel else default


class SequenceKernel(Kernel):
    kind: ClassVar[str] = "sequence"
    fallback_child_realization: ClassVar[ChildRealization] = "boundary"

    async def execute(
        self,
        node: ResolvedComponentNode,
        runtime_context: KernelRuntimeContext,
        runtime_sources: Mapping[str, Any],
        runtime: KernelDispatcher,
    ) -> KernelSubtreeResult:
        child_results: list[KernelSubtreeResult] = []
        encountered_failure = False

        for child in node.children:
            child_node_id = _node_id(child)
            always_run = bool(child.local_metadata and child.local_metadata.always_run)

            if encountered_failure and not always_run:
                child_result = KernelSubtreeResult(
                    component_id=child.component_id,
                    node_id=child_node_id,
                    kernel_kind=_kernel_kind(child, "skipped"),
                    status="skipped",
                )
                runtime_context.store_node_result(child_node_id, child_result)
                child_results.append(child_result)
                continue

            if child.local_metadata and not evaluate_condition(
                child.local_metadata.run_if,
                runtime_sources,
                runtime_context.node_results,
            ):
                child_result = KernelSubtreeResult(
                    component_id=child.component_id,
                    node_id=child_node_id,
                    kernel_kind=_kernel_kind(child, "skipped"),
                    status="skipped",
                )
                runtime_context.store_node_result(child_node_id, child_result)
                child_results.append(child_result)
                continue

            realization = self.get_child_realization(node, child)
            try:
                if realization == "inline":
                    child_result = KernelSubtreeResult(
                        component_id=child.component_id,
                        node_id=child_node_id,
                        kernel_kind=_kernel_kind(child, "inline"),
                        status="skipped",
                        control={"inline": True},
                    )
                elif realization == "reference":
                    child_result = await runtime.execute_reference_node(child, runtime_context, runtime_sources)
                else:
                    child_result = await runtime.execute_node(child, runtime_context, runtime_sources)
            except Exception as exc:
                child_result = KernelSubtreeResult(
                    component_id=child.component_id,
                    node_id=child_node_id,
                    kernel_kind=_kernel_kind(child, "unknown"),
                    status="failed",
                    control={"error": str(exc)},
                )

            runtime_context.store_node_result(child_node_id, child_result)
            child_results.append(child_result)
            if child_result.status == "failed" and not always_run:
                encountered_failure = True

        result = KernelSubtreeResult(
            component_id=node.component_id,
            node_id=_node_id(node),
            kernel_kind=_kernel_kind(node, self.kind),
            status="failed" if encountered_failure else "completed",
            children=child_results,
            control={"encountered_failure": encountered_failure},
        )
        runtime_context.store_node_result(result.node_id or result.component_id, result)
        return result
