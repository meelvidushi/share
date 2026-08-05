# Component Tree Kernels

Initial external kernels bundle for Context Mesh component-tree execution.

SDK dependency:

```text
distylai-toolkit-component-tree-kernel-sdk==0.1.1
```

## Kernels

| kind | module | class |
|---|---|---|
| `sequence` | `kernels.sequence.kernel` | `SequenceKernel` |
| `flat_prompt` | `kernels.flat_prompt.kernel` | `FlatPromptKernel` |
| `mock` | `kernels.mock.kernel` | `MockKernel` |

`mock` returns deterministic neutral results for every component in its subtree by generating placeholder outputs from each component's `output_schema`. It does not call an LLM.

## Runtime boundary

Kernels return nested neutral `KernelSubtreeResult` values. Host runtimes such as Eagle are responsible for:

- template context assembly
- PHI/tokenization boundaries
- LLM provider implementation
- execution-step persistence
- prior-auth / ELV / Coffey normalization

`flat_prompt` ignores any legacy GSI flag. It always renders package-local templates:

```text
kernels/flat_prompt/templates/system.jinja2
kernels/flat_prompt/templates/user.jinja2
```

Expected `flat_prompt` template context keys are supplied by the host runtime. Common Eagle keys include:

- `patient_documents`
- `combined_documents`
- `tool_results`
- `parser_results`
- `additional_evidence`
- `use_structured_data`
- `is_day_by_day_execution`
- `segmented_date`
- `previous_soap_notes`
- `routine_inputs`
- `policy_instructions`
- `system_instructions`
- `context_definitions`

## Validation

```bash
python -m pytest tests
```
