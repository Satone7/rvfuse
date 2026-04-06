# Selective DFG Generation: Hot-BB-Driven Filtering

## Problem

The DFG tool currently generates Data Flow Graphs for **all** basic blocks in a `.disas` file. For real workloads (e.g., YOLO inference), this can mean thousands of BBs — most of which are cold code with no fusion research value. We need to generate DFGs only for the hottest BBs identified by Step 5's BBV analysis.

## Solution

Refactor `analyze_bbv.py` to output structured JSON alongside its text report. Extend the DFG tool to read this JSON and filter BBs by `--top N` (default) or `--coverage X%`. Add a shell script to orchestrate the two-step pipeline.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Filtering logic location | DFG tool | "Which BBs deserve a DFG" is DFG's decision; analyze_bbv stays single-responsibility |
| Report data format | Structured `list[ReportEntry]` consumed by text + JSON formatters | Clean separation, no duplication |
| Default filter mode | `--top N` | Consistent with existing `analyze_bbv --top`; intuitive |
| Coverage boundary | Include the BB that first exceeds threshold | Natural behavior, doesn't truncate same-tier hotspots |
| Orchestration | Shell script (`profile_to_dfg.sh`) | Consistent with existing `prepare_model.sh`, `verify_bbv.sh` |
| Both tools communicate via | JSON file on disk | Shell script mediates; no Python import coupling |

## Section 1: analyze_bbv.py Refactor

### New data structure

```python
@dataclass
class ReportEntry:
    rank: int
    address: int
    count: int
    pct: float              # % of total executions
    cumulative_pct: float   # running cumulative %
    location: str
```

### New/changed functions

- **`build_report_data(resolved) -> list[ReportEntry]`**: Sorts by count descending, computes `pct` and `cumulative_pct` for every block. Returns full untruncated list.
- **`generate_report(entries, top_n=20) -> str`**: Signature changes to accept `list[ReportEntry]`. Only renders the top N entries as text. Logic unchanged otherwise.
- **`generate_report_json(entries) -> str`**: New function. Outputs **all** entries as JSON:
  ```json
  {
    "total_blocks": 1500,
    "total_executions": 999999,
    "blocks": [
      {
        "rank": 1,
        "address": "0x111f4",
        "count": 12345,
        "pct": 45.67,
        "cumulative_pct": 45.67,
        "location": "conv2d (src/conv.c:100)"
      }
    ]
  }
  ```

### CLI changes

- New `--json-output <path>` flag. When specified, writes JSON report to the given file.
- Text report (`-o`) and JSON report (`--json-output`) can be output independently or simultaneously.
- `parse_bbv()`, `resolve_addresses()`, and all address resolution logic remain untouched.

## Section 2: DFG Tool Report-Driven Filtering

### New CLI arguments

| Flag | Type | Description |
|------|------|-------------|
| `--report` | path | Path to analyze_bbv JSON output |
| `--coverage` | int | Coverage threshold (e.g., `80` = 80%) |

### Filtering behavior

1. If `--report` is given, read JSON and extract addresses of selected BBs:
   - **With `--coverage X`**: include BBs up to and including the one whose `cumulative_pct` first reaches or exceeds X.
   - **Without `--coverage`**: include top `--top` BBs (default 20, reuses existing parameter).
2. Match selected addresses against `parse_disas()` results. Only matched BBs proceed to DFG generation.
3. Addresses from the report that have no matching BB in `.disas` produce a warning but do not abort.
4. If no addresses match, print error and exit.
5. `--report` and `--bb-filter` are mutually exclusive (error if both given).

### summary.json additions

```json
{
  "filter_mode": "top",
  "filter_value": 20,
  "blocks_from_report": 20,
  "blocks_matched": 18,
  "blocks_skipped_not_in_disas": 2
}
```

## Section 3: Shell Orchestration Script

**File**: `tools/profile_to_dfg.sh`

### Interface

```bash
./tools/profile_to_dfg.sh \
  --bbv output/yolo.bbv.0.bb \
  --elf output/yolo_inference \
  --sysroot output/sysroot \
  --top 50 \
  --output-dir output/dfg
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--bbv` | Yes | Path to `.bb` BBV file |
| `--elf` | Yes | Path to RISC-V ELF binary |
| `--disas` | No | Path to `.disas` file (auto-inferred from `.bb` if omitted) |
| `--sysroot` | No | Sysroot for shared library resolution |
| `--top` | No | Top N BBs (default: 20) |
| `--coverage` | No | Coverage threshold % |
| `--output-dir` | No | DFG output directory |
| `--jobs` | No | Parallel jobs for DFG (default: 1) |

### Flow

1. Parse arguments (hand-written while/case, no dependencies)
2. Auto-infer `--disas` from `--bbv` path if not specified (`yolo.bbv.0.bb` → try `yolo.bbv.0.disas`, then `yolo.bbv.disas`)
3. Call `analyze_bbv.py` with `--json-output` pointing to a temp file (`mktemp`)
4. Call `python -m tools.dfg` with `--report` and filter arguments
5. Clean up temp file
6. Print completion summary

## Section 4: Testing

### analyze_bbv.py tests

- Adapt existing tests for `generate_report()` signature change (now accepts `list[ReportEntry]`)
- New `test_build_report_data`: verify sorting, pct calculation, cumulative_pct accumulation
- New `test_generate_report_json`: verify JSON structure and field completeness
- New `test_json_output_cli`: verify `--json-output` flag writes to file

### DFG tool tests

- New `--report` + `--top` integration test: construct `.disas` + JSON report, verify only top N BBs processed
- New `--report` + `--coverage` integration test: verify coverage threshold behavior (includes BB that first exceeds threshold)
- New boundary test: report addresses not in `.disas` produce warning and are skipped
- New mutual exclusion test: `--report` + `--bb-filter` produces error

### Unchanged

- Existing DFG parser, builder, output, agent unit tests are unaffected
- `parse_bbv()` and `resolve_addresses()` logic is untouched
