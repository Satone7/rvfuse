#!/usr/bin/env bash
set -euo pipefail

# profile_to_dfg.sh — Run BBV profiling through selective DFG generation.
#
# Usage:
#   ./tools/profile_to_dfg.sh --bbv <file.bb> --elf <binary> [options]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
BBV=""
ELF=""
DISAS=""
SYSROOT=""
TOP=20
COVERAGE=""
OUTPUT_DIR=""
JOBS=1

# -- parse arguments (hand-written, no dependencies) --------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bbv)      BBV="$2";      shift 2 ;;
        --elf)      ELF="$2";      shift 2 ;;
        --disas)    DISAS="$2";    shift 2 ;;
        --sysroot)  SYSROOT="$2";  shift 2 ;;
        --top)      TOP="$2";      shift 2 ;;
        --coverage) COVERAGE="$2";  shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --jobs|-j)  JOBS="$2";     shift 2 ;;
        -h|--help)
            sed -n '2,/^$/s/^# \?//p' "$0"
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# -- validate required arguments ----------------------------------------------
if [[ -z "$BBV" ]]; then
    echo "Error: --bbv is required" >&2
    exit 1
fi
if [[ -z "$ELF" ]]; then
    echo "Error: --elf is required" >&2
    exit 1
fi
if [[ ! -f "$BBV" ]]; then
    echo "Error: BBV file not found: $BBV" >&2
    exit 1
fi

# -- auto-infer disas path from bbv path -------------------------------------
if [[ -z "$DISAS" ]]; then
    # yolo.bbv.0.bb → try yolo.bbv.0.disas, then yolo.bbv.disas
    BASE="${BBV%.bb}"
    if [[ -f "${BASE}.disas" ]]; then
        DISAS="${BASE}.disas"
    else
        # Strip PID: yolo.bbv.0.bb → yolo.bbv.disas
        BASE2="$(echo "$BBV" | sed 's/\.[0-9]\+\.bb$/.disas/')"
        if [[ -f "$BASE2" ]]; then
            DISAS="$BASE2"
        else
            echo "Error: cannot auto-infer .disas path from $BBV" >&2
            exit 1
        fi
    fi
fi

if [[ ! -f "$DISAS" ]]; then
    echo "Error: disas file not found: $DISAS" >&2
    exit 1
fi

# -- build analyze_bbv arguments ---------------------------------------------
ANALYZE_ARGS=("--bbv" "$BBV" "--elf" "$ELF")

JSON_TMP="$(mktemp /tmp/bbv_report_XXXXXX.json)"
trap 'rm -f "$JSON_TMP"' EXIT

ANALYZE_ARGS+=("--json-output" "$JSON_TMP")

if [[ -n "$SYSROOT" ]]; then
    ANALYZE_ARGS+=("--sysroot" "$SYSROOT")
fi
if [[ "$TOP" != "0" ]]; then
    ANALYZE_ARGS+=("--top" "$TOP")
fi

# -- step 1: analyze_bbv ------------------------------------------------------
echo "=== Step 1: Running analyze_bbv.py ==="
echo "  BBV:      $BBV"
echo "  ELF:      $ELF"
echo "  JSON out: $JSON_TMP"

python3 "$SCRIPT_DIR/analyze_bbv.py" "${ANALYZE_ARGS[@]}"
echo ""

# -- build DFG arguments -----------------------------------------------------
DFG_ARGS=("--disas" "$DISAS" "--report" "$JSON_TMP" "--jobs" "$JOBS")

if [[ -n "$OUTPUT_DIR" ]]; then
    DFG_ARGS+=("--output-dir" "$OUTPUT_DIR")
fi
if [[ -n "$COVERAGE" ]]; then
    DFG_ARGS+=("--coverage" "$COVERAGE")
else
    DFG_ARGS+=("--top" "$TOP")
fi

# -- step 2: DFG generation --------------------------------------------------
echo "=== Step 2: Running DFG generation ==="
echo "  Disas:     $DISAS"
echo "  Report:    $JSON_TMP"
echo "  Mode:      $([ -n "$COVERAGE" ] && echo "coverage=$COVERAGE%" || echo "top=$TOP")"
if [[ -n "$OUTPUT_DIR" ]]; then
    echo "  Output:    $OUTPUT_DIR"
fi

cd "$PROJECT_ROOT" && python3 -m tools.dfg "${DFG_ARGS[@]}"

echo ""
echo "=== Done ==="
