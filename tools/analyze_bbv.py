#!/usr/bin/env python3
"""
analyze_bbv.py - Enhanced BBV hotspot analysis tool

Parses QEMU BBV plugin output, maps basic blocks to source code using
cross-compilation tools (objdump, addr2line), aggregates by function,
and generates a ranked hotspot report.

Usage:
    python3 analyze_bbv.py --bbv output/yolo.bbv --elf output/yolo_inference \
        --objdump riscv64-linux-gnu-objdump --addr2line riscv64-linux-gnu-addr2line \
        --top-funcs 10 --top-blocks 20 --output output/yolo_hotspot_report.txt
"""

import argparse
import subprocess
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_args():
    """Parse command-line arguments for the BBV hotspot analysis tool."""
    parser = argparse.ArgumentParser(
        description="Enhanced BBV hotspot analysis tool for RISC-V binaries"
    )
    parser.add_argument(
        "--bbv", required=True, help="Path to .bbv file from QEMU BBV plugin"
    )
    parser.add_argument(
        "--elf", required=True, help="Path to ELF executable (with debug info)"
    )
    parser.add_argument(
        "--objdump",
        default="riscv64-linux-gnu-objdump",
        help="Path to objdump tool (default: riscv64-linux-gnu-objdump)",
    )
    parser.add_argument(
        "--addr2line",
        default="riscv64-linux-gnu-addr2line",
        help="Path to addr2line tool (default: riscv64-linux-gnu-addr2line)",
    )
    parser.add_argument(
        "--top-funcs",
        type=int,
        default=10,
        help="Number of top hot functions to report (default: 10)",
    )
    parser.add_argument(
        "--top-blocks",
        type=int,
        default=20,
        help="Number of top hot basic blocks to report (default: 20)",
    )
    parser.add_argument(
        "--output", required=True, help="Output report file path"
    )
    return parser.parse_args()


def parse_bbv_file(bbv_path):
    """
    Parse .bbv file format from QEMU BBV plugin.

    Expected format (repeating per interval):
        # Interval N
        BB: start_pc end_pc exec_count
        BB: ...

    PC values are hexadecimal. Execution counts are aggregated across all
    intervals for each unique (start_pc, end_pc) pair.

    Returns:
        dict mapping (start_pc, end_pc) -> total_exec_count
    """
    bb_counts = defaultdict(int)

    with open(bbv_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("# Interval"):
                # Interval marker; just continue to next line
                continue
            elif line.startswith("BB:"):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        start_pc = int(parts[1], 16)
                        end_pc = int(parts[2], 16)
                        exec_count = int(parts[3])
                        bb_counts[(start_pc, end_pc)] += exec_count
                    except (ValueError, IndexError):
                        # Skip malformed BB lines
                        continue

    return bb_counts


def get_disassembly_with_source(elf_path, objdump_tool, pc_range):
    """
    Run objdump -d --source to get disassembly with source annotations
    for a given PC range.

    Args:
        elf_path: Path to the ELF binary
        objdump_tool: Path to the objdump executable
        pc_range: Tuple of (start_pc, end_pc) as integers

    Returns:
        String containing disassembly output, or an error message
    """
    start_pc, end_pc = pc_range

    try:
        result = subprocess.run(
            [
                objdump_tool,
                "-d",
                "--source",
                "--start-address=" + hex(start_pc),
                "--stop-address=" + hex(end_pc),
                elf_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return "Timeout getting disassembly"
    except FileNotFoundError:
        return "Error: {} not found".format(objdump_tool)
    except OSError as e:
        return "Error running objdump: {}".format(e)


def get_function_info(addr2line_tool, elf_path, pc):
    """
    Use addr2line to get function name and source location for a PC address.

    Args:
        addr2line_tool: Path to the addr2line executable
        elf_path: Path to the ELF binary
        pc: Program counter value as integer

    Returns:
        Tuple of (function_name, source_location). Returns ("unknown", "unknown")
        for unresolved addresses.
    """
    try:
        result = subprocess.run(
            [addr2line_tool, "-e", elf_path, "-f", hex(pc)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 2:
            func_name = lines[0].strip()
            source_loc = lines[1].strip()
            # Handle "??:?" or "??:0" as unknown locations
            if source_loc.startswith("??"):
                source_loc = "unknown"
            if func_name == "??":
                func_name = "unknown"
            return func_name, source_loc
        return "unknown", "unknown"
    except subprocess.TimeoutExpired:
        return "timeout", "timeout"
    except FileNotFoundError:
        return "error: {} not found".format(addr2line_tool), "error"
    except OSError as e:
        return "error: {}".format(e), "error"


def aggregate_by_function(bb_counts, addr2line_tool, elf_path):
    """
    Aggregate basic block execution counts by function using addr2line.

    For each basic block, looks up the function name via addr2line using the
    start PC of the block. Accumulates execution counts per function.

    Args:
        bb_counts: Dict mapping (start_pc, end_pc) -> exec_count
        addr2line_tool: Path to addr2line executable
        elf_path: Path to the ELF binary

    Returns:
        Tuple of:
            - function_counts: dict mapping function_name -> total_exec_count
            - bb_info: list of (pc_range, exec_count, func_name, source_loc)
    """
    function_counts = defaultdict(int)
    bb_info = []

    for pc_range, exec_count in bb_counts.items():
        start_pc = pc_range[0]
        func_name, source_loc = get_function_info(addr2line_tool, elf_path, start_pc)

        function_counts[func_name] += exec_count
        bb_info.append((pc_range, exec_count, func_name, source_loc))

    return function_counts, bb_info


def generate_report(args, function_counts, bb_info, bb_counts, objdump_tool, elf_path):
    """
    Generate the hotspot analysis report.

    Report sections:
        1. Header with file paths and summary counts
        2. Hot Functions Top N table
        3. Hot Basic Blocks Top N table (with assembly snippets for top 5)
        4. Execution Phase Analysis (estimated percentages based on keywords)

    Args:
        args: Parsed command-line arguments
        function_counts: Dict of function_name -> total_exec_count
        bb_info: List of (pc_range, exec_count, func_name, source_loc)
        bb_counts: Dict of (start_pc, end_pc) -> exec_count
        objdump_tool: Path to objdump executable
        elf_path: Path to the ELF binary

    Returns:
        Complete report as a single string
    """
    # Sort functions and blocks by execution count (descending)
    sorted_funcs = sorted(function_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_blocks = sorted(bb_info, key=lambda x: x[1], reverse=True)

    lines = []

    # Header
    lines.append("=" * 60)
    lines.append("ONNX Runtime YOLO Inference Hotspot Analysis Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append("BBV file:              {}".format(args.bbv))
    lines.append("ELF file:              {}".format(args.elf))
    lines.append("Total unique basic blocks: {}".format(len(bb_counts)))
    lines.append("Total unique functions:    {}".format(len(function_counts)))
    lines.append("")

    # Total execution count
    total_exec = sum(function_counts.values())
    lines.append("Total aggregated executions: {:,}".format(total_exec))
    lines.append("")

    # Hot Functions section
    lines.append("-- Hot Functions Top {} --".format(args.top_funcs))
    lines.append(
        "{:>14}  {:<40} {}".format("Exec Count", "Function Name", "Source Location")
    )
    lines.append("-" * 90)

    for func_name, count in sorted_funcs[: args.top_funcs]:
        # Find the best source location for this function (first non-unknown)
        source_locs = [
            info[3] for info in bb_info if info[2] == func_name and info[3] != "unknown"
        ]
        best_loc = source_locs[0] if source_locs else "unknown"
        lines.append("{:>14,}  {:<40} {}".format(count, func_name, best_loc))

    lines.append("")

    # Hot Basic Blocks section with assembly snippets
    lines.append("-- Hot Basic Blocks Top {} --".format(args.top_blocks))
    lines.append(
        "{:>14}  {:<12} {:<30} {}".format("Exec Count", "PC Address", "Function Name", "Source")
    )
    lines.append("-" * 90)

    for i, (pc_range, count, func_name, source_loc) in enumerate(
        sorted_blocks[: args.top_blocks]
    ):
        start_pc = pc_range[0]
        lines.append(
            "{:>14,}  0x{:08x}  {:<30} {}".format(
                count, start_pc, func_name, source_loc
            )
        )

        # Include assembly snippet for top 5 blocks
        if i < 5:
            disasm = get_disassembly_with_source(elf_path, objdump_tool, pc_range)
            disasm_lines = disasm.split("\n")[:8]
            for dl in disasm_lines:
                if dl.strip():
                    lines.append("    {}".format(dl))
            lines.append("")

    lines.append("")

    # Execution Phase Analysis
    lines.append("-- Execution Phase Analysis --")
    lines.append(
        "{:<16} {:>12}  {}".format("Phase", "Est. Exec%", "Primary Operations")
    )
    lines.append("-" * 65)

    # Classify functions into phases based on name keywords
    inference_keywords = ["matmul", "conv", "relu", "gemm", "softmax", "batchnorm"]
    model_load_keywords = ["protobuf", "parse", "model", "onnx", "session", "graph"]
    postprocess_keywords = ["nms", "sort", "detection", "yolo", "postprocess"]

    inference_count = sum(
        c
        for f, c in sorted_funcs
        if any(kw in f.lower() for kw in inference_keywords)
    )
    model_load_count = sum(
        c
        for f, c in sorted_funcs
        if any(kw in f.lower() for kw in model_load_keywords)
    )
    postprocess_count = sum(
        c
        for f, c in sorted_funcs
        if any(kw in f.lower() for kw in postprocess_keywords)
    )

    inference_pct = inference_count / total_exec * 100 if total_exec > 0 else 0
    model_load_pct = model_load_count / total_exec * 100 if total_exec > 0 else 0
    postprocess_pct = postprocess_count / total_exec * 100 if total_exec > 0 else 0

    lines.append(
        "{:<16} ~{:>10.1f}%  {}".format(
            "Model Load", model_load_pct, "protobuf parsing, model initialization"
        )
    )
    lines.append(
        "{:<16} ~{:>10.1f}%  {}".format(
            "Inference", inference_pct, "Conv/MatMul/activation functions"
        )
    )
    lines.append(
        "{:<16} ~{:>10.1f}%  {}".format(
            "Postprocess", postprocess_pct, "NMS/sorting/detection output"
        )
    )
    other_pct = 100.0 - model_load_pct - inference_pct - postprocess_pct
    lines.append(
        "{:<16} ~{:>10.1f}%  {}".format(
            "Other", max(other_pct, 0), "runtime overhead, memory management"
        )
    )

    lines.append("")
    lines.append("=" * 60)
    lines.append("End of Report")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    """Main entry point for the BBV hotspot analysis tool."""
    args = parse_args()

    # Verify input files exist
    if not Path(args.bbv).exists():
        print("Error: BBV file not found: {}".format(args.bbv), file=sys.stderr)
        sys.exit(1)
    if not Path(args.elf).exists():
        print("Error: ELF file not found: {}".format(args.elf), file=sys.stderr)
        sys.exit(1)

    print("Parsing BBV file: {}".format(args.bbv))
    bb_counts = parse_bbv_file(args.bbv)
    print("Found {} unique basic blocks".format(len(bb_counts)))

    print("Aggregating by function (this may take a while for large BBV files)...")
    function_counts, bb_info = aggregate_by_function(
        bb_counts, args.addr2line, args.elf
    )
    print("Found {} unique functions".format(len(function_counts)))

    print("Generating report...")
    report = generate_report(
        args, function_counts, bb_info, bb_counts, args.objdump, args.elf
    )

    # Write report to output file
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)

    print("Report written to: {}".format(args.output))
    print(report)


if __name__ == "__main__":
    main()
