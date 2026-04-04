#!/usr/bin/env python3
"""Parse QEMU BBV output and generate hotspot reports.

Maps basic block addresses to source locations using addr2line
and prints the most frequently executed blocks.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_bbv(bbv_path):
    """Parse BBV file into list of (address, count) tuples."""
    blocks = []
    with open(bbv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            addr = int(parts[0], 16) if parts[0].startswith("0x") else int(parts[0])
            blocks.append((addr, int(parts[1])))
    return blocks


def resolve_addresses(blocks, elf_path):
    """Resolve addresses to source locations via addr2line."""
    if not blocks:
        return []

    addresses = [f"0x{a:x}" for a, _ in blocks]
    cmd = ["addr2line", "-f", "-e", elf_path] + addresses

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"Warning: addr2line failed: {exc}", file=sys.stderr)
        return [(a, c, "??") for a, c in blocks]

    if result.returncode != 0:
        print(f"Warning: addr2line exited with code {result.returncode}", file=sys.stderr)
        return [(a, c, "??") for a, c in blocks]

    lines = result.stdout.strip().split("\n")
    resolved = []
    for i, (addr, count) in enumerate(blocks):
        if 2 * i + 1 < len(lines):
            func = lines[2 * i].strip()
            loc = lines[2 * i + 1].strip()
            resolved.append((addr, count, f"{func} ({loc})"))
        else:
            resolved.append((addr, count, "??"))
    return resolved


def generate_report(resolved, top_n=20):
    """Generate a sorted hotspot report string."""
    sorted_blocks = sorted(resolved, key=lambda x: x[1], reverse=True)
    total = sum(c for _, c, _ in resolved)
    show = min(top_n, len(sorted_blocks))

    lines = [
        "=" * 72,
        "BBV Hotspot Report",
        "=" * 72,
        f"Total basic blocks: {len(resolved)}",
        f"Total executions:   {total}",
        f"Showing top {show} blocks",
        "",
        f"{'Rank':<6}{'Address':<18}{'Count':<14}{'% Total':<10}Location",
        "-" * 72,
    ]
    for rank, (addr, count, location) in enumerate(sorted_blocks[:show], 1):
        pct = (count / total * 100) if total else 0
        lines.append(
            f"{rank:<6}0x{addr:016x}  {count:<14}{pct:>6.2f}%    {location}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze QEMU BBV output and generate hotspot report"
    )
    parser.add_argument("--bbv", required=True, help="Path to .bbv file")
    parser.add_argument("--elf", required=True, help="Path to RISC-V ELF binary")
    parser.add_argument("--top", type=int, default=20, help="Top N blocks")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    blocks = parse_bbv(args.bbv)
    if not blocks:
        print("Error: no basic blocks found in BBV file", file=sys.stderr)
        sys.exit(1)
    print(f"Parsed {len(blocks)} basic blocks from {args.bbv}")

    resolved = resolve_addresses(blocks, args.elf)
    report = generate_report(resolved, args.top)

    if args.output:
        Path(args.output).write_text(report + "\n")
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
