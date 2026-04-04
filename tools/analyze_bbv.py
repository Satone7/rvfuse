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
    """Parse BBV file into list of (address, count) tuples.

    Supports two formats:
    - Native QEMU BBV: T:bb_id:insn_count (with companion .disas for addresses)
    - Simple: address count (e.g. 0x10000 42)
    """
    bbv_file = Path(bbv_path)
    blocks = []

    with open(bbv_file) as f:
        first_line = f.readline().strip()

    # Detect native QEMU BBV format (T:bb_id:insn_count)
    if first_line.startswith("T:"):
        disas_path = bbv_file.with_suffix(".disas")
        if not disas_path.exists():
            print(
                f"Error: native BBV format detected but disas file not found: {disas_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        bb_addr_map = {}
        with open(disas_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) >= 1:
                    try:
                        bb_id = int(parts[0])
                    except ValueError:
                        continue
                    if len(parts) >= 2 and parts[1].startswith("0x"):
                        bb_addr_map[bb_id] = int(parts[1], 16)
                    elif len(parts) >= 2:
                        try:
                            bb_addr_map[bb_id] = int(parts[1], 16)
                        except ValueError:
                            pass

        with open(bbv_file) as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("T:"):
                    continue
                parts = line.split(":")
                if len(parts) < 3:
                    continue
                try:
                    bb_id = int(parts[1])
                    count = int(parts[2])
                except ValueError:
                    continue
                addr = bb_addr_map.get(bb_id)
                if addr is not None:
                    blocks.append((addr, count))
        return blocks

    # Simple format: address count
    with open(bbv_file) as f:
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
    batch_size = 500
    all_resolved = []

    for batch_start in range(0, len(addresses), batch_size):
        batch_addrs = addresses[batch_start : batch_start + batch_size]
        batch_blocks = blocks[batch_start : batch_start + batch_size]
        cmd = ["addr2line", "-f", "-e", elf_path] + batch_addrs

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"Warning: addr2line failed: {exc}", file=sys.stderr)
            all_resolved.extend([(a, c, "??") for a, c in batch_blocks])
            continue

        if result.returncode != 0:
            print(f"Warning: addr2line exited with code {result.returncode}", file=sys.stderr)
            all_resolved.extend([(a, c, "??") for a, c in batch_blocks])
            continue

        lines = result.stdout.strip().split("\n")
        for i, (addr, count) in enumerate(batch_blocks):
            if 2 * i + 1 < len(lines):
                func = lines[2 * i].strip()
                loc = lines[2 * i + 1].strip()
                all_resolved.append((addr, count, f"{func} ({loc})"))
            else:
                all_resolved.append((addr, count, "??"))
    return all_resolved


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
