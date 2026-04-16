#!/usr/bin/env python3
"""Parse QEMU BBV output and generate hotspot reports.

Maps basic block addresses to source locations using addr2line
and prints the most frequently executed blocks.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReportEntry:
    rank: int
    address: int
    count: int
    pct: float
    cumulative_pct: float
    location: str
    bb_id: int | None = None


def parse_bbv(bbv_path):
    """Parse BBV file into list of (address, count) tuples and address-to-bb_id map.

    Supports two formats:
    - Native QEMU BBV: T:id1:count1 :id2:count2 ... (with companion .disas)
    - Simple: address count (e.g. 0x10000 42)

    Returns:
        (addr_counts, addr_to_bb_id) where addr_counts is list of (address, count)
        and addr_to_bb_id maps address -> bb_id (None for simple format).
    """
    import re

    bbv_file = Path(bbv_path)

    with open(bbv_file) as f:
        first_line = f.readline().strip()

    # Detect native QEMU BBV format (T:id:count ...)
    if first_line.startswith("T:"):
        disas_path = bbv_file.with_suffix(".disas")
        # QEMU BBV names .bb as <base>.<pid>.bb but .disas as <base>.disas
        if not disas_path.exists():
            m = re.match(r"(.+)\.\d+\.bb$", bbv_file.name)
            if m:
                candidate = bbv_file.parent / (m.group(1) + ".disas")
                if candidate.exists():
                    disas_path = candidate
        if not disas_path.exists():
            print(
                f"Error: native BBV format detected but disas file not found: {disas_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Parse disas: "BB <id> (vaddr: <hex>, <N> insns):"
        bb_addr_map = {}
        bb_re = re.compile(r"BB\s+(\d+)\s+\(vaddr:\s+(0x[0-9a-fA-F]+)")
        with open(disas_path) as f:
            for line in f:
                m = bb_re.match(line.strip())
                if m:
                    bb_addr_map[int(m.group(1))] = int(m.group(2), 16)

        # Build reverse map: address -> bb_id
        addr_to_bb_id = {v: k for k, v in bb_addr_map.items()}

        # Parse BBV: each line is "T:id1:count1 :id2:count2 ..."
        # Aggregate counts across all intervals per address
        addr_counts = {}
        with open(bbv_file) as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("T:"):
                    continue
                for token in line.split():
                    token = token.lstrip(":")
                    if token.startswith("T:"):
                        token = token[2:]
                    parts = token.split(":")
                    if len(parts) < 2:
                        continue
                    try:
                        bb_id = int(parts[0])
                        count = int(parts[1])
                    except ValueError:
                        continue
                    addr = bb_addr_map.get(bb_id)
                    if addr is not None:
                        addr_counts[addr] = addr_counts.get(addr, 0) + count
        return list(addr_counts.items()), addr_to_bb_id

    # Simple format: address count (no bb_id available)
    blocks = []
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
    return blocks, None


def resolve_addresses(blocks, elf_path, sysroot=None):
    """Resolve addresses to source locations via addr2line.

    For PIE executables, first detects the runtime base address by analyzing
    address distribution. For shared library addresses, uses stricter matching
    requiring multiple addresses to agree on the same base.
    """
    if not blocks:
        return []

    # Collect .so files from sysroot
    so_files = _collect_so_files(sysroot)

    # Determine main ELF .text range via readelf (static offsets)
    main_ranges = _get_elf_text_range(elf_path)

    # For PIE executables, detect runtime base from address distribution
    pie_base = _detect_pie_base(blocks, main_ranges, elf_path)

    # Separate main binary vs shared library addresses
    main_blocks = []
    so_blocks = []
    for addr, count in blocks:
        if pie_base:
            # PIE: check if addr - pie_base falls within static .text range
            offset = addr - pie_base
            if any(lo <= offset < hi for lo, hi in main_ranges):
                main_blocks.append((addr, count))
            else:
                so_blocks.append((addr, count))
        elif any(lo <= addr < hi for lo, hi in main_ranges):
            # Non-PIE: direct address match
            main_blocks.append((addr, count))
        else:
            so_blocks.append((addr, count))

    resolved = []

    # Resolve main binary addresses (adjust for PIE)
    if main_blocks:
        if pie_base:
            # Convert runtime addresses to file offsets for PIE
            offset_blocks = [(addr - pie_base, count, addr) for addr, count in main_blocks]
            main_resolved = _batch_resolve([(off, c) for off, c, _ in offset_blocks], elf_path)
            for (_, count, orig_addr), (_, _, loc) in zip(offset_blocks, main_resolved):
                resolved.append((orig_addr, count, loc))
        else:
            resolved.extend(_batch_resolve(main_blocks, elf_path))

    # Resolve shared library addresses with stricter matching
    if so_blocks:
        resolved.extend(_resolve_so_addresses_strict(so_blocks, so_files))

    return resolved


def _detect_pie_base(blocks, static_ranges, elf_path):
    """Detect PIE executable runtime base from address distribution.

    For PIE, runtime addresses are base + static_offset. Analyzes low-address
    cluster to estimate base. Returns None for non-PIE or indeterminate cases.
    """
    if not static_ranges:
        return None

    # Check if ELF is PIE (DYN type with INTERP)
    try:
        result = subprocess.run(
            ["file", elf_path],
            capture_output=True, text=True, timeout=10,
        )
        if "DYN" not in result.stdout or "PIE" not in result.stdout.lower():
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Get the lowest .text static address
    text_start = min(lo for lo, hi in static_ranges)

    # Find addresses that could be from main binary (lower region)
    # PIE binaries are typically loaded in 0x55* region
    sorted_addrs = sorted([a for a, _ in blocks])
    low_addrs = [a for a in sorted_addrs if (a >> 40) <= 0x60]

    if not low_addrs:
        return None

    # Estimate base: lowest address - text_start, page-aligned
    lowest_addr = min(low_addrs)
    estimated_base = lowest_addr - text_start
    # Align to page boundary (PIE loads are page-aligned)
    pie_base = estimated_base & ~0xFFF

    # Validate: most low addresses should map to within .text range
    valid_count = 0
    for addr in low_addrs[:100]:  # Check sample
        offset = addr - pie_base
        if any(lo <= offset < hi for lo, hi in static_ranges):
            valid_count += 1

    if valid_count < len(low_addrs[:100]) * 0.5:
        # Less than 50% valid, likely not correct base
        return None

    return pie_base


def build_report_data(resolved, addr_to_bb_id=None):
    """Sort resolved blocks by count descending, compute pct and cumulative_pct.

    Returns full untruncated list of ReportEntry objects.
    """
    if not resolved:
        return []
    sorted_blocks = sorted(resolved, key=lambda x: x[1], reverse=True)
    total = sum(c for _, c, _ in resolved)
    entries = []
    cumulative = 0.0
    for rank, (addr, count, location) in enumerate(sorted_blocks, 1):
        pct = (count / total * 100) if total else 0.0
        cumulative += pct
        bb_id = addr_to_bb_id.get(addr) if addr_to_bb_id else None
        entries.append(ReportEntry(
            rank=rank,
            address=addr,
            count=count,
            pct=pct,
            cumulative_pct=cumulative,
            location=location,
            bb_id=bb_id,
        ))
    return entries


def _collect_so_files(sysroot):
    """Recursively find unique .so files in sysroot, sorted by size descending."""
    if not sysroot:
        return []
    sysroot_path = Path(sysroot)
    if not sysroot_path.is_dir():
        return []
    # Collect .so files, filtering out broken symlinks first
    so_candidates = list(sysroot_path.rglob("*.so*"))
    valid_so = []
    for sf in so_candidates:
        try:
            if sf.exists() and sf.is_file():
                valid_so.append(sf)
        except OSError:
            continue
    all_so = sorted(valid_so, key=lambda p: p.stat().st_size, reverse=True)
    seen_inodes = set()
    unique = []
    for sf in all_so:
        try:
            st = sf.stat()
            key = (st.st_dev, st.st_ino)
            if key not in seen_inodes:
                seen_inodes.add(key)
                unique.append(sf)
        except OSError:
            continue
    return unique


def _get_elf_text_range(elf_path):
    """Get .text section address ranges from ELF via readelf."""
    ranges = []
    try:
        result = subprocess.run(
            ["readelf", "-S", elf_path],
            capture_output=True, text=True, timeout=30,
        )
        for line in result.stdout.splitlines():
            if ".text" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == ".text" and i + 3 < len(parts):
                        try:
                            addr_start = int(parts[i + 2], 16)
                            addr_size = int(parts[i + 3], 16)
                            ranges.append((addr_start, addr_start + addr_size))
                        except ValueError:
                            pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ranges


def _batch_resolve(blocks, elf_path):
    """Resolve a batch of addresses against a single ELF via addr2line."""
    if not blocks:
        return []

    addresses = [f"0x{a:x}" for a, _ in blocks]
    batch_size = 500
    resolved = []

    for batch_start in range(0, len(addresses), batch_size):
        batch_addrs = addresses[batch_start : batch_start + batch_size]
        batch_blocks = blocks[batch_start : batch_start + batch_size]
        cmd = ["addr2line", "-f", "-e", elf_path] + batch_addrs

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"Warning: addr2line failed: {exc}", file=sys.stderr)
            resolved.extend([(a, c, "??") for a, c in batch_blocks])
            continue

        if result.returncode != 0:
            resolved.extend([(a, c, "??") for a, c in batch_blocks])
            continue

        lines = result.stdout.strip().split("\n")
        for i, (addr, count) in enumerate(batch_blocks):
            if 2 * i + 1 < len(lines):
                func = lines[2 * i].strip()
                loc = lines[2 * i + 1].strip()
                resolved.append((addr, count, f"{func} ({loc})"))
            else:
                resolved.append((addr, count, "??"))
    return resolved


def _get_so_loads(so_files):
    """Parse executable LOAD segments from .so files via readelf."""
    so_loads = []
    for so_path in so_files:
        try:
            result = subprocess.run(
                ["readelf", "-l", str(so_path)],
                capture_output=True, text=True, timeout=10,
            )
            lines = result.stdout.splitlines()
            for i, line in enumerate(lines):
                if "LOAD" not in line:
                    continue
                parts1 = line.split()
                parts2 = lines[i + 1].split() if i + 1 < len(lines) else []
                if len(parts1) < 3 or len(parts2) < 3:
                    continue
                try:
                    vaddr = int(parts1[2], 16)
                    memsz = int(parts2[1], 16)
                    # Flags may be split: ["R", "E", "0x1000"] or joined: ["RWE", "0x1000"]
                    flag_fields = parts2[2:]
                    is_exec = any("E" in f for f in flag_fields)
                    if is_exec:
                        so_loads.append((so_path, vaddr, memsz))
                        break
                except (ValueError, IndexError):
                    continue
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    return so_loads


def _resolve_so_addresses(blocks, so_files):
    """Resolve shared library addresses by matching against .so LOAD segments.

    For each .so, tries to find a page-aligned runtime base address that
    covers the most unresolved blocks. Removes matched blocks and continues
    with the next .so.
    """
    so_loads = _get_so_loads(so_files)
    if not so_loads:
        return [(a, c, f"<unknown-so>@0x{a:x}") for a, c in blocks]

    remaining = list(blocks)
    resolved = []

    # Try largest .so first
    so_loads.sort(key=lambda x: x[2], reverse=True)

    for so_path, load_vaddr, load_size in so_loads:
        if not remaining:
            break

        best_base = None
        best_matches = []

        # Sample addresses to find the best runtime base
        sample = remaining[:: max(1, len(remaining) // 50)]
        for addr, _ in sample:
            for page_off in range(0, 0x10000, 0x1000):
                cb = ((addr - load_vaddr) - page_off) & ~0xFFF
                if cb <= 0:
                    continue
                matches = [
                    (a, c) for a, c in remaining
                    if load_vaddr <= a - cb < load_vaddr + load_size
                ]
                if len(matches) > len(best_matches):
                    best_base = cb
                    best_matches = matches

        if not best_matches:
            continue

        # Resolve matched addresses via addr2line on the .so
        file_blocks = [(a - best_base, c, a) for a, c in best_matches]
        so_resolved = _batch_resolve(
            [(fv, c) for fv, c, _ in file_blocks], str(so_path)
        )
        so_name = so_path.name
        matched_addrs = set(a for a, _ in best_matches)
        for (_, count, orig_addr), (_, _, loc) in zip(file_blocks, so_resolved):
            if loc != "??":
                resolved.append((orig_addr, count, f"[{so_name}] {loc}"))
            else:
                resolved.append((orig_addr, count, f"[{so_name}] 0x{orig_addr:x}"))

        remaining = [(a, c) for a, c in remaining if a not in matched_addrs]

    resolved.extend([(a, c, f"<unknown-so>@0x{a:x}") for a, c in remaining])
    return resolved


def _resolve_so_addresses_strict(blocks, so_files):
    """Resolve shared library addresses with stricter matching.

    Key improvements over _resolve_so_addresses:
    1. Prioritize application-specific libraries (llama, ggml, onnx, etc.)
    2. Base address must be in high memory region (0x7f* prefix typical)
    3. Multiple addresses must agree on the same base (clustering)
    4. Minimum match threshold before accepting a .so

    Returns resolved addresses with stricter validation to avoid false matches.
    """
    so_loads = _get_so_loads(so_files)
    if not so_loads:
        return [(a, c, f"<unknown-so>@0x{a:x}") for a, c in blocks]

    # Pre-filter: only consider addresses in high memory region (0x7f*)
    # Lower addresses (0x55*) likely belong to main PIE binary
    high_region_blocks = [(a, c) for a, c in blocks if (a >> 32) >= 0x7f]
    low_region_blocks = [(a, c) for a, c in blocks if (a >> 32) < 0x7f]

    resolved = []
    remaining = list(high_region_blocks)

    # Group .so by name for matching (avoid duplicates from symlinks)
    unique_so = {}
    for so_path, load_vaddr, load_size in so_loads:
        name = so_path.name
        # Keep the actual file (not symlink) with largest size
        if name not in unique_so or load_size > unique_so[name][2]:
            unique_so[name] = (so_path, load_vaddr, load_size)

    so_loads_unique = list(unique_so.values())

    # PRIORITY: Application-specific libraries first, then system libraries
    # Application libs: llama, ggml, onnx, ort, ml-related
    app_lib_patterns = ['llama', 'ggml', 'onnx', 'ort', 'mlas', 'proto', 'sentencepiece']
    app_libs = []
    sys_libs = []
    for so_path, load_vaddr, load_size in so_loads_unique:
        name_lower = so_path.name.lower()
        if any(p in name_lower for p in app_lib_patterns):
            app_libs.append((so_path, load_vaddr, load_size))
        else:
            sys_libs.append((so_path, load_vaddr, load_size))

    # Sort: app libs by size (largest first), sys libs by size (largest first)
    app_libs.sort(key=lambda x: x[2], reverse=True)
    sys_libs.sort(key=lambda x: x[2], reverse=True)
    # Combined: app libs first
    so_loads_ordered = app_libs + sys_libs

    for so_path, load_vaddr, load_size in so_loads_ordered:
        if not remaining:
            break

        # Find candidate bases by clustering address differences
        # A valid base should have many addresses mapping to within LOAD range
        candidate_bases = {}
        # Use larger sample: top addresses by execution count + random sample
        # Sort remaining by count descending to prioritize hot addresses
        sorted_remaining = sorted(remaining, key=lambda x: -x[1])
        # Take top 50 hot addresses plus a random sample
        hot_sample = sorted_remaining[:50]
        random_sample = remaining[:: max(1, len(remaining) // 10)][:50]
        sample = hot_sample + random_sample

        for addr, _ in sample:
            # Calculate potential base (page-aligned)
            raw_base = addr - load_vaddr
            # Align down to page
            for page_adj in range(0, 0x10000, 0x1000):
                base = (raw_base - page_adj) & ~0xFFF
                if base <= 0:
                    continue
                # Validate base is in reasonable high memory region (user space)
                # Addresses typically in 0x7f*, 0x79*, 0x7a* ranges depending on ASLR
                if (base >> 32) < 0x7f:
                    continue
                # Count how many addresses match this base and total weight
                matches = [(a, c) for a, c in remaining
                           if load_vaddr <= a - base < load_vaddr + load_size]
                if matches:
                    # Weight by execution count
                    total_weight = sum(c for a, c in matches)
                    # Bonus for valid addr2line symbols
                    symbol_bonus = 0
                    for check_addr, _ in matches[:3]:
                        offset = check_addr - base
                        result = subprocess.run(
                            ['addr2line', '-f', '-e', str(so_path), f'{offset:#x}'],
                            capture_output=True, text=True, timeout=5
                        )
                        if result.returncode == 0:
                            lines = result.stdout.strip().split('\n')
                            if lines and lines[0] != '??':
                                symbol_bonus += total_weight * 0.5
                                break
                    score = total_weight + symbol_bonus
                    candidate_bases[base] = candidate_bases.get(base, 0) + score

        if not candidate_bases:
            continue

        # Pick the base with most matches
        # Use lower threshold for app libs (1%), higher for sys libs (10%)
        is_app_lib = any(p in so_path.name.lower() for p in app_lib_patterns)
        threshold = 0.01 if is_app_lib else 0.10

        best_base = max(candidate_bases.items(), key=lambda x: x[1])
        if best_base[1] < len(remaining) * threshold:
            continue  # Too few matches, likely false positive

        base = best_base[0]
        matches = [(a, c) for a, c in remaining
                   if load_vaddr <= a - base < load_vaddr + load_size]

        if not matches:
            continue

        # Resolve matched addresses
        file_blocks = [(a - base, c, a) for a, c in matches]
        so_resolved = _batch_resolve(
            [(fv, c) for fv, c, _ in file_blocks], str(so_path)
        )
        so_name = so_path.name
        matched_addrs = set(a for a, _ in matches)
        for (_, count, orig_addr), (_, _, loc) in zip(file_blocks, so_resolved):
            if loc != "??":
                resolved.append((orig_addr, count, f"[{so_name}] {loc}"))
            else:
                resolved.append((orig_addr, count, f"[{so_name}] 0x{orig_addr:x}"))

        remaining = [(a, c) for a, c in remaining if a not in matched_addrs]

    # Unmatched high-region addresses
    resolved.extend([(a, c, f"<unknown-so>@0x{a:x}") for a, c in remaining])

    # Low-region addresses (likely main binary) - mark separately
    resolved.extend([(a, c, f"<main-binary-region>@0x{a:x}") for a, c in low_region_blocks])

    return resolved


def generate_report(entries, top_n=20):
    """Generate a sorted hotspot report string from ReportEntry list.

    Args:
        entries: list[ReportEntry] from build_report_data (must be pre-sorted).
        top_n: maximum number of entries to render in the text table.

    Returns:
        Formatted report string.
    """
    if not entries:
        total_blocks = 0
        total_executions = 0
    else:
        total_blocks = len(entries)
        total_executions = sum(e.count for e in entries)

    show = min(top_n, len(entries))

    lines = [
        "=" * 72,
        "BBV Hotspot Report",
        "=" * 72,
        f"Total basic blocks: {total_blocks}",
        f"Total executions:   {total_executions}",
        f"Showing top {show} blocks",
        "",
        f"{'Rank':<6}{'Count':<14}{'% Total':<10}Location",
        "-" * 72,
    ]
    for entry in entries[:show]:
        lines.append(
            f"{entry.rank:<6}{entry.count:<14}{entry.pct:>6.2f}%    "
            f"0x{entry.address:x} {entry.location}"
        )
    return "\n".join(lines)


def generate_report_json(entries):
    """Output all entries as JSON for DFG tool consumption.

    Args:
        entries: list[ReportEntry] from build_report_data (full untruncated list).

    Returns:
        JSON string with total_blocks, total_executions, and blocks array.
    """
    total_executions = sum(e.count for e in entries)
    data = {
        "total_blocks": len(entries),
        "total_executions": total_executions,
        "blocks": [
            {
                "rank": e.rank,
                "address": f"0x{e.address:x}",
                "count": e.count,
                "pct": round(e.pct, 2),
                "cumulative_pct": round(e.cumulative_pct, 2),
                "location": e.location,
                "bb_id": e.bb_id,
            }
            for e in entries
        ],
    }
    return json.dumps(data, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze QEMU BBV output and generate hotspot report"
    )
    parser.add_argument("--bbv", required=True, help="Path to .bbv file")
    parser.add_argument("--elf", required=True, help="Path to RISC-V ELF binary")
    parser.add_argument(
        "--sysroot", default=None,
        help="Sysroot directory with shared libraries for resolving .so addresses",
    )
    parser.add_argument("--top", type=int, default=20, help="Top N blocks")
    parser.add_argument(
        "--json-output",
        help="Write JSON report to this file (all blocks, not truncated)",
    )
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    blocks, addr_to_bb_id = parse_bbv(args.bbv)
    if not blocks:
        print("Error: no basic blocks found in BBV file", file=sys.stderr)
        sys.exit(1)
    print(f"Parsed {len(blocks)} basic blocks from {args.bbv}")

    resolved = resolve_addresses(blocks, args.elf, args.sysroot)
    entries = build_report_data(resolved, addr_to_bb_id)
    report = generate_report(entries, args.top)

    if args.output:
        Path(args.output).write_text(report + "\n")
        print(f"Report written to {args.output}")
    else:
        print(report)

    # JSON report
    if args.json_output:
        json_report = generate_report_json(entries)
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json_report + "\n")
        print(f"JSON report written to {args.json_output}")


if __name__ == "__main__":
    main()
