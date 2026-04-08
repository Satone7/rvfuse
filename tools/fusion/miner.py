"""Deterministic fusion pattern mining pipeline."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dfg.instruction import ISARegistry

from fusion.pattern import (
    Pattern,
    classify_register,
    is_control_flow,
    normalize_chain,
)

logger = logging.getLogger("fusion")


def enumerate_chains(
    dfg_data: dict,
    registry: ISARegistry,
) -> list[list[tuple[str, str]]]:
    """Extract all valid linear RAW chains from a single DFG.

    Args:
        dfg_data: Parsed DFG JSON dict with 'nodes' and 'edges'.
        registry: ISA registry for register classification.

    Returns:
        List of chains. Each chain is a list of (mnemonic, operands) tuples.
    """
    nodes = dfg_data["nodes"]
    edges = dfg_data["edges"]

    if len(nodes) < 2:
        return []

    # Build adjacency: edge_map[(src_idx, dst_idx)] = [register, ...]
    edge_map: dict[tuple[int, int], list[str]] = defaultdict(list)
    for e in edges:
        edge_map[(e["src"], e["dst"])].append(e["register"])

    # Determine register class for each node
    def node_reg_class(idx: int) -> str | None:
        mn = nodes[idx]["mnemonic"]
        flow = registry.get_flow(mn)
        if flow is None:
            return None
        resolved = flow.resolve(nodes[idx]["operands"])
        all_regs = resolved.dst_regs + resolved.src_regs
        return classify_register(all_regs[0]) if all_regs else None

    results: list[list[tuple[str, str]]] = []

    def _valid_pair(i: int, j: int) -> bool:
        mn_i, mn_j = nodes[i]["mnemonic"], nodes[j]["mnemonic"]
        if is_control_flow(mn_i) or is_control_flow(mn_j):
            return False
        rc_i, rc_j = node_reg_class(i), node_reg_class(j)
        if rc_i is None or rc_j is None or rc_i != rc_j:
            return False
        return True

    # Scan for length-2 and length-3 chains
    for i in range(len(nodes) - 1):
        if not _valid_pair(i, i + 1):
            continue
        if (i, i + 1) not in edge_map:
            continue

        # Length-2 chain
        chain_2 = [
            (nodes[i]["mnemonic"], nodes[i]["operands"]),
            (nodes[i + 1]["mnemonic"], nodes[i + 1]["operands"]),
        ]
        results.append(chain_2)

        # Try to extend to length-3
        if i + 2 < len(nodes) and _valid_pair(i + 1, i + 2):
            if (i + 1, i + 2) in edge_map:
                chain_3 = chain_2 + [
                    (nodes[i + 2]["mnemonic"], nodes[i + 2]["operands"]),
                ]
                results.append(chain_3)

    return results


def _build_bbv_map(hotspot: dict) -> dict[str, int]:
    """Build vaddr -> execution count mapping from hotspot JSON."""
    result: dict[str, int] = {}
    for block in hotspot.get("blocks", []):
        result[block["address"]] = block["count"]
    return result


def aggregate_patterns(
    dfg_list: list[dict],
    hotspot: dict,
    registry: ISARegistry,
    top: int | None = None,
) -> list[dict]:
    """Enumerate, normalize, aggregate, and rank patterns across multiple DFGs."""
    bbv_map = _build_bbv_map(hotspot)
    groups: dict[tuple, dict] = {}

    for dfg_data in dfg_list:
        vaddr = dfg_data["vaddr"]
        frequency = bbv_map.get(vaddr, 0)
        chains = enumerate_chains(dfg_data, registry)

        for chain in chains:
            node_indices = list(range(len(chain)))
            all_edges = dfg_data["edges"]
            chain_edges = [
                {"src": src, "dst": dst, "register": e["register"]}
                for e in all_edges
                if (src := e["src"]) in node_indices
                and (dst := e["dst"]) in node_indices
                and dst == src + 1
            ]
            try:
                pattern = normalize_chain(chain, chain_edges, registry)
            except ValueError:
                continue

            key = pattern.template_key
            if key not in groups:
                groups[key] = {
                    "pattern": pattern,
                    "occurrence_count": 0,
                    "total_frequency": 0,
                    "source_bbs": [],
                }
            groups[key]["occurrence_count"] += 1
            groups[key]["total_frequency"] += frequency
            if vaddr not in groups[key]["source_bbs"]:
                groups[key]["source_bbs"].append(vaddr)

    results = []
    for key, group in groups.items():
        p = group["pattern"]
        results.append({
            "opcodes": p.opcodes,
            "register_class": p.register_class,
            "length": p.length,
            "occurrence_count": group["occurrence_count"],
            "total_frequency": group["total_frequency"],
            "chain_registers": p.chain_registers,
            "source_bbs": group["source_bbs"],
        })

    results.sort(key=lambda x: x["total_frequency"], reverse=True)
    if top is not None:
        results = results[:top]
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results


def mine(
    dfg_dir: Path,
    hotspot_path: Path,
    registry: ISARegistry,
    output_path: Path,
    top: int | None = None,
) -> list[dict]:
    """Run the full mining pipeline: load DFGs, aggregate, write output."""
    dfg_files = sorted(dfg_dir.glob("*.json"))
    dfg_list = []
    for f in dfg_files:
        try:
            data = json.loads(f.read_text())
            if "nodes" in data and "edges" in data:
                dfg_list.append(data)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Skipping invalid DFG file: %s", f)

    hotspot = json.loads(hotspot_path.read_text())
    patterns = aggregate_patterns(dfg_list, hotspot, registry, top=top)

    output = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source_df_count": len(dfg_list),
        "pattern_count": len(patterns),
        "patterns": patterns,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    logger.info("Mined %d patterns from %d DFGs (top=%s)", len(patterns), len(dfg_list), top)
    return patterns
