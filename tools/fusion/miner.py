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
