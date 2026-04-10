"""Subgraph-based fusion pattern mining pipeline.

Enumerates connected DFG subgraphs, normalizes them to pattern templates,
aggregates across basic blocks by BBV frequency, and writes a ranked
pattern catalog JSON.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dfg.instruction import ISARegistry

from fusion.pattern import normalize_subgraph
from fusion.subgraph import enumerate_subgraphs

logger = logging.getLogger("fusion")


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
    max_nodes: int = 8,
    min_size: int = 2,
    top: int | None = None,
) -> tuple[list[dict], int]:
    """Enumerate, normalize, aggregate, and rank patterns across multiple DFGs.

    Args:
        dfg_list: List of DFG JSON dicts.
        hotspot: BBV hotspot JSON with blocks[].address and blocks[].count.
        registry: ISA registry for register flow resolution.
        max_nodes: Maximum subgraph size to enumerate.
        min_size: Minimum subgraph size to include in output (default 2).
        top: Maximum number of patterns to return.

    Returns:
        A tuple of (patterns_list, total_subgraphs_enumerated).
    """
    bbv_map = _build_bbv_map(hotspot)
    groups: dict[tuple, dict] = {}
    total_subgraphs = 0

    for dfg_data in dfg_list:
        vaddr = dfg_data["vaddr"]
        bb_id = dfg_data.get("bb_id", 0)
        frequency = bbv_map.get(vaddr, 0)
        nodes = dfg_data["nodes"]

        subgraphs = enumerate_subgraphs(dfg_data, max_nodes=max_nodes)
        total_subgraphs += len(subgraphs)

        for sg in subgraphs:
            if len(sg) < min_size:
                continue

            try:
                pattern = normalize_subgraph(sg, dfg_data, registry)
            except ValueError:
                continue

            key = pattern.template_key
            if key not in groups:
                groups[key] = {
                    "pattern": pattern,
                    "occurrence_count": 0,
                    "total_frequency": 0,
                    "source_bbs": [],
                    "examples": [],
                }
            groups[key]["occurrence_count"] += 1
            groups[key]["total_frequency"] += frequency
            if vaddr not in groups[key]["source_bbs"]:
                groups[key]["source_bbs"].append(vaddr)
                example = {
                    "bb_id": bb_id,
                    "vaddr": vaddr,
                    "instructions": [
                        {"index": nodes[i]["index"], "mnemonic": nodes[i]["mnemonic"],
                         "operands": nodes[i]["operands"]}
                        for i in sorted(sg)
                    ],
                }
                groups[key]["examples"].append(example)

    results = []
    for key, group in groups.items():
        p = group["pattern"]
        results.append({
            "opcodes": [mn for layer in p.topology for mn in layer],
            "register_class": p.register_class,
            "size": p.size,
            "topology": p.topology,
            "edges": [
                {
                    "src_layer": e.src_layer,
                    "src_opcode": e.src_opcode,
                    "dst_layer": e.dst_layer,
                    "dst_opcode": e.dst_opcode,
                    "src_role": e.src_role,
                    "dst_role": e.dst_role,
                }
                for e in p.edges
            ],
            "occurrence_count": group["occurrence_count"],
            "total_frequency": group["total_frequency"],
            "source_bbs": group["source_bbs"],
            "examples": group["examples"],
        })

    results.sort(key=lambda x: x["total_frequency"], reverse=True)
    if top is not None:
        results = results[:top]
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results, total_subgraphs


def mine(
    dfg_dir: Path,
    hotspot_path: Path,
    registry: ISARegistry,
    output_path: Path,
    top: int | None = None,
    max_nodes: int = 8,
    min_size: int = 2,
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
    patterns, total_subgraphs = aggregate_patterns(
        dfg_list, hotspot, registry,
        max_nodes=max_nodes, min_size=min_size, top=top,
    )

    output = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source_df_count": len(dfg_list),
        "pattern_count": len(patterns),
        "patterns": patterns,
        "enumeration_stats": {
            "total_subgraphs": total_subgraphs,
            "max_nodes": max_nodes,
            "min_size": min_size,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    logger.info(
        "Mined %d patterns from %d DFGs (top=%s, max_nodes=%s, min_size=%s), "
        "%d total subgraphs enumerated",
        len(patterns), len(dfg_list), top, max_nodes, min_size, total_subgraphs,
    )
    return patterns
