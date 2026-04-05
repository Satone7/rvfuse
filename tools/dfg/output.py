"""Output serialization for DFG: DOT and JSON formats."""

from __future__ import annotations

import html
import json
from pathlib import Path

from dfg.instruction import DFG


def dfg_to_dot(dfg: DFG) -> str:
    """Convert a DFG to Graphviz DOT format string."""
    bb_id = dfg.bb.bb_id
    lines: list[str] = [f"digraph DFG_BB{bb_id} {{", '    node [shape=record];']

    for node in dfg.nodes:
        addr = f"0x{node.instruction.address:x}"
        mnemonic = node.instruction.mnemonic
        operands = node.instruction.operands
        safe_operands = html.escape(operands.replace("{", "(").replace("}", ")"), quote=False)
        label = f"{{{addr}: {mnemonic} {safe_operands}}}"
        lines.append(f'    {node.index} [label="{label}"];')

    lines.append("")

    for edge in dfg.edges:
        lines.append(f'    {edge.src_index} -> {edge.dst_index} [label="{edge.register}"];')

    lines.append("}")
    return "\n".join(lines)


def dfg_to_json(dfg: DFG) -> str:
    """Convert a DFG to an indented JSON string."""
    data = {
        "bb_id": dfg.bb.bb_id,
        "vaddr": f"0x{dfg.bb.vaddr:x}",
        "source": dfg.source,
        "nodes": [
            {
                "index": node.index,
                "address": f"0x{node.instruction.address:x}",
                "mnemonic": node.instruction.mnemonic,
                "operands": node.instruction.operands,
            }
            for node in dfg.nodes
        ],
        "edges": [
            {
                "src": edge.src_index,
                "dst": edge.dst_index,
                "register": edge.register,
            }
            for edge in dfg.edges
        ],
    }
    return json.dumps(data, indent=2)


def write_dfg_files(dfg: DFG, output_dir: Path) -> None:
    """Write both DOT and JSON files for a single DFG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bb_id = dfg.bb.bb_id

    dot_path = output_dir / f"bb_{bb_id:03d}.dot"
    dot_path.write_text(dfg_to_dot(dfg))

    json_path = output_dir / f"bb_{bb_id:03d}.json"
    json_path.write_text(dfg_to_json(dfg))


def write_summary(stats: dict, output_path: Path) -> None:
    """Write summary.json with stats dict."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2))
