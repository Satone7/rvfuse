"""Output serialization for DFG: DOT and JSON formats."""

from __future__ import annotations

import html
import json
import logging
import shutil
import subprocess
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
        safe_operands = html.escape(operands.replace("{", "(").replace("}", ")"), quote=True)
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
    bb_id = dfg.bb.bb_id

    dot_dir = output_dir / "dot"
    dot_dir.mkdir(parents=True, exist_ok=True)
    (dot_dir / f"bb_{bb_id:03d}.dot").write_text(dfg_to_dot(dfg))

    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    (json_dir / f"bb_{bb_id:03d}.json").write_text(dfg_to_json(dfg))


def write_summary(stats: dict, output_path: Path) -> None:
    """Write summary.json with stats dict."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2))


def convert_dot_to_png(dot_dir: Path, png_dir: Path) -> int:
    """Convert all .dot files in dot_dir to .png files in png_dir.

    Uses the graphviz ``dot`` CLI.  Returns the number of files
    successfully converted.

    If ``dot`` is not found on PATH, returns 0 immediately.
    Individual conversion failures are logged as warnings.
    """
    if shutil.which("dot") is None:
        logging.getLogger("dfg").warning(
            "graphviz 'dot' not found — skipping PNG generation"
        )
        return 0

    png_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    for dot_file in sorted(dot_dir.glob("*.dot")):
        png_file = png_dir / f"{dot_file.stem}.png"
        try:
            subprocess.run(
                ["dot", "-Tpng", str(dot_file), "-o", str(png_file)],
                check=True,
                capture_output=True,
                text=True,
            )
            converted += 1
        except subprocess.CalledProcessError:
            logging.getLogger("dfg").warning(
                "Failed to convert %s to PNG", dot_file.name,
            )
    return converted
