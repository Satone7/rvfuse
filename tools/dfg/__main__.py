"""CLI entry point for the DFG generation module.

Usage:
    python -m tools.dfg --disas <file.disas> [options]
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

# Ensure the tools/ directory is on sys.path so that `dfg.` imports resolve
# when running as `python -m tools.dfg` from the worktree root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dfg.agent import AgentDispatcher
from dfg.dfg import build_dfg
from dfg.instruction import ISARegistry
from dfg.output import write_dfg_files, write_summary
from dfg.parser import parse_disas

logger = logging.getLogger("dfg")

# Mapping from ISA extension name (uppercase) to (module path, builder function).
_ISA_MODULES: dict[str, tuple[str, str]] = {
    "I": ("dfg.isadesc.rv64i", "build_registry"),
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="tools.dfg",
        description="Generate Data Flow Graphs (DFG) from .disas basic-block files.",
    )
    parser.add_argument(
        "--disas",
        required=True,
        type=Path,
        help="Path to .disas input file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: dfg/ next to input file)",
    )
    parser.add_argument(
        "--isa",
        default="I",
        help="Comma-separated ISA extensions (default: I)",
    )
    parser.add_argument(
        "--no-agent",
        action="store_true",
        default=False,
        help="Disable agent check/fallback",
    )
    parser.add_argument(
        "--bb-filter",
        type=int,
        default=None,
        help="Only process specified BB ID (debugging)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    return parser


def load_isa_registry(extensions: str) -> ISARegistry:
    """Build an ISA registry from a comma-separated list of extension names.

    Only extensions present in _ISA_MODULES are loaded; unknown ones produce
    a warning and are skipped.
    """
    registry = ISARegistry()
    for ext in extensions.split(","):
        ext = ext.strip().upper()
        if ext in _ISA_MODULES:
            module_path, func_name = _ISA_MODULES[ext]
            mod = importlib.import_module(module_path)
            builder = getattr(mod, func_name)
            builder(registry)
            logger.info("Loaded ISA extension: %s (%d mnemonics)", ext, len(registry._flows))
        else:
            logger.warning("Unknown ISA extension '%s' -- skipping", ext)
    return registry


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point. Returns exit code."""
    args = build_arg_parser().parse_args(argv)

    # -- logging ---------------------------------------------------------------
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    # -- validate input --------------------------------------------------------
    disas_path: Path = args.disas
    if not disas_path.is_file():
        logger.error("Input file not found: %s", disas_path)
        return 1

    # -- output directory ------------------------------------------------------
    output_dir: Path = args.output_dir or disas_path.resolve().parent / "dfg"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- build ISA registry ----------------------------------------------------
    registry = load_isa_registry(args.isa)

    # -- parse -----------------------------------------------------------------
    blocks = parse_disas(disas_path)
    if not blocks:
        logger.error("No basic blocks found in %s", disas_path)
        return 1
    logger.info("Parsed %d basic block(s) from %s", len(blocks), disas_path)

    # -- bb filter -------------------------------------------------------------
    if args.bb_filter is not None:
        blocks = [bb for bb in blocks if bb.bb_id == args.bb_filter]
        if not blocks:
            logger.error("No basic block with ID %d found", args.bb_filter)
            return 1

    # -- agent -----------------------------------------------------------------
    agent = AgentDispatcher(enabled=not args.no_agent)

    # -- process each BB -------------------------------------------------------
    unsupported_instructions: set[str] = set()
    script_generated = 0
    agent_generated = 0
    agent_checked_pass = 0
    agent_checked_fail = 0

    for bb in blocks:
        # Check for unsupported mnemonics
        unknown = [insn.mnemonic for insn in bb.instructions
                    if not registry.is_known(insn.mnemonic)]

        if unknown:
            unsupported_instructions.update(unknown)
            logger.warning(
                "BB %d: %d unsupported instruction(s): %s -- trying agent",
                bb.bb_id, len(unknown), ", ".join(unknown),
            )
            agent_dfg = agent.generate(bb)
            if agent_dfg is not None:
                write_dfg_files(agent_dfg, output_dir)
                agent_generated += 1
                logger.info("BB %d: DFG generated by agent", bb.bb_id)
            else:
                logger.warning(
                    "BB %d: agent generation failed -- skipping", bb.bb_id,
                )
            continue

        # All instructions supported: build DFG via script
        dfg = build_dfg(bb, registry, source="script")
        script_generated += 1

        # Advisory agent check
        check_result = agent.check(dfg)
        if check_result.is_pass:
            agent_checked_pass += 1
        else:
            agent_checked_fail += 1
            logger.warning(
                "BB %d: agent check failed -- %s",
                bb.bb_id,
                "; ".join(issue.get("msg", "") for issue in check_result.issues),
            )

        write_dfg_files(dfg, output_dir)
        logger.info(
            "BB %d: %d nodes, %d edges (source=%s, check=%s)",
            bb.bb_id, len(dfg.nodes), len(dfg.edges),
            dfg.source, check_result.verdict,
        )

    # -- write summary ---------------------------------------------------------
    stats = {
        "total_blocks": len(blocks),
        "script_generated": script_generated,
        "agent_generated": agent_generated,
        "agent_checked_pass": agent_checked_pass,
        "agent_checked_fail": agent_checked_fail,
        "unsupported_instructions": sorted(unsupported_instructions),
    }
    write_summary(stats, output_dir / "summary.json")

    # -- print summary ---------------------------------------------------------
    print(f"DFG generation complete.")
    print(f"  Total blocks processed : {stats['total_blocks']}")
    print(f"  Script-generated DFGs  : {stats['script_generated']}")
    print(f"  Agent-generated DFGs   : {stats['agent_generated']}")
    print(f"  Agent check passed     : {stats['agent_checked_pass']}")
    print(f"  Agent check failed     : {stats['agent_checked_fail']}")
    if unsupported_instructions:
        print(f"  Unsupported mnemonics  : {', '.join(sorted(unsupported_instructions))}")
    print(f"  Output directory       : {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
