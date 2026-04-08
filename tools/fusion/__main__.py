"""CLI entry point for fusion pattern mining.

Usage:
    python -m tools.fusion discover --dfg-dir <dir> --report <json> --output <json>
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

# Ensure tools/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dfg.instruction import ISARegistry

from fusion.miner import mine


_ISA_MODULES: dict[str, tuple[str, str]] = {
    "I": ("dfg.isadesc.rv64i", "build_registry"),
    "F": ("dfg.isadesc.rv64f", "build_registry"),
    "M": ("dfg.isadesc.rv64m", "build_registry"),
}


def load_isa_registry(extensions: str) -> ISARegistry:
    """Build an ISA registry from comma-separated extension names."""
    registry = ISARegistry()
    for ext in extensions.split(","):
        ext = ext.strip().upper()
        if ext in _ISA_MODULES:
            module_path, func_name = _ISA_MODULES[ext]
            mod = importlib.import_module(module_path)
            builder = getattr(mod, func_name)
            builder(registry)
        else:
            logging.warning("Unknown ISA extension '%s' -- skipping", ext)
    return registry


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tools.fusion",
        description="Discover fusible instruction patterns from DFG output.",
    )
    parser.add_argument(
        "command",
        choices=["discover"],
        help="Command to run (currently: discover)",
    )
    parser.add_argument(
        "--dfg-dir",
        required=True,
        type=Path,
        help="Directory containing DFG JSON files",
    )
    parser.add_argument(
        "--report",
        required=True,
        type=Path,
        help="Path to BBV hotspot JSON report",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for pattern catalog JSON",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top patterns to include (default: 20)",
    )
    parser.add_argument(
        "--isa",
        default="I,F,M",
        help="Comma-separated ISA extensions (default: I,F,M)",
    )
    parser.add_argument(
        "--no-agent",
        action="store_true",
        default=False,
        help="Skip agent analysis, run miner only",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Claude model name for agent analysis",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s: %(message)s",
    )

    registry = load_isa_registry(args.isa)

    patterns = mine(
        dfg_dir=args.dfg_dir,
        hotspot_path=args.report,
        registry=registry,
        output_path=args.output,
        top=args.top,
    )

    # Agent analysis
    if not args.no_agent and patterns:
        from fusion.agent import run_agent
        run_agent(
            patterns=patterns,
            output_path=args.output,
            model=args.model,
        )

    # Text summary to stdout
    print(f"\nFusion Pattern Mining Results")
    print(f"  Patterns found: {len(patterns)}")
    if patterns:
        print(f"  Top pattern: {' → '.join(patterns[0]['opcodes'])} "
              f"(frequency: {patterns[0]['total_frequency']:,})")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
