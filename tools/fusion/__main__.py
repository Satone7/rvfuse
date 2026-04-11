"""CLI entry point for fusion pattern mining and scoring.

Usage:
    python -m tools.fusion discover --dfg-dir <dir> --report <json> --output <json>
    python -m tools.fusion score --catalog <json> --output <json>
    python -m tools.fusion validate --opcode N --funct3 M --funct7 K --reg-class X
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

from fusion.constraints import ConstraintConfig
from fusion.miner import mine


_ISA_MODULES: dict[str, tuple[str, str]] = {
    "I": ("dfg.isadesc.rv64i", "build_registry"),
    "F": ("dfg.isadesc.rv64f", "build_registry"),
    "M": ("dfg.isadesc.rv64m", "build_registry"),
    "V": ("dfg.isadesc.rv64v", "build_registry"),
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


def _build_constraint_config(args: argparse.Namespace) -> ConstraintConfig:
    """Build ConstraintConfig from CLI args.

    Priority: --enable/disable flags > --constraints-config file > defaults
    """
    config = ConstraintConfig.defaults()

    if args.constraints_config:
        config = ConstraintConfig.from_file(args.constraints_config)

    for name in (args.enable_constraint or []):
        if name in ConstraintConfig.ALL_CONSTRAINTS:
            config.enabled[name] = True
        else:
            logging.warning("Unknown constraint '%s' in --enable-constraint", name)

    for name in (args.disable_constraint or []):
        if name in ConstraintConfig.ALL_CONSTRAINTS:
            config.enabled[name] = False
        else:
            logging.warning("Unknown constraint '%s' in --disable-constraint", name)

    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tools.fusion",
        description="Discover fusible instruction patterns from DFG output.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["discover", "score", "validate", "scheme"],
        default=None,
        help="Command to run",
    )
    parser.add_argument(
        "--dfg-dir",
        required=False,
        default=None,
        type=Path,
        help="Directory containing DFG JSON files (required for discover)",
    )
    parser.add_argument(
        "--report",
        required=False,
        default=None,
        type=Path,
        help="Path to BBV hotspot JSON report (required for discover)",
    )
    parser.add_argument(
        "--output",
        required=False,
        default=None,
        type=Path,
        help="Output path for pattern catalog JSON (required for discover and score)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top patterns to include (default: 20)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=8,
        dest="max_nodes",
        help="Maximum subgraph size to enumerate (default: 8)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=2,
        dest="min_size",
        help="Minimum subgraph size to include in output (default: 2)",
    )
    parser.add_argument("--catalog", type=Path, default=None,
        help="Path to Feature 1 pattern catalog JSON (required for score)")
    parser.add_argument("--min-score", type=float, default=0.0,
        help="Minimum score threshold (default: 0.0)")
    parser.add_argument("--feasibility-only", action="store_true", default=False,
        help="Only check constraints, skip scoring")
    parser.add_argument("--weight-freq", type=float, default=None, dest="weight_freq",
        help="Weight for frequency score (default: 0.4)")
    parser.add_argument("--weight-tight", type=float, default=None, dest="weight_tight",
        help="Weight for tightness score (default: 0.3)")
    parser.add_argument("--weight-hw", type=float, default=None, dest="weight_hw",
        help="Weight for hardware score (default: 0.3)")
    parser.add_argument(
        "--isa",
        default="I,F,M,V",
        help="Comma-separated ISA extensions (default: I,F,M,V)",
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
    parser.add_argument(
        "--opcode",
        type=lambda x: int(x, 0),
        default=None,
        help="Opcode value (hex or decimal, required for validate)",
    )
    parser.add_argument(
        "--funct3",
        type=lambda x: int(x, 0),
        default=None,
        help="Funct3 value (hex or decimal, optional)",
    )
    parser.add_argument(
        "--funct7",
        type=lambda x: int(x, 0),
        default=None,
        help="Funct7 value (hex or decimal, optional)",
    )
    parser.add_argument(
        "--reg-class",
        choices=["integer", "float", "vector"],
        default="integer",
        help="Register class for the pattern (default: integer)",
    )
    parser.add_argument(
        "--constraints-config",
        type=Path,
        default=None,
        help="Path to JSON file with constraint enable/disable config",
    )
    parser.add_argument(
        "--enable-constraint",
        action="append",
        default=None,
        help="Enable a specific constraint (repeatable)",
    )
    parser.add_argument(
        "--disable-constraint",
        action="append",
        default=None,
        help="Disable a specific constraint (repeatable)",
    )
    parser.add_argument(
        "--list-constraints",
        action="store_true",
        default=False,
        help="List all constraints with their default status and descriptions",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Path to fusion_candidates.json (required for scheme command)",
    )
    parser.add_argument(
        "--scheme-dir",
        type=Path,
        default=None,
        help="Output directory for scheme files (default: output/fusion_schemes/)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel Claude processes for scheme generation (default: 4)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s: %(message)s",
    )

    registry = load_isa_registry(args.isa)

    # Handle --list-constraints
    if args.list_constraints:
        config = _build_constraint_config(args)
        print("\nConstraint Configuration:")
        print("-" * 70)
        for name, meta in ConstraintConfig.ALL_CONSTRAINTS.items():
            category, default, desc = meta
            status = "ON" if config.enabled[name] else "OFF"
            print(f"{name:30} {category:5} {status:3}  {desc}")
        print("-" * 70)
        return

    if args.command is None:
        sys.exit("error: a command is required (discover, score, validate, scheme). Run with --help for usage.")

    if args.command == "scheme":
        if not args.candidates:
            sys.exit("--candidates is required for scheme command")
        if not args.candidates.exists():
            sys.exit(f"Candidates file not found: {args.candidates}")

        from fusion.scheme_batch import run_scheme_batch

        scheme_dir = args.scheme_dir or Path("output/fusion_schemes")
        effective_model = args.model or "opus"

        run_scheme_batch(
            candidates_json=args.candidates,
            output_dir=scheme_dir,
            top=args.top,
            model=effective_model,
            workers=args.workers,
            timeout=300,
            registry=registry,
        )
        return

    if args.command == "validate":
        if args.opcode is None:
            print("Usage error: --opcode is required for validate command", file=sys.stderr)
            print("Run 'python -m tools.fusion validate --help' for usage.", file=sys.stderr)
            sys.exit(2)

        from fusion.scheme_validator import validate_encoding
        import json

        result = validate_encoding(
            opcode=args.opcode,
            funct3=args.funct3,
            funct7=args.funct7,
            reg_class=args.reg_class,
            registry=registry,
        )

        output = {
            "passed": result.passed,
            "conflicts": result.conflicts,
            "warnings": result.warnings,
            "suggested_alternatives": result.suggested_alternatives,
        }
        print(json.dumps(output, indent=2))
        return

    if args.command == "score":
        if not args.catalog:
            sys.exit("--catalog is required for score command")
        if not args.output:
            sys.exit("--output is required for score command")

        from fusion.scorer import score as run_score

        config = _build_constraint_config(args)

        weights = None
        if args.weight_freq is not None or args.weight_tight is not None or args.weight_hw is not None:
            weights = {
                "frequency": args.weight_freq if args.weight_freq is not None else 0.4,
                "tightness": args.weight_tight if args.weight_tight is not None else 0.3,
                "hardware": args.weight_hw if args.weight_hw is not None else 0.3,
            }

        candidates = run_score(
            catalog_path=args.catalog, registry=registry, output_path=args.output,
            top=args.top, min_score=args.min_score, weights=weights,
            config=config,
            feasibility_only=args.feasibility_only,
        )

        feasible = sum(1 for c in candidates if c.get("hardware", {}).get("status") == "feasible")
        constrained = sum(1 for c in candidates if c.get("hardware", {}).get("status") == "constrained")
        infeasible = sum(1 for c in candidates if c.get("hardware", {}).get("status") == "infeasible")

        print(f"\nFusion Candidate Scoring Results")
        print(f"  Candidates: {len(candidates)} (feasible={feasible}, constrained={constrained}, infeasible={infeasible})")
        if candidates:
            top = candidates[0]
            print(f"  Top: {' → '.join(top['pattern']['opcodes'])} "
                  f"(score={top['score']:.4f}, {top['hardware']['status']})")
        print(f"  Output: {args.output}")
        return

    # Validate discover-specific arguments
    if not args.dfg_dir:
        sys.exit("--dfg-dir is required for discover command")
    if not args.report:
        sys.exit("--report is required for discover command")
    if not args.output:
        sys.exit("--output is required for discover command")

    patterns = mine(
        dfg_dir=args.dfg_dir,
        hotspot_path=args.report,
        registry=registry,
        output_path=args.output,
        top=args.top,
        max_nodes=args.max_nodes,
        min_size=args.min_size,
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
        top = patterns[0]
        top_ops = " + ".join(mn for layer in top.get("topology", []) for mn in layer)
        print(f"  Top pattern: {top_ops} "
              f"(frequency: {top['total_frequency']:,}, size: {top['size']})")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
