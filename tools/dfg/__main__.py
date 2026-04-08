"""CLI entry point for the DFG generation module.

Usage:
    python -m tools.dfg --disas <file.disas> [options]
"""

from __future__ import annotations

import argparse
import fcntl
import importlib
import io
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Ensure the tools/ directory is on sys.path so that `dfg.` imports resolve
# when running as `python -m tools.dfg` from the worktree root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dfg.agent import AgentDispatcher
from dfg.dfg import build_dfg
from dfg.instruction import ISARegistry
from dfg.output import convert_dot_to_png, write_dfg_files, write_summary
from dfg.filter import select_addresses
from dfg.parser import parse_disas

logger = logging.getLogger("dfg")

# Mapping from ISA extension name (uppercase) to (module path, builder function).
_ISA_MODULES: dict[str, tuple[str, str]] = {
    "I": ("dfg.isadesc.rv64i", "build_registry"),
    "F": ("dfg.isadesc.rv64f", "build_registry"),
    "M": ("dfg.isadesc.rv64m", "build_registry"),
    "V": ("dfg.isadesc.rv64v", "build_registry"),
}

# Max debug log size before rotation (100 MB).
_DEBUG_LOG_MAX_BYTES = 100_000_000


@dataclass
class ProcessResult:
    """Result of processing a single basic block."""
    bb_id: int
    source: str  # "script", "agent", or "none"
    agent_check_result: str  # "pass", "fail", or "none"
    node_count: int
    edge_count: int
    unsupported: list[str]
    debug_log: str = ""  # buffered debug log text for this BB


class _BbLogBuffer:
    """Context manager that captures all DEBUG+ logs in-memory for one BB.

    Usage::

        with _BbLogBuffer() as buf:
            ...  # processing
        text = buf.text   # all log output as a string
    """

    def __init__(self) -> None:
        self._buffer = io.StringIO()
        self._handler: logging.StreamHandler | None = None

    def __enter__(self) -> _BbLogBuffer:
        self._handler = logging.StreamHandler(self._buffer)
        self._handler.setLevel(logging.DEBUG)
        self._handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        logging.getLogger().addHandler(self._handler)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
            self._handler.close()

    @property
    def text(self) -> str:
        return self._buffer.getvalue()


def _flush_bb_log(log_path: Path, text: str) -> None:
    """Append *text* to *log_path* under an exclusive file lock.

    If the file exceeds ``_DEBUG_LOG_MAX_BYTES`` after writing, it is
    rotated (renamed to ``debug.log.1``, ``debug.log.2``, etc.).
    Rotation is performed after releasing the lock and closing the file.
    """
    if not text:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    need_rotation = False
    with open(log_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
            f.flush()
            if f.tell() > _DEBUG_LOG_MAX_BYTES:
                need_rotation = True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    if need_rotation:
        _rotate_log(log_path)


def _rotate_log(log_path: Path) -> None:
    """Rotate log files: debug.log -> debug.log.1 -> debug.log.2 ..."""
    n = 1
    while log_path.with_suffix(f".log.{n}").exists():
        n += 1
    for i in range(n, 0, -1):
        src = log_path.with_suffix(f".log.{i - 1}") if i > 1 else log_path
        dst = log_path.with_suffix(f".log.{i}")
        try:
            os.rename(src, dst)
        except FileNotFoundError:
            pass


def process_single_bb(
    bb,
    registry: ISARegistry,
    agent: AgentDispatcher,
    output_dir: Path,
    buffer_logs: bool = False,
) -> ProcessResult:
    """Process a single basic block through the DFG pipeline.

    When *buffer_logs* is True, all log output for this BB is captured
    in memory and returned as ``ProcessResult.debug_log`` instead of
    going to the file handler.  The caller is responsible for flushing
    the buffered text to the shared log file under a file lock.
    """
    log = logging.getLogger("dfg")

    if buffer_logs:
        buf = _BbLogBuffer()
        with buf:
            result = _process_bb_impl(bb, registry, agent, output_dir, log)
        result.debug_log = buf.text
        return result
    else:
        return _process_bb_impl(bb, registry, agent, output_dir, log)


def _process_bb_impl(
    bb,
    registry: ISARegistry,
    agent: AgentDispatcher,
    output_dir: Path,
    log: logging.Logger,
) -> ProcessResult:
    """Core BB processing logic."""
    log.debug("--- BB %d start %s ---", bb.bb_id, datetime.now().isoformat())
    log.debug("BB %d: %d instruction(s)", bb.bb_id, len(bb.instructions))
    for i, insn in enumerate(bb.instructions):
        log.debug("  [%d] 0x%x: %s %s", i, insn.address, insn.mnemonic, insn.operands)

    unknown = [
        insn.mnemonic for insn in bb.instructions
        if not registry.is_known(insn.mnemonic)
    ]

    if unknown:
        log.warning(
            "BB %d: %d unsupported instruction(s): %s -- trying agent",
            bb.bb_id, len(unknown), ", ".join(unknown),
        )
        agent_dfg = agent.generate(bb)
        if agent_dfg is not None:
            write_dfg_files(agent_dfg, output_dir)
            log.info("BB %d: DFG generated by agent", bb.bb_id)
            return ProcessResult(
                bb_id=bb.bb_id,
                source="agent",
                agent_check_result="none",
                node_count=len(agent_dfg.nodes),
                edge_count=len(agent_dfg.edges),
                unsupported=unknown,
            )
        else:
            log.warning("BB %d: agent generation failed -- skipping", bb.bb_id)
            return ProcessResult(
                bb_id=bb.bb_id,
                source="none",
                agent_check_result="none",
                node_count=0,
                edge_count=0,
                unsupported=unknown,
            )

    dfg = build_dfg(bb, registry, source="script")
    log.debug(
        "BB %d: DFG built — %d nodes, %d edges",
        bb.bb_id, len(dfg.nodes), len(dfg.edges),
    )
    for edge in dfg.edges:
        log.debug(
            "BB %d: edge %d -> %d (register=%s)",
            bb.bb_id, edge.src_index, edge.dst_index, edge.register,
        )

    check_result = agent.check(dfg)
    if check_result.is_pass:
        agent_checked = "pass"
    else:
        agent_checked = "fail"
        log.warning(
            "BB %d: agent check failed -- %s",
            bb.bb_id,
            "; ".join(issue.get("msg", "") for issue in check_result.issues),
        )

    write_dfg_files(dfg, output_dir)
    log.info(
        "BB %d: %d nodes, %d edges (source=%s, check=%s)",
        bb.bb_id, len(dfg.nodes), len(dfg.edges),
        dfg.source, check_result.verdict,
    )

    return ProcessResult(
        bb_id=bb.bb_id,
        source="script",
        agent_check_result=agent_checked,
        node_count=len(dfg.nodes),
        edge_count=len(dfg.edges),
        unsupported=[],
    )


def _process_bb_worker(
    bb_id: int,
    vaddr: int,
    instructions: list[tuple[int, str, str, str]],
    isa_extensions: str,
    no_agent: bool,
    model: str | None,
    output_dir_str: str,
) -> ProcessResult:
    """Top-level worker for ProcessPoolExecutor.

    Buffers all logs in memory so they can be flushed atomically by
    the main process under a file lock — prevents interleaving.
    """
    from dfg.instruction import Instruction, BasicBlock

    # Set up minimal logging in the worker process (no file handlers).
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(logging.NullHandler())
    root.setLevel(logging.DEBUG)

    bb = BasicBlock(
        bb_id=bb_id,
        vaddr=vaddr,
        instructions=[
            Instruction(address=a, mnemonic=m, operands=o, raw_line=r)
            for a, m, o, r in instructions
        ],
    )
    registry = load_isa_registry(isa_extensions)
    agent = AgentDispatcher(enabled=not no_agent, model=model)
    output_dir = Path(output_dir_str)
    return process_single_bb(bb, registry, agent, output_dir, buffer_logs=True)


def parse_args(argv: list[str] | None = None):
    """Build the argument parser, parse *argv*, validate, and return namespace."""
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
        "--report",
        type=Path,
        default=None,
        help="Path to analyze_bbv JSON hotspot report for selective BB filtering",
    )
    parser.add_argument(
        "--coverage",
        type=int,
        default=None,
        help="Coverage threshold %% (e.g. 80 = include BBs up to 80%% cumulative execution)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top-ranked BBs to include when using --report (default: 20)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Claude model name passed to agent (e.g. claude-opus-4-6)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable full-pipeline detailed logging to rotating files",
    )
    parser.add_argument(
        "--jobs", "-j",
        type=int,
        default=1,
        help="Number of parallel processes for BB handling (default: 1)",
    )
    parsed = parser.parse_args(argv) if argv is not None else parser.parse_args()
    if parsed.report is not None and parsed.bb_filter is not None:
        parser.error("--report and --bb-filter are mutually exclusive")
    if parsed.coverage is not None and parsed.report is None:
        parser.error("--coverage requires --report")
    return parsed


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
    args = parse_args(argv)

    if args.jobs < 1:
        print("ERROR: --jobs must be >= 1", file=sys.stderr)
        return 1

    # -- validate input --------------------------------------------------------
    disas_path: Path = args.disas
    if not disas_path.is_file():
        print(f"ERROR: Input file not found: {disas_path}", file=sys.stderr)
        return 1

    # -- output directory (needed before logging setup in debug mode) ----------
    output_dir: Path = args.output_dir or disas_path.resolve().parent / "dfg"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- logging ---------------------------------------------------------------
    if args.debug:
        console_level = logging.INFO
    else:
        console_level = logging.DEBUG if args.verbose else logging.INFO

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root_logger.addHandler(console_handler)

    # In debug mode, per-BB logs are buffered in memory and flushed atomically
    # under a file lock — no RotatingFileHandler needed.
    debug_log_path: Path | None = None
    if args.debug:
        debug_log_path = output_dir / "debug.log"

    # -- build ISA registry ----------------------------------------------------
    registry = load_isa_registry(args.isa)

    # -- parse -----------------------------------------------------------------
    blocks = parse_disas(disas_path)
    if not blocks:
        logger.error("No basic blocks found in %s", disas_path)
        return 1
    logger.info("Parsed %d basic block(s) from %s", len(blocks), disas_path)

    # -- report-driven filtering -----------------------------------------------
    blocks_from_report = 0
    blocks_matched = 0
    blocks_skipped_not_in_disas = 0
    filter_mode = "none"
    filter_value = None

    if args.report is not None:
        try:
            selected_addrs = select_addresses(
                args.report,
                top_n=args.top,
                coverage=args.coverage,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Failed to load report: %s", exc)
            return 1
        blocks_from_report = len(selected_addrs)
        if not selected_addrs:
            logger.error("Report contains no blocks to process")
            return 1
        blocks_skipped_not_in_disas = blocks_from_report - len(
            {bb.vaddr for bb in blocks if bb.vaddr in selected_addrs}
        )
        blocks = [bb for bb in blocks if bb.vaddr in selected_addrs]
        blocks_matched = len(blocks)
        if not blocks:
            logger.error(
                "None of the %d addresses from the report matched BBs in %s",
                blocks_from_report, disas_path,
            )
            return 1
        if blocks_skipped_not_in_disas > 0:
            logger.warning(
                "%d address(es) from report not found in .disas — skipped",
                blocks_skipped_not_in_disas,
            )
        filter_mode = "coverage" if args.coverage is not None else "top"
        filter_value = args.coverage if args.coverage is not None else args.top
        logger.info(
            "Report filter (%s=%s): %d/%d BBs matched",
            filter_mode, filter_value, blocks_matched, blocks_from_report,
        )

    # -- bb filter (debugging, single BB) --------------------------------------
    elif args.bb_filter is not None:
        blocks = [bb for bb in blocks if bb.bb_id == args.bb_filter]
        if not blocks:
            logger.error("No basic block with ID %d found", args.bb_filter)
            return 1

    # -- agent -----------------------------------------------------------------
    agent = AgentDispatcher(enabled=not args.no_agent, model=args.model)

    # -- process BBs -----------------------------------------------------------
    results: list[ProcessResult] = []

    if args.jobs > 1 and len(blocks) > 1:
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Processing BBs...", total=len(blocks))
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                futures = {
                    executor.submit(
                        _process_bb_worker,
                        bb.bb_id, bb.vaddr,
                        [(ins.address, ins.mnemonic, ins.operands, ins.raw_line) for ins in bb.instructions],
                        args.isa,
                        args.no_agent, args.model,
                        str(output_dir),
                    ): bb
                    for bb in blocks
                }
                for future in as_completed(futures):
                    result = future.result()
                    # Flush buffered debug log under file lock.
                    if debug_log_path is not None:
                        _flush_bb_log(debug_log_path, result.debug_log)
                    logger.info(
                        "BB %d: %d nodes, %d edges (source=%s)",
                        result.bb_id, result.node_count, result.edge_count, result.source,
                    )
                    results.append(result)
                    progress.update(task, advance=1)

        results.sort(key=lambda r: r.bb_id)
    else:
        for bb in blocks:
            result = process_single_bb(
                bb, registry, agent, output_dir,
                buffer_logs=(debug_log_path is not None),
            )
            if debug_log_path is not None:
                _flush_bb_log(debug_log_path, result.debug_log)
            results.append(result)

    # -- collect stats ---------------------------------------------------------
    unsupported_instructions: set[str] = set()
    script_generated = 0
    agent_generated = 0
    agent_checked_pass = 0
    agent_checked_fail = 0
    for r in results:
        if r.source == "script":
            script_generated += 1
        elif r.source == "agent":
            agent_generated += 1
        if r.agent_check_result == "pass":
            agent_checked_pass += 1
        elif r.agent_check_result == "fail":
            agent_checked_fail += 1
        unsupported_instructions.update(r.unsupported)

    # -- convert DOT to PNG ----------------------------------------------------
    dot_dir = output_dir / "dot"
    png_dir = output_dir / "png"
    png_count = 0
    if dot_dir.exists():
        png_count = convert_dot_to_png(dot_dir, png_dir)

    # -- write summary ---------------------------------------------------------
    stats = {
        "filter_mode": filter_mode,
        "filter_value": filter_value,
        "blocks_from_report": blocks_from_report,
        "blocks_matched": blocks_matched,
        "blocks_skipped_not_in_disas": blocks_skipped_not_in_disas,
        "total_blocks": len(blocks),
        "script_generated": script_generated,
        "agent_generated": agent_generated,
        "agent_checked_pass": agent_checked_pass,
        "agent_checked_fail": agent_checked_fail,
        "unsupported_instructions": sorted(unsupported_instructions),
        "png_generated": png_count,
        "jobs_used": args.jobs,
    }
    write_summary(stats, output_dir / "summary.json")

    # -- print summary ---------------------------------------------------------
    print("DFG generation complete.")
    print(f"  Total blocks processed : {stats['total_blocks']}")
    print(f"  Script-generated DFGs  : {stats['script_generated']}")
    print(f"  Agent-generated DFGs   : {stats['agent_generated']}")
    print(f"  Agent check passed     : {stats['agent_checked_pass']}")
    print(f"  Agent check failed     : {stats['agent_checked_fail']}")
    if unsupported_instructions:
        print(f"  Unsupported mnemonics  : {', '.join(sorted(unsupported_instructions))}")
    if png_count > 0:
        print(f"  PNG images generated   : {png_count}")
    print(f"  Output directory       : {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
