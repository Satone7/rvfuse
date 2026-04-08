"""DFG builder: constructs a data-flow graph from a basic block.

Builds RAW (read-after-write) dependency edges by tracking which
instruction last wrote each register.
"""

from __future__ import annotations

from dfg.instruction import (
    BasicBlock,
    DFG,
    DFGEdge,
    DFGNode,
    ISARegistry,
    VectorConfig,
    _expand_grouping,
)


def _get_vec_config(bb: BasicBlock, idx: int) -> VectorConfig | None:
    """Get the effective VectorConfig for instruction at *idx*.

    Walks change_points to find the most recent config change before *idx*.
    Returns bb.vec_config if no change_points apply, or None if no config.
    """
    if bb.vec_config is None:
        return None
    config = bb.vec_config
    for cp_idx, cp_config in config.change_points:
        if cp_idx < idx:
            config = cp_config
    return config


def build_dfg(
    bb: BasicBlock,
    registry: ISARegistry,
    source: str = "script",
) -> DFG:
    """Build a DFG from a basic block using the given ISA registry.

    Args:
        bb: Basic block with instructions to analyse.
        registry: ISA registry providing register-flow rules per mnemonic.
        source: Tag indicating how the DFG was produced (default "script").

    Returns:
        A DFG containing one node per instruction and RAW dependency edges.
    """
    nodes: list[DFGNode] = []
    edges: list[DFGEdge] = []
    last_writer: dict[str, int] = {}

    for idx, insn in enumerate(bb.instructions):
        node = DFGNode(instruction=insn, index=idx)
        nodes.append(node)

        flow = registry.get_flow(insn.mnemonic)
        if flow is None:
            # Unknown instruction -- still record as a node, no edges.
            continue

        resolved = flow.resolve(insn.operands)

        # Apply LMUL register grouping expansion for vector registers.
        vec_config = _get_vec_config(bb, idx)
        resolved = _expand_grouping(resolved, vec_config)

        # Create edges for source registers (RAW dependencies).
        for reg in resolved.src_regs:
            if reg in ("zero", "x0"):
                continue
            if reg in last_writer and last_writer[reg] != idx:
                edges.append(DFGEdge(
                    src_index=last_writer[reg],
                    dst_index=idx,
                    register=reg,
                ))

        # Update last-writer map for destination registers.
        for reg in resolved.dst_regs:
            if reg in ("zero", "x0"):
                continue
            last_writer[reg] = idx

        # Wire config_regs (e.g. vl, vtype from VSETVLI) into the DFG so
        # that CSR side-effects appear as data-flow edges.
        for reg in resolved.config_regs:
            if reg not in ("zero", "x0"):
                last_writer[reg] = idx

    return DFG(bb=bb, nodes=nodes, edges=edges, source=source)
