"""Subgraph pattern model and normalization for fusion candidate discovery.

Converts a connected DFG subgraph (set of node indices) into a normalized
SubgraphPattern template:
  1. Extract edges within the subgraph
  2. Map concrete registers to operand roles via ISARegistry
  3. Compute topological layers (same-layer opcodes sorted alphabetically)
  4. Build a hashable template_key for cross-BB deduplication
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Register classification patterns (matching dfg/instruction.py)
_INTEGER_RE = re.compile(r"^(x\d+|zero|ra|sp|gp|tp|[ast]\d+|s\d+|jt?\d*)$")
_FLOAT_RE = re.compile(r"^(f\d+|ft\d+|fs\d+|fa\d+|fv\d+)$")


def classify_register(reg_name: str) -> str | None:
    """Return 'integer', 'float', or None for a register name."""
    if _INTEGER_RE.match(reg_name):
        return "integer"
    if _FLOAT_RE.match(reg_name):
        return "float"
    return None


def _classify_subgraph(
    node_indices: set[int],
    nodes: list[dict],
    registry,
) -> str:
    """Determine the dominant register class of a subgraph."""
    for idx in sorted(node_indices):
        mn = nodes[idx]["mnemonic"]
        flow = registry.get_flow(mn)
        if flow is None:
            continue
        resolved = flow.resolve(nodes[idx]["operands"])
        all_regs = resolved.dst_regs + resolved.src_regs
        if all_regs:
            cls = classify_register(all_regs[0])
            if cls:
                return cls
    return "unknown"


def _compute_node_layer(
    node_indices: set[int],
    intra_edges: list[dict],
) -> dict[int, int]:
    """Map each node index to its topological layer number."""
    remaining = set(node_indices)
    in_degree: dict[int, int] = {n: 0 for n in remaining}
    fwd: dict[int, set[int]] = {n: set() for n in remaining}
    for e in intra_edges:
        if e["src"] in remaining and e["dst"] in remaining:
            in_degree[e["dst"]] = in_degree.get(e["dst"], 0) + 1
            fwd[e["src"]].add(e["dst"])

    layer_map: dict[int, int] = {}
    layer_num = 0
    while remaining:
        layer_nodes = sorted(n for n in remaining if in_degree.get(n, 0) == 0)
        if not layer_nodes:
            layer_nodes = sorted(remaining)
        for n in layer_nodes:
            layer_map[n] = layer_num
            remaining.discard(n)
            for nb in fwd.get(n, set()):
                in_degree[nb] -= 1
        layer_num += 1

    return layer_map


def _topological_layers(
    node_indices: set[int],
    intra_edges: list[dict],
    nodes: list[dict],
) -> list[list[str]]:
    """Compute topological layers of the subgraph."""
    layer_map = _compute_node_layer(node_indices, intra_edges)
    layers_dict: dict[int, list[str]] = {}
    for idx, layer in layer_map.items():
        layers_dict.setdefault(layer, []).append(nodes[idx]["mnemonic"])
    return [sorted(opcodes) for _, opcodes in sorted(layers_dict.items())]


def _find_role(
    reg_name: str,
    resolved_regs: list[str],
    role_names: list[str],
) -> str | None:
    """Find the role name corresponding to a concrete register name."""
    role_map = dict(zip(resolved_regs, role_names))
    return role_map.get(reg_name)


def _normalize_edges(
    intra_edges: list[dict],
    node_indices: set[int],
    nodes: list[dict],
    layer_map: dict[int, int],
    registry,
) -> list[NormalizedEdge]:
    """Convert concrete DFG edges to role-abstracted NormalizedEdge list."""
    result: list[NormalizedEdge] = []
    for e in intra_edges:
        if e["src"] not in node_indices or e["dst"] not in node_indices:
            continue
        reg_name = e["register"]
        src_mn = nodes[e["src"]]["mnemonic"]
        dst_mn = nodes[e["dst"]]["mnemonic"]
        src_ops = nodes[e["src"]]["operands"]
        dst_ops = nodes[e["dst"]]["operands"]

        src_flow = registry.get_flow(src_mn)
        dst_flow = registry.get_flow(dst_mn)
        if src_flow is None or dst_flow is None:
            continue

        src_resolved = src_flow.resolve(src_ops)
        dst_resolved = dst_flow.resolve(dst_ops)

        src_role = _find_role(reg_name, src_resolved.dst_regs, src_flow.dst_regs)
        dst_role = _find_role(reg_name, dst_resolved.src_regs, dst_flow.src_regs)

        if src_role and dst_role:
            result.append(NormalizedEdge(
                src_layer=layer_map[e["src"]],
                src_opcode=src_mn,
                dst_layer=layer_map[e["dst"]],
                dst_opcode=dst_mn,
                src_role=src_role,
                dst_role=dst_role,
            ))
    return result


@dataclass
class NormalizedEdge:
    """A role-abstracted edge between two instructions in the pattern."""

    src_layer: int
    src_opcode: str
    dst_layer: int
    dst_opcode: str
    src_role: str
    dst_role: str


@dataclass
class SubgraphPattern:
    """A normalized fusible subgraph template.

    Attributes:
        topology: Topological layers of opcodes.
            e.g. [["flw", "flw"], ["fmadd.s"]]
        edges: Role-abstracted edges between instructions.
        register_class: "integer", "float", or "unknown".
        size: Number of instructions in the subgraph.
    """

    topology: list[list[str]]
    edges: list[NormalizedEdge]
    register_class: str
    size: int

    @property
    def template_key(self) -> tuple:
        """Hashable key for grouping identical patterns across BBs."""
        topo_tuple = tuple(tuple(layer) for layer in self.topology)
        edge_tuple = tuple(
            (e.src_layer, e.src_opcode, e.dst_layer, e.dst_opcode, e.src_role, e.dst_role)
            for e in sorted(self.edges, key=lambda e: (e.src_layer, e.dst_layer, e.src_role, e.dst_role))
        )
        return (topo_tuple, self.register_class, edge_tuple)


def normalize_subgraph(
    node_indices: frozenset[int] | set[int],
    dfg_data: dict,
    registry,
) -> SubgraphPattern:
    """Normalize a DFG subgraph into a SubgraphPattern template.

    Args:
        node_indices: Set of DFG node indices forming the subgraph.
        dfg_data: DFG JSON dict with 'nodes' and 'edges'.
        registry: ISA registry for register flow resolution.

    Returns:
        A SubgraphPattern with normalized topology, edges, and register class.

    Raises:
        ValueError: If any mnemonic is unknown in the registry.
    """
    node_set = set(node_indices)
    nodes = dfg_data["nodes"]
    all_edges = dfg_data["edges"]

    # Validate all mnemonics are known
    for idx in node_set:
        mn = nodes[idx]["mnemonic"]
        if registry.get_flow(mn) is None:
            raise ValueError(f"unknown instruction: {mn}")

    # Filter edges to those within the subgraph
    intra_edges = [
        e for e in all_edges
        if e["src"] in node_set and e["dst"] in node_set
    ]

    # Classify register class
    reg_class = _classify_subgraph(node_set, nodes, registry)

    # Compute topological layers and node-to-layer mapping
    layer_map = _compute_node_layer(node_set, intra_edges)
    layers = _topological_layers(node_set, intra_edges, nodes)

    # Normalize edges
    norm_edges = _normalize_edges(intra_edges, node_set, nodes, layer_map, registry)

    return SubgraphPattern(
        topology=layers,
        edges=norm_edges,
        register_class=reg_class,
        size=len(node_set),
    )
