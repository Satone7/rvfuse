"""Connected subgraph enumeration from DFG node/edge data.

Given a DFG (nodes + edges), enumerates all connected subgraphs up to
a maximum node count. Uses seed-based expansion with canonical ordering
(seed = minimum node index) to guarantee each subgraph is produced once.
"""

from __future__ import annotations

from collections import defaultdict


def enumerate_subgraphs(
    dfg_data: dict,
    max_nodes: int = 8,
) -> list[frozenset[int]]:
    """Enumerate all connected subgraphs in a DFG.

    Args:
        dfg_data: DFG JSON dict with 'nodes' (list of {index, mnemonic, operands})
            and 'edges' (list of {src, dst, register}).
        max_nodes: Maximum number of nodes per subgraph (inclusive).

    Returns:
        List of frozensets, each containing node indices of a connected subgraph.
        Single-node subgraphs are always included. No duplicates.
    """
    nodes = dfg_data["nodes"]
    edges = dfg_data["edges"]

    if not nodes:
        return []

    # Build undirected adjacency (DFG edges are directional but connectivity
    # is undirected -- both endpoints share the dependency relationship).
    adj: dict[int, set[int]] = defaultdict(set)
    for e in edges:
        adj[e["src"]].add(e["dst"])
        adj[e["dst"]].add(e["src"])

    results: list[frozenset[int]] = []
    seen: set[frozenset[int]] = set()

    for seed in range(len(nodes)):
        # Only enumerate subgraphs where `seed` is the minimum index.
        # This guarantees each subgraph is generated exactly once.
        stack: list[frozenset[int]] = [frozenset([seed])]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            results.append(current)

            if len(current) >= max_nodes:
                continue

            # Collect all neighbors of the current subgraph that have
            # index > seed (to maintain canonical ordering).
            candidates: set[int] = set()
            for node in current:
                for nb in adj[node]:
                    if nb not in current and nb > seed:
                        candidates.add(nb)

            for nb in candidates:
                expanded = current | {nb}
                if expanded not in seen:
                    stack.append(expanded)

    return results
