"""Weighted-sum scoring for fusion candidate patterns.

Scores each candidate on three axes:
  - frequency:   how often the pattern occurs (log-normalized)
  - tightness:   data-dependency density within the chain
  - hardware:    ConstraintChecker verdict mapped to 0 / 0.5 / 1.0

The final score is the weighted sum of the three sub-scores and lies in
[0, 1].
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

from dfg.instruction import ISARegistry
from fusion.constraints import ConstraintChecker, ConstraintConfig, Verdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "frequency": 0.4,
    "tightness": 0.3,
    "hardware": 0.3,
}

# Bonus multiplier for longer chains (key: chain length in instructions).
_CHAIN_FACTOR: dict[int, float] = {
    2: 1.0,
    3: 1.2,
}


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class Scorer:
    """Score and rank fusion candidate patterns."""

    def __init__(
        self,
        registry: ISARegistry,
        max_frequency: int = 1,
        weights: dict[str, float] | None = None,
        config: ConstraintConfig | None = None,
    ) -> None:
        self._registry = registry
        self._max_frequency = max(1, max_frequency)
        self._weights = weights if weights is not None else dict(DEFAULT_WEIGHTS)
        self._checker = ConstraintChecker(registry, config=config)

    # -- sub-scores -----------------------------------------------------------

    def _freq_score(self, frequency: int) -> float:
        """Log-normalized frequency in [0, 1].

        Uses ``log(1 + freq) / log(1 + max_freq)`` so that a pattern with
        ``max_frequency`` occurrences scores exactly 1.0.
        """
        if frequency <= 0:
            return 0.0
        return math.log(1 + frequency) / math.log(1 + self._max_frequency)

    def _tight_score(
        self,
        chain_registers: list[list[str]],
        length: int,
    ) -> float:
        """Data-dependency tightness in [0, 1].

        *raw_density* is the number of chain edges divided by
        ``(length - 1)``, the maximum possible edges for a chain of that
        size.  A bonus *chain_factor* is applied for longer chains.
        The result is capped at 1.0.
        """
        max_edges = max(1, length - 1)
        raw_density = len(chain_registers) / max_edges
        factor = _CHAIN_FACTOR.get(length, _CHAIN_FACTOR.get(max(_CHAIN_FACTOR), 1.0))
        return min(raw_density * factor, 1.0)

    @staticmethod
    def _hw_score(verdict: Verdict) -> float:
        """Map a constraint verdict to a numeric score.

        feasible   -> 1.0
        constrained -> 0.5
        infeasible  -> 0.0
        """
        mapping: dict[str, float] = {
            "feasible": 1.0,
            "constrained": 0.5,
            "infeasible": 0.0,
        }
        return mapping.get(verdict.status, 0.0)

    # -- single pattern -------------------------------------------------------

    def score_pattern(self, pattern: dict[str, Any]) -> dict[str, Any]:
        """Score a single fusion candidate pattern.

        Args:
            pattern: Dict with keys ``opcodes``, ``register_class``,
                ``chain_registers``, ``total_frequency``, and
                ``occurrence_count``.

        Returns:
            A result dict containing the final ``score``, sub-score
            ``score_breakdown``, ``tightness`` details, and ``hardware``
            verdict.
        """
        freq = pattern.get("total_frequency", 0)
        occurrences = pattern.get("occurrence_count", 0)
        chain_regs = pattern.get("chain_registers", [])
        length = len(pattern.get("opcodes", []))

        # Sub-scores
        freq_score = self._freq_score(freq)
        tight_score = self._tight_score(chain_regs, length)

        verdict = self._checker.check(pattern)
        hw_score = self._hw_score(verdict)

        # Weighted sum.  Infeasible patterns always score 0.0 regardless of
        # other sub-scores -- a hardware veto means the pattern cannot be
        # fused.
        w = self._weights
        if hw_score == 0.0:
            final_score = 0.0
        else:
            final_score = (
                w.get("frequency", 0.0) * freq_score
                + w.get("tightness", 0.0) * tight_score
                + w.get("hardware", 0.0) * hw_score
            )

        return {
            "pattern": {
                "opcodes": pattern["opcodes"],
                "register_class": pattern.get("register_class"),
                "chain_registers": chain_regs,
            },
            "input_frequency": freq,
            "input_occurrence_count": occurrences,
            "tightness": {
                "raw_density": len(chain_regs) / max(1, length - 1) if length > 1 else 0.0,
                "chain_factor": _CHAIN_FACTOR.get(length, _CHAIN_FACTOR.get(max(_CHAIN_FACTOR), 1.0)),
                "score": tight_score,
            },
            "hardware": {
                "status": verdict.status,
                "reasons": verdict.reasons,
                "violations": verdict.violations,
            },
            "score": final_score,
            "score_breakdown": {
                "freq_score": freq_score,
                "tight_score": tight_score,
                "hw_score": hw_score,
            },
        }

    # -- batch ----------------------------------------------------------------

    def score_patterns(
        self,
        patterns: list[dict[str, Any]],
        top: int | None = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Score, rank, and filter a list of patterns.

        Results are sorted by score descending.  Optional *top* limits the
        output to the N highest-scoring patterns.  *min_score* excludes
        patterns whose score falls below the threshold.

        Args:
            patterns: List of pattern dicts (same shape as ``score_pattern``).
            top: Maximum number of results to return.
            min_score: Minimum score threshold.

        Returns:
            A list of result dicts, sorted descending by score.
        """
        results = [self.score_pattern(p) for p in patterns]
        results.sort(key=lambda r: r["score"], reverse=True)

        # Apply min_score filter
        filtered = [r for r in results if r["score"] >= min_score]

        # Apply top-N
        if top is not None:
            filtered = filtered[:top]

        return filtered


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def score(
    catalog_path: str | Path,
    registry: ISARegistry,
    output_path: str | Path | None = None,
    top: int | None = None,
    min_score: float = 0.0,
    weights: dict[str, float] | None = None,
    config: ConstraintConfig | None = None,
    feasibility_only: bool = False,
) -> list[dict[str, Any]]:
    """Load a pattern catalog JSON, score all patterns, write results.

    Args:
        catalog_path: Path to the JSON catalog produced by the miner.
        registry: ISA registry with encoding metadata.
        output_path: Optional path to write scored results JSON.
        top: Keep only the top N patterns.
        min_score: Discard patterns scoring below this threshold.
        weights: Optional custom scoring weights.
        config: Optional constraint configuration.
        feasibility_only: Only check constraints, skip scoring.

    Returns:
        The list of scored result dicts.
    """
    from datetime import datetime, timezone

    catalog_path = Path(catalog_path)
    with open(catalog_path) as f:
        catalog = json.load(f)

    patterns = catalog.get("patterns", [])
    if not patterns:
        return []

    max_freq = max(p.get("total_frequency", 0) for p in patterns)

    if feasibility_only:
        checker = ConstraintChecker(registry, config=config)
        results = []
        for p in patterns:
            verdict = checker.check(p)
            results.append({
                "pattern": {
                    "opcodes": p["opcodes"],
                    "register_class": p.get("register_class"),
                    "chain_registers": p.get("chain_registers", []),
                },
                "input_frequency": p.get("total_frequency", 0),
                "input_occurrence_count": p.get("occurrence_count", 0),
                "hardware": {
                    "status": verdict.status,
                    "reasons": verdict.reasons,
                    "violations": verdict.violations,
                },
                "score": 0.0 if verdict.status == "infeasible" else (
                    0.5 if verdict.status == "constrained" else 1.0),
            })
        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        if top is not None:
            results = results[:top]
    else:
        scorer = Scorer(registry, max_frequency=max_freq, weights=weights, config=config)
        results = scorer.score_patterns(patterns, top=top, min_score=min_score)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "source_pattern_count": len(patterns),
            "candidate_count": len(results),
            "candidates": results,
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
            f.write("\n")

    logger.info(
        "Scored %d patterns -> %d candidates (top=%s, min_score=%s)",
        len(patterns), len(results), top, min_score,
    )

    return results


if __name__ == "__main__":
    # Minimal CLI: python -m fusion.scorer <catalog.json> [output.json] [--top N] [--min-score F]
    import argparse

    parser = argparse.ArgumentParser(description="Score fusion candidate patterns")
    parser.add_argument("catalog", help="Path to pattern catalog JSON")
    parser.add_argument("output", nargs="?", help="Path to write scored results JSON")
    parser.add_argument("--top", type=int, default=None, help="Keep only top N patterns")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum score threshold")
    args = parser.parse_args()

    scored = score(args.catalog, args.output, top=args.top, min_score=args.min_score)
    print(f"Scored {len(scored)} patterns (from catalog).")
    if scored:
        print(f"  Top score: {scored[0]['score']:.4f}")
        print(f"  Min score:  {scored[-1]['score']:.4f}")
