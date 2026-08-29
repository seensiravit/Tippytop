"""Convergence state for validation-driven autonomous search."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConvergenceTracker:
    epsilon: float
    patience: int
    best: float
    stagnant: int = 0

    def observe(self, score: float) -> tuple[bool, bool]:
        """Return ``(is_new_best, is_significant)`` and update stagnation."""
        previous_best = self.best
        is_new_best = score > previous_best
        is_significant = score > previous_best + self.epsilon
        if is_new_best:
            self.best = score
        # Small new bests are retained, but only gains above epsilon reset patience.
        if is_significant:
            self.stagnant = 0
        else:
            self.stagnant += 1
        return is_new_best, is_significant

    @property
    def converged(self) -> bool:
        return self.stagnant >= self.patience
