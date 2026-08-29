"""Stop conditions: convergence (spec ε/N) and budget (iterations / wall-clock)."""
from __future__ import annotations
import time
from typing import Callable


class ConvergenceTracker:
    """Converged when best valid primary gained <= eps over the last n improvements.

    We track the best-so-far trajectory; only iterations that produced a valid
    score advance it. Converged once we have >= n+1 points and the total gain
    across the last n steps is <= eps.
    """

    def __init__(self, eps: float = 0.002, n: int = 3):
        self.eps, self.n = eps, n
        self.best_trajectory: list[float] = []

    def update(self, valid_primary: float | None) -> None:
        if valid_primary is None:
            return
        prev = self.best_trajectory[-1] if self.best_trajectory else float("-inf")
        self.best_trajectory.append(max(prev, valid_primary))

    def converged(self) -> tuple[bool, str]:
        traj = self.best_trajectory
        if len(traj) < self.n + 1:
            return False, ""
        if traj[-1] - traj[-1 - self.n] <= self.eps:
            return True, "converged"
        return False, ""


class BudgetManager:
    def __init__(self, max_iters: int, wall_budget_s: float,
                 clock: Callable[[], float] = time.time):
        self.max_iters = max_iters
        self.wall_budget_s = wall_budget_s
        self._clock = clock
        self.t_start = clock()

    @property
    def elapsed_s(self) -> float:
        return self._clock() - self.t_start

    def exhausted(self, iters_done: int) -> tuple[bool, str]:
        if iters_done >= self.max_iters:
            return True, "max_iters"
        if self.elapsed_s >= self.wall_budget_s:
            return True, "wall_clock"
        return False, ""
