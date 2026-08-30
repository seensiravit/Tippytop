from tippytop.agent.convergence import ConvergenceTracker, BudgetManager


def test_converges_when_flat():
    t = ConvergenceTracker(eps=0.002, n=3)
    for v in [0.60, 0.6005, 0.6008, 0.601]:
        t.update(v)
    done, why = t.converged()
    assert done and why == "converged"


def test_big_jump_resets():
    t = ConvergenceTracker(eps=0.002, n=3)
    for v in [0.60, 0.601, 0.602, 0.70]:   # last step is a big gain
        t.update(v)
    assert not t.converged()[0]


def test_not_enough_points():
    t = ConvergenceTracker(eps=0.002, n=3)
    t.update(0.6); t.update(0.6)
    assert not t.converged()[0]


def test_failed_iters_do_not_advance():
    t = ConvergenceTracker(eps=0.002, n=3)
    t.update(0.6); t.update(None); t.update(None)
    assert not t.converged()[0]      # only one real point


def test_budget_max_iters_and_wall():
    clock = {"t": 0.0}
    b = BudgetManager(max_iters=5, wall_budget_s=100, clock=lambda: clock["t"])
    assert b.exhausted(4) == (False, "")
    assert b.exhausted(5)[0] and b.exhausted(5)[1] == "max_iters"
    clock["t"] = 100
    assert b.exhausted(1)[0] and b.exhausted(1)[1] == "wall_clock"
