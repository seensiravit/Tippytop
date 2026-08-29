from tippytop.convergence import ConvergenceTracker


def test_convergence_resets_only_for_significant_gain() -> None:
    tracker = ConvergenceTracker(epsilon=0.002, patience=3, best=0.6000)
    assert tracker.observe(0.6010) == (True, False)
    assert tracker.stagnant == 1
    assert tracker.observe(0.6040) == (True, True)
    assert tracker.stagnant == 0
    tracker.observe(0.6030)
    tracker.observe(0.6045)
    tracker.observe(0.6044)
    assert tracker.converged
