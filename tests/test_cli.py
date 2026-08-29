from pathlib import Path

from tippytop.cli import build_parser


def test_runtime_limits_can_be_set_for_run_and_resume() -> None:
    parser = build_parser()

    run = parser.parse_args(["run", "--llm-timeout", "450", "--learn-from", "old-run"])
    resume = parser.parse_args(
        [
            "resume",
            "a-run",
            "--max-iterations",
            "12",
            "--max-hours",
            "4",
            "--experiment-timeout",
            "900",
            "--llm-timeout",
            "450",
        ]
    )

    assert run.llm_timeout == 450
    assert run.prior_run == Path("old-run")
    assert resume.run_dir == Path("a-run")
    assert resume.max_iterations == 12
    assert resume.max_hours == 4
    assert resume.experiment_timeout == 900
    assert resume.llm_timeout == 450
