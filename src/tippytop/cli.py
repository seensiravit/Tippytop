"""Command-line interface for Tippytop."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .agent import run_agent
from .artifacts import RunStore, read_json
from .config import RunConfig
from .doctor import format_doctor, run_doctor
from .submission import finalize_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tippytop")
    subcommands = parser.add_subparsers(dest="command", required=True)

    doctor = subcommands.add_parser("doctor", help="verify data, environment, disk, and LLM service")
    _add_connection_options(doctor)
    doctor.add_argument("--data-dir", type=Path)
    doctor.add_argument("--skip-llm", action="store_true")

    run = subcommands.add_parser("run", help="start an autonomous research run")
    _add_connection_options(run)
    run.add_argument("--data-dir", type=Path)
    run.add_argument("--runs-dir", type=Path)
    run.add_argument(
        "--learn-from",
        dest="prior_run",
        type=Path,
        help="seed a fresh run with sanitized validation-only lessons from a prior run",
    )
    run.add_argument("--max-iterations", type=int, default=50)
    run.add_argument("--max-hours", type=float, default=6.0)
    run.add_argument("--epsilon", type=float, default=0.002)
    run.add_argument("--patience", type=int, default=3)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--experiment-timeout", type=int, default=1800)
    run.add_argument("--llm-timeout", type=int, default=600)
    run.add_argument(
        "--offline",
        action="store_true",
        help="verify and reproduce the baseline without generating experiments",
    )
    run.add_argument("--no-finalize", action="store_true", help="select a model without scoring test")

    resume = subcommands.add_parser("resume", help="resume an interrupted run")
    resume.add_argument("run_dir", type=Path)
    resume.add_argument("--max-iterations", type=int)
    resume.add_argument("--max-hours", type=float)
    resume.add_argument("--experiment-timeout", type=int)
    resume.add_argument("--llm-timeout", type=int)
    resume.add_argument("--no-finalize", action="store_true")

    submit = subcommands.add_parser("submit", help="score test once and validate the final submission")
    submit.add_argument("--run", dest="run_dir", type=Path, required=True)
    return parser


def _add_connection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--api-key")


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "doctor":
        config = RunConfig.from_env(
            data_dir=arguments.data_dir,
            base_url=arguments.base_url,
            model=arguments.model,
            api_key=arguments.api_key,
        )
        checks = run_doctor(config, check_llm=not arguments.skip_llm)
        print(format_doctor(checks))
        return 0 if checks["ok"] else 1

    if arguments.command == "run":
        config = RunConfig.from_env(
            data_dir=arguments.data_dir,
            runs_dir=arguments.runs_dir,
            prior_run=arguments.prior_run,
            base_url=arguments.base_url,
            model=arguments.model,
            api_key=arguments.api_key,
            max_iterations=arguments.max_iterations,
            max_hours=arguments.max_hours,
            epsilon=arguments.epsilon,
            patience=arguments.patience,
            seed=arguments.seed,
            experiment_timeout=arguments.experiment_timeout,
            llm_timeout=arguments.llm_timeout,
            offline=arguments.offline,
        )
        store = run_agent(config, finalize=not arguments.no_finalize)
        print(store.path)
        return 0

    run_dir = arguments.run_dir
    persisted = read_json(run_dir / "config.json")
    config = RunConfig.from_dict(persisted)
    if arguments.command == "resume":
        overrides = {
            name: getattr(arguments, name)
            for name in ("max_iterations", "max_hours", "experiment_timeout", "llm_timeout")
            if getattr(arguments, name) is not None
        }
        if overrides:
            config = replace(config, **overrides)
        store = run_agent(config, run_dir=run_dir, finalize=not arguments.no_finalize)
        print(store.path)
        return 0

    store = RunStore.open(run_dir, config)
    state = read_json(run_dir / "state.json")
    result = finalize_run(store, config, state)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
