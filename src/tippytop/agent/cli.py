"""`tippytop agent` subcommand."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path

from .. import config as pkg
from .config import AgentConfig
from .env import load_dotenv
from .llm import build_llm_client
from .orchestrator import run_agent


def register_agent_subparser(sub) -> None:
    p = sub.add_parser("agent", help="run the autonomous ML research agent")
    p.add_argument("--llm", default="mock", choices=["mock", "gemini"],
                   help="LLM backend (default: mock, no network)")
    p.add_argument("--max-iters", type=int, default=50)
    p.add_argument("--wall-hours", type=float, default=6.0)
    p.add_argument("--iter-timeout", type=int, default=900,
                   help="per-solution subprocess timeout (s)")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--run-id", default=None)
    p.add_argument("--final-out", default=None,
                   help="path for the final test submission CSV")
    p.set_defaults(func=cmd_agent)


def cmd_agent(a) -> int:
    run_id = a.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = str(a.data_dir or pkg.DATA_DIR)
    run_dir = pkg.RESULTS_DIR / "runs" / run_id
    final_out = Path(a.final_out) if a.final_out else (pkg.SUBMISSIONS_DIR / f"agent_{run_id}_test.csv")

    cfg = AgentConfig(
        data_dir=data_dir, run_dir=run_dir, run_id=run_id,
        max_iters=a.max_iters, wall_hours=a.wall_hours,
        iter_timeout_s=a.iter_timeout, temperature=a.temperature,
        final_out=final_out)

    if a.llm == "gemini":
        load_dotenv()                    # pick up GEMINI_API_KEY from .env if present
    llm = build_llm_client(a.llm)
    print(f"[agent] run_id={run_id} llm={a.llm} data={data_dir}")
    state = run_agent(cfg, llm)
    print(f"[agent] done. report: {run_dir / 'report.md'}")
    if state.final_out:
        print(f"[agent] final submission: {state.final_out}")
    return 0
