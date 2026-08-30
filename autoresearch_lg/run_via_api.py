"""Kick off a run from the terminal that's ALSO watchable live in LangGraph
Studio — unlike `cli.py run` (a direct `compiled.stream()` call in its own
Python process, invisible to Studio), this goes through the running
`langgraph dev` server's REST API, so it's a real Studio thread.

Requires `langgraph dev` already running (see README's Studio section).

    python -m autoresearch_lg.run_via_api --tag aug29 --model claude-sonnet-5

Prints the thread id and the Studio URL to open and watch it live.
"""
from __future__ import annotations

import argparse

from langgraph_sdk import get_sync_client

from . import bootstrap, cli


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-dir", default="./KuaiRand-Pure/data")
    ap.add_argument("--model", default=bootstrap.DEFAULT_MODEL)
    ap.add_argument("--max-iterations", type=int, default=50)
    ap.add_argument("--max-wall-hours", type=float, default=6.0)
    ap.add_argument("--epsilon", type=float, default=0.002)
    ap.add_argument("--n-plateau", type=int, default=3)
    ap.add_argument("--retry-cap", type=int, default=3)
    ap.add_argument("--tune-cap", type=int, default=3)
    ap.add_argument("--server-url", default="http://127.0.0.1:2024")
    args = ap.parse_args()

    root = cli.repo_root()
    branch = cli.tools.current_branch(root)
    expected = f"autoresearch/{args.tag}"
    if branch != expected:
        print(f"WARNING: current branch is '{branch}', expected '{expected}'.")

    state = cli._load_state(root, args)

    client = get_sync_client(url=args.server_url)
    thread = client.threads.create(graph_id="autoresearch")
    studio_url = f"https://smith.langchain.com/studio/?baseUrl={args.server_url}&threadId={thread['thread_id']}"
    print(f"thread: {thread['thread_id']}")
    print(f"watch live: {studio_url}")

    last_history_len = state["iteration"]
    for chunk in client.runs.stream(
        thread["thread_id"], "autoresearch", input=state, stream_mode="values",
    ):
        data = chunk.data
        if not isinstance(data, dict) or "history" not in data:
            continue
        if len(data["history"]) > last_history_len:
            last_history_len = len(data["history"])
            last = data["history"][-1]
            print(f"\n--- experiment {last_history_len} "
                  f"[{last['mode']}/{last['outcome']}] {last['concept_id']} ---")
            print(f"idea: {last['description']}")
            print(f"result: valid={last['metrics']['valid_primary']:.4f} "
                  f"test={last['metrics']['test_primary']:.4f}")


if __name__ == "__main__":
    main()
