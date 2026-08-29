from pathlib import Path

from tippytop.artifacts import RunStore, read_json, read_jsonl
from tippytop.config import RunConfig


def test_store_redacts_secrets_and_appends_events(tmp_path: Path) -> None:
    config = RunConfig(runs_dir=tmp_path, api_key="top-secret")
    store = RunStore.create(config)
    store.write_json("value.json", {"token": "Bearer top-secret"})
    store.event("failure", detail="top-secret leaked")
    assert read_json(store.path / "value.json")["token"] == "Bearer <redacted>"
    assert read_jsonl(store.path / "events.jsonl")[0]["detail"] == "<redacted> leaked"
    assert not list(store.path.glob("*.tmp"))
