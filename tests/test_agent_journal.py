import json
from tippytop.agent.journal import Journal, IterationRecord


def _rec(i, prim=None, **kw):
    m = None if prim is None else {"GAUC": prim, "nDCG@5": prim, "primary": prim}
    return IterationRecord(run_id="r", iter=i, phase="IMPROVE",
                           hypothesis=f"try {i}", valid_metrics=m, **kw)


def test_append_writes_jsonl(tmp_path):
    j = Journal(tmp_path)
    j.append(_rec(0, 0.60, total_tokens=100))
    j.append(_rec(1, 0.61, accepted=True, total_tokens=120))
    lines = j.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec1 = json.loads(lines[1])
    assert rec1["iter"] == 1 and rec1["accepted"] and rec1["valid_metrics"]["primary"] == 0.61


def test_report_has_deliverable_fields(tmp_path):
    j = Journal(tmp_path)
    j.append(_rec(0, 0.60))
    j.write_report(run_id="r", llm_model="mock", stop_reason="converged",
                   best_iter=0, best_metrics={"primary": 0.60},
                   final_test_metrics={"primary": 0.61}, interventions=0)
    txt = j.report_path.read_text(encoding="utf-8")
    assert "Manual interventions" in txt
    assert "Best valid primary" in txt
    assert "test" in txt and "converged" in txt
