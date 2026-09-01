
# Builds the submission from the AGENT'S OWN model, by running the exact code
# path the harness scores: baseline.run_fm().
#
# It does NOT re-implement training, and it does not name any model class. The
# kit's `submit.py --make` does both -- it hardcodes `B.FM(dim, k=16, lr=0.001)`
# and its own 40-epoch loop -- which fails two different ways once an agent is
# editing baseline.py:
#
#   1. LOUD: the agent renames the class (FM -> FFM) and --make dies with
#      AttributeError. Observed on the first real run.
#   2. SILENT, and far worse: the agent keeps a class called FM but changes k,
#      lr, epochs, or anything else inside run_fm. --make then succeeds, writes
#      a perfectly valid CSV, and submits the ORIGINAL baseline configuration.
#      Nothing anywhere reports a problem; the score is just wrong.
#
# run_fm returns metrics, not per-row scores, so we capture the scores on their
# way into evaluate(). run_fm's last act is
# `evaluate(ute, yte, m.predict(Xte))` -- those are exactly the test-split
# predictions, in row order. Wrapping evaluate is the only way to get them
# without requiring the agent to change run_fm's return signature, which would
# be one more contract for it to break.
import sys
import baseline as B
from data import load
from submit import write_submission

split, out_path, data_dir, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
splits = load(data_dir)
rows = splits[split]

captured = []
_real_evaluate = B.evaluate


def _recording_evaluate(users, labels, scores, *a, **kw):
    captured.append((len(scores), list(scores)))
    return _real_evaluate(users, labels, scores, *a, **kw)


B.evaluate = _recording_evaluate
try:
    B.run_fm(splits, seed=seed, verbose=False)
finally:
    B.evaluate = _real_evaluate

# Match by row count: valid is 124,909 rows and test is 170,588, so the split
# is unambiguous. Last match wins -- run_fm evaluates validation every epoch,
# and the final call for a split is the one made with the restored best state.
matches = [sc for n, sc in captured if n == len(rows)]
if not matches:
    sizes = sorted({n for n, _ in captured})
    raise SystemExit(
        f"run_fm never evaluated {len(rows)} rows (the {split} split). It "
        f"evaluated {sizes}. The agent's run_fm no longer scores that split, so "
        "no submission can be built from it.")
write_submission(out_path, rows, matches[-1])
print(f"wrote {out_path}: {len(rows):,d} rows (split={split}, agent's own run_fm)")
