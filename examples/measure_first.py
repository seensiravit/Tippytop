"""Phase 0 in ~20 lines: measure the instrument before trusting any comparison.

    python examples/measure_first.py --data_dir <KuaiRand-Pure/data>

Runs no experiments. Reports whether the split can move the metric, what the
paired-bootstrap interval on a real comparison looks like, and how many seeds an
effect of a given size actually needs.
"""
import argparse
import numpy as np
from tippytop.data.dataset import load_dataset
from tippytop.training.runner import train_model
from tippytop.stats import (diagnose_eval_split, per_user_stats, paired_bootstrap,
                            format_comparison, is_detectable, selection_inflation)

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True)
ap.add_argument("--split", default="valid")
a = ap.parse_args()

data = load_dataset(a.data_dir)
users, labels = data.users(a.split), data.y(a.split)

print("1. Can the metric move on this split?")
print("  ", diagnose_eval_split(users, labels))

print("\n2. Paired bootstrap: fm vs fm_listwise (one seed each)")
s_a = train_model("fm", data, seed=0).predict(data, a.split)
s_b = train_model("fm_listwise", data, seed=0).predict(data, a.split)
r = paired_bootstrap(per_user_stats(users, labels, s_b),
                     per_user_stats(users, labels, s_a))
print("  ", format_comparison("fm_listwise", "fm", r))

print("\n3. How many seeds would a claimed effect need?")
for d in (0.0023, 0.0015, 0.0002):
    print("  ", is_detectable(d)["verdict"], f"(delta {d})")

print("\n4. What does best-of-N selection manufacture on its own?")
print("  ", selection_inflation(50, seeds_per_candidate=1)["warning"])
print("  ", selection_inflation(50, seeds_per_candidate=3)["warning"])
