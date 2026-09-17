"""What did the shared preprocessing convention cost MobileViT-S?

The fairness rule requires one preprocessing convention for every architecture,
so all five models are trained and evaluated with ImageNet channel statistics.
Four of them report exactly those in their pretrained config. MobileViT-S does
not: timm gives it mean=(0,0,0), std=(1,1,1), meaning raw [0,1] inputs. It is the
one model asked to work outside the range its pretrained weights expect.

The rule is not the problem and is not being changed here. Giving one model a
preprocessing of its own would break the comparison that the whole benchmark
exists to make. The question is narrower: how much did the convention cost the
model it did not suit? Unmeasured, that is a caveat. Measured, it is a number.

The expectation is "not much", because the recipe fine-tunes the full network
rather than freezing a backbone, so several epochs of gradient descent can absorb
an affine shift on the input. Expectations are not measurements, which is why
this runs.

WHAT THIS DOES NOT TOUCH. train_one() writes a checkpoint to CHECKPOINT_DIR and
appends a row to results/runs.jsonl. Both are redirected here, because an
ablation that overwrote the real MobileViT checkpoints or added rows to the
published training log would corrupt the thing it is measuring. The redirection
is asserted before a single epoch runs.

Usage:
    python scripts/preprocessing_ablation.py [--seeds 0,1,2,3,4]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench.data as bench_data       # noqa: E402
import bench.train as bench_train     # noqa: E402
from bench.config import RESULTS_DIR, SPLIT_CSV  # noqa: E402

MODEL = "mobilevit_s"
NATIVE_MEAN = (0.0, 0.0, 0.0)
NATIVE_STD = (1.0, 1.0, 1.0)
SANDBOX = os.path.join(RESULTS_DIR, "ablation")
OUT = os.path.join(RESULTS_DIR, "preprocessing_ablation.json")


def published_accuracies() -> dict[int, float]:
    """MobileViT-S test accuracy per seed, under the shared convention."""
    path = os.path.join(RESULTS_DIR, "runs.jsonl")
    out = {}
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            if row["model"] == MODEL:
                out[row["seed"]] = row["test_acc"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    baseline = published_accuracies()
    missing = [s for s in seeds if s not in baseline]
    if missing:
        print(f"no published run for {MODEL} seeds {missing}", file=sys.stderr)
        return 2

    real_ckpt = bench_train.CHECKPOINT_DIR
    real_runs = bench_train.RUNS_JSONL
    # Captured BEFORE the patch, or the report would record the value it
    # was compared against as the value it was replaced by.
    shared_mean = list(bench_data.NORM_MEAN)
    shared_std = list(bench_data.NORM_STD)

    # Redirect the two side effects, then normalisation.
    os.makedirs(SANDBOX, exist_ok=True)
    bench_train.CHECKPOINT_DIR = SANDBOX
    bench_train.RUNS_JSONL = os.path.join(SANDBOX, "runs.jsonl")
    bench_data.NORM_MEAN = NATIVE_MEAN
    bench_data.NORM_STD = NATIVE_STD

    # Refuse to start unless every redirection took. A silent failure here would
    # overwrite trained checkpoints that cost hours, and the corruption would be
    # invisible until someone re-exported.
    assert bench_train.CHECKPOINT_DIR != real_ckpt, "checkpoint dir not redirected"
    assert bench_train.RUNS_JSONL != real_runs, "runs.jsonl not redirected"
    assert bench_data.NORM_MEAN == NATIVE_MEAN, "normalisation not patched"
    probe = bench_data.EuroSATFold("val", SPLIT_CSV)
    assert probe._mean.flatten().tolist() == list(NATIVE_MEAN), (
        "EuroSATFold still holds the shared normalisation; patch the module "
        "globals before constructing a fold")
    del probe
    print(f"sandboxed: checkpoints -> {SANDBOX}, runs -> {bench_train.RUNS_JSONL}")
    print(f"normalisation: mean={NATIVE_MEAN} std={NATIVE_STD} (MobileViT native)\n",
          flush=True)

    device = bench_train.pick_device(args.device)
    native = {}
    for seed in seeds:
        row = bench_train.train_one(MODEL, seed, device, SPLIT_CSV)
        native[seed] = row["test_acc"]

    deltas = [(native[s] - baseline[s]) * 100.0 for s in seeds]
    report = {
        "model": MODEL,
        "shared_convention": {"mean": shared_mean, "std": shared_std},
        "native_convention": {"mean": list(NATIVE_MEAN), "std": list(NATIVE_STD)},
        "per_seed": [{"seed": s,
                      "shared_acc": baseline[s],
                      "native_acc": native[s],
                      "delta_pp": (native[s] - baseline[s]) * 100.0} for s in seeds],
        "mean_shared_acc": st.mean(baseline[s] for s in seeds),
        "mean_native_acc": st.mean(native[s] for s in seeds),
        "mean_delta_pp": st.mean(deltas),
        "stdev_delta_pp": st.stdev(deltas) if len(deltas) > 1 else 0.0,
    }
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    print()
    print(f"{MODEL}: shared ImageNet convention vs its native [0,1]")
    for r in report["per_seed"]:
        print(f"  seed {r['seed']}  shared {r['shared_acc'] * 100:.2f}%  "
              f"native {r['native_acc'] * 100:.2f}%  "
              f"delta {r['delta_pp']:+.2f} pp")
    print(f"\nmean delta: {report['mean_delta_pp']:+.2f} pp "
          f"(sd {report['stdev_delta_pp']:.2f})")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
