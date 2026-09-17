"""How much on-die power did the reported energy figures leave out?

The benchmark integrates CPU package power and reports that as energy. The
README has always said so, and said it excludes DRAM, display and PSU losses.
What it could not say is how much was excluded, which left a reader unable to
judge whether the omission was a rounding error or half the answer.

powermetrics reports three on-die rails, and the sampler captured all three:

    CPU Power: ... mW
    GPU Power: ... mW
    ANE Power: ... mW

so the CPU share of on-die compute power is recoverable from the raw log. It
does not recover DRAM or PSU, which no on-die counter sees; those remain
genuinely unmeasured and the README still says so.

It doubles as a check on the "all benchmarking is CPU-only" claim. If the GPU
had been quietly doing work during the matrix, this is where it would show.

results/power_log.txt is not committed: it is ~250 MB, above GitHub's file limit,
for the reason recorded in .gitignore. This script regenerates
results/power_composition.json from it for anyone who re-runs the benchmark with
the sampler, in the same way the idle-baseline figure is derived.

Usage:
    python scripts/power_composition.py
"""

from __future__ import annotations

import json
import os
import re
import sys

LOG = os.path.join("results", "power_log.txt")
OUT = os.path.join("results", "power_composition.json")
RAIL = re.compile(r"^(CPU|GPU|ANE) Power: (\d+) mW")


def main() -> int:
    if not os.path.exists(LOG):
        print(f"{LOG} not found. It is not committed; re-run the benchmark with "
              f"scripts/energy_sampler.sh to regenerate it.", file=sys.stderr)
        return 2

    totals = {"CPU": 0, "GPU": 0, "ANE": 0}
    nonzero = {"CPU": 0, "GPU": 0, "ANE": 0}
    samples = 0
    current: dict[str, int] = {}

    with open(LOG) as fh:
        for line in fh:
            m = RAIL.match(line)
            if not m:
                continue
            current[m.group(1)] = int(m.group(2))
            # powermetrics emits the three rails together; a complete triple is
            # one sample. Counting rails separately would let a truncated final
            # block skew one mean against the others.
            if len(current) == 3:
                for rail, mw in current.items():
                    totals[rail] += mw
                    nonzero[rail] += mw > 0
                samples += 1
                current = {}

    if not samples:
        print(f"no complete power samples found in {LOG}", file=sys.stderr)
        return 2

    on_die = sum(totals.values())
    report = {
        "samples": samples,
        "mean_watts": {r: totals[r] / samples / 1000.0 for r in totals},
        "nonzero_sample_fraction": {r: nonzero[r] / samples for r in nonzero},
        "cpu_share_of_on_die_power": totals["CPU"] / on_die,
        "excluded_gpu_ane_share": (totals["GPU"] + totals["ANE"]) / on_die,
    }

    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    print(f"samples: {samples:,}")
    for r in ("CPU", "GPU", "ANE"):
        print(f"  mean {r} power: {report['mean_watts'][r]:7.3f} W "
              f"({report['nonzero_sample_fraction'][r] * 100:5.1f}% of samples non-zero)")
    print(f"\nCPU share of on-die compute power: "
          f"{report['cpu_share_of_on_die_power'] * 100:.2f}%")
    print(f"GPU + ANE, excluded from reported energy: "
          f"{report['excluded_gpu_ane_share'] * 100:.2f}%")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
