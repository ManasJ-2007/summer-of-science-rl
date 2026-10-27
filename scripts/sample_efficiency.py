"""Steps-to-solve on CartPole-v1, computed from the saved per-episode returns (results/raw/*.json).

CartPole gives +1 per step, so the cumulative sum of episode returns is the number of environment steps.
"Solved" = first time the 20-episode moving average reaches 475.

Usage:  python scripts/sample_efficiency.py   ->   results/sample_efficiency.json
"""
import json
import os

import numpy as np

RAW = os.path.join(os.path.dirname(__file__), "..", "results", "raw")
SOLVED, WINDOW = 475, 20


def steps_to_solve(returns):
    r = np.asarray(returns)
    if len(r) < WINDOW:
        return None
    ma = np.convolve(r, np.ones(WINDOW) / WINDOW, mode="valid")
    hit = np.flatnonzero(ma >= SOLVED)
    return int(np.cumsum(r)[hit[0] + WINDOW - 1]) if len(hit) else None


def main():
    out = {}
    for exp in ("cartpole", "reinforce", "ablation", "dqn_tuning"):
        path = os.path.join(RAW, f"{exp}.json")
        if not os.path.exists(path):
            continue
        for name, runs in json.load(open(path)).items():
            steps = [steps_to_solve(run["returns"]) for run in runs]
            solved = [s for s in steps if s is not None]
            out[f"{exp} / {name}"] = {
                "seeds_solved": f"{len(solved)}/{len(steps)}",
                "median_steps_to_solve": int(np.median(solved)) if solved else None,
                "steps_per_seed": steps,
            }
    with open(os.path.join(RAW, "..", "sample_efficiency.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for k, v in out.items():
        print(f"{k:45s} solved {v['seeds_solved']}  median steps {v['median_steps_to_solve']}  {v['steps_per_seed']}")


if __name__ == "__main__":
    main()
