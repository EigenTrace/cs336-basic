"""Plot learning curves from the JSON logs written by training_loop.

Usage (from repo root):
    uv run python -m assignment.part3.plot_curves                  # all runs in logs/
    uv run python -m assignment.part3.plot_curves lr1e-03 lr3e-03  # only these runs
"""

import json
import os
import sys

import matplotlib.pyplot as plt

LOG_DIR = "logs"
OUT_DIR = "figures"


def load_run(name: str) -> list[dict]:
    """Load one run's history from logs/<name>.json."""
    with open(os.path.join(LOG_DIR, f"{name}.json")) as f:
        return json.load(f)


def plot_curves(run_names: list[str], out_name: str = "learning_curves") -> None:
    """Plot val/train loss vs. gradient step and vs. wall-clock time."""
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, (ax_step, ax_time) = plt.subplots(1, 2, figsize=(13, 5))

    for name in run_names:
        h = load_run(name)
        if not h:
            print(f"skip empty run: {name}")
            continue

        steps = [r["step"] for r in h]
        wall = [r["wall_clock"] / 60 for r in h]          # minutes
        val = [r["val_loss"] for r in h]
        train = [r["train_loss"] for r in h]

        # left: loss vs. gradient step
        line, = ax_step.plot(steps, val, marker="o", ms=3, label=f"{name} (val)")
        ax_step.plot(steps, train, ls="--", alpha=0.4,
                     color=line.get_color(), label=f"{name} (train)")

        # right: loss vs. wall-clock time
        ax_time.plot(wall, val, marker="o", ms=3, color=line.get_color(), label=name)

        print(f"{name:20s} final val {val[-1]:.4f}  "
              f"min val {min(val):.4f}  wall {wall[-1]:.1f} min")

    ax_step.set_xlabel("gradient step")
    ax_step.set_ylabel("loss (per token)")
    ax_step.set_title("Loss vs. gradient steps")

    ax_time.set_xlabel("wall-clock time (minutes)")
    ax_time.set_ylabel("validation loss")
    ax_time.set_title("Validation loss vs. wall-clock time")

    for ax in (ax_step, ax_time):
        ax.set_yscale("log")      # log scale: diverging runs span orders of magnitude
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"{out_name}.png")
    fig.savefig(path, dpi=150)
    print(f"\nsaved {path}")
    plt.show()


if __name__ == "__main__":
    names = sys.argv[1:]
    if not names:
        # default: every run in logs/
        names = sorted(f[:-5] for f in os.listdir(LOG_DIR) if f.endswith(".json"))
    plot_curves(names)
