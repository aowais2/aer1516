"""Compare RRT vs RRT* path cost over many random instances.

This script runs 1,000 independent planning trials on a fixed "difficult" environment,
computes the average path length for RRT and RRT*, and plots a bar chart showing the
relative improvement achieved by RRT*.

It also draws the environment (start/goal + obstacles) for reference.

Note: This experiment uses a fixed random seed for reproducibility.
"""

import random
import math
import matplotlib.pyplot as plt
import numpy as np

from dubins_path_problem import RRT_dubins_problem
from rrt_planner import rrt_planner
from rrt_star_planner import rrt_star_planner


def run_experiment(num_trials=1000, base_seed=12345):
    """Run num_trials of RRT and RRT* and collect average path costs."""

    # Environment: obstacles arranged to make the straight-line path difficult
    obstacle_list = [
        (4, 4, 1.5),
        (6, 6, 1.5),
        (8, 4, 1.5),
        (6, 2, 1.5),
        (2, 6, 1.5),
        (10, 8, 1.5),
        (8, 10, 1.5),
        (4, 10, 1.5),
        (2, 8, 1.5),
        (7, 7, 1.0),
    ]

    start = [0.0, 0.0, 0.0]
    goal = [12.0, 12.0, np.deg2rad(90.0)]
    map_area = [-2.0, 15.0, -2.0, 15.0]

    rrt_costs = []
    rrtstar_costs = []

    for i in range(num_trials):
        seed = base_seed + i
        random.seed(seed)
        np.random.seed(seed)

        prob = RRT_dubins_problem(
            start=start,
            goal=goal,
            obstacle_list=obstacle_list,
            map_area=map_area,
            max_iter=10000,
        )
        path = rrt_planner(prob, display_map=False)
        if path is not None and len(path) > 0:
            rrt_costs.append(path[-1].cost)

        random.seed(seed)
        np.random.seed(seed)

        prob = RRT_dubins_problem(
            start=start,
            goal=goal,
            obstacle_list=obstacle_list,
            map_area=map_area,
            max_iter=10000,
        )
        path = rrt_star_planner(prob, display_map=False)
        if path is not None and len(path) > 0:
            rrtstar_costs.append(path[-1].cost)

    return {
        "seed": base_seed,
        "num_trials": num_trials,
        "rrt_costs": rrt_costs,
        "rrtstar_costs": rrtstar_costs,
        "start": start,
        "goal": goal,
        "obstacle_list": obstacle_list,
        "map_area": map_area,
    }


def plot_results(results):
    """Plot the environment and a bar chart comparing average costs."""
    rrt_costs = results["rrt_costs"]
    rrtstar_costs = results["rrtstar_costs"]

    avg_rrt = np.mean(rrt_costs) if rrt_costs else float("nan")
    avg_rrtstar = np.mean(rrtstar_costs) if rrtstar_costs else float("nan")

    fig, (ax_env, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))

    # Environment plot
    ax_env.set_title("Obstacle Environment")
    ax_env.set_aspect("equal")
    ax_env.set_xlim(results["map_area"][0], results["map_area"][1])
    ax_env.set_ylim(results["map_area"][2], results["map_area"][3])
    for ox, oy, size in results["obstacle_list"]:
        circle = plt.Circle((ox, oy), size, color="tab:blue", alpha=0.4)
        ax_env.add_patch(circle)
    ax_env.plot(results["start"][0], results["start"][1], "go", label="Start")
    ax_env.plot(results["goal"][0], results["goal"][1], "ro", label="Goal")
    ax_env.grid(True)
    ax_env.legend()

    # Bar chart
    ax_bar.set_title("Average Path Cost: RRT vs RRT*")
    bars = ax_bar.bar(
        [0, 1], [avg_rrt, avg_rrtstar], width=0.5, color=["tab:orange", "tab:green"]
    )
    ax_bar.set_xticks([0, 1])
    ax_bar.set_xticklabels(["RRT", "RRT*"])
    ax_bar.set_ylabel("Average path length")
    for bar in bars:
        height = bar.get_height()
        ax_bar.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    note = (
        f"Trials={results['num_trials']}, seed={results['seed']}\n"
        f"RRT avg={avg_rrt:.2f}, RRT* avg={avg_rrtstar:.2f}"
    )
    fig.suptitle(note)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    results = run_experiment(num_trials=1000, base_seed=12345)
    plot_results(results)
