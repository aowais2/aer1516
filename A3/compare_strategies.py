#!/usr/bin/env python3
"""
Script to compare exploration strategies across maps.

For each map (open_room, office, cave), this script runs the random baseline,
nearest-frontier, and custom strategies, collects coverage history, and plots
coverage (%) versus step count.

Usage: python compare_strategies.py
"""

import matplotlib.pyplot as plt
from run_exploration import run_exploration

# Maps and strategies to compare
maps = ['open_room', 'office', 'cave']
strategies = ['random', 'nearest', 'custom']

# Create a figure with subplots for each map
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Coverage (%) vs Step Count: Random vs Nearest vs Custom Strategies')

for i, map_name in enumerate(maps):
    ax = axes[i]
    
    for strategy in strategies:
        # Run exploration with fixed seed for reproducibility, no visualization
        result = run_exploration(
            map_name, 
            strategy, 
            visualize=False, 
            seed=42, 
            enforce_time=False
        )
        
        # Extract coverage history (list of coverage percentages)
        coverage_history = result['coverage_history']
        steps = list(range(len(coverage_history)))
        
        # Plot coverage vs step count
        ax.plot(steps, coverage_history, label=strategy, linewidth=2)
    
    ax.set_xlabel('Step Count')
    ax.set_ylabel('Coverage (%)')
    ax.set_title(f'{map_name.replace("_", " ").title()}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()