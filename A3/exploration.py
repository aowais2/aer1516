"""
AER1516 Winter 2026 — Assignment 3: Autonomous Frontier-Based Exploration

** STUDENT FILE ** — Implement the TODOs below.

This file contains:
  - FrontierRegion: data class for frontier clusters (provided)
  - detect_frontiers_random / select_goal_random: random baseline (provided)
  - plan_path: Optional bonus — upgrade Dijkstra to A* (not graded)
  - detect_frontiers: Part 1 — frontier detection & clustering (TODO)
  - select_goal_nearest: Part 2a — nearest-frontier goal selection (TODO)
  - select_goal_custom: Part 2b — custom goal selection strategy (TODO)
  - exploration_step: Part 3 — robust exploration loop callback (TODO)

Grading breakdown:
  Part 1 (detect_frontiers):     12 points
  Part 2a (select_goal_nearest): 10 points
  Part 2b (select_goal_custom):   6 points
  Part 3 (exploration_step):     12 points
  Code subtotal:                 40 points
  Report:                        10 points
  Total:                         50 points

Note: Upgrading plan_path from Dijkstra to A* is optional and not graded,
but it dramatically speeds up planning, which lets your robot explore more
within the time limit. A slow planner means the robot wastes wall-clock time
thinking instead of moving.
"""

import math
import heapq
import random
from collections import deque
from matplotlib.pyplot import grid
import numpy as np
from config import FREE, OCCUPIED, UNKNOWN, FRONTIER_MIN_SIZE, CELL_SIZE, CONNECTIVITY
from planner import inflate_grid, validate_path, path_cost
from planner import plan_path as _plan_path_dijkstra


# =============================================================================
# Data class (provided — do not modify)
# =============================================================================

class FrontierRegion:
    """A cluster of contiguous frontier cells."""

    def __init__(self, cells):
        """
        Args:
            cells: list of (row, col) tuples belonging to this frontier cluster.
        """
        self.cells = cells
        self.size = len(cells)
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        self.centroid = (int(round(sum(rows) / len(rows))),
                         int(round(sum(cols) / len(cols))))


# =============================================================================
# Random baseline (provided — do not modify)
# =============================================================================

def detect_frontiers_random(occ_grid):
    """
    Random baseline: sample random free cells as single-cell 'frontier regions'.
    This is NOT a real frontier detector — it just picks random explored cells.
    """
    free_cells = list(zip(*np.where(occ_grid.grid == FREE)))
    if not free_cells:
        return []
    sampled = random.sample(free_cells, min(10, len(free_cells)))
    return [FrontierRegion([cell]) for cell in sampled]


def select_goal_random(frontier_regions, occ_grid, state):
    """
    Random baseline: pick a random frontier region's centroid.
    """
    if not frontier_regions:
        return None
    valid = [fr for fr in frontier_regions
             if fr.centroid not in state.blacklisted_goals]
    if not valid:
        return None
    return random.choice(valid).centroid


# =============================================================================
# Optional Bonus: Path Planning — Upgrade Dijkstra to A* (not graded)
# =============================================================================

def plan_path(occ_grid, start_rc, goal_rc):
    """
    Plan a path on the 8-connected grid with obstacle inflation.

    This part is OPTIONAL and not graded. The provided Dijkstra planner
    (planner.py) is correct but slow — it explores uniformly in all directions
    without a heuristic. Upgrading to A* is strongly recommended because it
    dramatically speeds up planning for distant goals.

    Hint: Look at planner.py to see how Dijkstra is implemented. The only
    change needed for A* is to add h(n) to the priority queue ordering:
      Dijkstra: priority = g(n)
      A*:       priority = g(n) + h(n)
    where h(n) is an admissible heuristic (e.g., Euclidean distance to goal).

    You can use inflate_grid() from planner.py to get the traversability mask.

    Args:
        occ_grid: OccupancyGrid instance.
        start_rc: (row, col) start position.
        goal_rc: (row, col) goal position.

    Returns:
        (path, cost) where path is a list of (row, col) from start to goal
        (inclusive), and cost is the total path cost in cell-distance units.
        Returns (None, None) if no path exists.
    """
    # ---- START YOUR CODE (Optional Bonus) ----

    blocked = inflate_grid(occ_grid)
    H, W = occ_grid.height, occ_grid.width

    sr, sc = start_rc
    gr, gc = goal_rc

    # Check bounds
    if not (0 <= sr < H and 0 <= sc < W):
        return None, None
    if not (0 <= gr < H and 0 <= gc < W):
        return None, None
    # Require start and goal to be FREE cells (known open space)
    if occ_grid.grid[sr, sc] != FREE:
        return None, None
    if occ_grid.grid[gr, gc] != FREE:
        return None, None

    # Allow start and goal even if in inflation zone
    blocked[sr, sc] = False
    blocked[gr, gc] = False

    # 8-connected neighbors
    if CONNECTIVITY == 8:
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]
    else:
        neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    SQRT2 = math.sqrt(2)

    # A*: priority queue ordered by f = g + h (heuristic)
    open_set = [(0.0, sr, sc)]  # (f_cost, r, c)
    g_cost = {(sr, sc): 0.0}
    came_from = {}
    closed = set()

    while open_set:
        f, r, c = heapq.heappop(open_set)

        if (r, c) == (gr, gc):
            # Reconstruct path
            path = [(r, c)]
            while (r, c) in came_from:
                r, c = came_from[(r, c)]
                path.append((r, c))
            path.reverse()
            return path, g_cost[(gr, gc)]

        if (r, c) in closed:
            continue
        closed.add((r, c))

        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if blocked[nr, nc]:
                continue
            if (nr, nc) in closed:
                continue

            # Movement cost
            step_cost = SQRT2 if (dr != 0 and dc != 0) else 1.0
            new_g = g_cost[(r, c)] + step_cost

            if new_g < g_cost.get((nr, nc), float('inf')):
                g_cost[(nr, nc)] = new_g
                came_from[(nr, nc)] = (r, c)
                # A* heuristic: Chebyshev distance to goal (8-connected grid)
                h = max(abs(gr - nr), abs(gc - nc))
                f = new_g + h
                heapq.heappush(open_set, (f, nr, nc))

    return None, None
    #return _plan_path_dijkstra(occ_grid, start_rc, goal_rc)
    # ---- END YOUR CODE (Optional Bonus) ----


# =============================================================================
# Part 1: Frontier Detection (12 points)
# =============================================================================

def detect_frontiers(occ_grid):
    """
    Detect frontier cells and cluster them into contiguous regions.

    A frontier cell is a FREE cell that has at least one UNKNOWN 4-neighbor
    (up, down, left, right).

    Steps:
      1. Find all frontier cells in the occupancy grid.
      2. Cluster frontier cells into contiguous regions using BFS/flood-fill
         (use 4-connectivity for clustering).
      3. Filter out regions with fewer than FRONTIER_MIN_SIZE cells.
      4. Return a list of FrontierRegion objects.

    Args:
        occ_grid: OccupancyGrid instance with .grid (H x W numpy array),
                  .is_free(r,c), .is_unknown(r,c), .is_in_bounds(r,c).

    Returns:
        List of FrontierRegion objects, each with .cells, .centroid, .size.
    """
    # ---- START YOUR CODE (Part 1) ----

    grid = occ_grid.grid
    H, W = grid.shape

    # Step 1: Find all frontier cells.
    # A frontier cell is FREE and has at least one UNKNOWN 4-neighbor.
    # Hint: iterate over all cells, or use numpy boolean operations on
    # shifted arrays for efficiency.
    frontier_cells = set()
    # 4-neighbor offsets: up, down, left, right
    # neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # TODO: for each cell (r, c) in the grid, check if it is FREE and
    #       if any of its 4-neighbors is UNKNOWN. If so, add (r, c)
    #       to frontier_cells.
    unknown_mask = (grid == UNKNOWN)
    # Create dilated mask for 4-connectivity (up, down, left, right)
    dilated_unknown = np.zeros_like(unknown_mask, dtype=bool)
    dilated_unknown[:-1, :] |= unknown_mask[1:, :]  # up
    dilated_unknown[1:, :] |= unknown_mask[:-1, :]  # down
    dilated_unknown[:, :-1] |= unknown_mask[:, 1:]  # left
    dilated_unknown[:, 1:] |= unknown_mask[:, :-1]  # right
    # Frontier cells: FREE cells adjacent to UNKNOWN
    frontier_mask = (grid == FREE) & dilated_unknown
    frontier_cells = set(zip(*np.where(frontier_mask)))


    # Step 2: Cluster frontier cells into contiguous regions using BFS.
    # Use 4-connectivity (same neighbor offsets as above).
    # For each unvisited frontier cell, run a BFS/flood-fill to collect
    # all connected frontier cells into one cluster.
    visited = set()
    clusters = []
    # TODO: for each cell in frontier_cells that is not in visited:
    #       - start a BFS using a deque
    #       - collect all connected frontier cells into a cluster list
    #       - mark each visited cell in the visited set
    #       - append the cluster to clusters
    visited = set()
    clusters = []
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for start_cell in frontier_cells:
        if start_cell in visited:
            continue
        
        # BFS to find connected component
        cluster = []
        queue = deque([start_cell])
        visited.add(start_cell)
        
        while queue:
            r, c = queue.popleft()
            cluster.append((r, c))
            
            # Check 4-neighbors
            for dr, dc in neighbor_offsets:
                nr, nc = r + dr, c + dc
                if (0 <= nr < H and 0 <= nc < W and 
                    (nr, nc) in frontier_cells and 
                    (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        clusters.append(cluster)

    # Step 3: Filter and return.
    # Only keep clusters with >= FRONTIER_MIN_SIZE cells.
    # Wrap each valid cluster in a FrontierRegion object.
    # TODO: return [FrontierRegion(cluster) for cluster in clusters
    #               if len(cluster) >= FRONTIER_MIN_SIZE]

    # STUB (remove this line when you implement the above):
    valid_clusters = [cluster for cluster in clusters if len(cluster) >= FRONTIER_MIN_SIZE]
    return [FrontierRegion(cluster) for cluster in valid_clusters]
    #return detect_frontiers_random(occ_grid)

    # ---- END YOUR CODE (Part 1) ----


# =============================================================================
# Part 2a: Nearest-Frontier Goal Selection (10 points)
# =============================================================================

def select_goal_nearest(frontier_regions, occ_grid, state):
    """
    Select the nearest reachable frontier region by path cost.

    Steps:
      1. Get the robot's current grid position from state (robot_x, robot_y).
      2. For each frontier region (skip those whose centroid is blacklisted):
         a. Snap the centroid to the nearest FREE cell if needed.
         b. Plan a path using plan_path(occ_grid, robot_rc, goal_rc).
         c. Track the goal with minimum path cost.
      3. Return the goal (row, col) with the shortest path, or None if
         no frontier is reachable.

    Args:
        frontier_regions: List of FrontierRegion objects.
        occ_grid: OccupancyGrid instance.
        state: ExplorationState with .robot_x, .robot_y, .blacklisted_goals.

    Returns:
        (row, col) of the best goal, or None if no reachable frontier exists.
    """
    # ---- START YOUR CODE (Part 2a) ----

    if not frontier_regions:
        return None

    # TODO: Convert robot's world position (meters) to grid coordinates.
    # Recall: row corresponds to y, col corresponds to x.
    robot_row = int(state.robot_y / CELL_SIZE)
    robot_col = int(state.robot_x / CELL_SIZE)
    robot_rc = (robot_row, robot_col)

    # Find the nearest frontier by path cost.
    best_goal = None
    best_cost = float('inf')

    for fr in frontier_regions:
        # Skip blacklisted goals
        if fr.centroid in state.blacklisted_goals:
            continue

        goal = fr.centroid

        # Snap centroid to nearest FREE cell if it's not FREE already
        if not occ_grid.is_free(goal[0], goal[1]):
            # BFS outward to find nearest FREE cell
            queue = deque([(goal[0], goal[1], 0)])
            visited = {goal}
            snapped = False
            
            while queue and not snapped:
                r, c, dist = queue.popleft()
                if dist > 3:
                    break
                
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in visited and occ_grid.is_in_bounds(nr, nc):
                        visited.add((nr, nc))
                        if occ_grid.is_free(nr, nc):
                            goal = (nr, nc)
                            snapped = True
                            break
                        queue.append((nr, nc, dist + 1))

        # Skip if the snapped goal is also blacklisted
        if goal in state.blacklisted_goals:
            continue

        # Calculate Euclidean distance
        # dist = math.sqrt((robot_row - goal[0])**2 + (robot_col - goal[1])**2)
        # if dist < best_dist:
        #     best_dist = dist
        #     best_goal = goal
        path, cost = plan_path(occ_grid, robot_rc, goal)
        if path is None:
            continue

        if cost < best_cost:
            best_cost = cost
            best_goal = goal

    return best_goal

    # ---- END YOUR CODE (Part 2a) ----


# =============================================================================
# Part 2b: Custom Goal Selection Strategy (6 points)
# =============================================================================

def select_goal_custom(frontier_regions, occ_grid, state):
    """
    Your custom goal selection strategy.

    Requirements:
      - Must outperform select_goal_nearest on at least ONE map.
      - Should consider factors beyond pure distance (e.g., frontier size,
        information gain, unexplored area density).

    Briefly describe your strategy in a comment or docstring.

    Args:
        frontier_regions: List of FrontierRegion objects.
        occ_grid: OccupancyGrid instance.
        state: ExplorationState with .robot_x, .robot_y, .blacklisted_goals.

    Returns:
        (row, col) of the best goal, or None if no reachable frontier exists.
    """
    # ---- START YOUR CODE (Part 2b) ----

    # Your custom strategy should outperform nearest-frontier on at least
    # one map. Here are some ideas to consider (you are not limited to these):
    #
    # - Cost-utility: maximize frontier_size / path_cost — prefer large,
    #   close frontiers over small, distant ones.
    # - Information gain: estimate how much unknown area surrounds each
    #   frontier (e.g., count UNKNOWN cells in a local neighborhood).
    # - Avoid dead-ends: penalize frontiers deep in narrow corridors.
    # - Momentum: prefer frontiers in the robot's current travel direction.
    #
    # You have access to the same tools as in Part 2a: plan_path(),
    # state.robot_x/y, state.blacklisted_goals, occ_grid, etc.
    #
    # Describe your strategy briefly:
    # Strategy: This strategy prioritizes the nearest reachable frontier while 
    # adding a logarithmic bonus for larger clusters, ensuring the robot doesn't 
    # ignore massive unknown areas for tiny, insignificant gaps.

    if not frontier_regions:
        return None

    robot_row = int(state.robot_y / CELL_SIZE)
    robot_col = int(state.robot_x / CELL_SIZE)
    robot_rc = (robot_row, robot_col)

    best_goal = None
    best_score = float('inf')

    for fr in frontier_regions:
        # Skip blacklisted goals
        if fr.centroid in state.blacklisted_goals:
            continue

        goal = fr.centroid

        if not occ_grid.is_free(goal[0], goal[1]):
            # Find the cell in this cluster closest to the robot
            goal = min(fr.cells, key=lambda c: abs(c[0]-robot_row) + abs(c[1]-robot_col))
            
            # If that cell still isn't 'FREE' (e.g. it's UNKNOWN), do a SMALL snap
            if not occ_grid.is_free(goal[0], goal[1]):
                queue = deque([goal])
                visited = {goal}
                snapped = None
                
                # Small radius (5) is enough if we start from an actual frontier cell
                while queue:
                    r, c = queue.popleft()
                    if occ_grid.is_free(r, c):
                        snapped = (r, c)
                        break
                    if abs(r - goal[0]) + abs(c - goal[1]) > 5:
                        continue
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r + dr, c + dc
                        if (nr, nc) not in visited and occ_grid.is_in_bounds(nr, nc):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                if snapped:
                    goal = snapped
                else:
                    continue

        if goal in state.blacklisted_goals:
            continue

        path, cost = plan_path(occ_grid, robot_rc, goal)
        if path is None:
            # Important: blacklist unreachable cave walls to save time next loop
            state.blacklisted_goals.add(fr.centroid)
            continue

        score = cost - (20 * math.log(fr.size + 1))
        
        if score < best_score:
            best_score = score
            best_goal = goal

    return best_goal

    # ---- END YOUR CODE (Part 2b) ----


# =============================================================================
# Part 3: Robust Exploration Loop (12 points)
# =============================================================================

def exploration_step(state, occ_grid, env, frontier_regions, goal_selector):
    """
    Called once per iteration by the framework. Make exploration decisions by
    modifying state in-place.

    Your responsibilities:
      1. If state.current_path exists, validate it using validate_path().
         If invalid, set state.current_path = None (the framework will not
         execute an invalid path, but clearing it triggers replanning).
      2. If no current path, call goal_selector(frontier_regions, occ_grid, state)
         to pick a goal, then plan a path to it with plan_path().
         - If the path is None (unreachable), blacklist that goal and try another.
         - If no reachable frontiers remain, set state.exploration_complete = True.
      3. When you have a valid path, set:
           state.current_path = path           (list of (row,col))
           state.current_path_index = 1        (skip start cell, which is current pos)
         The framework will then execute the path segment (move up to
         CELLS_PER_STEP cells).

    Args:
        state: ExplorationState — mutable. Key fields:
            .robot_x, .robot_y: current position (meters)
            .current_path: list of (row,col) or None
            .current_path_index: int — next cell to visit in current_path
            .blacklisted_goals: set of (row,col)
            .exploration_complete: bool — set True to end exploration
            .step_count: int
        occ_grid: OccupancyGrid instance.
        env: Environment instance — use env.world_to_grid(x, y) for coordinate
             conversion.
        frontier_regions: List of FrontierRegion objects (already detected by
                          framework).
        goal_selector: Callable — one of select_goal_nearest or select_goal_custom.
            Call as: goal = goal_selector(frontier_regions, occ_grid, state)
    """
    # ---- START YOUR CODE (Part 3) ----

    # Step 1: Validate current path.
    if state.current_path is not None:
        remaining = state.current_path[state.current_path_index:]
        if not validate_path(occ_grid, remaining):
            state.current_path = None

    # Step 2: If we still have a valid path, let the framework execute it.
    if state.current_path is not None:
        return

    # Step 3: No valid path — we need to select a new goal and plan.
    if not frontier_regions:
        state.exploration_complete = True
        return

    robot_rc = env.world_to_grid(state.robot_x, state.robot_y)

    # Step 4: Try to find a reachable goal.
    while True:
        goal = goal_selector(frontier_regions, occ_grid, state)
        if goal is None:
            state.exploration_complete = True
            return
        
        path, cost = plan_path(occ_grid, robot_rc, goal)
        if path is not None:
            state.current_path = path
            state.current_path_index = 1
            return

        state.blacklisted_goals.add(goal)

    # ---- END YOUR CODE (Part 3) ----
