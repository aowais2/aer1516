## Assignment 3

This assignmement is for Full frontier-based exploration pipeline from perception to action.

### 1. Occupancy grid and frontier definition
- The robot maintains a grid with `FREE`, `OCCUPIED`, and `UNKNOWN`.
- A **frontier cell** is a `FREE` cell adjacent to at least one `UNKNOWN` cell.
- Frontiers are where explored space meets unexplored space.

### 2. Frontier clustering
- Individual frontier cells are grouped into connected **frontier regions**.
- This uses 4-connectivity BFS/flood-fill.
- Clustering helps avoid chasing isolated frontier pixels and instead targets coherent exploration goals.

### 3. Frontier region representation
- Each region is stored as a `FrontierRegion` with:
  - `.cells`
  - `.size`
  - `.centroid`
- The centroid is a simple representative goal for the region.

### 4. Goal selection strategies
- `select_goal_nearest` chooses a frontier based on proximity from the robot.
  - It converts robot pose to grid coordinates.
  - It ignores already blacklisted goals.
  - It may snap a centroid to the nearest known `FREE` cell.
- `select_goal_custom` is a place to improve on nearest.
  - The current implementation uses **cost-utility**:
    - prefers larger frontiers relative to path cost.
  - This teaches how to trade off exploration value vs travel cost.

### 5. Path planning
- `plan_path` computes a traversable route from robot to frontier goal.
- The file upgrades planning to A* with an admissible heuristic.
- It uses `inflate_grid` to make obstacles safer and avoid close collisions.

### 6. Robust exploration loop
- `exploration_step` is the core control loop.
- It must:
  - validate the existing path as the map changes,
  - keep a valid `state.current_path` if available,
  - otherwise select a new goal and plan to it,
  - blacklist unreachable goals,
  - mark exploration complete when nothing reachable remains.
- This shows how planning and execution are connected through state.

### 7. Practical algorithm lessons
- Frontier exploration is not just finding frontiers; it is:
  - detecting where exploration is useful,
  - grouping and scoring goals,
  - planning safe motion,
  - handling dynamic discovery and replanning.
- The file illustrates the full perception-action loop.

### 8. Key learning outcomes
- How to detect and cluster frontiers in an occupancy map.
- Why frontier selection matters: nearest vs utility-based.
- How path cost and goal quality should both influence decisions.
- Why exploration needs blacklisting and replanning.
- How to keep the robot moving while avoiding dead ends.

## What are blacklisted goals?

`state.blacklisted_goals` is a set of `(row, col)` frontier goal coordinates that the robot should stop trying.

### Why we blacklist goals
- A goal is blacklisted when the robot tries to plan a path to it and fails.
- That means the frontier may be unreachable right now, so retrying it immediately is wasteful.
- Blacklisting prevents repeated failed planning attempts.

## Where blacklisting happens

In your `exploration_step` implementation:

- You call `goal_selector(frontier_regions, occ_grid, state)` to choose a goal.
- Then you call `plan_path(occ_grid, robot_rc, goal)`.
- If `plan_path` returns `None`, you add that goal to `state.blacklisted_goals`.

So the blacklisting decision is made in `exploration_step`, based on failed path planning.

## How blacklisted goals are used

- Both `select_goal_nearest` and `select_goal_random` skip any frontier whose centroid is already in `state.blacklisted_goals`.
- In your `select_goal_custom`, you also skip blacklisted goals in the same way.

## Summary

- `blacklisted_goals` = unreachable frontier goals
- Determined by `exploration_step` when path planning fails
- Used by goal selection to avoid retrying dead goals
