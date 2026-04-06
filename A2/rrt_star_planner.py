"""
Assignment #2 Template file
"""

import random
import math
import numpy as np

"""
Problem Statement
--------------------
Implement the planning algorithm called Rapidly-Exploring Random Trees* (RRT*)
for the problem setup provided by the RRT_dubins_problem class.

INSTRUCTIONS
--------------------
1. The only file to be submitted is this file: rrt_star_planner.py. Your
   implementation can be tested by running dubins_path_problem.py (check the
   main function).
2. Read all class and function documentation in dubins_path_problem.py carefully.
   There are plenty of helper functions in the class to ease implementation.
3. Your solution must meet all the conditions specified below.
4. Below are some DOs and DONTs for this problem.

Conditions
-------------------
There are several conditions that must be satisfied for an acceptable solution.
These may or may not be verified by the auto-grading script.

1. The solution loop must not run for more than a certain number of random iterations
   (specified by the class member max_iter). This is mainly a safety
   measure to avoid time-out-related issues and will be set generously.
2. The planning function must return a list of nodes that represent a collision-free path
   from the start node to the goal node. The path states (path_x, path_y, path_yaw)
   specified by each node must define a Dubins-style path and traverse from node i-1 -> node i.
   (READ the documentation for the Node class to understand the terminology).
3. The returned path should have the start node at index 0 and the goal node at index -1.
   The parent node for node i from the list should be node i-1 from the list (i.e.,
   the path should be a valid, continuous list of connected nodes).
4. The node locations must not lie outside the map boundaries specified by
   RRT_dubins_problem.map_area.

DOs and DONTs
-------------------
1. DO NOT rename the file rrt_star_planner.py for submission.
2. DO NOT change the rrt_star_planner function signature.
3. DO NOT import anything other than what is already imported in this file.
4. YOU MAY write additional helper functions in this file to reduce code repetition,
   but these functions can only be used inside the rrt_star_planner function
   (since only the rrt_star_planner function will be imported for grading).
"""


def rrt_star_planner(rrt_dubins, display_map=False):
   """
   Execute RRT* planning using Dubins-style paths. Make sure to populate the node_list.

   Inputs
   -------------
   rrt_dubins  - (RRT_dubins_problem) Class containing the planning
               problem specification.
   display_map - (boolean) Flag for animation on or off (OPTIONAL).

   Outputs
   --------------
   (list of Node) This must be a valid list of connected nodes that form
                  a continuous path from the start node to the goal node.

   NOTE: In order for the rrt_dubins.draw_graph function to work properly, it is
   important to populate rrt_dubins.node_list with all valid RRT nodes.
   """
   # LOOP for max iterations
   goal_node = None
   for i in range(rrt_dubins.max_iter):

      # TODO: Generate a random vehicle state (x, y, yaw)
      # Hint: Ensure the state is within the bounds of rrt_dubins's map_area
      if random.random() < 0.1:  # 10% goal bias
         random_node = rrt_dubins.Node(
            rrt_dubins.goal.x, rrt_dubins.goal.y, rrt_dubins.goal.yaw
         )
      else:
         x = random.uniform(rrt_dubins.map_area[0], rrt_dubins.map_area[1])
         y = random.uniform(rrt_dubins.map_area[2], rrt_dubins.map_area[3])
         yaw = random.uniform(0, 2 * math.pi)
         random_node = rrt_dubins.Node(x, y, yaw)

      # TODO: Add any additional code you require for RRT*


      # TODO: Find an existing node nearest to the random vehicle state
      # example of usage
      # new_node = rrt_dubins.propagate(
      #     rrt_dubins.Node(0, 0, 0), rrt_dubins.Node(1, 1, 0)
      # )
      nearest = None
      nearest_dist = float("inf")
      for node in rrt_dubins.node_list:
         d = math.hypot(node.x - random_node.x, node.y - random_node.y)
         if d < nearest_dist:
            nearest_dist = d
            nearest = node

      if nearest is None:
         continue

      # TODO: Check if the path between nearest node and random state has obstacle collision
      # Add the node to node_list if it is valid
      # example of usage
      # if rrt_dubins.check_collision(new_node):
      #     rrt_dubins.node_list.append(new_node)  # Storing all valid nodes
      new_node = rrt_dubins.propagate(nearest, random_node)
      if new_node is None or not rrt_dubins.check_collision(new_node):
         continue

      # RRT* neighborhood radius (based on map size and current tree size)
      n = max(1, len(rrt_dubins.node_list))
      area = (rrt_dubins.map_area[1] - rrt_dubins.map_area[0]) * (
         rrt_dubins.map_area[3] - rrt_dubins.map_area[2]
      )
      gamma = 2.0 * math.sqrt(area / math.pi)
      radius = max(1.0, gamma * math.sqrt(math.log(n + 1) / (n + 1)))

      # Gather nearby nodes for parent selection + rewiring
      near_nodes = []
      for node in rrt_dubins.node_list:
         d = math.hypot(node.x - new_node.x, node.y - new_node.y)
         if d <= radius:
            near_nodes.append(node)

      # Choose best parent among nearby nodes (minimize cost-to-come)
      best_parent = nearest
      best_cost = new_node.cost
      for near in near_nodes:
         if near is nearest:
            continue
         tentative_cost = rrt_dubins.calc_new_cost(near, new_node)
         if tentative_cost < best_cost:
            candidate = rrt_dubins.propagate(near, new_node)
            if candidate is not None and rrt_dubins.check_collision(candidate):
               best_parent = near
               best_cost = tentative_cost

      # Re-create new_node from best parent to ensure correct path/cost
      new_node = rrt_dubins.propagate(best_parent, new_node)
      if new_node is None or not rrt_dubins.check_collision(new_node):
         continue

      # Add the new node to the tree
      rrt_dubins.node_list.append(new_node)

      # Rewire nearby nodes if going through the new node improves their cost
      for near in near_nodes:
         if near is new_node:
            continue
         new_cost = rrt_dubins.calc_new_cost(new_node, near)
         if new_cost < near.cost:
            candidate = rrt_dubins.propagate(new_node, near)
            if candidate is not None and rrt_dubins.check_collision(candidate):
               near.parent = new_node
               near.cost = candidate.cost
               near.path_x = candidate.path_x
               near.path_y = candidate.path_y
               near.path_yaw = candidate.path_yaw

      # Draw current view of the map
      # PRESS ESCAPE TO EXIT
      if display_map:
         rrt_dubins.draw_graph()

      # TODO: Check if new_node is close enough to the goal
      # Replace 'True' with your goal check logic
      if new_node.is_state_identical(rrt_dubins.goal):
         goal_node = new_node
         print("Iters:", i, ", number of nodes:", len(rrt_dubins.node_list))
         break

   else:
      # This else block executes if the for loop finishes without breaking
      print("Reached max iterations without finding a path")

   # TODO: Return path, which is a list of nodes leading to the goal...
   if goal_node is not None:
      path = []
      node = goal_node
      while node is not None:
         path.append(node)
         node = node.parent
      path.reverse()
      return path
   return None
