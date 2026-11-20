import math
import heapq

from grid_map import get_neighbours


def heuristic(cell, goal, kind):
    # estimate of the cost from a cell to the goal (in cell units)
    dr = abs(cell[0] - goal[0])
    dc = abs(cell[1] - goal[1])
    if kind == "manhattan":
        return dr + dc
    return math.sqrt(dr * dr + dc * dc)         # euclidean


def a_star(grid, start, goal, kind="euclidean"):
    # gives back the path, the cost and the cells expanded in order
    queue = []                                  # entries are (f, g, cell)
    heapq.heappush(queue, (heuristic(start, goal, kind), 0.0, start))
    g_cost = {start: 0.0}
    came_from = {start: None}
    closed = set()
    explored = []

    while len(queue) > 0:
        f, g, current = heapq.heappop(queue)
        if current in closed:
            continue
        closed.add(current)
        explored.append(current)

        if current == goal:
            break

        for (neighbour, step_cost) in get_neighbours(current, grid):
            new_g = g + step_cost
            if neighbour not in g_cost or new_g < g_cost[neighbour]:
                g_cost[neighbour] = new_g
                came_from[neighbour] = current
                new_f = new_g + heuristic(neighbour, goal, kind)
                heapq.heappush(queue, (new_f, new_g, neighbour))

    if goal not in closed:
        return None, None, explored

    path = []
    cell = goal
    while cell is not None:
        path.append(cell)
        cell = came_from[cell]
    path.reverse()
    return path, g_cost[goal], explored