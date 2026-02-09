from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

Coord = Tuple[int, int]


def heuristic(a: Coord, b: Coord) -> int:
    # Manhattan distance for 4-connected grid
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid: List[List[int]], start: Coord, goal: Coord) -> Optional[List[Coord]]:
    rows, cols = len(grid), len(grid[0])
    open_heap: List[Tuple[int, Coord]] = []
    heapq.heappush(open_heap, (0, start))

    came_from: Dict[Coord, Optional[Coord]] = {start: None}
    g_score: Dict[Coord, int] = {start: 0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            return reconstruct_path(came_from, current)

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr][nc] == 1:
                continue

            neighbor = (nr, nc)
            tentative = g_score[current] + 1
            if tentative < g_score.get(neighbor, 1_000_000):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                f_score = tentative + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, neighbor))

    return None


def reconstruct_path(came_from: Dict[Coord, Optional[Coord]], current: Coord) -> List[Coord]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def print_grid(grid: List[List[int]], path: Optional[List[Coord]], start: Coord, goal: Coord) -> None:
    path_set = set(path or [])
    for r, row in enumerate(grid):
        line = []
        for c, cell in enumerate(row):
            if (r, c) == start:
                line.append("S")
            elif (r, c) == goal:
                line.append("G")
            elif (r, c) in path_set:
                line.append("*")
            elif cell == 1:
                line.append("#")
            else:
                line.append(".")
        print(" ".join(line))


def main() -> None:
    # 0 = free, 1 = obstacle
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
    ]

    start = (0, 0)
    goal = (7, 7)

    path = astar(grid, start, goal)
    print_grid(grid, path, start, goal)
    if path:
        print(f"\nPath length: {len(path) - 1}")
    else:
        print("\nNo path found.")


if __name__ == "__main__":
    main()
