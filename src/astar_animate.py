from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import heapq

Coord = Tuple[int, int]


def heuristic(a: Coord, b: Coord) -> int:
    # Manhattan distance for 4-connected grid
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(
    grid: List[List[int]],
    start: Coord,
    goal: Coord,
    random_tie_break: bool,
) -> Optional[List[Coord]]:
    rows, cols = len(grid), len(grid[0])
    open_heap: List[Tuple[float, float, Coord]] = []
    heapq.heappush(open_heap, (0.0, 0.0, start))

    came_from: Dict[Coord, Optional[Coord]] = {start: None}
    g_score: Dict[Coord, int] = {start: 0}

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
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
                tie = random.random() if random_tie_break else 0.0
                heapq.heappush(open_heap, (float(f_score), tie, neighbor))

    return None


def reconstruct_path(came_from: Dict[Coord, Optional[Coord]], current: Coord) -> List[Coord]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def generate_grid(
    rows: int,
    cols: int,
    obstacle_ratio: float,
    start: Coord,
    goal: Coord,
) -> List[List[int]]:
    grid: List[List[int]] = []
    for r in range(rows):
        row = []
        for c in range(cols):
            if (r, c) in (start, goal):
                row.append(0)
            else:
                row.append(1 if random.random() < obstacle_ratio else 0)
        grid.append(row)
    return grid


def render_grid(
    grid: List[List[int]],
    path: Optional[List[Coord]],
    start: Coord,
    goal: Coord,
    drone: Optional[Coord],
    dynamic_obs: Optional[Coord] = None,
) -> str:
    path_set = set(path or [])
    lines: List[str] = []
    for r, row in enumerate(grid):
        line = []
        for c, cell in enumerate(row):
            if dynamic_obs is not None and (r, c) == dynamic_obs:
                line.append("X")
            elif drone is not None and (r, c) == drone:
                line.append("D")
            elif (r, c) == start:
                line.append("S")
            elif (r, c) == goal:
                line.append("G")
            elif (r, c) in path_set:
                line.append("*")
            elif cell == 1:
                line.append("#")
            else:
                line.append(".")
        lines.append(" ".join(line))
    return "\n".join(lines)


def densify_path(path: List[Coord], steps: int) -> List[Coord]:
    if steps <= 1:
        return path
    dense: List[Coord] = []
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        for s in range(steps):
            t = s / steps
            r = round(r1 + (r2 - r1) * t)
            c = round(c1 + (c2 - c1) * t)
            dense.append((r, c))
    dense.append(path[-1])
    # Remove duplicates caused by rounding
    out: List[Coord] = []
    for p in dense:
        if not out or p != out[-1]:
            out.append(p)
    return out


def smooth_path(path: List[Coord], rounds: int) -> List[Coord]:
    # Simple corner-cutting (Chaikin) on grid points, then round back to cells.
    pts = [(float(r), float(c)) for r, c in path]
    for _ in range(max(0, rounds)):
        if len(pts) < 3:
            break
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            p = pts[i]
            q = pts[i + 1]
            new_pts.append((0.75 * p[0] + 0.25 * q[0], 0.75 * p[1] + 0.25 * q[1]))
            new_pts.append((0.25 * p[0] + 0.75 * q[0], 0.25 * p[1] + 0.75 * q[1]))
        new_pts.append(pts[-1])
        pts = new_pts
    rounded: List[Coord] = []
    for r, c in pts:
        rc = (int(round(r)), int(round(c)))
        if not rounded or rc != rounded[-1]:
            rounded.append(rc)
    return rounded


def export_path(path: List[Coord], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "path_points.csv"
    json_path = out_dir / "path_points.json"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "row", "col"])
        for i, (r, c) in enumerate(path):
            writer.writerow([i, r, c])

    with json_path.open("w") as f:
        json.dump(
            [{"step": i, "row": r, "col": c} for i, (r, c) in enumerate(path)],
            f,
            indent=2,
        )


def time_parameterize(
    path: List[Coord],
    vmax: float,
    amax: float,
    dt: float,
) -> List[Dict[str, float]]:
    # 1D timing along path length with trapezoidal velocity profile.
    if len(path) == 0:
        return []

    distances = [0.0]
    for i in range(1, len(path)):
        pr, pc = path[i - 1]
        r, c = path[i]
        ds = ((r - pr) ** 2 + (c - pc) ** 2) ** 0.5
        distances.append(distances[-1] + ds)

    total = distances[-1]
    if total == 0:
        return [
            {
                "step": 0,
                "t_sec": 0.0,
                "row": path[0][0],
                "col": path[0][1],
                "vx_cell_per_s": 0.0,
                "vy_cell_per_s": 0.0,
                "speed": 0.0,
                "accel": 0.0,
            }
        ]

    t_accel = vmax / amax
    d_accel = 0.5 * amax * t_accel * t_accel
    if 2 * d_accel >= total:
        # Triangle profile
        t_accel = (total / amax) ** 0.5
        t_flat = 0.0
        vmax_eff = amax * t_accel
    else:
        # Trapezoid profile
        d_flat = total - 2 * d_accel
        t_flat = d_flat / vmax
        vmax_eff = vmax

    total_time = 2 * t_accel + t_flat
    samples: List[Dict[str, float]] = []
    t = 0.0
    s = 0.0
    i = 0
    while t <= total_time + 1e-6:
        if t < t_accel:
            v = amax * t
            a = amax
            s = 0.5 * amax * t * t
        elif t < t_accel + t_flat:
            v = vmax_eff
            a = 0.0
            s = d_accel + vmax_eff * (t - t_accel)
        else:
            t_dec = t - t_accel - t_flat
            v = max(vmax_eff - amax * t_dec, 0.0)
            a = -amax
            s = d_accel + (t_flat * vmax_eff) + (vmax_eff * t_dec - 0.5 * amax * t_dec * t_dec)

        # Find nearest path index for this distance
        while i + 1 < len(distances) and distances[i + 1] < s:
            i += 1
        r, c = path[i]
        if i == 0:
            vx = 0.0
            vy = 0.0
        else:
            pr, pc = path[i - 1]
            if distances[i] - distances[i - 1] > 0:
                vx = (r - pr) / (distances[i] - distances[i - 1]) * v
                vy = (c - pc) / (distances[i] - distances[i - 1]) * v
            else:
                vx = 0.0
                vy = 0.0

        samples.append(
            {
                "step": len(samples),
                "t_sec": round(t, 3),
                "row": r,
                "col": c,
                "vx_cell_per_s": round(vx, 3),
                "vy_cell_per_s": round(vy, 3),
                "speed": round(v, 3),
                "accel": round(a, 3),
            }
        )
        t += dt

    return samples


def export_path_named(
    path: List[Coord],
    out_dir: Path,
    name: str,
    dt: float,
    vmax: float,
    amax: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    json_path = out_dir / f"{name}.json"

    samples = time_parameterize(path, vmax=vmax, amax=amax, dt=dt)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["step", "t_sec", "row", "col", "vx_cell_per_s", "vy_cell_per_s", "speed", "accel"]
        )
        for s in samples:
            writer.writerow(
                [
                    s["step"],
                    s["t_sec"],
                    s["row"],
                    s["col"],
                    s["vx_cell_per_s"],
                    s["vy_cell_per_s"],
                    s["speed"],
                    s["accel"],
                ]
            )

    with json_path.open("w") as f:
        json.dump(samples, f, indent=2)


def path_length(path: List[Coord]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(path)):
        pr, pc = path[i - 1]
        r, c = path[i]
        total += ((r - pr) ** 2 + (c - pc) ** 2) ** 0.5
    return total


def clear_screen() -> None:
    # Try ANSI clear; if not a TTY, fall back to blank lines
    if sys.stdout.isatty():
        sys.stdout.write("\033[H\033[J")
        sys.stdout.flush()
    else:
        print("\n" * 40)


def animate_clear(
    grid: List[List[int]],
    path: List[Coord],
    start: Coord,
    goal: Coord,
    delay: float,
    dynamic_obs: Optional[Coord] = None,
) -> None:
    for i, point in enumerate(path):
        frame = render_grid(grid, path, start, goal, point, dynamic_obs=dynamic_obs)
        clear_screen()
        print(frame)
        print(f"\nStep {i + 1}/{len(path)}  Drone at {point}")
        time.sleep(delay)


def animate_frames(
    grid: List[List[int]],
    path: List[Coord],
    start: Coord,
    goal: Coord,
    delay: float,
    step: bool,
    dynamic_obs: Optional[Coord] = None,
) -> None:
    for i, point in enumerate(path):
        frame = render_grid(grid, path, start, goal, point, dynamic_obs=dynamic_obs)
        print(f"\n--- Frame {i + 1}/{len(path)}  Drone at {point} ---")
        print(frame)
        if step:
            input("Press Enter for next frame...")
        else:
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="A* path planning animation")
    parser.add_argument(
        "--mode",
        choices=["clear", "frames"],
        default="clear",
        help="clear: redraw in place, frames: print each frame",
    )
    parser.add_argument("--delay", type=float, default=0.2, help="seconds per frame")
    parser.add_argument(
        "--step",
        action="store_true",
        help="step through frames with Enter (frames mode only)",
    )
    parser.add_argument("--rows", type=int, default=8, help="grid rows")
    parser.add_argument("--cols", type=int, default=8, help="grid cols")
    parser.add_argument(
        "--random-map",
        action="store_true",
        help="generate a random obstacle map",
    )
    parser.add_argument(
        "--obstacle",
        type=float,
        default=0.25,
        help="obstacle ratio for random map (0-0.6 recommended)",
    )
    parser.add_argument(
        "--random-tie-break",
        action="store_true",
        help="randomize tie breaks among equal f-scores",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for reproducible runs",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=10,
        help="max attempts to find a path on random maps",
    )
    parser.add_argument(
        "--interp",
        type=int,
        default=1,
        help="interpolation steps between waypoints (>=1)",
    )
    parser.add_argument(
        "--smooth-rounds",
        type=int,
        default=0,
        help="number of smoothing rounds (0 = off)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.2,
        help="time step in seconds for trajectory timing",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=1.5,
        help="max speed (cells per second)",
    )
    parser.add_argument(
        "--amax",
        type=float,
        default=1.0,
        help="max accel (cells per second^2)",
    )
    parser.add_argument(
        "--dynamic-obs",
        action="store_true",
        help="enable a simple moving obstacle and replan in real time",
    )
    parser.add_argument(
        "--replan-every",
        type=int,
        default=3,
        help="replan every N steps when dynamic obstacle is on",
    )
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    start = (0, 0)
    goal = (args.rows - 1, args.cols - 1)

    if args.random_map:
        path = None
        grid = []
        for _ in range(max(1, args.retries)):
            grid = generate_grid(args.rows, args.cols, args.obstacle, start, goal)
            path = astar(grid, start, goal, args.random_tie_break)
            if path:
                break
    else:
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
        path = astar(grid, start, goal, args.random_tie_break)
    if not path:
        print("No path found.")
        return

    export_path(path, Path("outputs"))
    smooth = smooth_path(path, args.smooth_rounds) if args.smooth_rounds > 0 else path
    dense = densify_path(smooth, max(1, args.interp))
    if dense != path:
        export_path_named(dense, Path("outputs"), "path_points_smooth", args.dt, args.vmax, args.amax)

    if not args.dynamic_obs:
        if args.mode == "clear":
            animate_clear(grid, dense, start, goal, args.delay)
        else:
            animate_frames(grid, dense, start, goal, args.delay, args.step)
        length = path_length(dense)
        samples = time_parameterize(dense, vmax=args.vmax, amax=args.amax, dt=args.dt)
        total_time = samples[-1]["t_sec"] if samples else 0.0
        print(f"\nTrajectory length (cells): {length:.2f}")
        print(f"Estimated time (s): {total_time:.2f}")
        return

    # --- Dynamic obstacle demo with periodic replan ---
    # Simple obstacle: moves horizontally across row 3.
    obs_row = min(3, args.rows - 2)
    obs_cols = list(range(1, args.cols - 1))
    obs_index = 0
    current = start
    steps = 0

    while current != goal:
        # Update dynamic obstacle position
        obs = (obs_row, obs_cols[obs_index])
        obs_index = (obs_index + 1) % len(obs_cols)

        # Build a planning grid that includes the moving obstacle
        planning_grid = [row[:] for row in grid]
        planning_grid[obs[0]][obs[1]] = 1

        # Replan periodically or at the beginning
        if steps % max(1, args.replan_every) == 0:
            planned = astar(planning_grid, current, goal, args.random_tie_break)
            if not planned:
                print("No path found during replan.")
                break
            # Skip current position in the path
            planned = planned[1:]

        if not planned:
            print("No steps left in plan.")
            break

        next_step = planned.pop(0)
        current = next_step
        steps += 1

        frame = render_grid(grid, None, start, goal, current, dynamic_obs=obs)
        clear_screen()
        print(frame)
        print(f"\nRealtime step {steps}  Drone at {current}  Obstacle at {obs}")
        time.sleep(args.delay)

    print("\nDynamic obstacle demo complete.")
    print("\nExported:")
    print("outputs/path_points.csv")
    print("outputs/path_points.json")


if __name__ == "__main__":
    main()
