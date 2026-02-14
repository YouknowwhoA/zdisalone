# Drone Path Planning (Simulation)

Goal: Fly a simulated drone from A to B while avoiding fixed obstacles, then document the pipeline.

## Status
- [x] Environment setup
- [ ] Simulation takeoff
- [ ] Path planning (A* or RRT)
- [ ] Path following
- [ ] Demo + README polish

## Quick Start (No external dependencies)
Run a minimal A* path planning demo in a 2D grid:

```bash
python3 src/astar_demo.py
```

Animate the path and export waypoints:

```bash
python3 src/astar_animate.py
```

Random map + random tie-break (path varies each run):

```bash
python3 src/astar_animate.py --mode frames --random-map --random-tie-break --obstacle 0.25
```

Smoother motion (denser + smoothed path):

```bash
python3 src/astar_animate.py --mode frames --random-map --random-tie-break --obstacle 0.25 --interp 4 --smooth-rounds 2 --dt 0.2 --vmax 1.5 --amax 1.0
```

Realtime replan demo with a moving obstacle:

```bash
python3 src/astar_animate.py --dynamic-obs --delay 0.2 --replan-every 3
```
