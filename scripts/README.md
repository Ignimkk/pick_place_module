# 2026 KIIE Conference — Experiment Automation

Pipeline that runs **200 unattended pick-place trials** (2 scenarios × 5 modes ×
20 trials) on the UR16e simulation and produces conference-ready tables and
figures.

```
+-----------------------+   PoseStamped /pick_goal /place_goal   +-----------------+
|  experiment_runner.py +--------------------------------------->| pick_place_node |
|   (rclpy, this dir)   |   ros2 param set (mode params)         |   (existing C++)|
|                       |<-- per-trial CSV row appended ---------|                 |
+-----------+-----------+                                        +-----------------+
            |
            | append augmented row
            v
   cumulative.csv  ──►  aggregate_results.py  ──►  table4_main.{csv,tex}
                                                   table5_stats.{csv,tex}
                  ──►  plot_results.py        ──►  figure4_solver_time.{pdf,png}
                                                   figure5_accel_profile.{pdf,png}
```

## Files

| File                                        | Role                                           |
|---------------------------------------------|------------------------------------------------|
| [config/scenarios.yaml](../config/scenarios.yaml) | Scenarios (S1/S2) + mode definitions (A,B,C,C1,C2) |
| [scripts/experiment_runner.py](experiment_runner.py) | rclpy automation runner — 200 trials unattended |
| [scripts/aggregate_results.py](aggregate_results.py) | Tables 4 & 5 — CSV + LaTeX + Mann-Whitney U   |
| [scripts/plot_results.py](plot_results.py)            | Figures 4 & 5 — PDF + PNG @ 200 dpi           |

## Modes

| Key | Label                | experiment_mode | use_trajopt | use_reduced | use_free_t |
|-----|----------------------|-----------------|-------------|-------------|------------|
| A   | rrt_only             | rrt_only        | false       | —           | —          |
| B   | trajopt_only         | trajopt_only    | true        | true        | true       |
| C   | rrt_trajopt          | rrt_trajopt     | true        | true        | true       |
| C1  | rrt_trajopt_full     | rrt_trajopt     | true        | **false**   | true       |
| C2  | rrt_trajopt_fixedT   | rrt_trajopt     | true        | true        | **false**  |

Mode parameters are pushed at the start of each cell with
`rcl_interfaces/srv/SetParameters` against `/pick_place_node`.

## Dependencies

- ROS 2 Humble + MoveIt 2 + Gazebo (project's existing stack)
- `pick_place_module` (this package) — pre-built
- `trajopt_validation` action server — pre-built and running
- Python: `numpy`, `pandas`, `matplotlib`, `scipy`, `pyyaml`

```bash
pip install --user numpy pandas matplotlib scipy pyyaml
```

## Pre-conditions

Three terminals (or a single launch composition) must be up before invoking
the runner:

**Terminal 1 — UR sim + MoveIt2**
```bash
source /opt/ros/humble/setup.bash
source ur_setup_ws/install/setup.bash
ros2 launch ur_setup_bringup ur_sim_moveit_robotiq_ur16e.launch.py
```

**Terminal 2 — TrajOpt action server**
```bash
source ur_setup_ws/install/setup.bash
source trajopt_ws/install/setup.bash
ros2 run trajopt_validation trajopt_server_node
```

**Terminal 3 — pick_place_node + goal_relay**

Start with a known `experiment_csv_path` so the runner can read newly
appended rows from a single, stable file:

```bash
source ur_setup_ws/install/setup.bash
source trajopt_ws/install/setup.bash

EXP_CSV=$HOME/.ros/pick_place_exp/run_node.csv
mkdir -p "$(dirname "$EXP_CSV")"
rm -f "$EXP_CSV"

ros2 launch pick_place_module pick_place.launch.py \
    use_sim_time:=true \
    trigger_mode:=1 \
    enable_logger:=true \
    experiment_mode:=rrt_trajopt \
    experiment_csv_path:="$EXP_CSV"
```

> The runner reads new rows from `$EXP_CSV` after each trial, augments them
> with `(run_id, scenario, mode, trial_idx, seed, pick/place_xyz)`, and
> writes them into the cumulative CSV.

## Run all 200 trials

**Terminal 4 — runner**
```bash
source ur_setup_ws/install/setup.bash

ros2 run pick_place_module experiment_runner.py \
    --scenarios-file $(ros2 pkg prefix pick_place_module)/share/pick_place_module/config/scenarios.yaml \
    --node-csv  $HOME/.ros/pick_place_exp/run_node.csv \
    --out       $HOME/.ros/pick_place_exp/cumulative.csv \
    --traj-dir  $HOME/.ros/pick_place_exp/trajectories
```

`--traj-dir` is optional but **required for Figure 5**; the runner subscribes
to `/joint_trajectory_controller/joint_trajectory` and writes one CSV per
captured trajectory. Each file is named
`<run_id>_<scenario>_<mode>_<trial_idx>_<step>.csv`.

### Subset runs (debugging)

```bash
# Only S1 × {A, C}, 5 trials per cell:
ros2 run pick_place_module experiment_runner.py \
    --node-csv ... --out ... \
    --scenarios S1 --modes A,C --trials-per-cell 5
```

## Two-Mode Gazebo KPI Scripts

For a direct A/B comparison between TrajOpt without an RRT seed and RRT-warmed
TrajOpt, use the dedicated wrappers:

```bash
EXP_CSV=$HOME/.ros/pick_place_exp/node_data.csv

# Terminal 3 must launch pick_place_node with the same path:
ros2 launch pick_place_module pick_place.launch.py \
    use_sim_time:=true \
    trigger_mode:=1 \
    enable_logger:=true \
    experiment_mode:=trajopt_only \
    experiment_csv_path:="$EXP_CSV"

# Terminal 4:
python3 scripts/run_trajopt_only_experiments.py \
    --trials 20 \
    --node-csv "$EXP_CSV" \
    --output results/trajopt_only_gazebo_kpi.csv

python3 scripts/run_rrt_trajopt_experiments.py \
    --trials 20 \
    --node-csv "$EXP_CSV" \
    --output results/rrt_trajopt_gazebo_kpi.csv
```

Both scripts set `/pick_place_node` parameters for their mode, run deterministic
pick/place trials, and append the node KPI columns plus trial metadata to the
output CSV. The `--seed` default is identical for both scripts, so matching
trial indices use the same pick/place poses.

## Reproducibility

- The runner derives a **per-trial seed** from
  `master_seed + scenario_idx*1000 + mode_idx*100 + trial_idx` (`scenarios.yaml`
  → `master_seed: 20260504`). Seeds are written to the cumulative CSV — the
  same `--scenarios-file` and `--run-id` reproduce the same pick/place poses.
- The TrajOpt SLSQP solver and IK plugin do not draw additional randomness
  beyond what the C++ node already controls; pose sampling is the only
  experiment-level RNG consumer.

## Aggregate results

```bash
python3 aggregate_results.py $HOME/.ros/pick_place_exp/cumulative.csv \
    --out-dir $HOME/.ros/pick_place_exp/tables \
    --step pre_grasp        # optional: restrict to one step
```

Outputs:
- `table4_main.csv`  — main comparison (mean ± std), one block per scenario
- `table4_main.tex`  — LaTeX `booktabs` table, paste-ready
- `table5_stats.csv` — paired Mann-Whitney U: B vs C, C vs C1, C vs C2
- `table5_stats.tex` — LaTeX of Table 5 (significant *p* bolded)

## Plot results

```bash
python3 plot_results.py $HOME/.ros/pick_place_exp/cumulative.csv \
    --out-dir $HOME/.ros/pick_place_exp/figs \
    --traj-dir $HOME/.ros/pick_place_exp/trajectories
```

Outputs (PDF + PNG, 200 dpi, KIIE-column-width):
- `figure4_solver_time.{pdf,png}` — boxplot, 2 scenarios × 5 modes
- `figure5_accel_profile.{pdf,png}` — 6-joint $\ddot q$ profile of the
  representative trial (median solver time) per mode for S1, overlaid for A/B/C

Override which trial is overlaid:
```bash
python3 plot_results.py cumulative.csv --traj-dir trajectories/ \
    --fig5-scenario S2 --fig5-modes A,B,C --fig5-step approach
```

## Acceptance-criterion checklist

| Criterion                                                         | Status |
|--------------------------------------------------------------------|--------|
| Single invocation completes 200 trials unattended                  | ✓ runner |
| Re-run with same seed reproduces identical numerical results       | ✓ deterministic seed derivation |
| Tables 4/5 paste-ready as LaTeX                                    | ✓ booktabs |
| Figures with KIIE two-column figsize, both PDF and PNG @ 200 dpi   | ✓ |
| README covers procedure + dependencies                             | ✓ this file |
| Failed trials recorded as-is (no fabrication)                      | ✓ runner appends every row pick_place_node writes, success flag included |
