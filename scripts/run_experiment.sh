#!/bin/bash
# =============================================================
# run_experiment.sh
#
# 2026 KIIE conference one-shot launcher.
#
# 3 modes × 2 scenarios × 20 trials = 120 trials, fully unattended.
# (--trials-per-cell 로 조정 가능)
#
# 동작:
#   for mode in rrt_only trajopt_only rrt_trajopt:
#       1) pick_place.launch.py 를 해당 mode 로 백그라운드 기동
#       2) 해당 cell만 experiment_runner.py 로 실행
#       3) launch 종료
#   끝나면 aggregate_results.py + plot_results.py 자동 실행
#
# 사전조건 (사람이 따로 켜두는 터미널):
#   T1: ros2 launch ur_setup_bringup ur_sim_moveit_robotiq_ur16e.launch.py
#   T2: ros2 run trajopt_validation trajopt_server_node
#
# 실행:
#   bash scripts/run_experiment.sh                # 풀 실험
#   bash scripts/run_experiment.sh --smoke         # S1+modes A,C, trials=2 (~3분)
#   bash scripts/run_experiment.sh --modes A,C     # 일부 모드만
# =============================================================

set -euo pipefail

# ---------- 설정 ----------
PKG_DIR="${PKG_DIR:-/home/mk/dev_ws/robot_arm/UR/ur_setup_ws/src/pick_place_module}"
RESULTS_DIR="${RESULTS_DIR:-$PKG_DIR/results}"
EXP_DIR="${EXP_DIR:-$HOME/.ros/pick_place_exp}"
RUN_ID="${RUN_ID:-kiie_$(date +%Y%m%d_%H%M%S)}"
SETTLE_SEC="${SETTLE_SEC:-20}"      # launch 후 action server 준비 대기 (MoveGroupInterface init 포함)

CUMULATIVE_CSV="$EXP_DIR/cumulative.csv"
TRAJ_DIR="$EXP_DIR/trajectories"
TABLES_DIR="$EXP_DIR/tables"
FIGS_DIR="$EXP_DIR/figs"

mkdir -p "$RESULTS_DIR" "$EXP_DIR" "$TRAJ_DIR" "$TABLES_DIR" "$FIGS_DIR"

# ---------- CLI ----------
MODES_TO_RUN="A,B,C"
SCENARIOS_TO_RUN=""
TRIALS_PER_CELL=""
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --modes)            MODES_TO_RUN="$2"; shift 2;;
    --scenarios)        SCENARIOS_TO_RUN="$2"; shift 2;;
    --trials-per-cell)  TRIALS_PER_CELL="$2"; shift 2;;
    --smoke)            SMOKE=1; shift;;
    --run-id)           RUN_ID="$2"; shift 2;;
    -h|--help)
      grep '^#' "$0" | head -40
      exit 0;;
    *)  echo "unknown arg: $1"; exit 1;;
  esac
done

if [[ "$SMOKE" -eq 1 ]]; then
  MODES_TO_RUN="A,C"
  SCENARIOS_TO_RUN="S1"
  TRIALS_PER_CELL="2"
  RUN_ID="smoke_$(date +%H%M%S)"
fi

# ---------- mode → (experiment_mode, csv basename, extra params) ----------
mode_experiment_mode() {
  case "$1" in
    A) echo "rrt_only" ;;
    B) echo "trajopt_only" ;;
    C) echo "rrt_trajopt" ;;
    *) echo ""; return 1 ;;
  esac
}
mode_csv_name() {
  case "$1" in
    A) echo "rrt_only_results.csv" ;;
    B) echo "trajopt_only_results.csv" ;;
    C) echo "rrt_trajopt_results.csv" ;;
  esac
}
mode_extra_params() {
  # rcl SetParameters via runner는 alphabetic key=value 페어 — 여기서는 launch 후
  # 추가 파라미터를 ros2 param set 으로 적용.
  case "$1" in
    A) echo "" ;;
    B) echo "use_trajopt:=true trajopt_use_reduced:=true trajopt_use_free_t:=true" ;;
    C) echo "use_trajopt:=true trajopt_use_reduced:=true trajopt_use_free_t:=true" ;;
  esac
}

# ---------- ROS 환경 ----------
# set -u 가 활성화된 상태에서 setup.bash 를 소싱하면 내부의 미선언 변수
# (AMENT_TRACE_SETUP_FILES 등)가 오류를 일으키므로 소싱 구간만 -u 해제.
set +u
[[ -f /opt/ros/humble/setup.bash ]] && source /opt/ros/humble/setup.bash
[[ -f "$HOME/dev_ws/robot_arm/UR/ur_setup_ws/install/setup.bash" ]] && \
    source "$HOME/dev_ws/robot_arm/UR/ur_setup_ws/install/setup.bash"
[[ -f "$HOME/dev_ws/robot_arm/UR/trajopt_ws/install/setup.bash" ]] && \
    source "$HOME/dev_ws/robot_arm/UR/trajopt_ws/install/setup.bash"
set -u

SCEN_FILE="$(ros2 pkg prefix pick_place_module)/share/pick_place_module/config/scenarios.yaml"
[[ -f "$SCEN_FILE" ]] || SCEN_FILE="$PKG_DIR/config/scenarios.yaml"

echo "==============================================================="
echo "  KIIE 2026 experiment runner"
echo "  run_id       : $RUN_ID"
echo "  modes        : $MODES_TO_RUN"
echo "  scenarios    : ${SCENARIOS_TO_RUN:-<all from yaml>}"
echo "  trials/cell  : ${TRIALS_PER_CELL:-<default from yaml>}"
echo "  results dir  : $RESULTS_DIR"
echo "  cumulative   : $CUMULATIVE_CSV"
echo "==============================================================="

# ---------- 사전 점검: T1, T2 살아있나 ----------
if ! ros2 node list 2>/dev/null | grep -q "/move_group"; then
  echo "[FATAL] /move_group 노드가 없습니다. T1 (ur_sim_moveit_robotiq_ur16e.launch.py) 먼저 실행하세요."
  exit 2
fi
if ! ros2 action list 2>/dev/null | grep -q "/trajopt"; then
  echo "[WARN] /trajopt action server가 없습니다. mode B/C는 실패합니다. T2 (trajopt_server_node) 확인."
fi

# ---------- 모드 루프 ----------
LAUNCH_PID=""
cleanup() {
  if [[ -n "$LAUNCH_PID" ]] && kill -0 "$LAUNCH_PID" 2>/dev/null; then
    echo "[cleanup] killing launch pid=$LAUNCH_PID"
    kill -INT "$LAUNCH_PID" 2>/dev/null || true
    wait "$LAUNCH_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

IFS=',' read -ra MODES_ARR <<< "$MODES_TO_RUN"
for MODE in "${MODES_ARR[@]}"; do
  EXP_MODE=$(mode_experiment_mode "$MODE")
  [[ -z "$EXP_MODE" ]] && { echo "[skip] unknown mode key: $MODE"; continue; }
  CSV_NAME=$(mode_csv_name "$MODE")
  CSV_PATH="$RESULTS_DIR/$CSV_NAME"
  EXTRA=$(mode_extra_params "$MODE")

  echo
  echo ">>> [$(date +%H:%M:%S)] MODE $MODE  ($EXP_MODE)  →  $CSV_PATH"
  rm -f "$CSV_PATH"

  # 1) launch 백그라운드 기동
  ros2 launch pick_place_module pick_place.launch.py \
      use_sim_time:=true \
      trigger_mode:=1 \
      enable_logger:=true \
      experiment_mode:=$EXP_MODE \
      experiment_csv_path:="$CSV_PATH" \
      > "$EXP_DIR/launch_${MODE}.log" 2>&1 &
  LAUNCH_PID=$!
  echo "    launch pid=$LAUNCH_PID  (log: $EXP_DIR/launch_${MODE}.log)"
  sleep "$SETTLE_SEC"

  if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
    echo "[FATAL] launch가 죽었습니다. log 확인: $EXP_DIR/launch_${MODE}.log"
    exit 3
  fi
  if ! ros2 node list 2>/dev/null | grep -q "/pick_place_node"; then
    echo "[FATAL] /pick_place_node가 등록되지 않았습니다. SETTLE_SEC을 늘려보세요."
    exit 4
  fi

  # 2) 추가 파라미터 (B/C 의 trajopt 활성화)
  for KV in $EXTRA; do
    K="${KV%%:=*}"; V="${KV##*:=}"
    echo "    ros2 param set /pick_place_node $K $V"
    ros2 param set /pick_place_node "$K" "$V" >/dev/null
  done

  # 3) 해당 cell만 runner 실행
  RUNNER_ARGS=(
      --scenarios-file "$SCEN_FILE"
      --node-csv  "$CSV_PATH"
      --out       "$CUMULATIVE_CSV"
      --traj-dir  "$TRAJ_DIR"
      --modes     "$MODE"
      --run-id    "$RUN_ID"
  )
  [[ -n "$SCENARIOS_TO_RUN" ]] && RUNNER_ARGS+=(--scenarios "$SCENARIOS_TO_RUN")
  [[ -n "$TRIALS_PER_CELL"  ]] && RUNNER_ARGS+=(--trials-per-cell "$TRIALS_PER_CELL")

  python3 "$PKG_DIR/scripts/experiment_runner.py" "${RUNNER_ARGS[@]}" \
      || echo "[WARN] mode $MODE runner returned non-zero (실패 trial 포함 가능)"

  # 4) launch 종료
  echo "    stopping launch pid=$LAUNCH_PID"
  kill -INT "$LAUNCH_PID" 2>/dev/null || true
  wait "$LAUNCH_PID" 2>/dev/null || true
  LAUNCH_PID=""
  sleep 3
done

# ---------- 집계 + 그림 ----------
echo
echo "==============================================================="
echo "  집계 + 시각화"
echo "==============================================================="
python3 "$PKG_DIR/scripts/aggregate_results.py" "$CUMULATIVE_CSV" \
    --out-dir "$TABLES_DIR" \
    --step pre_grasp \
    --mode-order "$MODES_TO_RUN"

python3 "$PKG_DIR/scripts/plot_results.py" "$CUMULATIVE_CSV" \
    --out-dir "$FIGS_DIR" \
    --traj-dir "$TRAJ_DIR" \
    --mode-order "$MODES_TO_RUN" \
    --fig5-modes "$MODES_TO_RUN"

echo
echo "==============================================================="
echo "  DONE"
echo "  cumulative   : $CUMULATIVE_CSV"
echo "  per-mode CSV : $RESULTS_DIR/{rrt_only,trajopt_only,rrt_trajopt}_results.csv"
echo "  tables       : $TABLES_DIR"
echo "  figures      : $FIGS_DIR"
echo "==============================================================="
