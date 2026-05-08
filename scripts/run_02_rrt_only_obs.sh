#!/usr/bin/env bash
# =================================================================
# 02. mode=rrt_only  obstacle=obs
#
# 사전 조건 (사용자가 직접 실행):
#   T1: ur_setup_bringup 실행 중
#   T2: trajopt_server_node 실행 중
#   T3: ros2 launch pick_place_module pick_place.launch.py \
#           use_sim_time:=true \
#           experiment_mode:=rrt_only \
#           experiment_csv_path:=${HOME}/.ros/pick_place_exp/node_data.csv
#   장애물: obs 상태 확인
# =================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results"
OUTPUT_CSV="${RESULTS_DIR}/02_rrt_only_obs.csv"
NODE_CSV="${HOME}/.ros/pick_place_exp/node_data.csv"
TRIALS="${TRIALS:-20}"
SEED="${SEED:-20260504}"

# ── ROS2 환경 ──────────────────────────────────────────────────
set +u
source /opt/ros/humble/setup.bash
WS_SETUP="$(cd "${SCRIPT_DIR}/../../../.." && pwd)/install/setup.bash"
[[ -f "${WS_SETUP}" ]] && source "${WS_SETUP}"
set -u

# ── 노드 활성 확인 ─────────────────────────────────────────────
if ! ros2 node list 2>/dev/null | grep -q "/pick_place_node"; then
    echo "[오류] /pick_place_node 가 실행되지 않았습니다."
    echo "  → T3 터미널에서 pick_place.launch.py 를 먼저 실행하세요."
    exit 1
fi

# ── 실험 안내 ─────────────────────────────────────────────────
echo "================================================================="
echo "  실험 #2  |  mode=rrt_only  |  obstacle=obs"
echo "================================================================="
echo ""
echo "  [필수] launch 설정 확인:"
echo "    experiment_mode  := rrt_only"
echo "    experiment_csv_path := ${NODE_CSV}"
echo ""
echo "  [필수] 장애물 상태: obs"
echo ""
echo "  [출력] ${OUTPUT_CSV}"
echo "  [trials] ${TRIALS}  seed=${SEED}"
echo ""
read -rp "  위 조건 확인 후 Enter (취소: Ctrl+C) > "
echo ""

# ── 결과 디렉터리 생성 ─────────────────────────────────────────
mkdir -p "${RESULTS_DIR}"

# ── trial 실행 ────────────────────────────────────────────────
python3 "${SCRIPT_DIR}/trial_runner.py" \
    --mode      "rrt_only" \
    --obstacle  "yes" \
    --trials    "${TRIALS}" \
    --output    "${OUTPUT_CSV}" \
    --node-csv  "${NODE_CSV}" \
    --seed      "${SEED}"

echo ""
echo "[완료] 결과 저장: ${OUTPUT_CSV}"
echo "       분석: python3 ${SCRIPT_DIR}/analyze_results.py ${RESULTS_DIR}"
