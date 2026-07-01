# Gazebo grasp RTF 회복 문서

UR16e + Robotiq 2F-85 + alphabet pick-and-place 시뮬레이션에서, gripper 가
letter 를 잡는 순간 **RTF (Real-Time Factor) 가 0.1~0.2 까지 떨어지는** 문제를
해결한 과정과 최종 구조를 정리한다.

---

## 1. 문제 정의

- **증상**: grasp (gripper close) 직후부터 release (gripper open) 사이의 구간에서
  Gazebo RTF 가 0.1~0.2 수준으로 급락. wall-clock 기준 동작이 5~10 배 느려짐.
- **영향**:
  - `gripper_timeout_sec` 같은 wall-clock 타임아웃이 sim 시간 환산하면 부족해져
    "Failed to open gripper" 류 false-negative 가 발생.
  - 전체 pick-and-place 시퀀스 (9 letter) 의 실측 소요 시간이 비현실적으로 증가.
- **원인 (요약)**: Robotiq finger mesh ↔ alphabet collision 사이의 contact
  해석이 매 physics step (1 ms) 마다 수많은 contact point 를 만들어 LCP 솔버를
  부담시킴. DART (Fortress 기본 엔진) 는 이 부담을 단순 SDF 옵션으로 줄일 수 없음.

## 2. 시도했으나 효과가 없었던 접근들

| 시도 | 내용 | 결과 |
|------|------|------|
| max_contacts 제한 | 각 alphabet collision 에 `<max_contacts>2</max_contacts>` 삽입 | **무효** — Fortress 의 DART 엔진은 ODE 특화 옵션을 무시 |
| alphabet collision 단순화 (mesh → box) | STL collision 을 box 여러 개로 근사 | 약간 개선되었으나 RTF 회복엔 부족 |
| 솔버 iteration 감소, max_step_size 증가 | physics 정밀도 하향 | 전체 sim 부담은 조금 줄지만 grasp 시 RTF 회복에는 미미. 다른 motion 정밀도까지 손상 |
| Bullet 엔진 전환 | physics 플러그인 교체 | Fortress 의 bullet 은 featherstone 미포함, 효과 미지수 |

→ 결국 **fixed joint 로 contact 자체를 우회** 하는 정공법이 필요했다.

## 3. 최종 해법 — `DetachableJoint` 시스템 플러그인

Ignition Fortress 6.16 에 포함된
`ignition::gazebo::systems::DetachableJoint` 플러그인을 사용해, grasp 시점에
letter 를 robot 의 한 link 에 **runtime 으로 생성되는 fixed joint** 로
직접 묶는다. joint 가 두 body 의 상대 pose 를 강제하면 DART 는 그 사이의
contact 를 풀 필요가 없어 해석 비용이 사라진다.

### 3.1 아키텍처 개요

```
┌──────────────────────────────┐         ┌──────────────────────────────┐
│  edge_brain_dispatcher_node  │         │       pick_place_node        │
│  (ROS 2 / rclcpp)            │         │  (ROS 2 + MoveIt action)     │
└──────────┬───────────────────┘         └──────────────┬───────────────┘
           │ /pick_phase_done                           │
           │   (Bool, ROS)                              │
           │                                            │
           │ /grasp/attach/<letter>                     │ /grasp/release_all
           │   (Empty, ROS)                             │   (Empty, ROS)
           ▼                                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          ros_gz_bridge                                │
│  ROS → Ignition (`]`) for attach/release; Ign → ROS (`[`) for state │
└──────────────────────────────┬───────────────────────────────────────┘
                               │  ignition::transport
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│      DetachableJoint x 9 (alphabet_E1, D, G, E2, B, R, A, I, N)      │
│        parent_link = wrist_3_link  /  child_model = <letter>         │
│        attach_topic = /grasp/attach/<letter>                         │
│        detach_topic = /grasp/release_all   (공유)                     │
│        output_topic = /grasp/state/<letter>                          │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 한 letter 의 시퀀스 흐름

```
dispatcher          goal_relay_node      pick_place_node           DetachableJoint
────────────────────────────────────────────────────────────────────────────────
pub /pick_goal ──┐
pub /place_goal ─┴─→ receive (mode 1)
                     → call /pick action ──→ pick action runs
                                              (descend → close → retreat)
                                              result
                     pub /pick_phase_done(true) ─→ onPickPhaseDone
                                                    pub /grasp/attach/<letter>
                                                    ──────────────────────────→ joint 생성
                                                                                 (wrist_3 ↔ letter)
                     sleep 200ms (grace)
                     → call /place action ──→ place action runs
                                              (pre-place: long arm motion)
                                              ※ 이 구간에서 RTF 회복 효과 최대
                                              Step 5 (release):
                                                pub /grasp/release_all
                                                ─────────────────────────────→ joint 제거
                                                controlGripper(open)
                                              (retreat)
                                              result
                     pub /pickplace_done ─→ next step
```

## 4. 해결까지의 진단 단계

가장 큰 시간이 걸린 부분은 "왜 attach 가 작동하지 않는가" 의 디버깅이었다.
세 가지 별도의 함정을 차례로 발견했다.

### 함정 1 — plugin `name` 속성 임의 변경

처음에 9 개 인스턴스를 unique 하게 만들려고 plugin `name` 에 letter suffix 를
붙였더니 (`...DetachableJoint_alphabet_E1` 등), 플러그인 로더가
"Could not find a plugin with that name" 에러로 거부.

원인: `name` 은 SDF 식별자가 아니라 **C++ 클래스 이름** 으로 plugin loader 가
조회. 인스턴스 구분은 attach_topic / child_model 로 해야 함.

수정: `name="ignition::gazebo::systems::DetachableJoint"` 그대로 유지하고
각 인스턴스는 별도 `<gazebo>` 블록 안에 배치.

### 함정 2 — `parent_link` 가 URDF→SDF 변환에서 collapse

처음 `parent_link=robotiq_85_base_link`, 그 후 `tool0` 으로 시도. 두 경우 모두
`Link not found in model ur16e_testbed` 에러.

원인: URDF→SDF 변환 시 **fixed joint 로만 연결된 child link 는 부모로
머지된다**. tool0, robotiq_85_base_link, ur_to_robotiq_link 등이 모두
사라짐. 변환된 SDF 의 실제 link 를 확인 (`ign sdf -p` 후 link name 출력) :

| URDF link | SDF 변환 후 |
|---|---|
| `tool0` | ❌ wrist_3_link 로 머지 |
| `ur_to_robotiq_link` | ❌ 머지 |
| `robotiq_85_base_link` | ❌ 머지 |
| `robotiq_85_left_finger_link` | ❌ 머지 |
| `wrist_3_link` | ✅ 유지 (UR arm 의 마지막 revolute child) |
| `robotiq_85_*_knuckle_link` | ✅ 유지 (revolute) |
| `robotiq_85_*_finger_tip_link` | ✅ 유지 |
| `robotiq_85_*_inner_knuckle_link` | ✅ 유지 |

수정: `parent_link=$(arg tf_prefix)wrist_3_link`. gripper 위에 항상 살아있는
회전축 link 이고 grasp 위치와 충분히 가까워 fixed joint offset 도 합리적.

### 함정 3 — ROS topic ↔ ign topic 분리

`ros2 topic echo /grasp/state/<letter>` 가 비어있는 데 비해 `ign topic -l |
grep grasp` 에 state 토픽이 안 보임. plugin 이 출력 토픽을 publish 하지
않는다는 의미.

원인:
- dispatcher 의 ROS publisher 가 `/grasp/attach/<letter>` 를 발행해도
  DetachableJoint plugin 은 **ignition::transport** 를 통해 subscribe 하므로
  메시지가 닿지 않음.
- ROS 와 Ignition 은 다른 transport — bridge 가 명시적으로 필요.

수정: launch 의 `ros_gz_bridge` 인자에 9 개 attach 토픽과 release_all 토픽을
**ROS→Ign 방향 (`]`)** 으로 매핑 추가. 상태 모니터링용 state 토픽은 Ign→ROS
방향 (`[`) 매핑.

```python
attach_bridge_args = [
    f"/grasp/attach/{n}@std_msgs/msg/Empty]ignition.msgs.Empty"
    for n in alphabet_names
]
state_bridge_args = [
    f"/grasp/state/{n}@std_msgs/msg/String[ignition.msgs.StringMsg"
    for n in alphabet_names
]
# /grasp/release_all 도 ROS→Ign 방향
```

### 함정 4 — plugin 의 `attachRequested` 기본값이 `true`

위 세 함정 해결 후, 이번에는 **모든 alphabet 이 spawn 직후 robot 에 매달려
같이 떨어지는** 새 현상.

원인: DetachableJoint 헤더에서 `attachRequested{true}` 로 초기화됨.
즉 plugin 은 **첫 PreUpdate 에서 외부 트리거 없이 자동 attach 를 시도**한다.
alphabet 9 개가 sim 에 나타나는 순간 모두 wrist_3_link 에 fixed joint 로
묶여 robot 동작에 끌려감.

수정: alphabet spawn 전에 `/grasp/release_all` 을 미리 spam 발행하여 모든
plugin 의 `detachRequested` 도 `true` 로 만들어 둠. plugin 의 PreUpdate
한 tick 안에 attach 분기 → detach 분기가 **순차 실행** 되므로, alphabet 이
발견되어 attach 가 일어나는 같은 tick 에 즉시 detach 가 처리되어 letter 가
원위치에 그대로 머무름.

launch 의 `OnProcessExit(target_action=gz_spawn_entity)` 핸들러에
`ExecuteProcess` 를 추가 — `ign topic -t /grasp/release_all -m
ignition.msgs.Empty -p ''` 를 0.2 s 간격으로 15 회 반복 발행 (총 3 초).

```python
detach_pre_spam = ExecuteProcess(
    cmd=["bash", "-c",
         "for i in $(seq 1 15); do "
         "ign topic -t /grasp/release_all "
         "-m ignition.msgs.Empty -p '' >/dev/null 2>&1; "
         "sleep 0.2; done"],
    output="screen",
)
```

`detachRequested` 는 한 번 true 로 설정되면 실제 detach 가 실행될 때까지
**계속 true 로 유지** 되므로, spam 종료 후 alphabet spawn 까지 시간 간격이
있어도 효과는 유지됨.

## 5. 파일별 변경 사항

| 파일 | 변경 |
|---|---|
| `ur_setup_ws/urdf/ur16e_robotiq_2f85.urdf.xacro` | `<xacro:if sim_ignition>` 안에 9 개 `DetachableJoint` plugin 인스턴스 추가. `parent_link=wrist_3_link`, attach_topic 은 letter 별 unique, detach_topic 은 공유 (`/grasp/release_all`). |
| `ur_setup_ws/launch/ur_sim_moveit_robotiq_ur16e.launch.py` | (1) `ros_gz_bridge` 인자에 attach × 9 + release_all 의 ROS→Ign 매핑, state × 9 의 Ign→ROS 매핑 추가. (2) `OnProcessExit(gz_spawn_entity)` 핸들러에 `ExecuteProcess` 로 detach pre-spam 추가. (3) alphabet spawn TimerAction 의 period 를 5.0 s 로 늘려 pre-spam 윈도우와 안전한 순서 확보. |
| `pick_place_module/src/goal_relay_node.cpp` | `/pick_phase_done` (std_msgs/Bool) publisher 추가. mode 1 sequence 에서 pick action 결과 직후 발행 + 200 ms grace period (dispatcher 가 attach 처리할 시간). |
| `pick_place_module/src/pick_place_node.cpp` | place Step 5 (release) 의 `controlGripper(open)` **직전** 에 `/grasp/release_all` (std_msgs/Empty) publish. `release_pub_` 멤버 추가, 생성자에서 publisher 초기화. |
| `pick_place_module/src/edge_brain_dispatcher_node.cpp` | (1) `Step` 구조체에 `model_name` 필드 추가 (= Gazebo spawn name = attach 토픽의 suffix). (2) `/pick_phase_done` 구독 + `onPickPhaseDone` 콜백 — 성공 시 현재 letter 의 attach publisher 로 `Empty` 발행. (3) step 진입 시 letter 별 attach publisher 캐시. |

## 6. 검증 방법

```bash
# 1) launch
ros2 launch ur_setup_bringup ur_sim_moveit_robotiq_ur16e.launch.py \
  gazebo_gui:=false alphabet_count:=9

# 2) plugin 정상 로드 확인 — 9 개 모두 보여야 함
ign topic -l | grep '/grasp/state'

# 3) dispatcher 실행
ros2 run pick_place_module edge_brain_dispatcher_node \
  --ros-args -p letter_count:=9

# 4) attach/detach 상태 모니터링
ros2 topic echo /grasp/state/alphabet_E1
#   → pick 직후 data: "attached"
#   → place 직후 data: "detached"

# 5) RTF 측정
ign topic -e -t /world/testbed_world/stats --duration 5 | grep real_time_factor
```

기대 결과:
- attach 직전: contact 미존재, RTF ≈ 1.0
- attach 이후 (pre-place, transit): joint 가 contact 를 우회, **RTF ≈ 1.0 유지**
- detach 후 (gripper open): 정상 grasp release 와 동일하게 letter 가 pallet 위 안착

## 7. 한계 및 알려진 이슈

- DART 의 한계상 joint 로 묶인 두 body 사이도 일부 contact 가 발생할 수
  있으나, 솔버가 빠르게 수렴하여 RTF 영향은 미미함.
- attachRequested 기본값이 `true` 인 점은 Fortress 6.16 plugin 의 디자인
  특이점. Garden/Harmonic 으로 업그레이드 시 동작이 다를 수 있어 detach
  pre-spam 의 필요성을 재검토해야 함.
- pre-spam 동안 launch 콘솔에 `Child Model alphabet_X could not be found`
  경고가 잠시 출력됨 (`suppress_child_warning` 이 첫 PreUpdate 에서만 적용).
  기능에는 영향 없음.
- letter 가 9 개 모두 attach 된 적은 없음 — dispatcher 가 한 번에 하나만
  처리. `/grasp/release_all` 이 모든 plugin 에 broadcast 되어도, attach 되어
  있지 않은 8 개는 idempotent 하게 무시 (소스의 `Already detached` 경로).

## 8. 참고 — 소스 분석 인용

`ignition-gazebo-6.16.0/src/systems/detachable_joint/DetachableJoint.{hh,cc}`
에서 확인한 핵심 로직:

```cpp
// DetachableJoint.hh:138, 141
private: std::atomic<bool> attachRequested{true};   // ← 기본값 true (자동 attach)
private: std::atomic<bool> isAttached{false};

// DetachableJoint.cc PreUpdate (요약)
if (validConfig && !isAttached) {
    if (!attachRequested) return;
    // child_model / child_link 탐색 후 joint 생성
    // 성공 시: attachRequested=false, isAttached=true
}
if (isAttached) {
    if (detachRequested && detachableJointEntity != kNullEntity) {
        // joint 제거, isAttached=false, detachRequested=false
    }
}
```

이 두 분기가 **같은 tick 의 PreUpdate 안에서 순차 실행** 가능하다는 점이
detach pre-spam 트릭의 핵심 근거.
