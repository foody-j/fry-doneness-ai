# 방법 2: 색상 변화량 기반 튀김 완료 보조 판단

## 개요

딥러닝 없이 HSV 색상 변화량을 측정하여 튀김 익힘 정도를 수치화.
**시간 기반 판정은 MQTT 상위 레벨에서 처리**, 여기선 색상 계산만 담당.

---

## 카메라 구성

| 카메라 | 위치 | 용도 | 방식 |
|--------|------|------|------|
| **0, 2** | pot 위 (조리 중) | 색상 변화 측정 | HSV 분석 |
| **1, 3** | pot 밖 (거치대) | 바스켓 채움 여부 | YOLO 세그멘테이션 |

**이 문서는 camera 0, 2 전용 (색상 변화 측정)**

---

## 역할 분담

```
SimpleColorChecker (이 모듈)
├── 기준 색상 저장 (set_baseline)
├── 현재 색상 추출
├── color_diff 계산
└── progress_pct 계산

MQTT / 상위 시스템 (jetson2)
├── 타이머 관리 (target_time, min_time)
├── 색상 + 시간 조합 → 최종 상태 결정
├── "조리중" / "거의완료" / "완료" 판정
└── 알림 발행
```

---

## SimpleColorChecker API

### 생성자

```python
checker = SimpleColorChecker(color_threshold=25.0)
```

### 메서드

#### `reset()`
새 세션 시작. 기준 색상 초기화.

#### `set_baseline(image) -> dict`
1차 탈탈: 기준 색상 저장.

```python
result = checker.set_baseline(image)
# {
#     "baseline_set": True,
#     "color": {"h_mean": 18.3, "s_mean": 95.2, "v_mean": 180.5, ...}
# }
```

#### `measure(image) -> dict`
색상 변화량 측정.

```python
result = checker.measure(image)
# {
#     "color_diff": 23.5,
#     "current_color": {"h_mean": 22.1, "s_mean": 145.3, "v_mean": 165.8, ...},
#     "progress_pct": 94.0
# }
```

에러 시:
```python
{"error": "baseline_not_set"}
{"error": "food_region_not_found"}
```

---

## 사용 예시

### 오프라인 테스트

```bash
python -m src.simple_checker.demo \
    --session pot1/session_20260107_102151_merged/적어튀김 \
    --use_lift_sequence \
    --color_threshold 25
```

### 실시간 (jetson2 연동)

```python
from src.simple_checker.color_checker import SimpleColorChecker

checker = SimpleColorChecker(color_threshold=25.0)

# 조리 시작
def on_cooking_start(image):
    checker.reset()
    baseline = checker.set_baseline(image)
    print(f"Baseline: {baseline}")

# 탈탈 시점
def on_lift_event(image):
    result = checker.measure(image)
    if "error" not in result:
        print(f"color_diff: {result['color_diff']}")
        print(f"progress: {result['progress_pct']}%")
        # MQTT로 전달 → 상위 시스템에서 시간과 조합하여 판정
```

---

## 프로젝트 구조

```
src/simple_checker/
├── __init__.py
├── color_checker.py    # SimpleColorChecker
├── color_utils.py      # extract_food_region, calculate_color_distance
└── demo.py             # 오프라인 테스트
```

---

## 색상 거리 계산

```python
delta_H = |현재_H - 기준_H|
delta_S = |현재_S - 기준_S|
delta_V = |현재_V - 기준_V|

color_diff = 2.0 * delta_H + 0.5 * delta_S + 1.5 * delta_V
```

- H (색조): 익을수록 증가 (베이지 → 갈색)
- S (채도): 약간 변화
- V (명도): 익을수록 감소 (어두워짐)

---

## 설정

`color_threshold`만 필요. 생성자에서 지정하거나 config에서 로드.

```python
# 직접 지정
checker = SimpleColorChecker(color_threshold=25.0)

# config에서 로드 (jetson2)
threshold = config.get("color_threshold", 25.0)
checker = SimpleColorChecker(color_threshold=threshold)
```

---

## MQTT 연동 예시

```python
class FryingController:
    def __init__(self, config):
        self.checker = SimpleColorChecker(
            color_threshold=config.get("color_threshold", 25.0)
        )
        self.start_time = None
        self.target_time = config.get("target_time", 180)
        self.min_time = config.get("min_time", 120)

    def on_cooking_start(self, image):
        self.checker.reset()
        self.checker.set_baseline(image)
        self.start_time = time.time()

    def on_lift_event(self, image):
        result = self.checker.measure(image)
        if "error" in result:
            return

        elapsed = time.time() - self.start_time
        status = self._determine_status(elapsed, result["progress_pct"])

        self.mqtt_publish("frying/status", {
            "status": status,
            "elapsed": round(elapsed, 1),
            **result
        })

    def _determine_status(self, elapsed, progress_pct):
        if elapsed < self.min_time:
            return "조리중"
        if elapsed >= self.target_time and progress_pct >= 100:
            return "완료"
        if progress_pct >= 90:
            return "거의완료"
        return "조리중"
```

---

## RobotDetector (로봇 프레임 감지)

탈탈 시점을 자동으로 감지하기 위해 로봇 암 진입을 감지.

### 감지 원리

```
로봇 진입 → 딜레이 → 탈탈 캡처 → 색상 측정
```

- 로봇 암이 화면에 들어오면 금속 비율 증가
- pot별로 로봇 진입 방향이 다름

### pot별 설정

| pot | 카메라 | 로봇 진입 방향 | 감지 영역 | threshold |
|-----|--------|---------------|----------|-----------|
| pot1 | camera_0 | 우측 상단 | 우상단 1/4 | 0.005 |
| pot2 | camera_1 | 좌측 상단 | 좌상단 1/4 | 0.05 |

> pot2는 솥 테두리가 좌측 상단에 보여서 threshold가 높음

### API

```python
from src.simple_checker.robot_detector import RobotDetector, PotType

# pot1 (우측 상단 감지)
detector = RobotDetector(pot_type=PotType.POT1)

# pot2 (좌측 상단 감지)
detector = RobotDetector(pot_type=PotType.POT2)
```

#### `reset()`
상태 초기화.

#### `detect(image) -> dict`
로봇 프레임 감지.

```python
result = detector.detect(image)
# {
#     "robot_detected": True,
#     "metal_ratio": 0.0523,
#     "state": "robot_detected",
#     "state_changed": True,
#     "region": (x1, y1, x2, y2)
# }
```

### 상태 머신

```
idle ──[로봇 감지]──> robot_detected ──[로봇 사라짐]──> idle
```

### CLI 테스트

```bash
# pot1 테스트
python -m src.simple_checker.robot_detector shake/robot_frame pot1

# pot2 테스트
python -m src.simple_checker.robot_detector shake/taltal pot2
```

### 시각화

```python
from src.simple_checker.robot_detector import visualize_detection

vis = visualize_detection(image, result, save_path="output.jpg")
```

---

## 전체 파이프라인 (예정)

```
[로봇 감지] ──> [딜레이] ──> [탈탈 캡처] ──> [색상 측정]
     │              │              │              │
RobotDetector      N초         캡처 트리거    SimpleColorChecker
```

---

## 프로젝트 구조

```
src/simple_checker/
├── __init__.py
├── color_checker.py    # SimpleColorChecker
├── color_utils.py      # extract_food_region, calculate_color_distance
├── robot_detector.py   # RobotDetector (로봇 프레임 감지)
└── demo.py             # 오프라인 테스트
```

---

## 요약

| 모듈 | 역할 | 입력 | 출력 |
|------|------|------|------|
| **SimpleColorChecker** | 색상 변화량 측정 | BGR 이미지 | color_diff, progress_pct |
| **RobotDetector** | 로봇 프레임 감지 | BGR 이미지 | robot_detected, metal_ratio |

| pot | 색상 측정 카메라 | 로봇 감지 영역 | threshold |
|-----|-----------------|---------------|-----------|
| pot1 | camera_0 | 우상단 1/4 | 0.005 |
| pot2 | camera_1 | 좌상단 1/4 | 0.05 |
