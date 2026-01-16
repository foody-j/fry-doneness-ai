# RobotDetector - 로봇 프레임 감지

## 개요

튀김 조리 시 로봇 암이 바스켓을 잡으러 들어오는 순간을 감지.
로봇 감지 → 딜레이 → 탈탈 캡처 파이프라인의 트리거 역할.

---

## 감지 원리

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   로봇 암 진입 (금속)                                    │
│        ↓                                                │
│   감지 영역 내 밝은 금속 비율 증가                        │
│        ↓                                                │
│   metal_ratio > threshold → 로봇 감지!                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

- HSV 색상 공간에서 **낮은 채도 + 높은 명도** = 금속
- pot별로 로봇 진입 방향이 다름 → 감지 영역 분리

---

## pot별 설정

### pot1 (camera_0)

```
┌─────────────────┐
│         ■■■■■■■│ ← 우측 상단 감지
│         ■■■■■■■│
│                 │
│      [pot1]     │
│                 │
└─────────────────┘
```

| 항목 | 값 |
|------|-----|
| 카메라 | camera_0 |
| 로봇 진입 | 우측 상단 |
| 감지 영역 | 우상단 1/4 |
| threshold | 0.005 |

### pot2 (camera_1)

```
┌─────────────────┐
│■■■■■■■         │ ← 좌측 상단 감지
│■■■■■■■         │
│                 │
│      [pot2]     │
│                 │
└─────────────────┘
```

| 항목 | 값 |
|------|-----|
| 카메라 | camera_1 |
| 로봇 진입 | 좌측 상단 |
| 감지 영역 | 좌상단 1/4 |
| threshold | 0.05 |

> pot2는 솥 테두리(밝은 금속)가 좌측 상단에 항상 보여서 threshold가 높음

---

## API

### 생성자

```python
from src.simple_checker.robot_detector import RobotDetector, PotType

# pot1 (우측 상단 감지, threshold=0.005)
detector = RobotDetector(pot_type=PotType.POT1)

# pot2 (좌측 상단 감지, threshold=0.05)
detector = RobotDetector(pot_type=PotType.POT2)

# threshold 직접 지정
detector = RobotDetector(pot_type=PotType.POT1, metal_threshold=0.01)
```

### 메서드

#### `reset()`
상태 초기화. 새 조리 세션 시작 시 호출.

#### `detect(image) -> dict`
로봇 프레임 감지.

```python
result = detector.detect(image)
# {
#     "robot_detected": True,      # 로봇 감지 여부
#     "metal_ratio": 0.0523,       # 금속 비율
#     "state": "robot_detected",   # 현재 상태
#     "state_changed": True,       # 상태 변경 여부
#     "region": (640, 0, 1280, 360) # 감지 영역 (x1, y1, x2, y2)
# }
```

#### `detect_from_file(image_path) -> dict`
파일에서 이미지 로드 후 감지.

---

## 상태 머신

```
     ┌──────────────────────────────────────┐
     │                                      │
     ▼                                      │
  [idle] ──[robot_detected=True]──> [robot_detected]
     ▲                                      │
     │                                      │
     └──[robot_detected=False]──────────────┘
```

| 상태 | 설명 |
|------|------|
| `idle` | 대기 중 (로봇 없음) |
| `robot_detected` | 로봇 감지됨 |

`state_changed=True`일 때 트리거로 사용.

---

## 금속 감지 파라미터

```python
# HSV 범위 (기본값)
metal_hsv_lower = (0, 0, 120)    # H: 전체, S: 낮음, V: 높음
metal_hsv_upper = (180, 60, 255)
```

| 채널 | 범위 | 의미 |
|------|------|------|
| H (색조) | 0~180 | 전체 색상 |
| S (채도) | 0~60 | 낮은 채도 (금속) |
| V (명도) | 120~255 | 높은 명도 (밝음) |

---

## CLI 테스트

### 단일 이미지

```bash
python -m src.simple_checker.robot_detector <image_path> [pot1|pot2]
```

### 디렉토리 전체

```bash
# pot1 robot_frame 테스트
python -m src.simple_checker.robot_detector shake/robot_frame pot1

# pot2 taltal 테스트
python -m src.simple_checker.robot_detector shake/taltal pot2
```

출력 예시:
```
Pot type: pot1
Found 43 images

[ROBOT] camera_0_094319_892.jpg - metal: 0.01843
[ROBOT] camera_0_094830_847.jpg - metal: 0.06641
...
Total: 37/43 frames with robot
```

---

## 시각화

```python
from src.simple_checker.robot_detector import visualize_detection

image = cv2.imread("image.jpg")
result = detector.detect(image)

# 시각화 (감지 영역 + 상태 표시)
vis = visualize_detection(image, result, save_path="output.jpg")
```

출력 이미지:
- 녹색 박스: 로봇 감지됨
- 회색 박스: 로봇 없음
- 좌상단: 상태 + metal_ratio 표시

---

## 테스트 결과

### pot1

| 구분 | metal_ratio | 결과 |
|------|-------------|------|
| 로봇 없음 | 0.0004 ~ 0.001 | ✅ No robot |
| 로봇 있음 | 0.02 ~ 0.23 | ✅ ROBOT DETECTED |

### pot2

| 구분 | metal_ratio | 결과 |
|------|-------------|------|
| 로봇 없음 | 0.01 ~ 0.015 | ✅ No robot |
| 로봇 있음 | 0.08 ~ 0.16 | ✅ ROBOT DETECTED |

---

## 실시간 사용 예시

```python
from src.simple_checker.robot_detector import RobotDetector, PotType

detector = RobotDetector(pot_type=PotType.POT1)

def process_frame(frame):
    result = detector.detect(frame)

    # 로봇 진입 감지 (상태 변경 시)
    if result["state_changed"] and result["robot_detected"]:
        print("로봇 진입! 딜레이 후 탈탈 캡처 시작")
        # start_capture_timer(delay=N초)

    # 로봇 퇴장 감지
    if result["state_changed"] and not result["robot_detected"]:
        print("로봇 퇴장. 대기 상태로 복귀")
        detector.reset()
```

---

## 전체 파이프라인 (예정)

```
[로봇 감지] ──> [딜레이] ──> [탈탈 캡처] ──> [색상 측정]
     │              │              │              │
RobotDetector      N초         캡처 트리거    SimpleColorChecker
     │                             │
     └─────── state_changed ───────┘
```

---

## 파일 위치

```
src/simple_checker/
└── robot_detector.py    # RobotDetector, PotType, visualize_detection
```

---

## 참고

- [색상 기반 방법 전체 문서](method2_color_based.md)
- [개발 워크플로우](development_workflow.md)
