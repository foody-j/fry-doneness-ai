# 방법 2: 색상 차이 기반 튀김 완료 판단

## 개요

딥러닝 없이 전통적인 컴퓨터 비전으로 튀김 완료를 판단하는 간단한 방법.
투입 시점 대비 색상 변화량 + 최소 조리 시간 조건으로 완료 여부 결정.

---

## 핵심 아이디어

```
튀김 색상 변화: 밝은 베이지 → 황금색 → 진한 갈색

투입 직후 색상 vs 현재 색상 = 색상 차이
색상 차이 >= 임계값 AND 시간 >= 타겟 → 완료!
```

---

## 시스템 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                      입력                                    │
├─────────────────────────────────────────────────────────────┤
│  - 탈탈 이미지 (바스켓 들어올린 시점)                         │
│  - 타임스탬프                                                │
│  - 튀김 종류 (recipe)                                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   1차 탈탈 (투입 직후)                        │
├─────────────────────────────────────────────────────────────┤
│  1. 튀김 영역 추출 (HSV 마스킹)                               │
│  2. 기준 색상 저장:                                          │
│     - H (색조) 평균/표준편차                                  │
│     - S (채도) 평균                                          │
│     - V (명도) 평균                                          │
│  3. 시작 시간 기록                                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   N차 탈탈 (이후)                             │
├─────────────────────────────────────────────────────────────┤
│  1. 현재 튀김 색상 추출                                       │
│  2. 색상 변화량 계산:                                         │
│     delta_H = |현재_H - 기준_H|                               │
│     delta_S = |현재_S - 기준_S|                               │
│     delta_V = |현재_V - 기준_V|                               │
│     color_diff = weighted_sum(delta_H, delta_S, delta_V)     │
│  3. 경과 시간 계산:                                          │
│     elapsed = 현재_시간 - 시작_시간                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      완료 판단                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  if elapsed < target_time * 0.7:                            │
│      return "조리중"        # 아직 시간 부족                  │
│                                                             │
│  elif elapsed < target_time:                                │
│      if color_diff >= threshold * 0.8:                      │
│          return "거의완료"  # 색상 변화 충분, 시간 조금 부족   │
│      else:                                                  │
│          return "조리중"                                     │
│                                                             │
│  else:  # elapsed >= target_time                            │
│      if color_diff >= threshold:                            │
│          return "완료"      # 조건 모두 충족                  │
│      else:                                                  │
│          return "거의완료"  # 시간 됐지만 색상 부족           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                       출력                                   │
├─────────────────────────────────────────────────────────────┤
│  - 상태: "조리중" / "거의완료" / "완료"                       │
│  - 색상 변화량: 23.5 (디버깅용)                               │
│  - 경과 시간: 185초                                          │
│  - 남은 시간 예측: ~15초 (선택적)                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 색상 추출 방법

### 튀김 영역 마스킹

```python
def extract_food_region(image):
    """튀김 영역만 추출 (HSV 마스킹)"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 튀김 색상 범위 (밝은 베이지 ~ 진한 갈색)
    lower = np.array([8, 40, 80])    # 넓은 범위
    upper = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask, hsv
```

### 색상 통계 추출

```python
def extract_color_stats(hsv_image, mask):
    """마스크 영역의 HSV 통계 추출"""
    h, s, v = cv2.split(hsv_image)

    # 마스크 영역만
    h_values = h[mask > 0]
    s_values = s[mask > 0]
    v_values = v[mask > 0]

    if len(h_values) == 0:
        return None

    return {
        'h_mean': np.mean(h_values),
        'h_std': np.std(h_values),
        's_mean': np.mean(s_values),
        'v_mean': np.mean(v_values),
        'v_std': np.std(v_values),
        'pixel_count': len(h_values),
    }
```

---

## 색상 거리 계산

### 방법 1: 가중 유클리드 거리 (권장)

```python
def calculate_color_distance(start_color, current_color):
    """
    색상 변화량 계산

    H (색조): 익을수록 증가 (베이지→갈색)
    S (채도): 약간 변화
    V (명도): 익을수록 감소 (어두워짐)
    """
    delta_h = abs(current_color['h_mean'] - start_color['h_mean'])
    delta_s = abs(current_color['s_mean'] - start_color['s_mean'])
    delta_v = abs(current_color['v_mean'] - start_color['v_mean'])

    # 가중치 (H와 V가 더 중요)
    weights = {'h': 2.0, 's': 0.5, 'v': 1.5}

    distance = (
        weights['h'] * delta_h +
        weights['s'] * delta_s +
        weights['v'] * delta_v
    )

    return distance
```

### 방법 2: 정규화된 거리

```python
def calculate_normalized_distance(start_color, current_color):
    """0~100 범위로 정규화된 거리"""
    delta_h = abs(current_color['h_mean'] - start_color['h_mean']) / 180 * 100
    delta_s = abs(current_color['s_mean'] - start_color['s_mean']) / 255 * 100
    delta_v = abs(current_color['v_mean'] - start_color['v_mean']) / 255 * 100

    # 가중 평균
    distance = 0.5 * delta_h + 0.2 * delta_s + 0.3 * delta_v

    return distance  # 0~100
```

---

## 튀김별 설정

### 설정 파일 (`configs/recipes.yaml`)

```yaml
# 튀김 종류별 조리 파라미터
recipes:
  적어튀김:
    target_time: 180          # 목표 조리 시간 (초)
    min_time: 120             # 최소 조리 시간 (이전엔 완료 불가)
    color_threshold: 25       # 색상 변화 임계값
    description: "작은 생선 튀김"

  돈까스:
    target_time: 240
    min_time: 180
    color_threshold: 30
    description: "두꺼운 돈까스"

  치킨:
    target_time: 300
    min_time: 240
    color_threshold: 35
    description: "뼈있는 치킨"

  고구마튀김:
    target_time: 150
    min_time: 100
    color_threshold: 20
    description: "얇은 고구마"

  튀김만두:
    target_time: 180
    min_time: 120
    color_threshold: 25
    description: "냉동 만두"

  새우튀김:
    target_time: 120
    min_time: 80
    color_threshold: 22
    description: "새우"

# 기본값 (매핑 안 된 튀김)
default:
  target_time: 180
  min_time: 120
  color_threshold: 25
```

---

## 메인 클래스 설계

### `SimpleColorChecker`

```python
class SimpleColorChecker:
    """색상 차이 기반 튀김 완료 판단기"""

    def __init__(self, config_path: str = "configs/recipes.yaml"):
        self.config = load_config(config_path)
        self.reset()

    def reset(self):
        """새 세션 시작"""
        self.start_color = None
        self.start_time = None
        self.recipe = None
        self.history = []

    def set_recipe(self, recipe_name: str):
        """튀김 종류 설정"""
        self.recipe = recipe_name
        self.params = self.config['recipes'].get(
            recipe_name,
            self.config['default']
        )

    def on_first_lift(self, image: np.ndarray, timestamp: float):
        """
        1차 탈탈: 기준 색상 저장

        Args:
            image: BGR 이미지
            timestamp: 유닉스 타임스탬프 또는 초
        """
        mask, hsv = extract_food_region(image)
        self.start_color = extract_color_stats(hsv, mask)
        self.start_time = timestamp

        self.history.append({
            'lift': 1,
            'timestamp': timestamp,
            'color': self.start_color,
            'status': '조리시작',
        })

        return {
            'status': '조리시작',
            'color': self.start_color,
        }

    def check(self, image: np.ndarray, timestamp: float) -> dict:
        """
        탈탈 시점에 완료 체크

        Args:
            image: BGR 이미지
            timestamp: 현재 타임스탬프

        Returns:
            {
                'status': '조리중' | '거의완료' | '완료',
                'color_diff': float,
                'elapsed_sec': float,
                'progress_pct': float,
            }
        """
        if self.start_color is None:
            # 첫 탈탈이면 기준 저장
            return self.on_first_lift(image, timestamp)

        # 현재 색상 추출
        mask, hsv = extract_food_region(image)
        current_color = extract_color_stats(hsv, mask)

        if current_color is None:
            return {'status': '측정실패', 'error': '튀김 영역 검출 실패'}

        # 색상 변화량
        color_diff = calculate_color_distance(self.start_color, current_color)

        # 경과 시간
        elapsed = timestamp - self.start_time

        # 파라미터
        target_time = self.params['target_time']
        min_time = self.params['min_time']
        threshold = self.params['color_threshold']

        # 진행률 계산
        time_progress = min(elapsed / target_time, 1.0)
        color_progress = min(color_diff / threshold, 1.0)
        overall_progress = (time_progress + color_progress) / 2

        # 상태 판단
        status = self._determine_status(
            elapsed, color_diff,
            target_time, min_time, threshold
        )

        # 히스토리 저장
        result = {
            'status': status,
            'color_diff': round(color_diff, 2),
            'elapsed_sec': round(elapsed, 1),
            'progress_pct': round(overall_progress * 100, 1),
            'time_progress': round(time_progress * 100, 1),
            'color_progress': round(color_progress * 100, 1),
            'current_color': current_color,
        }

        self.history.append({
            'lift': len(self.history) + 1,
            'timestamp': timestamp,
            **result,
        })

        return result

    def _determine_status(
        self,
        elapsed: float,
        color_diff: float,
        target_time: float,
        min_time: float,
        threshold: float
    ) -> str:
        """상태 결정 로직"""

        # 최소 시간 미만: 무조건 조리중
        if elapsed < min_time:
            return "조리중"

        # 타겟 시간 미만
        if elapsed < target_time:
            if color_diff >= threshold * 0.9:
                return "거의완료"
            elif color_diff >= threshold * 0.6:
                return "조리중"  # 색상 변화 진행중
            else:
                return "조리중"

        # 타겟 시간 이상
        else:
            if color_diff >= threshold:
                return "완료"
            elif color_diff >= threshold * 0.7:
                return "거의완료"  # 시간은 됐는데 색상이 조금 부족
            else:
                # 시간은 지났는데 색상 변화 부족 → 이상 상황
                return "거의완료"  # 일단 거의완료로 (안전하게)
```

---

## 프로젝트 구조

```
src/
└── simple_checker/
    ├── __init__.py
    ├── color_checker.py      # SimpleColorChecker 클래스
    ├── color_utils.py        # 색상 추출/거리 계산 함수
    └── demo.py               # 테스트 스크립트

configs/
└── recipes.yaml              # 튀김별 설정
```

---

## 구현 순서

### Step 1: 색상 유틸리티 (`color_utils.py`)

- [ ] `extract_food_region()` - 튀김 영역 마스킹
- [ ] `extract_color_stats()` - HSV 통계 추출
- [ ] `calculate_color_distance()` - 색상 거리 계산
- [ ] 시각화 함수 (디버깅용)

### Step 2: 설정 파일 (`recipes.yaml`)

- [ ] 튀김별 target_time, min_time, threshold 정의
- [ ] 기본값 설정

### Step 3: 메인 클래스 (`color_checker.py`)

- [ ] `SimpleColorChecker` 클래스
- [ ] `on_first_lift()` - 기준 색상 저장
- [ ] `check()` - 완료 판단
- [ ] `_determine_status()` - 상태 결정 로직

### Step 4: 테스트 (`demo.py`)

- [ ] 기존 세션 데이터로 테스트
- [ ] 탈탈 이미지 순서대로 입력
- [ ] 결과 출력 및 시각화

### Step 5: 튜닝

- [ ] 실제 데이터로 임계값 조정
- [ ] 튀김별 파라미터 최적화

---

## 테스트 방법

### 단일 세션 테스트

```bash
python src/simple_checker/demo.py \
    --session pot1/session_20260107_102151/적어튀김 \
    --recipe 적어튀김
```

### 예상 출력

```
=== 색상 기반 튀김 완료 판단 테스트 ===
Recipe: 적어튀김
Target time: 180초
Color threshold: 25

[1차 탈탈] 10:23:43
  상태: 조리시작
  기준 색상: H=18.3, S=95.2, V=180.5

[2차 탈탈] 10:26:46 (+183초)
  색상 변화: 15.2
  시간 진행: 102%
  상태: 거의완료

[3차 탈탈] 10:28:15 (+272초)
  색상 변화: 28.7
  시간 진행: 151%
  상태: 완료 ✓
```

---

## 방법 1 vs 방법 2 비교

| 항목 | 방법 1 (딥러닝) | 방법 2 (색상 차이) |
|------|----------------|-------------------|
| **복잡도** | 높음 | 낮음 |
| **학습 데이터** | 필요 (수백~수천) | 불필요 |
| **정확도** | 높음 (학습 후) | 중간 |
| **해석 가능성** | 낮음 | 높음 |
| **튜닝 난이도** | 어려움 | 쉬움 |
| **배포** | GPU 권장 | CPU OK |
| **개발 시간** | 길다 | 짧다 |
| **유지보수** | 복잡 | 단순 |

---

## 권장 사용 시나리오

### 방법 2가 적합한 경우

- 빠른 프로토타입 필요
- 데이터가 부족할 때
- 단순한 튀김 종류 (색상 변화 명확)
- 해석 가능한 결과 필요

### 방법 1이 적합한 경우

- 높은 정확도 필요
- 충분한 학습 데이터 확보
- 복잡한 튀김 (색상 변화 미묘)
- GPU 인프라 있음

---

## 하이브리드 접근 (선택적)

```
방법 2 (색상 차이)로 빠른 체크
    ↓
불확실하면 방법 1 (딥러닝)로 정밀 판단
```

이렇게 하면 효율성과 정확도 둘 다 잡을 수 있음.

---

## 다음 단계

1. **Step 1~4 구현** - Codex에게 전달
2. **실제 데이터 테스트** - 임계값 튜닝
3. **방법 1과 비교** - 어느 게 더 나은지 평가
4. **최종 결정** - 하나 선택 또는 하이브리드
