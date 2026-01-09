# 튀김 조리 완료 판단 AI 시스템

## 프로젝트 개요

급식 주방용 튀김 완료 시점 자동 판단 AI 시스템.
GMSL 카메라로 촬영된 튀김 조리 과정을 분석하여 최적 완료 시점을 예측하고 알림을 제공한다.

## 두 가지 접근 방법

| | 방법 1 (딥러닝) | 방법 2 (색상 차이) |
|---|---|---|
| **핵심** | EfficientNet + LSTM | HSV 색상 변화량 |
| **학습 데이터** | 필요 | 불필요 |
| **정확도** | 높음 | 중간 |
| **복잡도** | 높음 | 낮음 |
| **배포** | GPU 권장 | CPU OK |

---

# 방법 1: 딥러닝 기반 (EfficientNet + LSTM)

## 핵심 인사이트

```
튀김이 기름에 잠김 → 튀김 안 보임, 기포만 보임
튀김을 들어올림 (탈탈) → 튀김이 명확히 보임 ⭐
```

**탈탈 이미지가 핵심 데이터!**

## 기술 스택

- **Framework**: PyTorch
- **Backbone**: EfficientNet-B2 (ImageNet pretrained)
- **Temporal**: Bi-LSTM (선택적)
- **기포 분석**: OpenCV (딥러닝 불필요)

## 모델 아키텍처 (3-스트림)

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ 탈탈 스트림  │  │ 기포 스트림  │  │ 튀김 종류   │
│   (메인)    │  │   (보조)    │  │  (조건부)   │
├─────────────┤  ├─────────────┤  ├─────────────┤
│EfficientNet │  │   OpenCV    │  │ Embedding   │
│    -B2      │  │ 상대적 변화  │  │ (16d)       │
│   (256d)    │  │ (16d)       │  │             │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       └────────────────┼────────────────┘
                        ↓
                  Concat (288d)
                        ↓
                  Bi-LSTM (선택적)
                        ↓
                  3단계 분류
```

## 프로젝트 구조

```
dku_frying_ai/
├── CLAUDE.md                     # 이 파일
├── requirements.txt              # 의존성
├── configs/
│   └── default.yaml              # 기본 설정
├── docs/
│   └── architecture.md           # 상세 아키텍처
├── data/
│   └── raw/pot1/                 # 원본 데이터
├── src/
│   ├── data/
│   │   ├── bubble_features.py    # 기포 특징 추출 (OpenCV)
│   │   ├── lift_detector.py      # 탈탈 시점 감지
│   │   └── dataset.py            # PyTorch Dataset
│   ├── models/
│   │   └── frying_model.py       # 메인 모델
│   └── training/
│       └── trainer.py            # 학습 파이프라인
├── checkpoints/                  # 모델 체크포인트
└── logs/                         # 학습 로그
```

## 주요 모듈

### 1. 기포 특징 추출 (`bubble_features.py`)
- OpenCV 기반 (딥러닝 불필요)
- 세션 시작 대비 **상대적 변화** 측정
- 특징: bubble_ratio, bubble_delta, surface_calm 등

### 2. 탈탈 시점 감지 (`lift_detector.py`)
- 바스켓이 올라온 프레임 자동 식별
- HSV 색상 + 엣지 기반 감지
- 세션에서 1차/2차/3차 탈탈 추출

### 3. 메인 모델 (`frying_model.py`)
- `FryingModel`: 전체 3-스트림 모델
- `FryingModelSimple`: 단순 버전 (이미지 + 튀김종류만)
- EfficientNet-B2 백본, 선택적 Bi-LSTM

### 4. 데이터셋 (`dataset.py`)
- `FryingDataset`: 단일 탈탈 이미지 샘플
- `FryingDatasetSequence`: 시퀀스 샘플
- 자동 라벨링 (마지막 탈탈 = 완료)

### 5. 학습 파이프라인 (`trainer.py`)
- AdamW + CosineAnnealingLR
- 클래스 가중치 적용
- TensorBoard 로깅
- Early stopping

## 사용법

### 설치
```bash
pip install -r requirements.txt
```

### 학습
```bash
python -m src.training.trainer \
    --data_root pot1 \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4
```

### 탈탈 감지 테스트
```bash
python src/data/lift_detector.py <image_directory>
```

### 기포 특징 테스트
```bash
python src/data/bubble_features.py <image_path>
```

## 분류 기준

| Class | 상태 | 탈탈 시점 | 알림 |
|-------|------|----------|------|
| 0 | 조리중 | 1차 탈탈 | 없음 |
| 1 | 거의완료 | 2차 탈탈 | 노란불 |
| 2 | 완료 | 3차 탈탈 | 빨간불 + 부저 |

## 데이터 구조

```
pot1/
└── session_YYYYMMDD_HHMMSS/
    └── {recipe_name}/
        ├── camera_0/           # 이미지 (3초 간격)
        │   └── camera_0_{timestamp}.jpg
        ├── meta/               # 메타데이터 (recipe만 사용)
        └── session_info.json
```

## 연속 세션 처리 정책

카메라 녹화가 끊어져 하나의 조리가 여러 세션으로 분리된 경우:

### 원칙
- pot1/pot2 구조 유지
- 연속 세션은 `session_*_merged`로 병합
- 원본 세션은 보존 (삭제하지 않음)

### 병합 스크립트 사용법
```bash
python scripts/merge_sessions.py \
  --pot pot1 \
  --food_type 적어튀김 \
  --output session_20260107_102151_merged \
  --sessions session_20260107_102151 session_20260107_103320
```

### 결과 구조
```
pot1/
├── session_20260107_102151/        # 원본 (보존)
├── session_20260107_103320/        # 원본 (보존)
└── session_20260107_102151_merged/ # 병합본
    └── 적어튀김/
        ├── camera_0/               # 모든 이미지 합쳐짐
        └── session_info.json       # merged_from 필드 포함
```

---

# 방법 2: 색상 차이 기반 (딥러닝 X)

## 핵심 로직

```
1차 탈탈: 기준 색상 저장 (HSV)
N차 탈탈: 색상 변화량 + 시간 체크 → 완료 판단

완료 조건:
  색상 변화 >= 임계값 AND 경과 시간 >= 타겟 타임
```

## 흐름

```
[1차 탈탈] 투입 직후
    ↓
기준 색상 저장 (H, S, V 평균)
    ↓
[2차 탈탈]
    색상 차이: 15.2
    경과 시간: 183초 (타겟: 180초)
    → "거의완료"
    ↓
[3차 탈탈]
    색상 차이: 28.7 ✓ (임계값: 25)
    경과 시간: 272초 ✓
    → "완료!"
```

## 프로젝트 구조

```
src/simple_checker/
├── color_utils.py      # 색상 추출/거리 계산
├── color_checker.py    # SimpleColorChecker 클래스
└── demo.py             # 테스트

configs/
└── recipes.yaml        # 튀김별 설정 (target_time, threshold)
```

## 튀김별 설정 예시

```yaml
recipes:
  적어튀김:
    target_time: 180      # 3분
    color_threshold: 25
  돈까스:
    target_time: 240      # 4분
    color_threshold: 30
  치킨:
    target_time: 300      # 5분
    color_threshold: 35
```

## 장점

- 학습 데이터 불필요
- 해석 가능 ("색상 28 변화, 임계값 25 초과")
- CPU만으로 실행 가능
- 튜닝 쉬움 (임계값만 조정)

---

## 참고 문서

- [방법 1 상세 아키텍처](docs/architecture.md)
- [방법 2 상세 플랜](docs/method2_color_based.md)
