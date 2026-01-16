# 개발 환경 및 워크플로우

## 전체 구조

```
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│         WSL2 (Windows PC)       │     │      Jetson Orin Nano (현장)     │
├─────────────────────────────────┤     ├─────────────────────────────────┤
│                                 │     │                                 │
│  Claude Code / Codex            │     │  Claude Code / Codex            │
│  (SSH 또는 로컬)                 │     │  (SSH로 접속해서 사용)           │
│                                 │     │                                 │
│  프로젝트:                       │     │  프로젝트:                       │
│  ├── dku_frying_ai/             │     │  └── jetson-food-ai/            │
│  │   └── 학습 & 알고리즘 개발    │     │      └── 실제 배포 & 운영        │
│  │                              │     │                                 │
│  └── jetson-food-ai/ (clone)    │     │                                 │
│      └── 동기화용               │     │                                 │
│                                 │     │                                 │
└─────────────────────────────────┘     └─────────────────────────────────┘
              │                                       │
              │            git push                   │
              └──────────────────┐                    │
                                 ▼                    │
                           ┌─────────┐                │
                           │ GitHub  │                │
                           └─────────┘                │
                                 │                    │
                                 │  git pull          │
                                 └────────────────────┘
```

---

## AI 어시스턴트 역할 분담

### Claude vs Codex 역할

| AI | 역할 | 설명 |
|----|------|------|
| **Claude** | 플래닝 & 검증 | 설계, 문서 작성, 코드 리뷰, 프롬프트 작성 |
| **Codex** | 코딩 | 실제 코드 구현, 수정, 테스트 |

### WSL2 환경

| AI | 담당 |
|----|------|
| Claude | 알고리즘 설계, 문서 작성, Codex에게 줄 프롬프트 작성 |
| Codex | SimpleColorChecker 구현, 학습 코드 작성, 테스트 실행 |

### Jetson 환경

| AI | 담당 |
|----|------|
| Claude | 통합 방법 검토, 코드 리뷰, 디버깅 방향 제시 |
| Codex | JETSON2_INTEGRATED.py 수정, git pull, 실행 테스트 |

### 작업 흐름 예시

```
1. [WSL2 Claude] 알고리즘 설계 & 프롬프트 작성
       ↓
2. [WSL2 Codex] 코드 구현
       ↓
3. [WSL2 Claude] 코드 검증 & sync 스크립트 실행
       ↓
4. [Jetson Claude] 통합 프롬프트 작성
       ↓
5. [Jetson Codex] JETSON2_INTEGRATED.py 수정
       ↓
6. [Jetson Claude] 테스트 결과 검증
```

---

## 프로젝트 구조

### dku_frying_ai (WSL2) - 학습 & 개발

```
/home/youngjin/dku_frying_ai/
├── CLAUDE.md                 # 프로젝트 개요
├── docs/
│   ├── development_workflow.md   # 이 문서
│   └── method2_color_based.md    # 색상 기반 방법 설계
├── src/
│   ├── simple_checker/       # ★ 핵심: 색상 변화 측정
│   │   ├── color_checker.py
│   │   ├── color_utils.py
│   │   └── demo.py
│   ├── data/                 # 데이터 처리
│   ├── models/               # 딥러닝 모델 (방법1)
│   └── training/             # 학습 파이프라인
├── scripts/
│   ├── sync_to_jetson.sh     # 동기화 스크립트
│   └── merge_sessions.py     # 세션 병합
├── pot1/                     # 학습 데이터 (gitignore)
├── checkpoints/              # 모델 (gitignore)
└── venv/                     # 가상환경 (gitignore)
```

### jetson-food-ai (Jetson) - 배포 & 운영

```
~/jetson-food-ai/
├── jetson1_monitoring/       # Jetson1용
└── jetson2_frying_ai/        # Jetson2용 (튀김 AI)
    ├── simple_checker/       # ← WSL2에서 동기화됨
    │   ├── color_checker.py
    │   ├── color_utils.py
    │   └── demo.py
    ├── JETSON2_INTEGRATED.py # 메인 실행 파일
    ├── frying_segmenter.py   # YOLO 세그멘테이션
    └── config_jetson2.json   # 설정
```

---

## 카메라 구성 (Jetson2)

| 카메라 | 위치 | 용도 | 처리 방식 |
|--------|------|------|-----------|
| 0 | pot1 위 (조리중) | 색상 변화 측정 | SimpleColorChecker |
| 1 | pot1 밖 (거치대) | 바스켓 채움 여부 | YOLO (FILLED/EMPTY) |
| 2 | pot2 위 (조리중) | 색상 변화 측정 | SimpleColorChecker |
| 3 | pot2 밖 (거치대) | 바스켓 채움 여부 | YOLO (FILLED/EMPTY) |

---

## 개발 워크플로우

### Step 1: 알고리즘 개발 (WSL2)

```bash
# WSL2에서
cd /home/youngjin/dku_frying_ai
source venv/bin/activate

# simple_checker 수정 후 테스트
python -m src.simple_checker.demo \
    --session pot1/session_20260107_102151_merged/적어튀김 \
    --use_lift_sequence \
    --color_threshold 25
```

### Step 2: 동기화 (WSL2 → GitHub)

```bash
# WSL2에서
./scripts/sync_to_jetson.sh

# 또는 수동으로:
cp -r src/simple_checker/* ../jetson-food-ai/jetson2_frying_ai/simple_checker/
cd ../jetson-food-ai
git add -A && git commit -m "Sync simple_checker" && git push
```

### Step 3: Jetson에 적용

```bash
# Jetson에 SSH 접속 후
cd ~/jetson-food-ai
git pull
```

### Step 4: Jetson Claude에게 통합 요청

WSL2 Claude가 작성한 프롬프트를 Jetson Claude에게 전달:

```
## Task: JETSON2_INTEGRATED.py에 SimpleColorChecker 통합

1. import 추가
2. __init__에 인스턴스 생성
3. update_frying_left()에 측정 코드 추가
4. GUI에 결과 표시
```

### Step 5: 현장 테스트 (Jetson)

```bash
# Jetson에서
python jetson2_frying_ai/JETSON2_INTEGRATED.py
```

---

## 동기화 스크립트 사용법

### scripts/sync_to_jetson.sh

```bash
# WSL2에서 실행
cd /home/youngjin/dku_frying_ai
./scripts/sync_to_jetson.sh
```

**하는 일:**
1. `src/simple_checker/` → `jetson-food-ai/jetson2_frying_ai/simple_checker/` 복사
2. git add & commit
3. git push (확인 후)

**실행 후:**
- Jetson에서 `git pull` 필요

---

## SimpleColorChecker 사용법

### 기본 사용

```python
from simple_checker.color_checker import SimpleColorChecker

# 생성
checker = SimpleColorChecker(color_threshold=25.0)

# 조리 시작: baseline 설정
checker.reset()
result = checker.set_baseline(frame)
# {"baseline_set": True, "color": {"h_mean": 18.3, ...}}

# 이후 측정
result = checker.measure(frame)
# {"color_diff": 23.5, "current_color": {...}, "progress_pct": 94.0}
```

### 반환값

| 키 | 설명 |
|----|------|
| color_diff | 기준 대비 색상 변화량 |
| current_color | 현재 HSV 값 |
| progress_pct | threshold 대비 진행률 (%) |

### 상태 판정 (MQTT 상위 시스템에서)

```python
elapsed = time.time() - start_time

if elapsed < min_time:
    status = "조리중"
elif elapsed >= target_time and result["progress_pct"] >= 100:
    status = "완료"
elif result["progress_pct"] >= 90:
    status = "거의완료"
else:
    status = "조리중"
```

---

## Git 레포지토리

| 레포 | 위치 | 용도 |
|------|------|------|
| dku_frying_ai | WSL2 | 학습, 알고리즘 개발 |
| jetson-food-ai | WSL2 (clone) + Jetson | 배포, 운영 |

---

## 자주 쓰는 명령어

### WSL2

```bash
# 가상환경 활성화
source venv/bin/activate

# 테스트
python -m src.simple_checker.demo --session <path> --color_threshold 25

# 동기화
./scripts/sync_to_jetson.sh
```

### Jetson

```bash
# 코드 업데이트
cd ~/jetson-food-ai && git pull

# 실행
python jetson2_frying_ai/JETSON2_INTEGRATED.py
```

---

## 충돌 방지 규칙 (중요!)

### 수정 권한

| 위치 | 수정 가능 | 수정 금지 |
|------|-----------|-----------|
| **WSL2** | `simple_checker/` | `JETSON2_INTEGRATED.py` |
| **Jetson** | `JETSON2_INTEGRATED.py` | `simple_checker/` |

### 이유

```
simple_checker/ = WSL2에서 sync로 덮어씌워짐
→ Jetson에서 수정하면 다음 sync 때 날아감!
```

### 규칙

1. **simple_checker 버그 발견** → WSL2에서 수정 → sync
2. **JETSON2 통합 문제** → Jetson에서 수정
3. **Jetson에서 simple_checker 절대 수정 금지!**

---

## 작업 완료 보고 형식

Jetson 작업 후 WSL2 Claude에게 보고할 때:

```
[JETSON 작업 완료]
작업: (뭐 했는지)
파일: (수정한 파일)
결과: 성공/실패
테스트: (실행 결과)
```

**보고 필요한 경우:**
- 에러 발생
- 설계 변경 필요
- 다음 단계 불명확

**보고 안 해도 되는 경우:**
- 단순 적용 성공
- 테스트 통과

---

## 문제 해결

### sync 후 Jetson에서 import 에러

```bash
# Jetson에서
cd ~/jetson-food-ai/jetson2_frying_ai
python -c "from simple_checker.color_checker import SimpleColorChecker; print('OK')"
```

### baseline 설정 안 됨

- 첫 프레임에서 튀김 영역이 감지되지 않으면 baseline 설정 실패
- `color_threshold` 조정 필요할 수 있음

---

## 참고 문서

- [색상 기반 방법 설계](method2_color_based.md)
- [프로젝트 개요](../CLAUDE.md)
