"""
탈탈 시점 감지 모듈 (Lift Detection)

바스켓이 튀김유에서 들어올려진 프레임을 자동으로 식별.
튀김이 명확히 보이는 "탈탈" 시점의 이미지를 추출하는데 사용.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path
from enum import Enum


class FrameType(Enum):
    """프레임 타입"""
    OIL_ONLY = 0      # 튀김유만 보임 (대기 또는 조리중)
    BASKET_LIFT = 1   # 바스켓 들어올림 (탈탈)
    BASKET_IN = 2     # 바스켓 투입/잠김
    DISCHARGE = 3     # 배출


@dataclass
class LiftDetectionResult:
    """탈탈 감지 결과"""
    frame_type: FrameType
    confidence: float
    basket_area_ratio: float    # 이미지 대비 바스켓 영역 비율
    metal_ratio: float          # 금속(바스켓) 픽셀 비율
    food_visible: bool          # 튀김이 보이는지


class LiftDetector:
    """탈탈 시점 감지기"""

    def __init__(
        self,
        # 바스켓(금속) 색상 범위 (HSV)
        metal_hsv_lower: Tuple[int, int, int] = (0, 0, 140),
        metal_hsv_upper: Tuple[int, int, int] = (180, 40, 255),
        # 튀김 색상 범위 (HSV) - 황금색~갈색
        food_hsv_lower: Tuple[int, int, int] = (12, 60, 90),
        food_hsv_upper: Tuple[int, int, int] = (25, 200, 200),
        # 감지 임계값
        lift_threshold: float = 0.02,  # 바스켓 면적 비율 임계값
        food_threshold: float = 0.01,  # 튀김 면적 비율 임계값
    ):
        self.metal_hsv_lower = np.array(metal_hsv_lower)
        self.metal_hsv_upper = np.array(metal_hsv_upper)
        self.food_hsv_lower = np.array(food_hsv_lower)
        self.food_hsv_upper = np.array(food_hsv_upper)
        self.lift_threshold = lift_threshold
        self.food_threshold = food_threshold

        # 이전 프레임 저장 (변화 감지용)
        self.prev_frame: Optional[np.ndarray] = None
        self.frame_history: List[LiftDetectionResult] = []

    def reset(self):
        """세션 초기화"""
        self.prev_frame = None
        self.frame_history = []

    def detect(self, image: np.ndarray) -> LiftDetectionResult:
        """
        프레임에서 탈탈 시점 감지

        Args:
            image: BGR 이미지

        Returns:
            LiftDetectionResult
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        total_pixels = h * w

        # 1. 금속(바스켓) 영역 검출
        metal_mask = cv2.inRange(hsv, self.metal_hsv_lower, self.metal_hsv_upper)

        # 모폴로지로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_CLOSE, kernel)
        metal_mask = cv2.morphologyEx(metal_mask, cv2.MORPH_OPEN, kernel)

        metal_pixels = np.sum(metal_mask > 0)
        metal_ratio = metal_pixels / total_pixels

        # 2. 튀김 영역 검출
        food_mask = cv2.inRange(hsv, self.food_hsv_lower, self.food_hsv_upper)
        food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_CLOSE, kernel)
        food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_OPEN, kernel)

        food_pixels = np.sum(food_mask > 0)
        food_ratio = food_pixels / total_pixels
        food_visible = food_ratio > self.food_threshold

        # 3. 바스켓 윤곽선 검출 (망 패턴)
        edges = cv2.Canny(gray, 50, 150)
        basket_contours = self._detect_basket_pattern(edges, metal_mask)
        basket_area_ratio = basket_contours / total_pixels if basket_contours > 0 else 0

        # 4. 프레임 변화량 계산 (이전 프레임 대비)
        frame_diff = 0
        if self.prev_frame is not None:
            diff = cv2.absdiff(gray, self.prev_frame)
            frame_diff = np.mean(diff) / 255.0

        self.prev_frame = gray.copy()

        # 5. 프레임 타입 판단
        frame_type, confidence = self._classify_frame(
            metal_ratio=metal_ratio,
            food_ratio=food_ratio,
            basket_area_ratio=basket_area_ratio,
            frame_diff=frame_diff,
            food_visible=food_visible,
        )

        result = LiftDetectionResult(
            frame_type=frame_type,
            confidence=confidence,
            basket_area_ratio=basket_area_ratio,
            metal_ratio=metal_ratio,
            food_visible=food_visible,
        )

        self.frame_history.append(result)
        return result

    def _detect_basket_pattern(
        self,
        edges: np.ndarray,
        metal_mask: np.ndarray
    ) -> float:
        """바스켓 망 패턴 검출"""
        # 금속 영역 내 엣지 밀도
        masked_edges = cv2.bitwise_and(edges, metal_mask)

        # 직선 검출 (망 패턴)
        lines = cv2.HoughLinesP(
            masked_edges, 1, np.pi/180, 50,
            minLineLength=30, maxLineGap=10
        )

        if lines is not None:
            # 직선이 많으면 바스켓 망 패턴일 가능성 높음
            return len(lines) * 100  # 대략적인 면적 추정
        return 0

    def _classify_frame(
        self,
        metal_ratio: float,
        food_ratio: float,
        basket_area_ratio: float,
        frame_diff: float,
        food_visible: bool,
    ) -> Tuple[FrameType, float]:
        """프레임 타입 분류"""
        confidence = 0.5

        # 튀김이 보이고 금속(바스켓)도 보이면 → 탈탈
        if food_visible and metal_ratio > self.lift_threshold:
            frame_type = FrameType.BASKET_LIFT
            confidence = min(0.9, 0.5 + food_ratio * 5 + metal_ratio * 3)

        # 금속만 보이고 튀김 안 보이면 → 바스켓 투입/잠김
        elif metal_ratio > self.lift_threshold and not food_visible:
            frame_type = FrameType.BASKET_IN
            confidence = min(0.8, 0.5 + metal_ratio * 3)

        # 금속도 안 보이고 튀김도 안 보이면 → 튀김유만
        elif metal_ratio < 0.01 and not food_visible:
            frame_type = FrameType.OIL_ONLY
            confidence = 0.8

        # 그 외
        else:
            frame_type = FrameType.OIL_ONLY
            confidence = 0.5

        return frame_type, confidence

    def detect_from_file(self, image_path: str) -> LiftDetectionResult:
        """파일에서 이미지 로드 후 감지"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return self.detect(image)

    def find_lift_frames(
        self,
        image_paths: List[str],
        min_confidence: float = 0.6
    ) -> List[Tuple[int, str, LiftDetectionResult]]:
        """
        세션에서 탈탈 프레임 찾기

        Args:
            image_paths: 이미지 경로 리스트 (시간순)
            min_confidence: 최소 신뢰도

        Returns:
            (인덱스, 경로, 결과) 튜플 리스트
        """
        self.reset()
        lift_frames = []

        for i, path in enumerate(image_paths):
            result = self.detect_from_file(path)

            if (result.frame_type == FrameType.BASKET_LIFT and
                result.confidence >= min_confidence):
                lift_frames.append((i, path, result))

        return lift_frames

    def get_lift_sequence(
        self,
        image_paths: List[str],
        min_gap: int = 5
    ) -> List[Tuple[int, str]]:
        """
        탈탈 시퀀스 추출 (1차, 2차, 3차 탈탈)

        중복 제거: 연속된 탈탈 프레임 중 대표 1장만 선택

        Args:
            image_paths: 이미지 경로 리스트
            min_gap: 탈탈 간 최소 프레임 간격

        Returns:
            (인덱스, 경로) 튜플 리스트
        """
        lift_frames = self.find_lift_frames(image_paths)

        if not lift_frames:
            return []

        # 연속된 탈탈 프레임 그룹화
        sequences = []
        current_seq = [lift_frames[0]]

        for i in range(1, len(lift_frames)):
            prev_idx = lift_frames[i-1][0]
            curr_idx = lift_frames[i][0]

            if curr_idx - prev_idx <= min_gap:
                # 연속된 프레임
                current_seq.append(lift_frames[i])
            else:
                # 새로운 탈탈 시퀀스
                sequences.append(current_seq)
                current_seq = [lift_frames[i]]

        sequences.append(current_seq)

        # 각 시퀀스에서 가장 높은 confidence 프레임 선택
        result = []
        for seq in sequences:
            best = max(seq, key=lambda x: x[2].confidence)
            result.append((best[0], best[1]))

        return result


def visualize_detection(
    image: np.ndarray,
    result: LiftDetectionResult,
    save_path: Optional[str] = None
) -> np.ndarray:
    """감지 결과 시각화"""
    vis = image.copy()

    # 프레임 타입에 따른 색상
    colors = {
        FrameType.OIL_ONLY: (128, 128, 128),     # 회색
        FrameType.BASKET_LIFT: (0, 255, 0),      # 녹색
        FrameType.BASKET_IN: (255, 165, 0),      # 주황
        FrameType.DISCHARGE: (255, 0, 0),        # 빨강
    }
    color = colors.get(result.frame_type, (255, 255, 255))

    # 정보 표시
    text_lines = [
        f"Type: {result.frame_type.name}",
        f"Confidence: {result.confidence:.2f}",
        f"Metal: {result.metal_ratio:.3f}",
        f"Food visible: {result.food_visible}",
    ]

    for i, text in enumerate(text_lines):
        cv2.putText(vis, text, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 테두리
    cv2.rectangle(vis, (0, 0), (vis.shape[1]-1, vis.shape[0]-1), color, 5)

    if save_path:
        cv2.imwrite(save_path, vis)

    return vis


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python lift_detector.py <image_path_or_directory>")
        sys.exit(1)

    path = Path(sys.argv[1])
    detector = LiftDetector()

    if path.is_file():
        # 단일 이미지
        result = detector.detect_from_file(str(path))
        print(f"Frame type: {result.frame_type.name}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Metal ratio: {result.metal_ratio:.4f}")
        print(f"Food visible: {result.food_visible}")

    elif path.is_dir():
        # 디렉토리 내 모든 이미지
        image_paths = sorted(path.glob("*.jpg"))
        print(f"Found {len(image_paths)} images")

        lift_sequence = detector.get_lift_sequence([str(p) for p in image_paths])

        print(f"\n=== Lift Sequence ({len(lift_sequence)} found) ===")
        for i, (idx, p) in enumerate(lift_sequence):
            print(f"  {i+1}차 탈탈: frame {idx} - {Path(p).name}")
