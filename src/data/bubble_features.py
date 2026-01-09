"""
기포 특징 추출 모듈 (OpenCV 기반)

튀김유 표면의 기포 패턴을 분석하여 조리 진행도를 추정하는 보조 지표.
튀김유는 재사용되므로 절대값이 아닌 세션 시작 대비 상대적 변화를 측정.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path


@dataclass
class BubbleFeatures:
    """기포 특징 데이터 클래스"""
    # 기포 관련
    bubble_count: int           # 기포 개수
    bubble_area_total: float    # 총 기포 면적
    bubble_area_mean: float     # 평균 기포 크기
    bubble_area_std: float      # 기포 크기 표준편차

    # HSV 색상 통계
    hsv_h_mean: float           # 색조 평균
    hsv_s_mean: float           # 채도 평균
    hsv_v_mean: float           # 명도 평균
    hsv_h_std: float            # 색조 표준편차
    hsv_v_std: float            # 명도 표준편차

    # 표면 상태
    edge_density: float         # 엣지 밀도 (거품 활동 지표)
    brightness_uniformity: float  # 밝기 균일도

    def to_vector(self) -> np.ndarray:
        """특징을 벡터로 변환 (모델 입력용)"""
        return np.array([
            self.bubble_count,
            self.bubble_area_total,
            self.bubble_area_mean,
            self.bubble_area_std,
            self.hsv_h_mean,
            self.hsv_s_mean,
            self.hsv_v_mean,
            self.hsv_h_std,
            self.hsv_v_std,
            self.edge_density,
            self.brightness_uniformity,
        ], dtype=np.float32)

    @staticmethod
    def feature_dim() -> int:
        """특징 벡터 차원"""
        return 11


@dataclass
class RelativeBubbleFeatures:
    """상대적 기포 특징 (세션 시작 대비)"""
    bubble_ratio: float         # 현재/시작 기포 비율 (1.0 → 0.3)
    bubble_delta: float         # 기포 감소량
    area_ratio: float           # 면적 비율
    edge_ratio: float           # 엣지 밀도 비율

    # 추세
    bubble_trend: float         # 최근 N프레임 기포 변화 추세
    surface_calm: float         # 표면 안정화 정도 (높을수록 완료 근접)

    def to_vector(self) -> np.ndarray:
        """특징을 벡터로 변환"""
        return np.array([
            self.bubble_ratio,
            self.bubble_delta,
            self.area_ratio,
            self.edge_ratio,
            self.bubble_trend,
            self.surface_calm,
        ], dtype=np.float32)

    @staticmethod
    def feature_dim() -> int:
        """특징 벡터 차원"""
        return 6


class BubbleFeatureExtractor:
    """기포 특징 추출기"""

    def __init__(
        self,
        min_blob_area: int = 50,
        max_blob_area: int = 5000,
        oil_hsv_lower: Tuple[int, int, int] = (15, 50, 50),
        oil_hsv_upper: Tuple[int, int, int] = (35, 255, 255),
    ):
        """
        Args:
            min_blob_area: 최소 기포 면적 (픽셀)
            max_blob_area: 최대 기포 면적 (픽셀)
            oil_hsv_lower: 튀김유 HSV 하한 (마스킹용)
            oil_hsv_upper: 튀김유 HSV 상한 (마스킹용)
        """
        self.min_blob_area = min_blob_area
        self.max_blob_area = max_blob_area
        self.oil_hsv_lower = np.array(oil_hsv_lower)
        self.oil_hsv_upper = np.array(oil_hsv_upper)

        # 세션 시작 시 기준값 저장용
        self.start_features: Optional[BubbleFeatures] = None
        self.feature_history: List[BubbleFeatures] = []

    def reset_session(self):
        """새 세션 시작 시 초기화"""
        self.start_features = None
        self.feature_history = []

    def extract_features(self, image: np.ndarray) -> BubbleFeatures:
        """
        이미지에서 기포 특징 추출

        Args:
            image: BGR 이미지 (numpy array)

        Returns:
            BubbleFeatures 객체
        """
        # HSV 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 튀김유 영역 마스크 (대략적)
        oil_mask = cv2.inRange(hsv, self.oil_hsv_lower, self.oil_hsv_upper)

        # 기포 감지를 위한 전처리
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 적응형 이진화로 기포 검출
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # 튀김유 영역 내에서만 기포 검출
        binary = cv2.bitwise_and(binary, oil_mask)

        # 모폴로지 연산으로 노이즈 제거
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 컨투어 검출 (기포)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 기포 필터링 및 특징 계산
        bubble_areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_blob_area <= area <= self.max_blob_area:
                bubble_areas.append(area)

        bubble_count = len(bubble_areas)
        bubble_area_total = sum(bubble_areas) if bubble_areas else 0
        bubble_area_mean = np.mean(bubble_areas) if bubble_areas else 0
        bubble_area_std = np.std(bubble_areas) if bubble_areas else 0

        # HSV 통계 (튀김유 영역)
        hsv_masked = cv2.bitwise_and(hsv, hsv, mask=oil_mask)
        h, s, v = cv2.split(hsv_masked)

        # 마스크된 영역에서만 통계 계산
        oil_pixels = oil_mask > 0
        if np.any(oil_pixels):
            hsv_h_mean = np.mean(h[oil_pixels])
            hsv_s_mean = np.mean(s[oil_pixels])
            hsv_v_mean = np.mean(v[oil_pixels])
            hsv_h_std = np.std(h[oil_pixels])
            hsv_v_std = np.std(v[oil_pixels])
        else:
            hsv_h_mean = hsv_s_mean = hsv_v_mean = 0
            hsv_h_std = hsv_v_std = 0

        # 엣지 밀도 (거품 활동 지표)
        edges = cv2.Canny(blurred, 50, 150)
        edges_masked = cv2.bitwise_and(edges, oil_mask)
        edge_density = np.sum(edges_masked > 0) / max(np.sum(oil_mask > 0), 1)

        # 밝기 균일도
        if np.any(oil_pixels):
            brightness_uniformity = 1.0 - (np.std(v[oil_pixels]) / 128.0)
            brightness_uniformity = max(0, min(1, brightness_uniformity))
        else:
            brightness_uniformity = 0

        features = BubbleFeatures(
            bubble_count=bubble_count,
            bubble_area_total=bubble_area_total,
            bubble_area_mean=bubble_area_mean,
            bubble_area_std=bubble_area_std,
            hsv_h_mean=hsv_h_mean,
            hsv_s_mean=hsv_s_mean,
            hsv_v_mean=hsv_v_mean,
            hsv_h_std=hsv_h_std,
            hsv_v_std=hsv_v_std,
            edge_density=edge_density,
            brightness_uniformity=brightness_uniformity,
        )

        # 히스토리에 추가
        self.feature_history.append(features)

        # 첫 프레임이면 시작 기준값으로 저장
        if self.start_features is None:
            self.start_features = features

        return features

    def get_relative_features(
        self,
        current: Optional[BubbleFeatures] = None,
        trend_window: int = 5
    ) -> RelativeBubbleFeatures:
        """
        세션 시작 대비 상대적 특징 계산

        Args:
            current: 현재 특징 (None이면 마지막 히스토리 사용)
            trend_window: 추세 계산에 사용할 프레임 수

        Returns:
            RelativeBubbleFeatures 객체
        """
        if current is None:
            if not self.feature_history:
                raise ValueError("No features extracted yet")
            current = self.feature_history[-1]

        if self.start_features is None:
            raise ValueError("No start features set")

        start = self.start_features

        # 비율 계산 (0으로 나누기 방지)
        bubble_ratio = current.bubble_count / max(start.bubble_count, 1)
        bubble_delta = start.bubble_count - current.bubble_count
        area_ratio = current.bubble_area_total / max(start.bubble_area_total, 1)
        edge_ratio = current.edge_density / max(start.edge_density, 0.001)

        # 추세 계산 (최근 N프레임)
        if len(self.feature_history) >= trend_window:
            recent = self.feature_history[-trend_window:]
            counts = [f.bubble_count for f in recent]
            # 선형 회귀로 추세 계산 (음수 = 감소 추세)
            x = np.arange(len(counts))
            if np.std(counts) > 0:
                bubble_trend = np.corrcoef(x, counts)[0, 1]
            else:
                bubble_trend = 0
        else:
            bubble_trend = 0

        # 표면 안정화 정도 (엣지 감소 + 균일도 증가)
        surface_calm = (1 - edge_ratio) * 0.5 + current.brightness_uniformity * 0.5
        surface_calm = max(0, min(1, surface_calm))

        return RelativeBubbleFeatures(
            bubble_ratio=bubble_ratio,
            bubble_delta=bubble_delta,
            area_ratio=area_ratio,
            edge_ratio=edge_ratio,
            bubble_trend=bubble_trend,
            surface_calm=surface_calm,
        )

    def extract_from_file(self, image_path: str) -> BubbleFeatures:
        """파일에서 이미지 로드 후 특징 추출"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return self.extract_features(image)

    def process_session(
        self,
        image_paths: List[str],
        return_relative: bool = True
    ) -> List[np.ndarray]:
        """
        세션의 모든 이미지에서 특징 추출

        Args:
            image_paths: 이미지 경로 리스트 (시간순 정렬)
            return_relative: True면 상대적 특징 반환

        Returns:
            특징 벡터 리스트
        """
        self.reset_session()

        features_list = []
        for path in image_paths:
            features = self.extract_from_file(path)

            if return_relative and self.start_features is not None:
                rel_features = self.get_relative_features(features)
                features_list.append(rel_features.to_vector())
            else:
                features_list.append(features.to_vector())

        return features_list


def visualize_bubbles(
    image: np.ndarray,
    extractor: BubbleFeatureExtractor,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    기포 검출 결과 시각화 (디버깅용)

    Args:
        image: BGR 이미지
        extractor: BubbleFeatureExtractor 인스턴스
        save_path: 저장 경로 (None이면 저장 안 함)

    Returns:
        시각화된 이미지
    """
    vis = image.copy()

    # HSV 변환 및 마스크
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    oil_mask = cv2.inRange(hsv, extractor.oil_hsv_lower, extractor.oil_hsv_upper)

    # 기포 검출
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    binary = cv2.bitwise_and(binary, oil_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 기포 표시
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if extractor.min_blob_area <= area <= extractor.max_blob_area:
            cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)

            # 중심점 표시
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)

    # 정보 표시
    features = extractor.feature_history[-1] if extractor.feature_history else None
    if features:
        info_text = f"Bubbles: {features.bubble_count}"
        cv2.putText(vis, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if save_path:
        cv2.imwrite(save_path, vis)

    return vis


if __name__ == "__main__":
    # 테스트 코드
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bubble_features.py <image_path>")
        sys.exit(1)

    extractor = BubbleFeatureExtractor()
    features = extractor.extract_from_file(sys.argv[1])

    print("=== Bubble Features ===")
    print(f"Bubble count: {features.bubble_count}")
    print(f"Bubble area total: {features.bubble_area_total:.1f}")
    print(f"Bubble area mean: {features.bubble_area_mean:.1f}")
    print(f"HSV H mean: {features.hsv_h_mean:.1f}")
    print(f"HSV S mean: {features.hsv_s_mean:.1f}")
    print(f"HSV V mean: {features.hsv_v_mean:.1f}")
    print(f"Edge density: {features.edge_density:.4f}")
    print(f"Brightness uniformity: {features.brightness_uniformity:.4f}")
    print(f"\nFeature vector: {features.to_vector()}")
