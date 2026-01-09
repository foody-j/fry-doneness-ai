"""
Simple color-based frying completion checker.
"""

import json
from typing import Dict, Optional

import yaml
import numpy as np

from .color_utils import (
    extract_food_region,
    extract_color_stats,
    calculate_color_distance,
)


class SimpleColorChecker:
    """색상 차이 기반 튀김 완료 판단기."""

    def __init__(
        self,
        config_path: str = "configs/recipes.yaml",
        verbose: bool = False,
    ):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.verbose = verbose
        self.reset()

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def reset(self):
        """새 세션 시작."""
        self.start_color: Optional[Dict[str, float]] = None
        self.start_time: Optional[float] = None
        self.recipe: Optional[str] = None
        self.params: Dict[str, float] = self.config.get("default", {})
        self.history = []

    def set_recipe(self, recipe_name: str):
        """튀김 종류 설정."""
        self.recipe = recipe_name
        recipes = self.config.get("recipes", {})
        self.params = recipes.get(recipe_name, self.config.get("default", {}))

    def on_first_lift(self, image: np.ndarray, timestamp: float) -> Dict:
        """1차 탈탈: 기준 색상 저장."""
        mask, hsv = extract_food_region(image)
        self.start_color = extract_color_stats(hsv, mask)
        self.start_time = timestamp

        self._log(f"First lift at t={timestamp:.1f}s")
        if self.start_color:
            self._log(
                f"  Base color: H={self.start_color['h_mean']:.1f}, "
                f"S={self.start_color['s_mean']:.1f}, V={self.start_color['v_mean']:.1f}"
            )

        self.history.append({
            "lift": 1,
            "timestamp": timestamp,
            "color": self.start_color,
            "status": "조리시작",
        })

        return {
            "status": "조리시작",
            "color": self.start_color,
        }

    def check(self, image: np.ndarray, timestamp: float) -> Dict:
        """탈탈 시점에 완료 체크."""
        if self.start_color is None or self.start_time is None:
            return self.on_first_lift(image, timestamp)

        mask, hsv = extract_food_region(image)
        current_color = extract_color_stats(hsv, mask)
        if current_color is None:
            return {"status": "측정실패", "error": "튀김 영역 검출 실패"}

        color_diff = calculate_color_distance(self.start_color, current_color)
        elapsed = timestamp - self.start_time

        target_time = float(self.params.get("target_time", 180))
        min_time = float(self.params.get("min_time", 120))
        threshold = float(self.params.get("color_threshold", 25))

        time_progress = min(elapsed / target_time, 1.0)
        color_progress = min(color_diff / threshold, 1.0)
        overall_progress = (time_progress + color_progress) / 2

        status = self._determine_status(
            elapsed, color_diff,
            target_time, min_time, threshold
        )

        result = {
            "status": status,
            "color_diff": round(color_diff, 2),
            "elapsed_sec": round(elapsed, 1),
            "progress_pct": round(overall_progress * 100, 1),
            "time_progress": round(time_progress * 100, 1),
            "color_progress": round(color_progress * 100, 1),
            "current_color": current_color,
        }

        self._log(
            f"Lift #{len(self.history)+1}: elapsed={elapsed:.1f}s, "
            f"color_diff={color_diff:.2f}, status={status}"
        )

        self.history.append({
            "lift": len(self.history) + 1,
            "timestamp": timestamp,
            **result,
        })

        return result

    def _determine_status(
        self,
        elapsed: float,
        color_diff: float,
        target_time: float,
        min_time: float,
        threshold: float,
    ) -> str:
        """상태 결정 로직."""
        if elapsed < min_time:
            return "조리중"

        if elapsed < target_time:
            if color_diff >= threshold * 0.9:
                return "거의완료"
            if color_diff >= threshold * 0.6:
                return "조리중"
            return "조리중"

        if color_diff >= threshold:
            return "완료"
        if color_diff >= threshold * 0.7:
            return "거의완료"
        return "거의완료"

    def _log(self, msg: str):
        """Print debug message if verbose mode."""
        if self.verbose:
            print(f"[ColorChecker] {msg}")

    def save_log(self, filepath: str):
        """Save check history to JSON file."""
        data = {
            "recipe": self.recipe,
            "params": self.params,
            "start_color": self.start_color,
            "history": self.history,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._log(f"Log saved to {filepath}")
