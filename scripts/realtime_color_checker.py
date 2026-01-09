#!/usr/bin/env python3
"""
Real-time color-based frying checker.

Detects lift frames and runs color-based status check on each lift.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.lift_detector import LiftDetector, FrameType
from src.simple_checker.color_checker import SimpleColorChecker


def _parse_source(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time color-based checker")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--recipe", required=True, help="Recipe name (e.g., 적어튀김)")
    parser.add_argument("--min_confidence", type=float, default=0.6)
    parser.add_argument("--max_lifts", type=int, default=0, help="Stop after N lifts (0=unlimited)")
    args = parser.parse_args()

    source = _parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open source: {args.source}")

    detector = LiftDetector()
    checker = SimpleColorChecker()
    checker.set_recipe(args.recipe)

    lift_count = 0
    last_lift_time = 0.0
    min_lift_interval = 1.0  # seconds

    print("=== Real-time color-based checker ===")
    print(f"Source: {args.source}")
    print(f"Recipe: {args.recipe}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = detector.detect(frame)
            now = time.time()

            if (
                result.frame_type == FrameType.BASKET_LIFT
                and result.confidence >= args.min_confidence
                and now - last_lift_time >= min_lift_interval
            ):
                last_lift_time = now
                lift_count += 1
                status = checker.check(frame, now)

                print(
                    f"[Lift {lift_count}] status={status.get('status')} "
                    f"color_diff={status.get('color_diff')} "
                    f"elapsed={status.get('elapsed_sec')}"
                )

                if args.max_lifts and lift_count >= args.max_lifts:
                    break
    finally:
        cap.release()


if __name__ == "__main__":
    main()
