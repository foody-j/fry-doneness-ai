#!/usr/bin/env python3
"""
Merge consecutive sessions into a single session directory.

Example:
  python scripts/merge_sessions.py \
    --pot pot1 \
    --food_type 적어튀김 \
    --output session_20260107_102151_merged \
    --sessions session_20260107_102151 session_20260107_103320
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def _parse_time(value: str) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _safe_copy(src: Path, dst_dir: Path) -> Path:
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst

    stem = src.stem
    suffix = src.suffix
    idx = 1
    while True:
        candidate = dst_dir / f"{stem}_dup{idx}{suffix}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
        idx += 1


def _load_session_info(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_session_info(path: Path, info: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def _collect_images(
    pot_dir: Path,
    food_type: str,
    session_ids: List[str],
    camera_dir: str,
) -> List[Path]:
    images = []
    for session_id in session_ids:
        image_dir = pot_dir / session_id / food_type / camera_dir
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image dir: {image_dir}")
        images.extend(sorted(image_dir.glob("*.jpg")))
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge consecutive sessions")
    parser.add_argument("--pot", required=True, help="Pot directory (e.g., pot1)")
    parser.add_argument("--food_type", required=True, help="Food type directory name")
    parser.add_argument("--output", required=True, help="Merged session dir name")
    parser.add_argument(
        "--sessions",
        nargs="+",
        required=True,
        help="Session directories to merge (e.g., session_... session_...)",
    )
    parser.add_argument("--camera_dir", default="camera_0")
    parser.add_argument("--data_root", default=".", help="Root containing pot dirs")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    pot_dir = data_root / args.pot
    output_dir = pot_dir / args.output / args.food_type
    output_cam_dir = output_dir / args.camera_dir
    output_cam_dir.mkdir(parents=True, exist_ok=True)

    # Copy images
    images = _collect_images(pot_dir, args.food_type, args.sessions, args.camera_dir)
    for img_path in images:
        _safe_copy(img_path, output_cam_dir)

    # Build merged session_info.json
    info_list = []
    for session_id in args.sessions:
        info_path = pot_dir / session_id / args.food_type / "session_info.json"
        info_list.append(_load_session_info(info_path))

    pot_name = args.pot
    food_name = args.food_type
    start_times = []
    end_times = []

    for info in info_list:
        start = _parse_time(info.get("start_time", ""))
        end = _parse_time(info.get("end_time", ""))
        if start:
            start_times.append(start)
        if end:
            end_times.append(end)
        pot_name = info.get("pot", pot_name)
        food_name = info.get("food_type", food_name)

    start_time = min(start_times).strftime("%Y-%m-%d %H:%M:%S") if start_times else ""
    end_time = max(end_times).strftime("%Y-%m-%d %H:%M:%S") if end_times else ""
    duration_sec = 0.0
    if start_times and end_times:
        duration_sec = (max(end_times) - min(start_times)).total_seconds()

    merged_info = {
        "session_id": args.output,
        "pot": pot_name,
        "food_type": food_name,
        "start_time": start_time,
        "end_time": end_time,
        "duration_sec": duration_sec,
        "merged_from": args.sessions,
    }

    _write_session_info(output_dir / "session_info.json", merged_info)
    print(f"Merged {len(images)} images into {output_cam_dir}")
    print(f"Wrote session_info.json to {output_dir}")


if __name__ == "__main__":
    main()
