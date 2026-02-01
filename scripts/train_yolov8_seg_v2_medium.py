"""
YOLOv8 Segmentation 학습 스크립트 v2 - Medium 모델
DKU_HDEV_FRY_FOOD 데이터셋 (확장된 버전)

=== 학습 파라미터 (보고서용) ===
- Model: YOLOv8m-seg (medium, pretrained on COCO)
- Dataset: DKU_HDEV_FRY_FOOD.v2i.yolov8 (Roboflow export)
  - Train: 708 images
  - Valid: 215 images
  - Test: 206 images
  - Classes: 1 (food - 튀김)
- Epochs: 100
- Image Size: 640x640
- Batch Size: 8 (medium 모델이라 배치 줄임)
- Optimizer: AdamW (default)
- Learning Rate: 0.01 (default, with cosine decay)
- Augmentation: YOLOv8 default (mosaic, mixup, hsv, flip, etc.)
- Device: NVIDIA RTX 5070 Ti (16GB)
- Framework: Ultralytics 8.4.9
================================
"""

from ultralytics import YOLO
from datetime import datetime
import os

# 학습 파라미터
PARAMS = {
    "model": "yolov8m-seg.pt",  # medium segmentation (pretrained)
    "data": "/home/youngjin/dku_frying_ai/DKU_HDEV_FRY_FOOD.v2i.yolov8/data.yaml",
    "epochs": 100,
    "imgsz": 640,
    "batch": 8,  # medium 모델이라 배치 사이즈 줄임
    "device": 0,  # GPU 0
    "workers": 4,
    "patience": 20,  # early stopping patience
    "save": True,
    "save_period": 10,  # 10 에폭마다 체크포인트 저장
    "project": "/home/youngjin/dku_frying_ai/runs/segment",
    "name": f"frying_seg_v2_medium_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "exist_ok": False,
    "pretrained": True,
    "verbose": True,
}

def main():
    print("=" * 60)
    print("YOLOv8m Segmentation 학습 시작 (v2 - Medium 모델)")
    print("=" * 60)
    print("\n[학습 파라미터]")
    for k, v in PARAMS.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # 모델 로드
    model = YOLO(PARAMS["model"])

    # 학습 시작
    results = model.train(
        data=PARAMS["data"],
        epochs=PARAMS["epochs"],
        imgsz=PARAMS["imgsz"],
        batch=PARAMS["batch"],
        device=PARAMS["device"],
        workers=PARAMS["workers"],
        patience=PARAMS["patience"],
        save=PARAMS["save"],
        save_period=PARAMS["save_period"],
        project=PARAMS["project"],
        name=PARAMS["name"],
        exist_ok=PARAMS["exist_ok"],
        pretrained=PARAMS["pretrained"],
        verbose=PARAMS["verbose"],
    )

    print("\n" + "=" * 60)
    print("학습 완료!")
    print(f"결과 저장 위치: {PARAMS['project']}/{PARAMS['name']}")
    print("=" * 60)

    return results

if __name__ == "__main__":
    main()
