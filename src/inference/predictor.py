"""
실시간 추론용 predictor

단일 이미지 또는 이미지 시퀀스에서 조리 상태를 예측.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

from src.models.frying_model import FryingModel, CookingState, get_food_type_id
from src.data.bubble_features import BubbleFeatureExtractor
from src.data.lift_detector import LiftDetector


class FryingPredictor:
    """튀김 조리 상태 추론기"""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        use_temporal: bool = False,
        image_size: Tuple[int, int] = (224, 224),
        use_bubble_features: bool = True,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.use_temporal = use_temporal
        self.image_size = image_size
        self.use_bubble_features = use_bubble_features

        self.model = FryingModel(
            num_food_types=10,
            num_classes=3,
            use_temporal=use_temporal,
            pretrained=False,
            freeze_backbone=False,
        ).to(device)
        self.model.eval()

        self._load_checkpoint(checkpoint_path)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.bubble_extractor = BubbleFeatureExtractor() if use_bubble_features else None
        self.lift_detector = LiftDetector()

    def _load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)

    def reset_session(self):
        if self.bubble_extractor:
            self.bubble_extractor.reset_session()

    def _load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return image

    def _preprocess_image(self, image_bgr: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(image_rgb)

    def _extract_bubble_features(self, image_bgr: np.ndarray) -> np.ndarray:
        if not self.bubble_extractor:
            return np.zeros(6, dtype=np.float32)

        features = self.bubble_extractor.extract_features(image_bgr)
        rel = self.bubble_extractor.get_relative_features(features)
        return rel.to_vector()

    def predict_image(
        self,
        image_path: str,
        food_type: str,
    ) -> Tuple[np.ndarray, int]:
        image_bgr = self._load_image(image_path)
        image = self._preprocess_image(image_bgr)
        bubble = self._extract_bubble_features(image_bgr)

        images = image.unsqueeze(0).to(self.device)
        bubble_features = torch.tensor(bubble, dtype=torch.float32).unsqueeze(0).to(self.device)
        food_type_id = torch.tensor([get_food_type_id(food_type)], dtype=torch.long).to(self.device)

        with torch.no_grad():
            probs, predicted = self.model.predict(images, bubble_features, food_type_id)

        return probs.squeeze(0).cpu().numpy(), int(predicted.item())

    def predict_sequence(
        self,
        image_paths: List[str],
        food_type: str,
        use_lift_sequence: bool = False,
        min_gap: int = 5,
        max_frames: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        if not image_paths:
            raise ValueError("No images provided for prediction")

        if use_lift_sequence:
            lift_sequence = self.lift_detector.get_lift_sequence(image_paths, min_gap=min_gap)
            image_paths = [p for _, p in lift_sequence] or image_paths

        if max_frames is not None and len(image_paths) > max_frames:
            image_paths = image_paths[-max_frames:]

        self.reset_session()
        images = []
        bubbles = []
        for path in image_paths:
            image_bgr = self._load_image(path)
            images.append(self._preprocess_image(image_bgr))
            bubbles.append(self._extract_bubble_features(image_bgr))

        images_tensor = torch.stack(images).unsqueeze(0).to(self.device)
        bubble_tensor = torch.tensor(bubbles, dtype=torch.float32).unsqueeze(0).to(self.device)
        food_type_id = torch.tensor([get_food_type_id(food_type)], dtype=torch.long).to(self.device)
        seq_len = torch.tensor([len(images)], dtype=torch.long).to(self.device)

        with torch.no_grad():
            if self.use_temporal:
                probs, predicted = self.model.predict(
                    images_tensor,
                    bubble_tensor,
                    food_type_id,
                    seq_lengths=seq_len,
                )
            else:
                probs, predicted = self.model.predict(
                    images_tensor,
                    bubble_tensor,
                    food_type_id,
                )

        return probs.squeeze(0).cpu().numpy(), int(predicted.item())


def _format_prediction(probs: np.ndarray, predicted: int) -> str:
    state = CookingState(predicted).name
    prob_str = ", ".join([f"{CookingState(i).name}:{p:.3f}" for i, p in enumerate(probs)])
    return f"Predicted: {state} | Probs: {prob_str}"


def main():
    parser = argparse.ArgumentParser(description="Frying model predictor")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--image_dir", type=str, help="Directory of images for sequence")
    parser.add_argument("--food_type", type=str, default="기타", help="Food type name")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--temporal", action="store_true", help="Use temporal model")
    parser.add_argument("--use_lift_sequence", action="store_true", help="Use lift detector on sequence")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--no_bubble", action="store_true", help="Disable bubble features")
    args = parser.parse_args()

    predictor = FryingPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_temporal=args.temporal,
        use_bubble_features=not args.no_bubble,
    )

    if args.image:
        probs, predicted = predictor.predict_image(args.image, args.food_type)
        print(_format_prediction(probs, predicted))
        return

    if args.image_dir:
        image_paths = sorted(Path(args.image_dir).glob("*.jpg"))
        probs, predicted = predictor.predict_sequence(
            [str(p) for p in image_paths],
            args.food_type,
            use_lift_sequence=args.use_lift_sequence,
            max_frames=args.max_frames,
        )
        print(_format_prediction(probs, predicted))
        return

    raise SystemExit("Provide --image or --image_dir")


if __name__ == "__main__":
    main()
