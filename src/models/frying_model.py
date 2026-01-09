"""
튀김 조리 완료 판단 모델

3-스트림 아키텍처:
1. 탈탈 이미지 스트림 (EfficientNet-B2) - 메인
2. 기포 특징 스트림 - 보조
3. 튀김 종류 임베딩 - 조건부 입력
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple, List
from enum import IntEnum


class CookingState(IntEnum):
    """조리 상태 클래스"""
    COOKING = 0       # 조리중 (1차 탈탈)
    ALMOST_DONE = 1   # 거의완료 (2차 탈탈)
    DONE = 2          # 완료 (3차 탈탈)


class ImageEncoder(nn.Module):
    """
    이미지 인코더 (EfficientNet-B2 기반)

    탈탈 이미지에서 튀김 상태 특징 추출
    """

    def __init__(
        self,
        pretrained: bool = True,
        feature_dim: int = 256,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # EfficientNet-B2 백본
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b2(weights=weights)

        # 백본 동결 (선택적)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 분류기 헤드 제거하고 특징 추출기로 사용
        backbone_out_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # 특징 변환 레이어
        self.feature_transform = nn.Sequential(
            nn.Linear(backbone_out_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True),
        )

        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width) 또는
               (batch, seq_len, channels, height, width)

        Returns:
            (batch, feature_dim) 또는 (batch, seq_len, feature_dim)
        """
        # 시퀀스 입력 처리
        if x.dim() == 5:
            batch, seq_len, c, h, w = x.shape
            x = x.view(batch * seq_len, c, h, w)
            features = self.backbone(x)
            features = self.feature_transform(features)
            features = features.view(batch, seq_len, -1)
        else:
            features = self.backbone(x)
            features = self.feature_transform(features)

        return features

    def unfreeze_backbone(self, layers: int = -1):
        """
        백본 일부 또는 전체 동결 해제

        Args:
            layers: 동결 해제할 레이어 수 (-1이면 전체)
        """
        params = list(self.backbone.parameters())
        if layers == -1:
            for param in params:
                param.requires_grad = True
        else:
            for param in params[-layers:]:
                param.requires_grad = True


class BubbleFeatureEncoder(nn.Module):
    """
    기포 특징 인코더

    상대적 기포 변화 특징을 처리
    """

    def __init__(
        self,
        input_dim: int = 6,  # RelativeBubbleFeatures 차원
        hidden_dim: int = 32,
        output_dim: int = 16,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) 또는 (batch, seq_len, input_dim)

        Returns:
            (batch, output_dim) 또는 (batch, seq_len, output_dim)
        """
        return self.encoder(x)


class FoodTypeEmbedding(nn.Module):
    """
    튀김 종류 임베딩

    튀김 종류별로 다른 완료 기준을 학습
    """

    def __init__(
        self,
        num_food_types: int = 10,
        embedding_dim: int = 16,
    ):
        super().__init__()

        self.embedding = nn.Embedding(num_food_types, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch,) 튀김 종류 인덱스

        Returns:
            (batch, embedding_dim)
        """
        return self.embedding(x)


class TemporalEncoder(nn.Module):
    """
    시계열 인코더 (Bi-LSTM)

    탈탈 시퀀스의 시간적 패턴 학습
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            lengths: (batch,) 각 시퀀스의 실제 길이

        Returns:
            (batch, output_dim)
        """
        if lengths is not None:
            # 패딩된 시퀀스 처리
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(x)

        # Bidirectional LSTM의 마지막 hidden state 결합
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]

        return hidden


class FryingModel(nn.Module):
    """
    튀김 조리 완료 판단 통합 모델

    3-스트림 아키텍처:
    - 탈탈 이미지 → ImageEncoder → 256d
    - 기포 특징 → BubbleFeatureEncoder → 16d
    - 튀김 종류 → FoodTypeEmbedding → 16d
    → Concat → (선택적) TemporalEncoder → Classifier → 3-class
    """

    def __init__(
        self,
        num_food_types: int = 10,
        num_classes: int = 3,
        image_feature_dim: int = 256,
        bubble_feature_dim: int = 6,
        bubble_output_dim: int = 16,
        food_embedding_dim: int = 16,
        use_temporal: bool = False,
        temporal_hidden_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.use_temporal = use_temporal

        # 1. 이미지 인코더
        self.image_encoder = ImageEncoder(
            pretrained=pretrained,
            feature_dim=image_feature_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

        # 2. 기포 특징 인코더
        self.bubble_encoder = BubbleFeatureEncoder(
            input_dim=bubble_feature_dim,
            output_dim=bubble_output_dim,
        )

        # 3. 튀김 종류 임베딩
        self.food_embedding = FoodTypeEmbedding(
            num_food_types=num_food_types,
            embedding_dim=food_embedding_dim,
        )

        # 결합된 특징 차원
        combined_dim = image_feature_dim + bubble_output_dim + food_embedding_dim

        # 4. 시계열 인코더 (선택적)
        if use_temporal:
            self.temporal_encoder = TemporalEncoder(
                input_dim=combined_dim,
                hidden_dim=temporal_hidden_dim,
                dropout=dropout,
            )
            classifier_input_dim = self.temporal_encoder.output_dim
        else:
            self.temporal_encoder = None
            classifier_input_dim = combined_dim

        # 5. 분류기
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self.num_classes = num_classes

    def forward(
        self,
        images: torch.Tensor,
        bubble_features: torch.Tensor,
        food_type: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            images: (batch, C, H, W) 또는 (batch, seq_len, C, H, W)
            bubble_features: (batch, bubble_dim) 또는 (batch, seq_len, bubble_dim)
            food_type: (batch,) 튀김 종류 인덱스
            seq_lengths: (batch,) 시퀀스 길이 (temporal 모드에서 사용)

        Returns:
            (batch, num_classes) 로짓
        """
        # 이미지 인코딩
        img_features = self.image_encoder(images)

        # 기포 특징 인코딩
        bubble_encoded = self.bubble_encoder(bubble_features)

        # 튀김 종류 임베딩
        food_emb = self.food_embedding(food_type)

        # 시퀀스 처리
        if self.use_temporal and images.dim() == 5:
            batch, seq_len = images.shape[:2]

            # food_emb를 시퀀스 길이만큼 확장
            food_emb = food_emb.unsqueeze(1).expand(-1, seq_len, -1)

            # 결합
            combined = torch.cat([img_features, bubble_encoded, food_emb], dim=-1)

            # 시계열 인코딩
            features = self.temporal_encoder(combined, seq_lengths)
        else:
            # 단일 이미지 또는 temporal 미사용
            if images.dim() == 5:
                # 시퀀스의 마지막 프레임만 사용
                img_features = img_features[:, -1, :]
                bubble_encoded = bubble_encoded[:, -1, :]

            combined = torch.cat([img_features, bubble_encoded, food_emb], dim=-1)
            features = combined

        # 분류
        logits = self.classifier(features)

        return logits

    def predict(
        self,
        images: torch.Tensor,
        bubble_features: torch.Tensor,
        food_type: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        예측 (소프트맥스 확률과 클래스 반환)

        Returns:
            (probs, predicted_class)
        """
        logits = self.forward(images, bubble_features, food_type, seq_lengths)
        probs = F.softmax(logits, dim=-1)
        predicted = torch.argmax(probs, dim=-1)
        return probs, predicted

    def unfreeze_backbone(self, layers: int = -1):
        """백본 동결 해제 (fine-tuning용)"""
        self.image_encoder.unfreeze_backbone(layers)


class FryingModelSimple(nn.Module):
    """
    간단한 버전 (단일 탈탈 이미지만 사용)

    시계열 없이 마지막 탈탈 이미지만으로 판단
    """

    def __init__(
        self,
        num_food_types: int = 10,
        num_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        # EfficientNet-B2
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b2(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 특징 차원
        backbone_features = self.backbone.classifier[1].in_features

        # 튀김 종류 임베딩
        self.food_embedding = nn.Embedding(num_food_types, 16)

        # 분류기 교체
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(backbone_features + 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        images: torch.Tensor,
        food_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images: (batch, C, H, W)
            food_type: (batch,)

        Returns:
            (batch, num_classes)
        """
        # 백본 특징 추출 (classifier 전까지)
        x = self.backbone.features(images)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        # 튀김 종류 임베딩
        food_emb = self.food_embedding(food_type)

        # 결합
        x = torch.cat([x, food_emb], dim=1)

        # 분류
        x = self.backbone.classifier(x)

        return x


# 튀김 종류 매핑
FOOD_TYPE_MAP = {
    '적어튀김': 0,
    '돈까스': 1,
    '치킨': 2,
    '고구마튀김': 3,
    '튀김만두': 4,
    '생선까스': 5,
    '새우튀김': 6,
    '오징어튀김': 7,
    '야채튀김': 8,
    '기타': 9,
}


def get_food_type_id(food_name: str) -> int:
    """튀김 이름 → ID 변환"""
    return FOOD_TYPE_MAP.get(food_name, FOOD_TYPE_MAP['기타'])


if __name__ == "__main__":
    # 테스트
    print("=== FryingModel Test ===")

    # 모델 생성
    model = FryingModel(
        num_food_types=10,
        num_classes=3,
        use_temporal=False,
        pretrained=True,
        freeze_backbone=True,
    )

    # 더미 입력
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    bubble_features = torch.randn(batch_size, 6)
    food_type = torch.randint(0, 10, (batch_size,))

    # Forward
    logits = model(images, bubble_features, food_type)
    print(f"Output shape: {logits.shape}")  # (4, 3)

    # 예측
    probs, predicted = model.predict(images, bubble_features, food_type)
    print(f"Probs: {probs}")
    print(f"Predicted: {predicted}")

    # 시계열 모델 테스트
    print("\n=== FryingModel (Temporal) Test ===")
    model_temporal = FryingModel(
        num_food_types=10,
        num_classes=3,
        use_temporal=True,
        pretrained=True,
    )

    seq_len = 3
    images_seq = torch.randn(batch_size, seq_len, 3, 224, 224)
    bubble_seq = torch.randn(batch_size, seq_len, 6)

    logits_temporal = model_temporal(images_seq, bubble_seq, food_type)
    print(f"Temporal output shape: {logits_temporal.shape}")  # (4, 3)

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
