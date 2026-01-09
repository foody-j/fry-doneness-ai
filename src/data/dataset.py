"""
튀김 조리 데이터셋

세션 데이터를 로드하고 학습용 데이터로 변환
"""

import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .lift_detector import LiftDetector, FrameType
from .bubble_features import BubbleFeatureExtractor, RelativeBubbleFeatures


@dataclass
class SessionInfo:
    """세션 정보"""
    session_id: str
    pot: str
    food_type: str
    start_time: str
    end_time: str
    duration_sec: float
    image_dir: Path
    meta_dir: Path


@dataclass
class LiftSample:
    """탈탈 샘플 (학습 데이터 단위)"""
    image_path: str
    food_type: str
    food_type_id: int
    lift_order: int          # 1차, 2차, 3차 탈탈
    label: int               # 0: 조리중, 1: 거의완료, 2: 완료
    session_id: str
    bubble_features: Optional[np.ndarray] = None


class FryingDataset(Dataset):
    """
    튀김 조리 데이터셋

    세션에서 탈탈 이미지를 추출하고 자동 라벨링
    """

    # 튀김 종류 → ID 매핑
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

    def __init__(
        self,
        data_root: str,
        transform: Optional[transforms.Compose] = None,
        use_bubble_features: bool = True,
        image_size: Tuple[int, int] = (224, 224),
        cache_lift_detection: bool = True,
    ):
        """
        Args:
            data_root: 데이터 루트 디렉토리 (pot1, pot2 등이 있는 곳)
            transform: 이미지 변환
            use_bubble_features: 기포 특징 사용 여부
            image_size: 이미지 크기
            cache_lift_detection: 탈탈 감지 결과 캐시 여부
        """
        self.data_root = Path(data_root)
        self.use_bubble_features = use_bubble_features
        self.image_size = image_size
        self.cache_lift_detection = cache_lift_detection

        # 이미지 변환
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transform

        # 탈탈 감지기 & 기포 추출기
        self.lift_detector = LiftDetector()
        self.bubble_extractor = BubbleFeatureExtractor() if use_bubble_features else None

        # 세션 로드 및 샘플 생성
        self.sessions: List[SessionInfo] = []
        self.samples: List[LiftSample] = []

        self._load_sessions()
        self._extract_samples()

    def _load_sessions(self):
        """세션 정보 로드"""
        # pot 디렉토리들 탐색
        if self.data_root.name.startswith("pot") and self.data_root.is_dir():
            pot_dirs = [self.data_root]
        else:
            pot_dirs = sorted(self.data_root.glob("pot*"))

        for pot_dir in pot_dirs:
            if not pot_dir.is_dir():
                continue

            # 세션 디렉토리들 탐색
            for session_dir in sorted(pot_dir.glob("session_*")):
                if not session_dir.is_dir():
                    continue

                # 음식 종류 디렉토리 (세션 내)
                for food_dir in session_dir.iterdir():
                    if not food_dir.is_dir():
                        continue

                    session_info_path = food_dir / "session_info.json"
                    if not session_info_path.exists():
                        continue

                    try:
                        with open(session_info_path, 'r', encoding='utf-8') as f:
                            info = json.load(f)

                        session = SessionInfo(
                            session_id=info.get('session_id', session_dir.name),
                            pot=info.get('pot', pot_dir.name),
                            food_type=info.get('food_type', food_dir.name),
                            start_time=info.get('start_time', ''),
                            end_time=info.get('end_time', ''),
                            duration_sec=info.get('duration_sec', 0),
                            image_dir=food_dir / "camera_0",
                            meta_dir=food_dir / "meta",
                        )
                        self.sessions.append(session)

                    except Exception as e:
                        print(f"Warning: Failed to load session {session_dir}: {e}")

        print(f"Loaded {len(self.sessions)} sessions")

    def _extract_samples(self):
        """각 세션에서 탈탈 샘플 추출"""
        for session in self.sessions:
            samples = self._extract_session_samples(session)
            self.samples.extend(samples)

        print(f"Extracted {len(self.samples)} samples")

    def _extract_session_samples(self, session: SessionInfo) -> List[LiftSample]:
        """단일 세션에서 샘플 추출"""
        samples = []

        # 이미지 파일 정렬 (시간순)
        image_paths = sorted(session.image_dir.glob("*.jpg"))
        if not image_paths:
            return samples

        # 탈탈 시점 감지
        self.lift_detector.reset()
        lift_sequence = self.lift_detector.get_lift_sequence(
            [str(p) for p in image_paths]
        )

        if not lift_sequence:
            # 탈탈 감지 실패 시 - 규칙 기반 대체
            # 세션을 3등분하여 샘플링
            n = len(image_paths)
            if n >= 3:
                indices = [n // 4, n // 2, 3 * n // 4]
                lift_sequence = [(i, str(image_paths[i])) for i in indices]

        # 기포 특징 추출 (세션 전체)
        bubble_features_list = None
        if self.use_bubble_features and self.bubble_extractor:
            try:
                # 튀김유만 보이는 프레임에서 기포 특징 추출
                oil_frames = [str(p) for p in image_paths[:10]]  # 초반 10프레임
                bubble_features_list = self.bubble_extractor.process_session(
                    oil_frames, return_relative=True
                )
            except Exception as e:
                print(f"Warning: Bubble feature extraction failed: {e}")

        # 튀김 종류 ID
        food_type_id = self.FOOD_TYPE_MAP.get(
            session.food_type,
            self.FOOD_TYPE_MAP['기타']
        )

        # 탈탈 샘플 생성 (자동 라벨링)
        num_lifts = len(lift_sequence)
        for i, (frame_idx, image_path) in enumerate(lift_sequence):
            # 자동 라벨링: 마지막 탈탈 = 완료, 그 전 = 거의완료, 처음 = 조리중
            if i == num_lifts - 1:
                label = 2  # 완료
            elif i == num_lifts - 2 and num_lifts >= 2:
                label = 1  # 거의완료
            else:
                label = 0  # 조리중

            # 기포 특징 (해당 시점의 상대적 변화)
            bubble_feat = None
            if bubble_features_list and len(bubble_features_list) > 0:
                # 마지막 기포 특징 사용 (간단화)
                bubble_feat = bubble_features_list[-1]

            sample = LiftSample(
                image_path=image_path,
                food_type=session.food_type,
                food_type_id=food_type_id,
                lift_order=i + 1,
                label=label,
                session_id=session.session_id,
                bubble_features=bubble_feat,
            )
            samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 이미지 로드
        image = cv2.imread(sample.image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {sample.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 변환 적용
        if self.transform:
            image = self.transform(image)

        # 기포 특징
        if sample.bubble_features is not None:
            bubble_features = torch.tensor(sample.bubble_features, dtype=torch.float32)
        else:
            # 기본값
            bubble_features = torch.zeros(6, dtype=torch.float32)

        return {
            'image': image,
            'bubble_features': bubble_features,
            'food_type': torch.tensor(sample.food_type_id, dtype=torch.long),
            'label': torch.tensor(sample.label, dtype=torch.long),
            'lift_order': sample.lift_order,
            'session_id': sample.session_id,
            'image_path': sample.image_path,
        }

    def get_class_weights(self) -> torch.Tensor:
        """클래스 불균형 해결용 가중치 계산"""
        labels = [s.label for s in self.samples]
        class_counts = np.bincount(labels, minlength=3)
        total = len(labels)

        # 역빈도 가중치
        weights = total / (3 * class_counts + 1e-6)
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_info(self) -> Dict[str, Any]:
        """데이터셋 정보 반환"""
        labels = [s.label for s in self.samples]
        food_types = [s.food_type for s in self.samples]

        return {
            'total_samples': len(self.samples),
            'total_sessions': len(self.sessions),
            'class_distribution': {
                'cooking': labels.count(0),
                'almost_done': labels.count(1),
                'done': labels.count(2),
            },
            'food_type_distribution': {
                ft: food_types.count(ft) for ft in set(food_types)
            },
        }


class FryingDatasetSequence(Dataset):
    """
    시퀀스 버전 데이터셋

    세션의 탈탈 시퀀스 전체를 하나의 샘플로
    """

    def __init__(
        self,
        data_root: str,
        transform: Optional[transforms.Compose] = None,
        max_seq_len: int = 4,
        image_size: Tuple[int, int] = (224, 224),
    ):
        self.data_root = Path(data_root)
        self.max_seq_len = max_seq_len
        self.image_size = image_size

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transform

        self.lift_detector = LiftDetector()
        self.bubble_extractor = BubbleFeatureExtractor()

        # 세션 단위 샘플
        self.session_samples: List[Dict] = []
        self._load_sessions()

    def _load_sessions(self):
        """세션 로드 및 시퀀스 샘플 생성"""
        if self.data_root.name.startswith("pot") and self.data_root.is_dir():
            pot_dirs = [self.data_root]
        else:
            pot_dirs = sorted(self.data_root.glob("pot*"))

        for pot_dir in pot_dirs:
            for session_dir in sorted(pot_dir.glob("session_*")):
                for food_dir in session_dir.iterdir():
                    if not food_dir.is_dir():
                        continue

                    image_dir = food_dir / "camera_0"
                    if not image_dir.exists():
                        continue

                    session_info_path = food_dir / "session_info.json"
                    food_type = food_dir.name

                    if session_info_path.exists():
                        with open(session_info_path, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                            food_type = info.get('food_type', food_dir.name)

                    # 이미지 정렬
                    image_paths = sorted(image_dir.glob("*.jpg"))
                    if len(image_paths) < 3:
                        continue

                    # 탈탈 시점 감지
                    self.lift_detector.reset()
                    lift_sequence = self.lift_detector.get_lift_sequence(
                        [str(p) for p in image_paths]
                    )

                    if len(lift_sequence) >= 2:
                        self.session_samples.append({
                            'session_id': session_dir.name,
                            'food_type': food_type,
                            'lift_sequence': lift_sequence,
                            'all_images': [str(p) for p in image_paths],
                        })

        print(f"Loaded {len(self.session_samples)} session sequences")

    def __len__(self) -> int:
        return len(self.session_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        session = self.session_samples[idx]
        lift_sequence = session['lift_sequence']

        # 이미지 시퀀스 로드
        images = []
        for _, path in lift_sequence[:self.max_seq_len]:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # 패딩
        seq_len = len(images)
        while len(images) < self.max_seq_len:
            images.append(torch.zeros_like(images[0]))

        images = torch.stack(images)

        # 기포 특징 (시퀀스)
        bubble_features = torch.zeros(self.max_seq_len, 6)

        # 튀김 종류
        food_type_id = FryingDataset.FOOD_TYPE_MAP.get(
            session['food_type'],
            FryingDataset.FOOD_TYPE_MAP['기타']
        )

        # 라벨: 마지막 탈탈 = 완료
        label = 2

        return {
            'images': images,
            'bubble_features': bubble_features,
            'food_type': torch.tensor(food_type_id, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'seq_len': torch.tensor(seq_len, dtype=torch.long),
            'session_id': session['session_id'],
        }


def create_dataloaders(
    data_root: str,
    batch_size: int = 16,
    val_split: float = 0.2,
    num_workers: int = 4,
    use_sequence: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    학습/검증 데이터로더 생성

    Args:
        data_root: 데이터 루트 경로
        batch_size: 배치 크기
        val_split: 검증 데이터 비율
        num_workers: 데이터 로딩 워커 수
        use_sequence: 시퀀스 데이터셋 사용 여부

    Returns:
        (train_loader, val_loader)
    """
    # 데이터 증강 (학습용)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if use_sequence:
        dataset = FryingDatasetSequence(data_root, transform=train_transform)
    else:
        dataset = FryingDataset(data_root, transform=train_transform)

    # 학습/검증 분할
    n_samples = len(dataset)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    # 검증셋 변환 교체 (증강 없이)
    # Note: random_split 후에는 subset이므로 직접 transform 변경 어려움
    # 실제로는 별도 데이터셋 생성 권장

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <data_root>")
        sys.exit(1)

    data_root = sys.argv[1]

    print("=== Loading Dataset ===")
    dataset = FryingDataset(data_root)

    print(f"\n=== Dataset Info ===")
    info = dataset.get_sample_info()
    print(f"Total samples: {info['total_samples']}")
    print(f"Total sessions: {info['total_sessions']}")
    print(f"Class distribution: {info['class_distribution']}")
    print(f"Food type distribution: {info['food_type_distribution']}")

    if len(dataset) > 0:
        print(f"\n=== Sample Data ===")
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Bubble features: {sample['bubble_features']}")
        print(f"Food type: {sample['food_type']}")
        print(f"Label: {sample['label']}")
        print(f"Lift order: {sample['lift_order']}")
