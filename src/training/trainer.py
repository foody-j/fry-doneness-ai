"""
학습 파이프라인

튀김 조리 완료 판단 모델 학습
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import json
import time
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.frying_model import FryingModel, FryingModelSimple, CookingState
from src.data.dataset import FryingDataset, create_dataloaders


class Trainer:
    """모델 학습기"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 손실 함수 (클래스 가중치 적용)
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 옵티마이저
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # 스케줄러
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6,
        )

        # 체크포인트 & 로그
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # 학습 상태
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_history: List[Dict] = []

    def train_epoch(self) -> Dict[str, float]:
        """한 에폭 학습"""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            # 데이터 로드
            images = batch['image'].to(self.device)
            bubble_features = batch['bubble_features'].to(self.device)
            food_type = batch['food_type'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward
            self.optimizer.zero_grad()

            if isinstance(self.model, FryingModelSimple):
                outputs = self.model(images, food_type)
            else:
                outputs = self.model(images, bubble_features, food_type)

            loss = self.criterion(outputs, labels)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 통계
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """검증"""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # 클래스별 정확도
        class_correct = [0, 0, 0]
        class_total = [0, 0, 0]

        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['image'].to(self.device)
            bubble_features = batch['bubble_features'].to(self.device)
            food_type = batch['food_type'].to(self.device)
            labels = batch['label'].to(self.device)

            if isinstance(self.model, FryingModelSimple):
                outputs = self.model(images, food_type)
            else:
                outputs = self.model(images, bubble_features, food_type)

            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 클래스별 통계
            for i in range(3):
                mask = labels == i
                class_total[i] += mask.sum().item()
                class_correct[i] += (predicted[mask] == labels[mask]).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        # 클래스별 정확도
        class_acc = {}
        for i, state in enumerate(CookingState):
            if class_total[i] > 0:
                class_acc[state.name] = 100. * class_correct[i] / class_total[i]
            else:
                class_acc[state.name] = 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'class_accuracy': class_acc,
        }

    def train(
        self,
        num_epochs: int = 100,
        early_stopping_patience: int = 15,
        save_every: int = 10,
    ):
        """전체 학습 루프"""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")

        patience_counter = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1

            # 학습
            train_metrics = self.train_epoch()

            # 검증
            val_metrics = self.validate()

            # 스케줄러 업데이트
            self.scheduler.step()

            # 로깅
            self._log_metrics(train_metrics, val_metrics)

            # 히스토리 저장
            self.train_history.append({
                'epoch': self.current_epoch,
                'train': train_metrics,
                'val': val_metrics,
                'lr': self.optimizer.param_groups[0]['lr'],
            })

            # 출력
            print(f"\nEpoch {self.current_epoch}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            print(f"  Class Acc: {val_metrics['class_accuracy']}")

            # Best 모델 저장
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint('best.pt')
                print(f"  -> New best model! (Acc: {self.best_val_acc:.2f}%)")
                patience_counter = 0
            else:
                patience_counter += 1

            # 정기 저장
            if self.current_epoch % save_every == 0:
                self.save_checkpoint(f'epoch_{self.current_epoch}.pt')

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {self.current_epoch}")
                break

        # 최종 저장
        self.save_checkpoint('final.pt')
        self._save_history()

        print(f"\nTraining complete!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")

    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """TensorBoard 로깅"""
        self.writer.add_scalar('Loss/train', train_metrics['loss'], self.current_epoch)
        self.writer.add_scalar('Loss/val', val_metrics['loss'], self.current_epoch)
        self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], self.current_epoch)
        self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], self.current_epoch)
        self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.current_epoch)

        for class_name, acc in val_metrics['class_accuracy'].items():
            self.writer.add_scalar(f'ClassAcc/{class_name}', acc, self.current_epoch)

    def save_checkpoint(self, filename: str):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
        }
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, filename: str):
        """체크포인트 로드"""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']

        print(f"Checkpoint loaded: {path}")
        print(f"Resuming from epoch {self.current_epoch}")

    def _save_history(self):
        """학습 히스토리 저장"""
        path = self.log_dir / 'train_history.json'
        with open(path, 'w') as f:
            json.dump(self.train_history, f, indent=2)


def train_model(
    data_root: str,
    checkpoint_dir: str = 'checkpoints',
    log_dir: str = 'logs',
    num_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    use_temporal: bool = False,
    device: str = 'cuda',
):
    """
    모델 학습 메인 함수

    Args:
        data_root: 데이터 루트 경로
        checkpoint_dir: 체크포인트 저장 경로
        log_dir: 로그 저장 경로
        num_epochs: 학습 에폭 수
        batch_size: 배치 크기
        learning_rate: 학습률
        use_temporal: 시계열 모델 사용 여부
        device: 학습 디바이스
    """
    # 디바이스 확인
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"Using device: {device}")

    # 데이터 로더 생성
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        val_split=0.2,
        num_workers=4,
        use_sequence=use_temporal,
    )

    # 클래스 가중치 계산
    dataset = train_loader.dataset.dataset  # unwrap Subset
    class_weights = dataset.get_class_weights()
    print(f"Class weights: {class_weights}")

    # 모델 생성
    print("\nCreating model...")
    if use_temporal:
        model = FryingModel(
            num_food_types=10,
            num_classes=3,
            use_temporal=True,
            pretrained=True,
            freeze_backbone=True,
        )
    else:
        model = FryingModel(
            num_food_types=10,
            num_classes=3,
            use_temporal=False,
            pretrained=True,
            freeze_backbone=True,
        )

    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Trainer 생성
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        class_weights=class_weights,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )

    # 학습
    print("\nStarting training...")
    trainer.train(
        num_epochs=num_epochs,
        early_stopping_patience=15,
    )

    return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train frying model')
    parser.add_argument('--data_root', type=str, required=True, help='Data root directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temporal', action='store_true', help='Use temporal model')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    train_model(
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_temporal=args.temporal,
        device=args.device,
    )
