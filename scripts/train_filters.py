"""
ADAMO ID — Script de fine-tuning para Filtros 1 y 2.

Entrena los modelos de Screen Capture (CMA) y Print Detection (EfficientNet-B4)
sobre datasets SIDTD + DLC-2021.

Uso:
    python scripts/train_filters.py --filter screen --dataset /path/to/sidtd --epochs 40
    python scripts/train_filters.py --filter print --dataset /path/to/dlc2021 --epochs 50
"""

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Dataset genérico para clasificación binaria ────────────────
class BinaryImageDataset(Dataset):
    """
    Espera estructura:
        dataset_dir/
            real/     ← imágenes genuinas
            fake/     ← imágenes falsas (screen recapture, printed, etc.)
    """

    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        real_dir = root / "real"
        fake_dir = root / "fake"

        if real_dir.exists():
            for f in real_dir.iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    self.samples.append((f, 0))  # 0 = real/original

        if fake_dir.exists():
            for f in fake_dir.iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    self.samples.append((f, 1))  # 1 = fake/attack

        logger.info("Dataset: %d samples (%d real, %d fake)",
                     len(self.samples),
                     sum(1 for _, l in self.samples if l == 0),
                     sum(1 for _, l in self.samples if l == 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label


# ── Entrenamiento ──────────────────────────────────────────────
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    freeze_epochs: int = 10,
    lr_frozen: float = 1e-3,
    lr_unfrozen: float = 1e-5,
    save_path: Path | None = None,
) -> nn.Module:
    """
    Entrena con estrategia de 2 fases:
    1. Backbone congelado por freeze_epochs (lr alta, solo head)
    2. Backbone descongelado parcial por el resto (lr baja, fine-tune)
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # ── Fase 1: backbone congelado ──────────────────────────────
    logger.info("=== Phase 1: Frozen backbone (%d epochs, lr=%.0e) ===", freeze_epochs, lr_frozen)
    for name, param in model.named_parameters():
        if "classifier" not in name and "fc" not in name:
            param.requires_grad = False

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_frozen, weight_decay=0.01)

    for epoch in range(freeze_epochs):
        _train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        _validate(model, val_loader, criterion, device, epoch)

    # ── Fase 2: backbone descongelado ───────────────────────────
    remaining = epochs - freeze_epochs
    if remaining > 0:
        logger.info("=== Phase 2: Unfrozen backbone (%d epochs, lr=%.0e) ===", remaining, lr_unfrozen)
        for param in model.parameters():
            param.requires_grad = True

        optimizer = AdamW(model.parameters(), lr=lr_unfrozen, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=remaining)

        best_acc = 0.0
        for epoch in range(remaining):
            _train_epoch(model, train_loader, optimizer, criterion, device, freeze_epochs + epoch)
            acc = _validate(model, val_loader, criterion, device, freeze_epochs + epoch)
            scheduler.step()

            if acc > best_acc and save_path:
                best_acc = acc
                torch.save(model.state_dict(), save_path)
                logger.info("Best model saved: %.2f%% → %s", acc * 100, save_path)

    if save_path and not save_path.exists():
        torch.save(model.state_dict(), save_path)
        logger.info("Final model saved → %s", save_path)

    return model


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int,
) -> None:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.perf_counter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    elapsed = time.perf_counter() - t0
    logger.info("Epoch %d TRAIN: loss=%.4f acc=%.2f%% (%.1fs)",
                epoch, total_loss / total, 100 * correct / total, elapsed)


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
) -> float:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    acc = correct / total if total > 0 else 0.0
    logger.info("Epoch %d VAL:   loss=%.4f acc=%.2f%%", epoch, total_loss / total, 100 * acc)
    return acc


# ── Main ───────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="ADAMO ID — Fine-tune detection models")
    parser.add_argument("--filter", choices=["screen", "print", "forgery"], required=True)
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset (real/ + fake/ subdirs)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--freeze-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    weights_dir = Path(__file__).resolve().parent.parent / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Construir modelo y transforms según filtro
    if args.filter == "screen":
        from app.filters.screen_capture import CMAScreenClassifier
        model = CMAScreenClassifier()
        save_path = weights_dir / "screen_capture_cma.pth"
        input_size = 224
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif args.filter == "print":
        from app.filters.printed_paper import PrintDetectionClassifier
        model = PrintDetectionClassifier()
        save_path = weights_dir / "print_detection_effnet.pth"
        input_size = 380
        transform = transforms.Compose([
            transforms.Resize(400),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif args.filter == "forgery":
        from app.filters.forgery_detection import ForgeryFeatureExtractor
        # Para forgery solo entrenamos el clasificador, no segmentación
        model = ForgeryFeatureExtractor()
        save_path = weights_dir / "forgery_hifi.pth"
        input_size = 256
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError(f"Unknown filter: {args.filter}")

    # Dataset
    dataset = BinaryImageDataset(Path(args.dataset), transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger.info("Training %s filter: %d train, %d val, %d epochs on %s",
                args.filter, train_size, val_size, args.epochs, args.device)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        epochs=args.epochs,
        freeze_epochs=args.freeze_epochs,
        save_path=save_path,
    )

    logger.info("Done! Weights saved to %s", save_path)


if __name__ == "__main__":
    main()
