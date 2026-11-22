"""Independent expert training with validation-only model selection."""

import random
from pathlib import Path

import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import write_json
from .data import modality_records
from .models import amp_context, checkpoint_metadata, image_transform, select_device


class ImageDataset(Dataset):
    def __init__(self, records, modality, transform):
        self.records, self.modality, self.transform = records, modality, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row[self.modality]) as source:
            image = self.transform(source.convert("RGB"))
        return image, row["label"]


def loss_function(logits, targets, settings):
    ce = F.cross_entropy(logits, targets, reduction="none",
                         label_smoothing=settings["label_smoothing"])
    if settings["loss"] == "poly1":
        # p_y is the probability of the hard reference label, computed in float32.
        py = torch.softmax(logits.float(), dim=1).gather(1, targets[:, None]).squeeze(1)
        ce = ce + settings["poly_epsilon"] * (1 - py)
    return ce.mean()


def train_expert(config, rows, modality, checkpoint_dir, device_name):
    settings = config["training"]
    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings["seed"])
    device = select_device(device_name)
    train_rows = modality_records(rows, "train", modality)
    val_rows = modality_records(rows, "val", modality)
    if not train_rows or not val_rows:
        raise ValueError(f"Nonempty train and val data are required for {modality}")
    if {r["label"] for r in train_rows} != {0, 1, 2}:
        raise ValueError(f"Training data for {modality} must include all three classes")
    model = timm.create_model(config["models"][modality]["tag"], pretrained=True, num_classes=3).to(device)
    classifier = model.get_classifier()
    if settings["mode"] == "head_only":
        model.requires_grad_(False)
        classifier.requires_grad_(True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    amp = settings["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loaders = {}
    for split, records in (("train", train_rows), ("val", val_rows)):
        transform = image_transform(config, modality, training=split == "train")
        loaders[split] = DataLoader(ImageDataset(records, modality, transform),
                                    batch_size=settings["batch_size"], shuffle=split == "train",
                                    num_workers=settings["workers"], pin_memory=device.type == "cuda")
    output = Path(checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / f"{modality}_best.pt"
    if checkpoint_path.exists():
        raise FileExistsError(f"Use a fresh checkpoint directory to preserve this run: {checkpoint_path}")
    best_loss, bad_epochs, history = float("inf"), 0, []
    for epoch in range(1, settings["epochs"] + 1):
        if settings["mode"] == "head_only":
            # Keep frozen BatchNorm statistics and stochastic backbone layers fixed.
            model.eval()
            classifier.train()
        else:
            model.train()
        train_loss, train_count = 0.0, 0
        for images, targets in loaders["train"]:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, amp):
                loss = loss_function(model(images), targets, settings)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(targets)
            train_count += len(targets)
        model.eval()
        val_loss, val_count = 0.0, 0
        with torch.inference_mode():
            for images, targets in loaders["val"]:
                images, targets = images.to(device), targets.to(device)
                with amp_context(device, amp):
                    loss = loss_function(model(images), targets, settings)
                val_loss += loss.item() * len(targets)
                val_count += len(targets)
        val_loss /= val_count
        record = {"epoch": epoch, "train_loss": train_loss / train_count, "val_loss": val_loss}
        history.append(record)
        print(f"{modality}: epoch {epoch}, train_loss={record['train_loss']:.5f}, val_loss={val_loss:.5f}")
        if val_loss < best_loss:
            best_loss, bad_epochs = val_loss, 0
            torch.save({"state_dict": model.state_dict(), "metadata": checkpoint_metadata(config, modality),
                        "training": settings, "epoch": epoch, "val_loss": val_loss,
                        "torch_version": str(torch.__version__), "timm_version": str(timm.__version__)},
                       checkpoint_path)
        else:
            bad_epochs += 1
        write_json(output / f"{modality}_history.json", history)
        if bad_epochs >= settings["patience"]:
            break
    return str(checkpoint_path)
