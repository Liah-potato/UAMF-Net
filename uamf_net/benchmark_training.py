"""Train independent experts using the settings in the recovered scripts."""

import random
from pathlib import Path

import numpy as np
import timm
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .metrics import classification_metrics
from .models import save_expert_checkpoint, select_device


PROFILES = {
    # Source workbook: 模型acc.xlsx, 病理!C5, CT!B3, 胃镜!B3 respectively.
    "pathology": {"tag": "convnext_base.fb_in1k", "epochs": 40, "batch_size": 16,
                  "lr": 1e-4, "weight_decay": 0.05, "accumulate": 1, "seed": 42,
                  "selection_metric": "accuracy", "evaluate_every": 1},
    "ct": {"tag": "maxvit_tiny_tf_224.in1k", "epochs": 50, "batch_size": 16,
           "lr": 5e-5, "weight_decay": 0.05, "accumulate": 1, "seed": None,
           "selection_metric": "macro_f1", "evaluate_every": 5},
    "gastro": {"tag": "fastvit_ma36.apple_in1k", "epochs": 50, "batch_size": 32,
               "lr": 1e-3, "weight_decay": 0.02, "accumulate": 2, "seed": 42,
               "selection_metric": "macro_f1", "evaluate_every": 1},
}


def expert_transform(spec, normalization, modality, training):
    size, crop = spec["resize_size"], spec["input_size"]
    operations = [transforms.Resize((size, size))]
    if training:
        if size != crop:
            operations.append(transforms.RandomCrop(crop))
        operations.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
        if modality == "pathology":
            operations.append(transforms.RandomRotation(180))
        elif modality == "gastro":
            operations.extend([transforms.RandomRotation(90),
                               transforms.ColorJitter(brightness=0.15, contrast=0.15)])
    elif size != crop:
        operations.append(transforms.CenterCrop(crop))
    operations.extend([transforms.ToTensor(), transforms.Normalize(**normalization)])
    return transforms.Compose(operations)


def train_independent_expert(config, modality, train_dir, selection_dir, class_folders, checkpoint_dir, device_name):
    profile = PROFILES[modality]
    spec = config["models"][modality]
    resize = 224 if modality == "ct" else 256
    if spec != {"tag": profile["tag"], "input_size": 224, "resize_size": resize}:
        raise ValueError(f"The recovered {modality} profile requires {profile['tag']} and {resize}→224 inputs.")
    if Path(train_dir).resolve() == Path(selection_dir).resolve():
        raise ValueError("Training and checkpoint selection require separate directories.")
    destination = Path(checkpoint_dir) / f"{modality}_best.pt"
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists: {destination}")

    seed = profile["seed"]
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    device = select_device(device_name)
    folders = {name: index for index, name in enumerate(class_folders)}
    loaders = {}
    for training, root in ((True, train_dir), (False, selection_dir)):
        dataset = datasets.ImageFolder(root, expert_transform(spec, config["normalization"], modality, training))
        if len(folders) != 3 or set(dataset.classes) != set(folders):
            raise ValueError(f"Expected the three declared class folders in {root}.")
        targets_by_index = tuple(folders[name] for name in dataset.classes)
        dataset.samples = [(path, targets_by_index[label]) for path, label in dataset.samples]
        dataset.targets = [label for _, label in dataset.samples]
        loaders[training] = DataLoader(dataset, batch_size=profile["batch_size"], shuffle=training,
                                       num_workers=config["training"]["workers"], pin_memory=device.type == "cuda")

    model_options = {"drop_path_rate": 0.2} if modality == "pathology" else {}
    model = timm.create_model(profile["tag"], pretrained=True, num_classes=3, **model_options).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=profile["lr"], weight_decay=profile["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=profile["epochs"])
    loss_function = CrossEntropyLoss(label_smoothing=0.1)
    best_state, best_metrics, best_epoch = None, None, None
    train_loader, selection_loader = loaders[True], loaders[False]
    accumulate = profile["accumulate"]
    for epoch in range(1, profile["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for index, (images, labels) in enumerate(train_loader):
            group_start = index - index % accumulate
            group_size = min(accumulate, len(train_loader) - group_start)
            loss = loss_function(model(images.to(device)), labels.to(device))
            (loss / group_size).backward()
            if (index + 1) % accumulate == 0 or index + 1 == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if epoch != 1 and epoch % profile["evaluate_every"] != 0:
            continue
        model.eval()
        truth, predictions = [], []
        with torch.inference_mode():
            for images, labels in selection_loader:
                predictions.extend(model(images.to(device)).argmax(1).cpu().tolist())
                truth.extend(labels.tolist())
        metrics = classification_metrics(truth, predictions)
        key = profile["selection_metric"]
        print(f"{modality}: epoch {epoch}/{profile['epochs']}, selection {key}={metrics[key]:.4f}")
        if best_metrics is None or metrics[key] > best_metrics[key]:
            best_state = {name: tensor.detach().to(device="cpu", copy=True)
                          for name, tensor in model.state_dict().items()}
            best_metrics, best_epoch = metrics, epoch

    path = save_expert_checkpoint(config, modality, best_state, checkpoint_dir, {
        "format": "independent_expert_training", "profile": profile,
        "selection_epoch": best_epoch, "selection_metrics": best_metrics,
        "class_folders": class_folders, "label_smoothing": 0.1,
    })
    return {"modality": modality, "checkpoint": str(path),
            "selection_epoch": best_epoch, "selection_metrics": best_metrics}
