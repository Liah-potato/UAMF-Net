"""Image experts, deterministic inference views, and strict task checkpoint loading."""

from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import timm
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .fusion import Prediction


def select_device(requested):
    name = ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("This implementation supports CPU or CUDA devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def amp_context(device, enabled):
    return torch.autocast("cuda", dtype=torch.float16) if enabled and device.type == "cuda" else nullcontext()


def image_transform(config, modality, training=False, view=0):
    spec = config["models"][modality]
    resize, crop = spec["resize_size"], spec["input_size"]
    if view == 4:
        resize = round(resize * 280 / 256)
    ops = [transforms.Resize((resize, resize), interpolation=InterpolationMode.BICUBIC)]
    ops.append(transforms.RandomCrop(crop) if training else transforms.CenterCrop(crop))
    if training:
        ops.append(transforms.RandomHorizontalFlip())
    elif view == 1:
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    elif view == 2:
        ops.append(transforms.RandomVerticalFlip(p=1.0))
    elif view == 3:
        ops.append(transforms.RandomRotation((90, 90), interpolation=InterpolationMode.BICUBIC))
    ops.extend([transforms.ToTensor(), transforms.Normalize(**config["normalization"])])
    return transforms.Compose(ops)


def checkpoint_metadata(config, modality):
    return {"modality": modality, "model": config["models"][modality],
            "classes": config["classes"], "normalization": config["normalization"]}


def load_expert_state(config, modality, state, path):
    model = timm.create_model(config["models"][modality]["tag"], pretrained=False, num_classes=3)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Weights in {path} do not match the configured {modality} model: {exc}") from exc
    return model


def import_expert_weights(config, modality, source, checkpoint_dir):
    """Validate an original state_dict and save a checkpoint for the fusion runner."""
    source = Path(source)
    destination = Path(checkpoint_dir) / f"{modality}_best.pt"
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists: {destination}")
    state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not state or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError(f"Expected a task model state_dict saved by the original scripts: {source}")
    if all(name.startswith("model.") for name in state):
        state = {name.removeprefix("model."): value for name, value in state.items()}
    model = load_expert_state(config, modality, state, source)
    return save_expert_checkpoint(
        config, modality, model.state_dict(), checkpoint_dir,
        {"format": "imported_state_dict", "source": source.name},
    )


def save_expert_checkpoint(config, modality, state, checkpoint_dir, provenance):
    destination = Path(checkpoint_dir) / f"{modality}_best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as staging_dir:
        temporary = Path(staging_dir) / destination.name
        torch.save({"state_dict": state,
                    "metadata": checkpoint_metadata(config, modality),
                    "provenance": provenance},
                   temporary)
        # Publish only a complete checkpoint without overwriting an existing run.
        destination.hardlink_to(temporary)
    return destination


class ImageExperts:
    def __init__(self, config, checkpoint_dir, device="auto"):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = select_device(device)
        self.loaded = {}

    def load(self, modality):
        if modality in self.loaded:
            return self.loaded[modality]
        path = self.checkpoint_dir / f"{modality}_best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Task checkpoint required: {path}. Use train or import-weights to supply it.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        expected = checkpoint_metadata(self.config, modality)
        if checkpoint.get("metadata") != expected:
            raise ValueError(f"Checkpoint architecture, class order, or preprocessing mismatch: {path}")
        model = load_expert_state(self.config, modality, checkpoint["state_dict"], path)
        model.to(self.device).eval()
        self.loaded[modality] = model
        return model

    def predict(self, modality, path):
        if path is None:
            if modality == "pathology":
                raise ValueError("A pathology image is required")
            return None
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Unreadable {modality} image: {path}") from exc
        model = self.load(modality)
        views = self.config["pathology_views"] if modality == "pathology" else 1
        probabilities = []
        # Sequential forwards: no claim of simultaneous TTA or measured speedup.
        with torch.inference_mode(), amp_context(self.device, self.config["training"]["amp"]):
            for view in range(views):
                inputs = image_transform(self.config, modality, view=view)(image).unsqueeze(0).to(self.device)
                probabilities.append(torch.softmax(model(inputs).float(), dim=1)[0])
        average = torch.stack(probabilities).mean(0).cpu().tolist()
        return Prediction(tuple(average))
