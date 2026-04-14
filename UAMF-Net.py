import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import re
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import timm
import scipy.stats

# ==========================================
# 0. Global Configuration
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3
WEIGHT_DIR = './weights_cache'
os.makedirs(WEIGHT_DIR, exist_ok=True)

# Toggle this: True to train experts, False to run inference/evaluation
TRAIN_MODE = False

# Sniper Protocol Hyperparameters (Aligned with Paper)
THRESHOLD_ENTROPY = 0.45
THRESHOLD_RECALL = 0.65
THRESHOLD_PRECIS = 0.80
MARGIN_REQUIREMENT = 0.15

# Dataset Paths - Strictly separated to avoid Data Leakage
TRAIN_DIRS = {
    "Pathology": "/root/autodl-tmp/READY_DATASET/Pathology/train",
    "CT": "/root/autodl-tmp/READY_DATASET/CT/train",
    "Gastro": "/root/autodl-tmp/READY_DATASET/Gastro/train"
}

TEST_DIRS = {
    "Pathology": "/root/autodl-tmp/READY_DATASET/Pathology/test",
    "CT": "/root/autodl-tmp/READY_DATASET/CT/test",
    "Gastro": "/root/autodl-tmp/READY_DATASET/Gastro/test"
}

# ==========================================
# 1. Transforms & TTA (Test-Time Augmentation)
# ==========================================
common_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
base_size, crop_size = 256, 224

train_tfm = transforms.Compose([
    transforms.Resize((base_size, base_size)),
    transforms.RandomCrop(crop_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), common_norm
])

# Standard single-pass for Macro modalities
standard_tfm = transforms.Compose([
    transforms.Resize((base_size, base_size)),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(), common_norm
])

# 5-view TTA for Pathology (Micro-modality)
tta_tfms = [
    transforms.Compose(
        [transforms.Resize((base_size, base_size)), transforms.CenterCrop(crop_size), transforms.ToTensor(),
         common_norm]),
    transforms.Compose([transforms.Resize((base_size, base_size)), transforms.CenterCrop(crop_size),
                        transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), common_norm]),
    transforms.Compose([transforms.Resize((base_size, base_size)), transforms.CenterCrop(crop_size),
                        transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), common_norm]),
    transforms.Compose([transforms.Resize((base_size, base_size)), transforms.CenterCrop(crop_size),
                        transforms.RandomRotation((90, 90)), transforms.ToTensor(), common_norm]),
    transforms.Compose(
        [transforms.Resize((280, 280)), transforms.CenterCrop(crop_size), transforms.ToTensor(), common_norm]),
]


# ==========================================
# 2. Optimization: O(1) File Indexing
# ==========================================
def build_file_index(root_dir):
    idx = {}
    if not root_dir or not os.path.exists(root_dir): return idx
    for r, _, fs in os.walk(root_dir):
        for f in fs:
            pid_match = re.findall(r'^(\d+)', f)
            if pid_match:
                idx[pid_match[0]] = os.path.join(r, f)
    return idx


# ==========================================
# 3. Model Engine (Freeze-and-Finetune)
# ==========================================
class SimpleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        if not root_dir or not os.path.exists(root_dir): return
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        c2i = {c: i for i, c in enumerate(classes)}
        for cls in classes:
            td = os.path.join(root_dir, cls)
            for f in os.listdir(td):
                if f.lower().endswith(('.jpg', '.png')):
                    self.samples.append((os.path.join(td, f), c2i[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224))
        if self.transform: img = self.transform(img)
        return img, label


def train_and_save_expert(modality, tag, data_dir):
    print(f"\n[Train] Fine-tuning {modality}...")
    save_path = os.path.join(WEIGHT_DIR, f"{modality}_finetuned.pth")

    model = timm.create_model(tag, pretrained=True, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    # Freeze-and-Finetune Strategy
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specific head layers
    for name, module in model.named_modules():
        if any(x in name for x in ['head', 'fc', 'classifier']):
            for param in module.parameters():
                param.requires_grad = True

    ds = SimpleDataset(data_dir, transform=train_tfm)
    if len(ds) == 0: return None

    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.train()
    for ep in range(5):
        for imgs, lbls in tqdm(loader, desc=f"Epoch {ep + 1}", leave=False):
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                loss = criterion(model(imgs.to(DEVICE)), lbls.to(DEVICE))
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), save_path)
    return model


def load_finetuned_expert(modality, tag):
    model = timm.create_model(tag, pretrained=False, num_classes=NUM_CLASSES)
    save_path = os.path.join(WEIGHT_DIR, f"{modality}_finetuned.pth")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    else:
        print(f"Warning: {save_path} missing!")
    model.to(DEVICE).eval()
    return model


# ==========================================
# 4. FP16 Inference Logic
# ==========================================
def predict_with_entropy(model, path):
    if not model or not path or not os.path.exists(path):
        return np.ones(NUM_CLASSES) / NUM_CLASSES, 1.1
    try:
        img = Image.open(path).convert('RGB')
    except:
        return np.ones(NUM_CLASSES) / NUM_CLASSES, 1.1

    probs_sum = torch.zeros(NUM_CLASSES).to(DEVICE)
    with torch.no_grad(), torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu',
                                         dtype=torch.float16):
        for t in tta_tfms:
            out = model(t(img).unsqueeze(0).to(DEVICE))
            probs_sum += F.softmax(out, dim=1)[0]

    prob_avg = (probs_sum / len(tta_tfms)).cpu().float().numpy()
    entropy = scipy.stats.entropy(prob_avg)
    return prob_avg, entropy


def predict_standard(model, path):
    if not model or not path or not os.path.exists(path):
        return np.ones(NUM_CLASSES) / NUM_CLASSES
    try:
        img = Image.open(path).convert('RGB')
    except:
        return np.ones(NUM_CLASSES) / NUM_CLASSES

    with torch.no_grad(), torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu',
                                         dtype=torch.float16):
        out = model(standard_tfm(img).unsqueeze(0).to(DEVICE))
        prob = F.softmax(out, dim=1)[0].cpu().float().numpy()
    return prob


# ==========================================
# 5. UAMF-Net Main Pipeline
# ==========================================
def run_uamf_net():
    # 1. Model Preparation
    tags = {"Pathology": "convnext_base.fb_in1k", "CT": "maxvit_tiny_tf_512.in1k", "Gastro": "fastvit_ma36.apple_in1k"}
    if TRAIN_MODE:
        m_path = train_and_save_expert("Pathology", tags["Pathology"], TRAIN_DIRS["Pathology"])
        m_ct = train_and_save_expert("CT", tags["CT"], TRAIN_DIRS["CT"])
        m_gas = train_and_save_expert("Gastro", tags["Gastro"], TRAIN_DIRS["Gastro"])
    else:
        m_path = load_finetuned_expert("Pathology", tags["Pathology"])
        m_ct = load_finetuned_expert("CT", tags["CT"])
        m_gas = load_finetuned_expert("Gastro", tags["Gastro"])

    # 2. Inference on TEST SET only
    print("\nRunning Inference on Test Set...")
    ct_index = build_file_index(TEST_DIRS["CT"])
    gas_index = build_file_index(TEST_DIRS["Gastro"])

    results, stats = [], {"Total": 0, "Base_Correct": 0, "Final_Correct": 0, "Rescue": 0}
    scan_root = TEST_DIRS["Pathology"]

    for class_folder in os.listdir(scan_root):
        class_path = os.path.join(scan_root, class_folder)
        if not os.path.isdir(class_path): continue
        try:
            gt = int(class_folder)
        except:
            continue

        for fname in tqdm(os.listdir(class_path), desc=f"Class {gt}"):
            if not fname.lower().endswith(('.jpg', '.png')): continue
            p_path = os.path.join(class_path, fname)
            stats["Total"] += 1

            # Path Branch + Entropy Gating
            prob_p, entropy_p = predict_with_entropy(m_path, p_path)
            pred_base = np.argmax(prob_p)
            conf_p = np.max(prob_p)
            if pred_base == gt: stats["Base_Correct"] += 1

            final_pred, status_log = pred_base, "Locked"

            if entropy_p >= THRESHOLD_ENTROPY:
                pid = re.findall(r'^(\d+)', fname)[0] if re.findall(r'^(\d+)', fname) else None
                c_path, g_path = ct_index.get(pid), gas_index.get(pid)

                prob_c = predict_standard(m_ct, c_path)
                prob_g = predict_standard(m_gas, g_path)

                # Sniper Protocol
                conf_c, conf_g = np.max(prob_c), np.max(prob_g)
                macro_pred, macro_conf, macro_src = (np.argmax(prob_g), conf_g, "Gastro") if conf_g > conf_c else (
                    np.argmax(prob_c), conf_c, "CT")

                margin = macro_conf - conf_p
                if macro_pred == 1 and pred_base != 1:
                    if macro_conf > THRESHOLD_RECALL and margin > MARGIN_REQUIREMENT:
                        final_pred, status_log = 1, f"Rescue_by_{macro_src}"
                elif pred_base == 1 and macro_pred == 2:
                    if macro_conf > THRESHOLD_PRECIS:
                        final_pred, status_log = 2, f"Correct_by_{macro_src}"

            if final_pred == gt: stats["Final_Correct"] += 1
            if final_pred == gt and pred_base != gt: stats["Rescue"] += 1
            if pred_base != final_pred or (final_pred != gt and entropy_p > 0.4):
                results.append(
                    {"File": fname, "GT": gt, "Base": pred_base, "Final": final_pred, "Entropy": f"{entropy_p:.3f}",
                     "Status": status_log})

    acc_base, acc_final = stats["Base_Correct"] / stats["Total"], stats["Final_Correct"] / stats["Total"]
    print(f"\nFinal Report | Baseline: {acc_base:.2%} | UAMF-Net: {acc_final:.2%} | Rescued: {stats['Rescue']}")

    if results: pd.DataFrame(results).to_csv("UAMF_Net_Results.csv", index=False)


if __name__ == "__main__":
    run_uamf_net()