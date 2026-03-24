#!/usr/bin/env python3
import argparse
import copy
import gc
import os
import time
import warnings
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as tq
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models.quantization import mobilenet_v2 as qmobilenet_v2
from torchvision.models.quantization import resnet18 as qresnet18

warnings.filterwarnings("ignore")


# ----------------------------
# Data
# ----------------------------
def make_loaders(data_dir: str, batch_size: int, num_workers: int):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


# ----------------------------
# Model builders
# ----------------------------
def build_resnet18_fp32():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m


def build_resnet18_quantizable():
    m = qresnet18(weights=None, quantize=False)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m


def build_mobilenetv2_fp32():
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 10)
    return m


def build_mobilenetv2_quantizable():
    m = qmobilenet_v2(weights=None, quantize=False)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 10)
    return m


def build_googlenet():
    m = models.googlenet(weights=None, aux_logits=False)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m


def build_vgg16_bn():
    m = models.vgg16_bn(weights=None)
    m.classifier[6] = nn.Linear(m.classifier[6].in_features, 10)
    return m


def get_fp32_model(name: str):
    if name == "resnet18":
        return build_resnet18_fp32()
    if name == "mobilenetv2":
        return build_mobilenetv2_fp32()
    if name == "googlenet":
        return build_googlenet()
    if name == "vgg16_bn":
        return build_vgg16_bn()
    raise ValueError(f"Unsupported model: {name}")


def get_quantizable_base_model(name: str):
    if name == "resnet18":
        return build_resnet18_quantizable()
    if name == "mobilenetv2":
        return build_mobilenetv2_quantizable()
    if name == "googlenet":
        return build_googlenet()
    if name == "vgg16_bn":
        return build_vgg16_bn()
    raise ValueError(f"Unsupported model: {name}")


def load_fp32_weights_into_quantizable(model_name: str, trained_fp32_model: nn.Module):
    qmodel = get_quantizable_base_model(model_name)
    qmodel.load_state_dict(trained_fp32_model.state_dict(), strict=False)
    return qmodel


# ----------------------------
# Quantization helpers
# ----------------------------
class QuantizedWrapper(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.quant = tq.QuantStub()
        self.model = backbone
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        if isinstance(x, tuple):
            x = x[0]
        x = self.dequant(x)
        return x

    def fuse_model(self, is_qat: bool = False):
        fuse_backbone(self.model, is_qat=is_qat)


def fuse_backbone(model: nn.Module, is_qat: bool = False):
    if isinstance(model, models.GoogLeNet):
        for mod in model.modules():
            if type(mod).__name__ == "BasicConv2d":
                try:
                    tq.fuse_modules(mod, ["conv", "bn"], inplace=True)
                except Exception:
                    pass
        return

    if isinstance(model, models.VGG):
        i = 0
        while i < len(model.features) - 2:
            if (
                isinstance(model.features[i], nn.Conv2d)
                and isinstance(model.features[i + 1], nn.BatchNorm2d)
                and isinstance(model.features[i + 2], nn.ReLU)
            ):
                try:
                    tq.fuse_modules(model.features, [str(i), str(i + 1), str(i + 2)], inplace=True)
                except Exception:
                    pass
                i += 3
            else:
                i += 1
        return


# ----------------------------
# Metrics
# ----------------------------
def get_example_batch(loader):
    x, y = next(iter(loader))
    return x[:1], y[:1]


def model_size_mb(model, path="tmp_model.pth") -> float:
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    os.remove(path)
    return size_mb


@torch.no_grad()
def evaluate(model, loader, device="cpu", is_quantized=False):
    model.eval()
    if not is_quantized:
        model.to(device)

    correct = 0
    total = 0

    for x, y in loader:
        if not is_quantized:
            x = x.to(device)
            y = y.to(device)

        out = model(x)
        if isinstance(out, tuple):
            out = out[0]

        preds = out.argmax(dim=1)
        if is_quantized:
            correct += (preds.cpu() == y).sum().item()
        else:
            correct += (preds == y).sum().item()
        total += y.size(0)

    return 100.0 * correct / total


@torch.no_grad()
def measure_latency(model, example_input, device="cpu", is_quantized=False, iters=30):
    model.eval()
    if not is_quantized:
        model.to(device)
        example_input = example_input.to(device)

    for _ in range(10):
        _ = model(example_input)

    if device == "cuda" and not is_quantized:
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        _ = model(example_input)

    if device == "cuda" and not is_quantized:
        torch.cuda.synchronize()

    return (time.time() - start) / iters


# ----------------------------
# Training
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    model.to(device)
    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total += y.size(0)
        correct += (out.argmax(1) == y).sum().item()

    return total_loss / total, 100.0 * correct / total


def train_model(model, train_loader, test_loader, device, epochs=3, lr=0.01, verbose=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te_acc = evaluate(model, test_loader, device=device, is_quantized=False)
        scheduler.step()
        if verbose:
            print(f"{model.__class__.__name__} epoch {epoch+1}/{epochs} "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.2f} test_acc={te_acc:.2f}")

        if te_acc > best_acc:
            best_acc = te_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


# ----------------------------
# Quantization flows
# ----------------------------
def run_dynamic_ptq(fp32_model):
    dyn = copy.deepcopy(fp32_model).cpu().eval()
    dyn = tq.quantize_dynamic(dyn, {nn.Linear}, dtype=torch.qint8)
    return dyn


def run_static_ptq(model_name: str, trained_fp32_model: nn.Module, calib_loader):
    torch.backends.quantized.engine = "fbgemm"

    if model_name in ["resnet18", "mobilenetv2"]:
        model = load_fp32_weights_into_quantizable(model_name, trained_fp32_model)
        model.eval().cpu()
        model.fuse_model()
        model.qconfig = tq.get_default_qconfig("fbgemm")
        prepared = tq.prepare(model, inplace=False)

        with torch.no_grad():
            for i, (x, _) in enumerate(calib_loader):
                _ = prepared(x)
                if i >= 20:
                    break

        quantized = tq.convert(prepared, inplace=False)
        return quantized

    base_model = copy.deepcopy(trained_fp32_model)
    wrapped = QuantizedWrapper(base_model).cpu().eval()
    wrapped.fuse_model(is_qat=False)
    wrapped.qconfig = tq.get_default_qconfig("fbgemm")
    prepared = tq.prepare(wrapped, inplace=False)

    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            _ = prepared(x)
            if i >= 20:
                break

    quantized = tq.convert(prepared, inplace=False)
    return quantized


def run_qat(model_name: str, trained_fp32_model: nn.Module, train_loader, device: str, qat_epochs=1, lr=1e-3):
    torch.backends.quantized.engine = "fbgemm"

    if model_name in ["resnet18", "mobilenetv2"]:
        qat_model = load_fp32_weights_into_quantizable(model_name, trained_fp32_model)
        qat_model.train()
        qat_model.fuse_model(is_qat=True)
        qat_model.qconfig = tq.get_default_qat_qconfig("fbgemm")
        qat_model = tq.prepare_qat(qat_model, inplace=False)
    else:
        base_model = copy.deepcopy(trained_fp32_model)
        qat_model = QuantizedWrapper(base_model)
        qat_model.train()
        qat_model.fuse_model(is_qat=True)
        qat_model.qconfig = tq.get_default_qat_qconfig("fbgemm")
        qat_model = tq.prepare_qat(qat_model, inplace=False)

    qat_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(qat_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    for _ in range(qat_epochs):
        qat_model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = qat_model(x)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    qat_model = qat_model.cpu().eval()
    quantized = tq.convert(qat_model, inplace=False)
    return quantized


# ----------------------------
# Plotting
# ----------------------------
def plot_results(df: pd.DataFrame):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    for tech in df["technique"].unique():
        sub = df[df["technique"] == tech]
        plt.plot(sub["model"], sub["accuracy"], marker="o", label=tech)
    plt.title("CNN Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    for tech in df["technique"].unique():
        sub = df[df["technique"] == tech]
        plt.plot(sub["model"], sub["latency_s"], marker="o", label=tech)
    plt.title("CNN Latency Comparison (CPU inference)")
    plt.ylabel("Latency (s)")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    for tech in df["technique"].unique():
        sub = df[df["technique"] == tech]
        plt.plot(sub["model"], sub["size_mb"], marker="o", label=tech)
    plt.title("CNN Model Size Comparison")
    plt.ylabel("Model Size (MB)")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ----------------------------
# Main
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="CNN quantization benchmark on CIFAR-10.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        choices=["resnet18", "mobilenetv2", "googlenet", "vgg16_bn"],
        help="One or more CNN models to benchmark.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="FP32 training epochs.")
    parser.add_argument("--qat-epochs", type=int, default=1, help="QAT fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--plots", action="store_true", help="Show accuracy/latency/size plots.")
    parser.add_argument("--save-csv", type=str, default="", help="Optional path to save the result table as CSV.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    train_loader, test_loader = make_loaders(args.data_dir, args.batch_size, args.num_workers)
    example_input, _ = get_example_batch(test_loader)

    rows: List[Dict] = []

    for model_name in args.models:
        fp32 = get_fp32_model(model_name)
        fp32 = train_model(fp32, train_loader, test_loader, device=device, epochs=args.epochs, lr=args.lr, verbose=args.verbose)

        fp32_acc = evaluate(fp32, test_loader, device=device, is_quantized=False)
        fp32_cpu = copy.deepcopy(fp32).cpu().eval()
        fp32_lat = measure_latency(fp32_cpu, example_input, device="cpu", is_quantized=False, iters=30)
        fp32_size = model_size_mb(fp32_cpu)
        rows.append({"model": model_name, "technique": "FP32", "accuracy": round(fp32_acc, 2), "latency_s": round(fp32_lat, 6), "size_mb": round(fp32_size, 3)})

        dyn = run_dynamic_ptq(fp32)
        dyn_acc = evaluate(dyn, test_loader, device="cpu", is_quantized=True)
        dyn_lat = measure_latency(dyn, example_input, device="cpu", is_quantized=True, iters=30)
        dyn_size = model_size_mb(dyn)
        rows.append({"model": model_name, "technique": "Dynamic PTQ", "accuracy": round(dyn_acc, 2), "latency_s": round(dyn_lat, 6), "size_mb": round(dyn_size, 3)})

        static_q = run_static_ptq(model_name, fp32, train_loader)
        static_acc = evaluate(static_q, test_loader, device="cpu", is_quantized=True)
        static_lat = measure_latency(static_q, example_input, device="cpu", is_quantized=True, iters=30)
        static_size = model_size_mb(static_q)
        rows.append({"model": model_name, "technique": "Static PTQ", "accuracy": round(static_acc, 2), "latency_s": round(static_lat, 6), "size_mb": round(static_size, 3)})

        qat_q = run_qat(model_name, fp32, train_loader, device=device, qat_epochs=args.qat_epochs, lr=1e-3)
        qat_acc = evaluate(qat_q, test_loader, device="cpu", is_quantized=True)
        qat_lat = measure_latency(qat_q, example_input, device="cpu", is_quantized=True, iters=30)
        qat_size = model_size_mb(qat_q)
        rows.append({"model": model_name, "technique": "QAT", "accuracy": round(qat_acc, 2), "latency_s": round(qat_lat, 6), "size_mb": round(qat_size, 3)})

        del fp32, fp32_cpu, dyn, static_q, qat_q
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values(["model", "technique"]).reset_index(drop=True)

    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        df.to_csv(args.save_csv, index=False)

    print(df.to_string(index=False))

    if args.plots:
        plot_results(df)


if __name__ == "__main__":
    main()
