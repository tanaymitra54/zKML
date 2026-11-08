"""Model definitions, public data loaders, ONNX export for EZKL sweeps."""
from __future__ import annotations

import pickle
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent / "data"
CIFAR_TAR = DATA_DIR / "cifar-10-python.tar.gz"
CIFAR_DIR = DATA_DIR / "cifar-10-batches-py"


@dataclass
class WorkloadBundle:
    name: str
    model: nn.Module
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    input_shape: tuple[int, ...]
    task: str  # "binary" | "multiclass"


class MLPTabular(nn.Module):
    """30-64-32-1 MLP for breast-cancer tabular binary classification."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyCNN(nn.Module):
    """2 conv + 2 FC on 16x16 RGB CIFAR-10 subset (10 classes)."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def mlp_tabular(seed: int = 0) -> WorkloadBundle:
    ds = load_breast_cancer()
    x = StandardScaler().fit_transform(ds.data).astype(np.float32)
    y = ds.target.astype(np.float32)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    model = MLPTabular()
    return WorkloadBundle(
        name="mlp_tabular",
        model=model,
        x_train=torch.from_numpy(x_tr),
        y_train=torch.from_numpy(y_tr).unsqueeze(1),
        x_test=torch.from_numpy(x_te),
        y_test=torch.from_numpy(y_te).unsqueeze(1),
        input_shape=(30,),
        task="binary",
    )


def _ensure_cifar() -> Path:
    if CIFAR_DIR.is_dir():
        return CIFAR_DIR
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CIFAR_TAR.is_file():
        raise FileNotFoundError(f"missing CIFAR archive at {CIFAR_TAR}")
    with tarfile.open(CIFAR_TAR, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    return CIFAR_DIR


def _load_cifar_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as fh:
        raw = pickle.load(fh, encoding="bytes")
    data = raw[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(raw[b"labels"], dtype=np.int64)
    return data, labels


def _resize_16(imgs: np.ndarray) -> np.ndarray:
    """Nearest-neighbour downsample 32x32 -> 16x16 (no torchvision)."""
    return imgs[:, :, ::2, ::2]


def tiny_cnn(seed: int = 0, n_train: int = 2000, n_test: int = 500) -> WorkloadBundle:
    root = _ensure_cifar()
    xs, ys = [], []
    for i in range(1, 6):
        d, l = _load_cifar_batch(root / f"data_batch_{i}")
        xs.append(d)
        ys.append(l)
    x_all = _resize_16(np.concatenate(xs))
    y_all = np.concatenate(ys)
    x_te32, y_te = _load_cifar_batch(root / "test_batch")
    x_te = _resize_16(x_te32)

    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(len(x_all), size=n_train, replace=False)
    te_idx = rng.choice(len(x_te), size=n_test, replace=False)
    model = TinyCNN()
    return WorkloadBundle(
        name="tiny_cnn",
        model=model,
        x_train=torch.from_numpy(x_all[tr_idx]),
        y_train=torch.from_numpy(y_all[tr_idx]),
        x_test=torch.from_numpy(x_te[te_idx]),
        y_test=torch.from_numpy(y_te[te_idx]),
        input_shape=(3, 16, 16),
        task="multiclass",
    )


def train_bundle(
    bundle: WorkloadBundle,
    epochs: int = 8,
    lr: float = 1e-2,
    seed: int = 0,
    batch_size: int = 64,
) -> nn.Module:
    torch.manual_seed(seed)
    model = bundle.model
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss() if bundle.task == "binary" else nn.CrossEntropyLoss()
    x, y = bundle.x_train, bundle.y_train
    n = len(x)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = x[idx], y[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb if bundle.task == "binary" else yb.long())
            loss.backward()
            opt.step()
    model.eval()
    return model


def export_onnx(model: nn.Module, sample: torch.Tensor, path: Path, opset: int = 17) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        sample,
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamo=False,
    )
    return path


def sample_input(bundle: WorkloadBundle, batch: int = 1) -> torch.Tensor:
    n = min(batch, len(bundle.x_test))
    return bundle.x_test[:n]


# Scaling variants: enlarge MLP width / CNN channels to push constraint rows.
def mlp_scaled(hidden: int, seed: int = 0) -> WorkloadBundle:
    """Tabular MLP with configurable width (input still 30, output 1)."""
    base = mlp_tabular(seed=seed)

    class ScaledMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(30, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    return WorkloadBundle(
        name=f"mlp_h{hidden}",
        model=ScaledMLP(),
        x_train=base.x_train,
        y_train=base.y_train,
        x_test=base.x_test,
        y_test=base.y_test,
        input_shape=(30,),
        task="binary",
    )


def cnn_scaled(c1: int, c2: int, fc: int, seed: int = 0) -> WorkloadBundle:
    base = tiny_cnn(seed=seed)

    class ScaledCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, c1, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(c1, c2, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(c2 * 4 * 4, fc),
                nn.ReLU(),
                nn.Linear(fc, 10),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.features(x))

    return WorkloadBundle(
        name=f"cnn_c{c1}_{c2}_fc{fc}",
        model=ScaledCNN(),
        x_train=base.x_train,
        y_train=base.y_train,
        x_test=base.x_test,
        y_test=base.y_test,
        input_shape=(3, 16, 16),
        task="multiclass",
    )


# Registry used by ``python -m zk.runner --config <name>`` (one subprocess per
# measurement, so peak RSS is attributed to that config alone).
FACTORIES = {
    "mlp_tabular": mlp_tabular,
    "tiny_cnn": tiny_cnn,
    "mlp_h128": lambda: mlp_scaled(128),
    "mlp_h256": lambda: mlp_scaled(256),
    "mlp_h384": lambda: mlp_scaled(384),
    "mlp_h512": lambda: mlp_scaled(512),
    "mlp_h768": lambda: mlp_scaled(768),
    "mlp_h1024": lambda: mlp_scaled(1024),
    "cnn_c4_8_fc32": lambda: cnn_scaled(4, 8, 32),
    "cnn_c8_16_fc64": lambda: cnn_scaled(8, 16, 64),
    "cnn_c16_32_fc128": lambda: cnn_scaled(16, 32, 128),
    "cnn_c32_64_fc256": lambda: cnn_scaled(32, 64, 256),
    "cnn_c48_96_fc384": lambda: cnn_scaled(48, 96, 384),
    "cnn_c64_128_fc512": lambda: cnn_scaled(64, 128, 512),
}

