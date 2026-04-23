import os
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class EuroSATDataset:
    """
    读取 EuroSAT_RGB 目录，输出展平后的图像向量和标签。

    目录结构默认如下：
        EuroSAT_RGB/
            AnnualCrop/
            Forest/
            ...

    每个子文件夹名视为一个类别名。
    """

    def __init__(
        self,
        root_dir: str,
        image_size: Tuple[int, int] = (64, 64),
        normalize: bool = True,
        flatten: bool = True,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.root_dir = root_dir
        self.image_size = image_size
        self.normalize = normalize
        self.flatten = flatten

        if class_names is None:
            class_names = self._discover_classes(root_dir)
        self.class_names = list(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        self.samples = self._collect_samples()

    def _discover_classes(self, root_dir: str) -> List[str]:
        classes = []
        for entry in os.listdir(root_dir):
            full_path = os.path.join(root_dir, entry)
            if os.path.isdir(full_path):
                classes.append(entry)
        classes.sort()
        if not classes:
            raise ValueError(f"No class folders found in: {root_dir}")
        return classes

    def _collect_samples(self) -> List[Tuple[str, int]]:
        samples = []
        for class_name in self.class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                raise ValueError(f"Class folder not found: {class_dir}")

            label = self.class_to_idx[class_name]
            file_names = sorted(os.listdir(class_dir))  # 固定顺序，增强可复现性
            for file_name in file_names:
                ext = os.path.splitext(file_name)[1].lower()
                if ext in IMG_EXTENSIONS:
                    img_path = os.path.join(class_dir, file_name)
                    samples.append((img_path, label))

        if not samples:
            raise ValueError(f"No images found under: {self.root_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, img_path: str) -> np.ndarray:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.image_size)
        arr = np.asarray(img, dtype=np.float32)

        if self.normalize:
            arr = arr / 255.0

        if self.flatten:
            arr = arr.reshape(-1)

        return arr

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        img_path, label = self.samples[index]
        x = self._load_image(img_path)
        return x, label

    def get_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for i in range(len(self)):
            x, y = self[i]
            xs.append(x)
            ys.append(y)
        X = np.stack(xs, axis=0)
        y = np.array(ys, dtype=np.int64)
        return X, y


class StandardScaler:
    """
    按特征维度做标准化：
        x' = (x - mean) / std
    只在训练集 fit，再用于验证/测试集 transform。
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> None:
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before calling transform().")
        return (X - self.mean) / (self.std + self.eps)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def load_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean
        self.std = std


class DataLoader:
    """
    简易版 DataLoader。
    每次迭代返回一个 batch 的 (X_batch, y_batch)。
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> None:
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(X)

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_indices = indices[start_idx:end_idx]
            yield self.X[batch_indices], self.y[batch_indices]

    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size


def load_full_dataset(
    root_dir: str,
    image_size: Tuple[int, int] = (64, 64),
    normalize: bool = True,
    flatten: bool = True,
):
    dataset = EuroSATDataset(
        root_dir=root_dir,
        image_size=image_size,
        normalize=normalize,
        flatten=flatten,
    )
    X, y = dataset.get_all_data()
    return X, y, dataset


def make_split_indices(
    n: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, np.ndarray]:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    indices = np.arange(n)
    rng = np.random.default_rng(random_seed)
    rng.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    split_indices = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }
    return split_indices


def build_dataloaders_from_split(
    X: np.ndarray,
    y: np.ndarray,
    split_indices: Dict[str, np.ndarray],
    batch_size: int = 64,
    standardize: bool = True,
):
    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    train_loader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(X_test, y_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler


def build_dataloaders_from_saved_split(
    X: np.ndarray,
    y: np.ndarray,
    split_indices: Dict[str, np.ndarray],
    batch_size: int = 64,
    standardize: bool = True,
    scaler_stats: Optional[Dict[str, np.ndarray]] = None,
):
    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    scaler = None
    if standardize:
        scaler = StandardScaler()
        if scaler_stats is None:
            raise ValueError("scaler_stats must be provided when standardize=True")
        scaler.load_stats(scaler_stats["mean"], scaler_stats["std"])
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    train_loader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(X_test, y_test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler


def save_split_and_scaler(
    save_dir: str,
    split_indices: Dict[str, np.ndarray],
    scaler: Optional[StandardScaler],
    dataset,
    root_dir: str,
    image_size: Tuple[int, int],
    standardize: bool,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    np.savez(
        os.path.join(save_dir, "split.npz"),
        train_idx=split_indices["train"],
        val_idx=split_indices["val"],
        test_idx=split_indices["test"],
    )

    if scaler is not None and scaler.mean is not None and scaler.std is not None:
        np.savez(
            os.path.join(save_dir, "scaler.npz"),
            mean=scaler.mean.astype(np.float32),
            std=scaler.std.astype(np.float32),
        )

    metadata = {
        "root_dir": root_dir,
        "image_size": list(image_size),
        "standardize": standardize,
        "class_names": dataset.class_names,
        "samples": [[path, int(label)] for path, label in dataset.samples],
    }
    with open(os.path.join(save_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_split_and_scaler(save_dir: str):
    split_file = np.load(os.path.join(save_dir, "split.npz"))
    split_indices = {
        "train": split_file["train_idx"],
        "val": split_file["val_idx"],
        "test": split_file["test_idx"],
    }

    scaler_path = os.path.join(save_dir, "scaler.npz")
    scaler_stats = None
    if os.path.exists(scaler_path):
        scaler_file = np.load(scaler_path)
        scaler_stats = {
            "mean": scaler_file["mean"],
            "std": scaler_file["std"],
        }

    with open(os.path.join(save_dir, "dataset_meta.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return split_indices, scaler_stats, metadata


if __name__ == "__main__":
    root_dir = "./EuroSAT_RGB"

    X, y, dataset = load_full_dataset(root_dir=root_dir, image_size=(64, 64))
    split_indices = make_split_indices(len(X), random_seed=42)
    train_loader, val_loader, test_loader, scaler = build_dataloaders_from_split(
        X, y, split_indices, batch_size=64, standardize=True
    )

    print("Class names:", dataset.class_names)
    print("Input dim:", X.shape[1])
    print("Num classes:", len(dataset.class_names))

    X_batch, y_batch = next(iter(train_loader))
    print("Train batch X shape:", X_batch.shape)
    print("Train batch y shape:", y_batch.shape)