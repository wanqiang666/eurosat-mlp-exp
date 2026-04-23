import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt

from model import MLPClassifier


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path)


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-8)


def load_training_config(save_dir: str):
    config_path = os.path.join(save_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def build_model_from_config(config):
    image_size = tuple(config["image_size"])
    hidden_dim = config["hidden_dim"]
    activation = config["activation"]

    input_dim = image_size[0] * image_size[1] * 3
    num_classes = 10  # EuroSAT 固定 10 类

    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        activation=activation,
    )
    return model


def get_first_layer_weights(model) -> np.ndarray:
    linear_layers = model.get_linear_layers()
    if len(linear_layers) == 0:
        raise ValueError("No Linear layers found in model.")
    W1 = linear_layers[0].W  # shape: (input_dim, hidden_dim)
    return W1


def rank_hidden_units(W1: np.ndarray, method: str = "l2"):
    """
    给隐藏单元排序，便于挑最有代表性的权重图。
    method:
        - "l2": 按每列权重的 L2 范数排序
        - "abs_mean": 按每列绝对值平均排序
    """
    if method == "l2":
        scores = np.linalg.norm(W1, axis=0)
    elif method == "abs_mean":
        scores = np.mean(np.abs(W1), axis=0)
    else:
        raise ValueError(f"Unsupported ranking method: {method}")

    ranked_indices = np.argsort(scores)[::-1]
    return ranked_indices, scores


def reshape_weight_to_image(weight_vector: np.ndarray, image_size):
    """
    将单个隐藏单元的权重向量恢复为图像尺寸
    """
    h, w = image_size
    expected_dim = h * w * 3
    if weight_vector.shape[0] != expected_dim:
        raise ValueError(
            f"Weight vector dim {weight_vector.shape[0]} does not match "
            f"image_size {image_size} with 3 channels."
        )

    weight_img = weight_vector.reshape(h, w, 3)
    return weight_img


def save_single_weight_image(weight_img: np.ndarray, save_path: str, title: str = ""):
    """
    保存单张 RGB 权重图（做 min-max 归一化后显示）
    """
    vis_img = minmax_normalize(weight_img)

    plt.figure(figsize=(4, 4))
    plt.imshow(vis_img)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_single_weight_heatmap(weight_img: np.ndarray, save_path: str, title: str = ""):
    """
    保存灰度热力图版本：
    将 RGB 权重图按通道平均后可视化，便于观察整体空间模式
    """
    gray = np.mean(weight_img, axis=2)

    plt.figure(figsize=(4, 4))
    plt.imshow(gray, cmap="coolwarm")
    plt.colorbar(fraction=0.046, pad=0.04)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_weight_grid(weight_images, unit_indices, scores, save_path: str, cols: int = 4):
    """
    保存若干 hidden units 的权重图总览
    """
    n = len(weight_images)
    rows = math.ceil(n / cols)

    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, (img, unit_idx) in enumerate(zip(weight_images, unit_indices)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(minmax_normalize(img))
        plt.title(f"Unit {unit_idx}\nscore={scores[unit_idx]:.3f}")
        plt.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_gray_heatmap_grid(weight_images, unit_indices, scores, save_path: str, cols: int = 4):
    """
    保存灰度热力图总览
    """
    n = len(weight_images)
    rows = math.ceil(n / cols)

    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, (img, unit_idx) in enumerate(zip(weight_images, unit_indices)):
        gray = np.mean(img, axis=2)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(gray, cmap="coolwarm")
        plt.title(f"Unit {unit_idx}\nscore={scores[unit_idx]:.3f}")
        plt.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def visualize_first_layer_weights(
    save_dir: str = "./outputs",
    model_name: str = "best_model.npz",
    out_dir: str = "./weight_vis",
    top_k: int = 12,
    ranking_method: str = "l2",
):
    ensure_dir(out_dir)

    # 1. 读配置并构建模型
    config = load_training_config(save_dir)
    image_size = tuple(config["image_size"])
    model_path = os.path.join(save_dir, model_name)

    model = build_model_from_config(config)
    model.load_weights(model_path)

    # 2. 提取第一层权重
    W1 = get_first_layer_weights(model)  # (input_dim, hidden_dim)
    hidden_dim = W1.shape[1]

    print("=" * 60)
    print("Visualizing first-layer weights")
    print(f"Model path   : {model_path}")
    print(f"W1 shape     : {W1.shape}")
    print(f"Image size   : {image_size}")
    print(f"Hidden dim   : {hidden_dim}")
    print("=" * 60)

    # 3. 排序选择最有代表性的 hidden units
    ranked_indices, scores = rank_hidden_units(W1, method=ranking_method)
    selected_indices = ranked_indices[:top_k]

    # 4. 保存单张图
    selected_weight_images = []
    summary = []

    for rank_id, unit_idx in enumerate(selected_indices, start=1):
        weight_vector = W1[:, unit_idx]
        weight_img = reshape_weight_to_image(weight_vector, image_size)
        selected_weight_images.append(weight_img)

        rgb_path = os.path.join(out_dir, f"unit_{unit_idx:03d}_rgb.png")
        gray_path = os.path.join(out_dir, f"unit_{unit_idx:03d}_heatmap.png")

        title = f"Unit {unit_idx} | score={scores[unit_idx]:.4f}"
        save_single_weight_image(weight_img, rgb_path, title=title)
        save_single_weight_heatmap(weight_img, gray_path, title=title)

        channel_mean = np.mean(weight_img, axis=(0, 1))
        summary.append(
            {
                "rank": rank_id,
                "unit_idx": int(unit_idx),
                "score": float(scores[unit_idx]),
                "channel_mean_r": float(channel_mean[0]),
                "channel_mean_g": float(channel_mean[1]),
                "channel_mean_b": float(channel_mean[2]),
                "rgb_path": rgb_path,
                "heatmap_path": gray_path,
            }
        )

    # 5. 保存总览图
    save_weight_grid(
        selected_weight_images,
        selected_indices,
        scores,
        save_path=os.path.join(out_dir, "top_units_rgb_grid.png"),
        cols=4,
    )

    save_gray_heatmap_grid(
        selected_weight_images,
        selected_indices,
        scores,
        save_path=os.path.join(out_dir, "top_units_heatmap_grid.png"),
        cols=4,
    )

    # 6. 保存 summary
    summary_path = os.path.join(out_dir, "weight_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved weight visualizations to: {out_dir}")
    print("Top selected units:", [int(i) for i in selected_indices])

    return {
        "W1_shape": W1.shape,
        "selected_indices": [int(i) for i in selected_indices],
        "summary_path": summary_path,
        "grid_rgb": os.path.join(out_dir, "top_units_rgb_grid.png"),
        "grid_heatmap": os.path.join(out_dir, "top_units_heatmap_grid.png"),
    }


if __name__ == "__main__":
    visualize_first_layer_weights(
        save_dir="./outputs/128_4",
        model_name="best_model.npz",
        out_dir="./weight_vis/128_4",
        top_k=4,
        ranking_method="l2",   # 可选: "l2" / "abs_mean"
    )