import os
import json
import numpy as np
import matplotlib.pyplot as plt

from data import (
    load_full_dataset,
    load_split_and_scaler,
    build_dataloaders_from_saved_split,
)
from model import MLPClassifier
from loss import (
    SoftmaxCrossEntropyLoss,
    compute_l2_loss,
    compute_accuracy,
)


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def evaluate_model(model, data_loader, criterion, weight_decay=0.0):
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    all_y_true = []
    all_y_pred = []

    for X_batch, y_batch in data_loader:
        logits = model.forward(X_batch)

        data_loss = criterion.forward(logits, y_batch)
        reg_loss = compute_l2_loss(model, weight_decay)
        loss = data_loss + reg_loss

        preds = np.argmax(logits, axis=1)
        acc = compute_accuracy(logits, y_batch)

        batch_size = X_batch.shape[0]
        total_loss += loss * batch_size
        total_acc += acc * batch_size
        total_samples += batch_size

        all_y_true.append(y_batch)
        all_y_pred.append(preds)

    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples

    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)

    return float(avg_loss), float(avg_acc), all_y_true, all_y_pred


def save_confusion_matrix_text(cm: np.ndarray, class_names, save_path: str) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Confusion Matrix\n")
        f.write("=" * 80 + "\n")
        f.write("Rows: Ground Truth, Columns: Prediction\n\n")

        f.write("Classes:\n")
        for i, name in enumerate(class_names):
            f.write(f"{i}: {name}\n")
        f.write("\n")

        header = " " * 15 + "".join([f"{i:>8}" for i in range(len(class_names))]) + "\n"
        f.write(header)

        for i, row in enumerate(cm):
            row_str = f"{class_names[i][:12]:>12} |" + "".join([f"{v:>8d}" for v in row]) + "\n"
            f.write(row_str)


def plot_confusion_matrix(cm: np.ndarray, class_names, save_path: str, normalize: bool = False) -> None:
    if normalize:
        cm_plot = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-12)
        title = "Normalized Confusion Matrix"
        value_format = ".2f"
    else:
        cm_plot = cm
        title = "Confusion Matrix"
        value_format = "d"

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_plot, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm_plot.max() / 2.0 if cm_plot.size > 0 else 0.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            text_value = format(cm_plot[i, j], value_format)
            plt.text(
                j,
                i,
                text_value,
                ha="center",
                va="center",
                color="white" if cm_plot[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def collect_misclassified_samples(model, data_loader):
    misclassified = []
    global_idx = 0

    for X_batch, y_batch in data_loader:
        logits = model.forward(X_batch)
        preds = np.argmax(logits, axis=1)

        for i in range(len(y_batch)):
            if preds[i] != y_batch[i]:
                misclassified.append(
                    {
                        "index": int(global_idx),
                        "true_label": int(y_batch[i]),
                        "pred_label": int(preds[i]),
                    }
                )
            global_idx += 1

    return misclassified


def run_evaluation(
    save_dir="./outputs",
    eval_save_dir="./eval_outputs",
    model_name="best_model.npz",
):
    ensure_dir(eval_save_dir)

    # 1. 读取训练配置
    with open(os.path.join(save_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    root_dir = config["root_dir"]
    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    hidden_dim = config["hidden_dim"]
    activation = config["activation"]
    weight_decay = config["weight_decay"]
    standardize = config["standardize"]

    # 2. 读取 split/scaler/meta
    split_indices, scaler_stats, metadata = load_split_and_scaler(save_dir)

    # 3. 重新加载完整数据，但不重新切分
    X, y, dataset = load_full_dataset(
        root_dir=root_dir,
        image_size=image_size,
        normalize=True,
        flatten=True,
    )

    # 4. 用保存好的 split/scaler 构建 dataloader
    train_loader, val_loader, test_loader, _ = build_dataloaders_from_saved_split(
        X=X,
        y=y,
        split_indices=split_indices,
        batch_size=batch_size,
        standardize=standardize,
        scaler_stats=scaler_stats,
    )

    input_dim = X.shape[1]
    num_classes = len(dataset.class_names)
    class_names = dataset.class_names

    # 5. 加载模型
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        activation=activation,
    )
    model_path = os.path.join(save_dir, model_name)
    model.load_weights(model_path)

    criterion = SoftmaxCrossEntropyLoss()

    # 6. 测试评估
    test_loss, test_acc, y_true, y_pred = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        weight_decay=weight_decay,
    )

    cm = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)

    np.save(os.path.join(eval_save_dir, "confusion_matrix.npy"), cm)
    save_confusion_matrix_text(
        cm,
        class_names,
        os.path.join(eval_save_dir, "confusion_matrix.txt"),
    )
    plot_confusion_matrix(
        cm,
        class_names,
        os.path.join(eval_save_dir, "confusion_matrix.png"),
        normalize=False,
    )
    plot_confusion_matrix(
        cm,
        class_names,
        os.path.join(eval_save_dir, "confusion_matrix_normalized.png"),
        normalize=True,
    )

    misclassified = collect_misclassified_samples(model, test_loader)
    with open(os.path.join(eval_save_dir, "misclassified_samples.json"), "w", encoding="utf-8") as f:
        json.dump(misclassified, f, indent=2)

    summary = {
        "model_path": model_path,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "num_test_samples": int(len(y_true)),
        "num_misclassified": int(len(misclassified)),
        "class_names": class_names,
    }

    with open(os.path.join(eval_save_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("Evaluation Finished")
    print(f"Model path         : {model_path}")
    print(f"Test loss          : {test_loss:.4f}")
    print(f"Test accuracy      : {test_acc:.4f}")
    print(f"Num test samples   : {len(y_true)}")
    print(f"Num misclassified  : {len(misclassified)}")
    print(f"Results saved to   : {eval_save_dir}")
    print("=" * 60)

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "confusion_matrix": cm,
        "misclassified": misclassified,
        "class_names": class_names,
    }


if __name__ == "__main__":
    run_evaluation(
        save_dir="./outputs/128_4",
        eval_save_dir="./eval_outputs/128_4",
        model_name="best_model.npz",
    )