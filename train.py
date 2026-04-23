import os
import json
import numpy as np
import matplotlib.pyplot as plt

from data import (
    load_full_dataset,
    make_split_indices,
    build_dataloaders_from_split,
    save_split_and_scaler,
)
from model import MLPClassifier
from optimizer import SGD
from loss import (
    SoftmaxCrossEntropyLoss,
    compute_l2_loss,
    add_l2_gradients,
    compute_accuracy,
)


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path)


def evaluate(model, data_loader, criterion, weight_decay=0.0):
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for X_batch, y_batch in data_loader:
        logits = model.forward(X_batch)

        data_loss = criterion.forward(logits, y_batch)
        reg_loss = compute_l2_loss(model, weight_decay)
        loss = data_loss + reg_loss

        acc = compute_accuracy(logits, y_batch)

        batch_size = X_batch.shape[0]
        total_loss += loss * batch_size
        total_acc += acc * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    return float(avg_loss), float(avg_acc)


def train_one_epoch(model, train_loader, criterion, optimizer, weight_decay=0.0):
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for X_batch, y_batch in train_loader:
        logits = model.forward(X_batch)

        data_loss = criterion.forward(logits, y_batch)
        reg_loss = compute_l2_loss(model, weight_decay)
        loss = data_loss + reg_loss

        optimizer.zero_grad()
        dlogits = criterion.backward()
        model.backward(dlogits)
        add_l2_gradients(model, weight_decay)
        optimizer.step()

        acc = compute_accuracy(logits, y_batch)

        batch_size = X_batch.shape[0]
        total_loss += loss * batch_size
        total_acc += acc * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    return float(avg_loss), float(avg_acc)


def save_history(history, save_dir):
    ensure_dir(save_dir)

    with open(os.path.join(save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    np.savez(
        os.path.join(save_dir, "history.npz"),
        train_loss=np.array(history["train_loss"], dtype=np.float32),
        train_acc=np.array(history["train_acc"], dtype=np.float32),
        val_loss=np.array(history["val_loss"], dtype=np.float32),
        val_acc=np.array(history["val_acc"], dtype=np.float32),
        lr=np.array(history["lr"], dtype=np.float32),
    )


def plot_history(history, save_dir):
    ensure_dir(save_dir)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Val Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train / Val Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["lr"], label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lr_curve.png"), dpi=200)
    plt.close()


def train(
    root_dir="./EuroSAT_RGB",
    image_size=(64, 64),
    batch_size=64,
    hidden_dim=128,
    activation="relu",
    lr=1e-2,
    lr_decay=0.95,
    min_lr=1e-5,
    weight_decay=1e-4,
    num_epochs=20,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    standardize=True,
    random_seed=42,
    save_dir="./outputs",
    best_model_name="best_model.npz",
):
    np.random.seed(random_seed)
    ensure_dir(save_dir)

    # 1. 读取完整数据
    X, y, dataset = load_full_dataset(
        root_dir=root_dir,
        image_size=image_size,
        normalize=True,
        flatten=True,
    )

    # 2. 只生成一次划分
    split_indices = make_split_indices(
        n=len(X),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
    )

    # 3. 用这份划分构建 dataloader，并由 train set 拟合 scaler
    train_loader, val_loader, test_loader, scaler = build_dataloaders_from_split(
        X=X,
        y=y,
        split_indices=split_indices,
        batch_size=batch_size,
        standardize=standardize,
    )

    # 4. 保存 split 和 scaler，供 eval 直接读取
    save_split_and_scaler(
        save_dir=save_dir,
        split_indices=split_indices,
        scaler=scaler,
        dataset=dataset,
        root_dir=root_dir,
        image_size=image_size,
        standardize=standardize,
    )

    input_dim = X.shape[1]
    num_classes = len(dataset.class_names)

    print("=" * 60)
    print("Data loaded successfully")
    print(f"Input dim   : {input_dim}")
    print(f"Num classes : {num_classes}")
    print(f"Class names : {dataset.class_names}")
    print("=" * 60)

    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        activation=activation,
    )
    criterion = SoftmaxCrossEntropyLoss()
    optimizer = SGD(
        model=model,
        lr=lr,
        lr_decay=lr_decay,
        min_lr=min_lr,
    )

    model.summary()

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = -1.0
    best_model_path = os.path.join(save_dir, best_model_name)

    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.get_lr()

        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            weight_decay=weight_decay,
        )

        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            weight_decay=weight_decay,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(best_model_path)
            print(f"[Epoch {epoch:03d}] Best model saved to: {best_model_path}")

        print(
            f"Epoch [{epoch:03d}/{num_epochs:03d}] "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        optimizer.decay_lr()

    save_history(history, save_dir)
    plot_history(history, save_dir)

    model.load_weights(best_model_path)
    test_loss, test_acc = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        weight_decay=weight_decay,
    )

    print("=" * 60)
    print("Training finished")
    print(f"Best val acc : {best_val_acc:.4f}")
    print(f"Test loss    : {test_loss:.4f}")
    print(f"Test acc     : {test_acc:.4f}")
    print("=" * 60)

    config = {
        "root_dir": root_dir,
        "image_size": list(image_size),
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "activation": activation,
        "lr": lr,
        "lr_decay": lr_decay,
        "min_lr": min_lr,
        "weight_decay": weight_decay,
        "num_epochs": num_epochs,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "standardize": standardize,
        "random_seed": random_seed,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }

    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return model, history


if __name__ == "__main__":
    train(
        root_dir="./EuroSAT_RGB",
        image_size=(64, 64),
        batch_size=64,
        hidden_dim=128,
        activation="relu",
        lr=1e-2,
        lr_decay=0.95,
        min_lr=1e-5,
        weight_decay=1e-4,
        num_epochs=10,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        standardize=True,
        random_seed=42,
        save_dir="./outputs/128_4",
        best_model_name="best_model.npz",
    )