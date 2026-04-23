import os
import csv
import json
import time
import random
import itertools
from typing import Dict, List, Any

import numpy as np

from train import train


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path)


def load_run_result(save_dir: str) -> Dict[str, Any]:
    """
    读取某次训练生成的 config.json，提取关键指标
    """
    config_path = os.path.join(save_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in: {save_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    result = {
        "save_dir": save_dir,
        "best_val_acc": config.get("best_val_acc", None),
        "test_loss": config.get("test_loss", None),
        "test_acc": config.get("test_acc", None),
        "lr": config.get("lr", None),
        "lr_decay": config.get("lr_decay", None),
        "min_lr": config.get("min_lr", None),
        "weight_decay": config.get("weight_decay", None),
        "hidden_dim": config.get("hidden_dim", None),
        "activation": config.get("activation", None),
        "batch_size": config.get("batch_size", None),
        "num_epochs": config.get("num_epochs", None),
        "random_seed": config.get("random_seed", None),
        "image_size": config.get("image_size", None),
    }
    return result


def save_results_json(results: List[Dict[str, Any]], save_path: str) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def save_results_csv(results: List[Dict[str, Any]], save_path: str) -> None:
    if not results:
        return

    fieldnames = list(results[0].keys())
    with open(save_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_best_result(results: List[Dict[str, Any]], title: str = "Best Result") -> None:
    if not results:
        print(f"{title}: no results")
        return

    valid_results = [r for r in results if r.get("best_val_acc") is not None]
    if not valid_results:
        print(f"{title}: no valid results")
        return

    best_result = max(valid_results, key=lambda x: x["best_val_acc"])

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    for k, v in best_result.items():
        print(f"{k}: {v}")
    print("=" * 70)


def run_single_experiment(
    run_id: int,
    save_dir: str,
    root_dir: str,
    image_size=(64, 64),
    batch_size: int = 64,
    hidden_dim: int = 128,
    activation: str = "relu",
    lr: float = 1e-2,
    lr_decay: float = 0.95,
    min_lr: float = 1e-5,
    weight_decay: float = 1e-4,
    num_epochs: int = 10,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    standardize: bool = True,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    跑单次实验，并返回结果字典
    """
    print("\n" + "-" * 70)
    print(f"Run {run_id}")
    print(f"save_dir      : {save_dir}")
    print(f"lr            : {lr}")
    print(f"hidden_dim    : {hidden_dim}")
    print(f"weight_decay  : {weight_decay}")
    print(f"activation    : {activation}")
    print(f"batch_size    : {batch_size}")
    print(f"num_epochs    : {num_epochs}")
    print(f"random_seed   : {random_seed}")
    print("-" * 70)

    start_time = time.time()

    train(
        root_dir=root_dir,
        image_size=image_size,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        activation=activation,
        lr=lr,
        lr_decay=lr_decay,
        min_lr=min_lr,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        standardize=standardize,
        random_seed=random_seed,
        save_dir=save_dir,
        best_model_name="best_model.npz",
    )

    result = load_run_result(save_dir)
    result["run_id"] = run_id
    result["elapsed_seconds"] = round(time.time() - start_time, 2)

    print(
        f"Run {run_id} done | "
        f"best_val_acc={result['best_val_acc']:.4f}, "
        f"test_acc={result['test_acc']:.4f}, "
        f"time={result['elapsed_seconds']}s"
    )

    return result


def grid_search(
    root_dir: str = "./EuroSAT_RGB",
    image_size=(64, 64),
    batch_size: int = 64,
    num_epochs: int = 10,
    lr_decay: float = 0.95,
    min_lr: float = 1e-5,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    standardize: bool = True,
    random_seed: int = 42,
    search_save_dir: str = "./search_runs/grid_search",
):
    """
    网格搜索：
    对给定参数网格做笛卡尔积，逐组训练。
    """

    ensure_dir(search_save_dir)

    param_grid = {
        "lr": [1e-2, 1e-3],
        "hidden_dim": [128, 256],
        "weight_decay": [1e-4, 1e-3],
        "activation": ["relu", "tanh"],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(itertools.product(*values))

    print("=" * 70)
    print("Grid Search Start")
    print(f"Total runs: {len(all_combinations)}")
    print("=" * 70)

    results = []

    for run_id, combo in enumerate(all_combinations, start=1):
        params = dict(zip(keys, combo))
        run_save_dir = os.path.join(search_save_dir, f"run_{run_id:03d}")

        result = run_single_experiment(
            run_id=run_id,
            save_dir=run_save_dir,
            root_dir=root_dir,
            image_size=image_size,
            batch_size=batch_size,
            hidden_dim=params["hidden_dim"],
            activation=params["activation"],
            lr=params["lr"],
            lr_decay=lr_decay,
            min_lr=min_lr,
            weight_decay=params["weight_decay"],
            num_epochs=num_epochs,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            standardize=standardize,
            random_seed=random_seed,
        )
        results.append(result)

    save_results_json(results, os.path.join(search_save_dir, "grid_results.json"))
    save_results_csv(results, os.path.join(search_save_dir, "grid_results.csv"))
    print_best_result(results, title="Best Grid Search Result")

    return results


def sample_log_uniform(low_exp: float, high_exp: float) -> float:
    """
    例如 low_exp=-4, high_exp=-1
    返回 10^U(low_exp, high_exp)
    """
    return float(10 ** np.random.uniform(low_exp, high_exp))




if __name__ == "__main__":

    grid_search(
        root_dir="./EuroSAT_RGB",
        image_size=(64, 64),
        batch_size=64,
        num_epochs=5,
        random_seed=42,
        search_save_dir="./search_runs/grid_search",
    )

