"""Train and evaluate a model: ``python -m eurosat.train --model cnn --data-root <EuroSATallBands>``.

Keeps the checkpoint with the best validation accuracy, then reports test accuracy, weighted F1,
macro one-vs-rest ROC-AUC (computed from predicted probabilities) and a confusion matrix.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from torch import nn

from .data import get_loaders
from .models import build, count_parameters


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def predict(model: nn.Module, loader, device: str):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def evaluate(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = probs.argmax(1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
        "roc_auc_macro_ovr": float(roc_auc_score(labels, probs, multi_class="ovr", average="macro")),
    }


def train(args) -> dict:
    set_seed(args.seed)
    torch.set_num_threads(args.threads) if args.threads else None
    loaders, label_map = get_loaders(args.data_root, args.cache_dir, args.batch_size, augment=not args.no_augment)
    class_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    kwargs = {"hidden_sizes": tuple(args.hidden)} if args.model == "mlp" else {}
    model = build(args.model, num_classes=len(class_names), **kwargs).to(args.device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val, best_state, history, patience = -1.0, None, [], 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        seen = correct = 0
        loss_sum = 0.0
        for x, y in loaders["train"]:
            x, y = x.to(args.device), y.to(args.device)
            optimiser.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimiser.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        scheduler.step()

        val_probs, val_labels = predict(model, loaders["validation"], args.device)
        val_acc = float((val_probs.argmax(1) == val_labels).mean())
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_acc": correct / seen, "val_acc": val_acc,
               "minutes": round((time.time() - start) / 60, 1)}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_acc > best_val:
            best_val, patience = val_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if args.patience and patience >= args.patience:
                print(f"early stopping after epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    test_probs, test_labels = predict(model, loaders["test"], args.device)
    metrics = evaluate(test_probs, test_labels)
    metrics.update({"model": args.model, "run": args.run_name, "parameters": count_parameters(model),
                    "best_val_acc": best_val, "epochs_run": len(history),
                    "train_minutes": round((time.time() - start) / 60, 1)})

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(
        classification_report(test_labels, test_probs.argmax(1), target_names=class_names, digits=3), encoding="utf-8")
    np.savetxt(out_dir / "confusion_matrix.csv", confusion_matrix(test_labels, test_probs.argmax(1)),
               fmt="%d", delimiter=",", header=",".join(class_names), comments="")
    if args.save_model:
        torch.save(best_state, out_dir / "model.pt")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a land-cover classifier on 13-band EuroSAT.")
    p.add_argument("--data-root", required=True, help="folder with train.csv, validation.csv, test.csv and class folders")
    p.add_argument("--cache-dir", default="data/cache", help="where the decoded uint16 arrays are stored")
    p.add_argument("--model", choices=["mlp", "cnn"], default="cnn")
    p.add_argument("--hidden", type=int, nargs="+", default=[2048, 1024], help="MLP hidden layer sizes")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=0, help="early-stopping patience in epochs (0 = off)")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--run-name", default=None)
    p.add_argument("--save-model", action="store_true")
    args = p.parse_args(argv)
    args.run_name = args.run_name or args.model
    return args


if __name__ == "__main__":
    train(parse_args())
