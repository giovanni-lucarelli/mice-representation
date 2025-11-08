#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render training histories with unified axes across runs")
    p.add_argument("--experiments", type=str, nargs="*", default=[], help="Experiment subdirs under checkpoints (default: all)")
    p.add_argument("--max-loss", type=float, default=None, help="Fixed Y max for loss (if omitted, computed globally)")
    p.add_argument("--max-acc", type=float, default=None, help="Fixed Y max for accuracy (if omitted, computed globally)")
    p.add_argument("--x-max", type=int, default=None, help="Fixed X max (epochs); default None leaves it free")
    p.add_argument("--latest-only", action="store_true", help="Only process latest run per experiment")
    return p.parse_args()


def _read_history_from_csv(csv_path: Path) -> Dict[str, List[float]] | None:
    try:
        import csv
        with open(csv_path, "r") as f:
            r = csv.DictReader(f)
            epochs, tl, vl, ta, va = [], [], [], [], []
            for row in r:
                epochs.append(int(row.get("epoch", "0") or 0))
                tl.append(float(row.get("train_loss", "nan") or "nan"))
                vl.append(float(row.get("val_loss", "nan") or "nan"))
                ta.append(float(row.get("train_acc", "nan") or "nan"))
                va.append(float(row.get("val_acc", "nan") or "nan"))
        return {"epochs": epochs, "train_losses": tl, "val_losses": vl, "train_acc": ta, "val_acc": va}
    except Exception:
        return None


def _read_history_from_ckpt(ckpt_path: Path) -> Dict[str, List[float]] | None:
    try:
        ckpt = torch.load(ckpt_path.as_posix(), map_location="cpu")
        hist = ckpt.get("history")
        if not isinstance(hist, dict):
            return None
        n = max(len(hist.get("train_losses", [])), len(hist.get("val_losses", [])), len(hist.get("train_accuracies", [])), len(hist.get("val_accuracies", [])))
        epochs = list(range(1, n + 1))
        return {
            "epochs": epochs,
            "train_losses": list(hist.get("train_losses", [])),
            "val_losses": list(hist.get("val_losses", [])),
            "train_acc": list(hist.get("train_accuracies", [])),
            "val_acc": list(hist.get("val_accuracies", [])),
        }
    except Exception:
        return None


def _gather_runs(exp_dir: Path, latest_only: bool) -> List[Path]:
    if not exp_dir.exists():
        return []
    subdirs = [d for d in exp_dir.iterdir() if d.is_dir()]
    # filter timestamp-like names (YYYYMMDD_HHMMSS or with suffix)
    def is_ts(name: str) -> bool:
        return len(name) >= 15 and name[8] == "_" and name[:8].isdigit()
    runs = [d for d in subdirs if is_ts(d.name)]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if latest_only and runs:
        return [runs[0]]
    return runs


def _compute_global_limits(
    histories: List[Dict[str, Any]],
    max_loss_opt: float | None,
    max_acc_opt: float | None,
) -> Tuple[float, float]:
    max_loss = 0.0
    max_acc = 0.0
    for h in histories:
        tl = [v for v in h.get("train_losses", []) if isinstance(v, (int, float))]
        vl = [v for v in h.get("val_losses", []) if isinstance(v, (int, float))]
        ta = [v for v in h.get("train_acc", []) if isinstance(v, (int, float))]
        va = [v for v in h.get("val_acc", []) if isinstance(v, (int, float))]
        if tl or vl:
            max_loss = max(max_loss, max((tl + vl)))
        if ta or va:
            max_acc = max(max_acc, max((ta + va)))
    if max_loss_opt is not None:
        max_loss = max_loss_opt
    if max_acc_opt is not None:
        max_acc = max_acc_opt
    if max_loss > 0:
        max_loss *= 1.05
    if max_acc > 0:
        max_acc *= 1.02
    return max_loss, max_acc


def _render_plot(h: Dict[str, Any], out_dir: Path, max_loss: float, max_acc: float, x_max: int | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = h.get("epochs", [])
    tl = h.get("train_losses", [])
    vl = h.get("val_losses", [])
    ta = h.get("train_acc", [])
    va = h.get("val_acc", [])

    plt.figure(figsize=(12, 4))
    # Losses
    plt.subplot(1, 2, 1)
    if tl:
        plt.plot(epochs, tl, 'b-', label='Training Loss')
    if vl:
        plt.plot(epochs, vl, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    if max_loss:
        plt.ylim(0, max_loss)
    if x_max:
        plt.xlim(0, x_max)

    # Accuracies
    plt.subplot(1, 2, 2)
    if ta:
        plt.plot(epochs, ta, 'b-', label='Train Acc')
    if va:
        plt.plot(epochs, va, 'r-', label='Val Acc')
    plt.title('Accuracy (Top-1)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(); plt.grid(True)
    if max_acc:
        plt.ylim(0, max_acc)
    if x_max:
        plt.xlim(0, x_max)

    plt.tight_layout()
    plt.savefig((out_dir / 'training_history_scaled.png').as_posix(), dpi=300, bbox_inches='tight')
    plt.close()


def main() -> None:
    args = parse_args()
    ROOT = Path(__file__).resolve().parent.parent
    checkpoints = ROOT / "checkpoints"

    exp_names = args.experiments
    if not exp_names:
        # all experiment subdirs
        exp_names = [d.name for d in checkpoints.iterdir() if d.is_dir()]

    all_histories: List[Dict[str, Any]] = []
    run_meta: List[Tuple[str, Path, Dict[str, Any]]] = []  # (experiment, run_dir, history)

    for exp in exp_names:
        exp_dir = checkpoints / exp
        for run_dir in _gather_runs(exp_dir, latest_only=args.latest_only):
            # Prefer CSV if present
            csv_path = run_dir / "artifacts" / "training_history.csv"
            h = _read_history_from_csv(csv_path)
            if h is None:
                # fallback to best_model
                ckpt = run_dir / "best_model.pth"
                if not ckpt.exists():
                    # pick any checkpoint
                    ckpts = sorted(run_dir.glob("*.pth"))
                    ckpt = ckpts[0] if ckpts else ckpt
                h = _read_history_from_ckpt(ckpt) if ckpt.exists() else None
            if h is None:
                print(f"Warning: no history found for {exp}/{run_dir.name}")
                continue
            all_histories.append(h)
            run_meta.append((exp, run_dir, h))

    if not run_meta:
        print("No runs found to process.")
        return

    max_loss, max_acc = _compute_global_limits(all_histories, args.max_loss, args.max_acc)
    print(f"Using max_loss={max_loss:.4f} max_acc={max_acc:.4f} x_max={args.x_max}")

    for exp, run_dir, h in run_meta:
        out_dir = run_dir / "artifacts"
        _render_plot(h, out_dir, max_loss=max_loss, max_acc=max_acc, x_max=args.x_max)
        print(f"Wrote scaled plot for {exp}/{run_dir.name} -> {out_dir / 'training_history_scaled.png'}")


if __name__ == "__main__":
    main()


