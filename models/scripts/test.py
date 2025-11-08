#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    load_and_resolve_configs,
    ensure_dirs,
    save_resolved_config,
    seed_everything,
    select_device,
)

from src.pipeline.mouse_transforms import mouse_transform
from src.datasets.DataManager import DataManager
from src.model.supervised_trainer import SupervisedTrainer
from src.model.ir_trainer import InstanceDiscriminationTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test model with YAML configs")
    p.add_argument("--project-config", type=str, default="configs/project.yaml")
    p.add_argument("--config", type=str, required=True, help="Experiment YAML file")
    p.add_argument("--checkpoint", type=str, default="best_model.pth", help="Checkpoint to load (name or absolute path)")
    p.add_argument(
        "--set",
        type=str,
        nargs="*",
        default=[],
        help="Override keys, e.g. device.device=cuda",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    resolved, flat = load_and_resolve_configs(
        project_yaml=args.project_config,
        experiment_yaml=args.config,
        overrides=args.__dict__.get("set", []),
    )

    # Ensure run directory structure (timestamped) even for testing
    dirs = ensure_dirs(resolved)
    flat["run_dir"] = dirs["run_dir"].as_posix()
    save_resolved_config(flat, dirs["run_dir"] / "resolved_config.yaml")

    # determinism and device
    seed_everything(resolved.experiment.device.seed)
    device_str = select_device(resolved.experiment.device)
    device = torch.device(device_str)

    # Data manager (evaluation transforms only)
    data_cfg = resolved.experiment.data
    eval_transform = mouse_transform(
        img_size=224,
        blur_sig=1.76,
        noise_std=0.25,
        to_gray=resolved.experiment.diet.grayscale,
        apply_blur=resolved.experiment.diet.blur,
        apply_noise=resolved.experiment.diet.noise,
        train=False,
    )
    dm = DataManager(
        data_path=str((resolved.root / Path(data_cfg.data_path)).resolve())
        if not Path(os.path.expanduser(str(data_cfg.data_path))).is_absolute()
        else str(Path(os.path.expanduser(str(data_cfg.data_path))).resolve()),
        batch_size=int(data_cfg.batch_size),
        num_workers=int(data_cfg.num_workers),
        train_split=float(data_cfg.train_split),
        val_split=float(data_cfg.val_split),
        split_seed=int(data_cfg.split_seed),
        use_cuda=resolved.experiment.device.use_cuda,
        persistent_workers=bool(data_cfg.persistent_workers),
        prefetch_factor=int(data_cfg.prefetch_factor),
        eval_transform=eval_transform,
    )
    dm.setup()

    # Model
    train_cfg = resolved.experiment.train

    opt_params = dict(getattr(train_cfg.optimizer, "params", {}) or {})
    scheduler_params = dict(getattr(train_cfg.scheduler, "params", {}) or {})
    loss_params = dict(getattr(train_cfg.loss, "params", {}) or {})

    # Pick trainer based on self_supervised flag
    self_supervised = bool(train_cfg.self_supervised)
    TrainerClass = InstanceDiscriminationTrainer if self_supervised else SupervisedTrainer

    model = TrainerClass(
        data_manager=dm,
        num_epochs=int(train_cfg.num_epochs),
        dropout_rate=float(train_cfg.dropout_rate),
        patience=int(train_cfg.early_stopping_patience),
        log_file=dirs["log_file"],
        checkpoint_dir=dirs["checkpoint_dir"],
        artifacts_dir=dirs["artifacts_dir"],
        use_cuda=resolved.experiment.device.use_cuda,
        save_every_n=int(train_cfg.save_every_n),
        optimizer=train_cfg.optimizer.name,
        optimizer_params=opt_params,
        scheduler=train_cfg.scheduler.name,
        scheduler_params=scheduler_params,
        loss=train_cfg.loss.name,
        loss_params=loss_params,
        autocast=bool(train_cfg.autocast),
    )

    # Load checkpoint
    ckpt_arg = args.checkpoint
    ckpt_path = ckpt_arg
    if not os.path.isabs(ckpt_arg):
        # Prefer current run dir
        candidate = dirs["checkpoint_dir"] / ckpt_arg
        if candidate.exists():
            ckpt_path = candidate.as_posix()
        else:
            # Fallback: search latest previous run under the same experiment subdir
            base_dir = dirs["checkpoint_dir"].parent
            try:
                subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
                # sort by mtime desc (latest first)
                subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                found = None
                for d in subdirs:
                    cand = d / ckpt_arg
                    if cand.exists():
                        found = cand
                        break
                if found is not None:
                    ckpt_path = found.as_posix()
                else:
                    # final attempt: if a pre-timestamp layout exists (file directly under base_dir)
                    legacy = base_dir / ckpt_arg
                    ckpt_path = legacy.as_posix()
            except Exception:
                ckpt_path = candidate.as_posix()
    model.load_model(ckpt_path)

    # For self-supervised evaluation, warm up memory bank to compute NN accuracy
    if self_supervised:
        try:
            warmup_epochs = int(getattr(train_cfg.loss, "params", {}).get("warmup_epochs", 1))
        except Exception:
            warmup_epochs = 1
        try:
            model._warmup_memory_bank(warmup_epochs)
        except Exception:
            pass

    # Evaluate on test set
    test_loss, test_acc = model.test()
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")


if __name__ == "__main__":
    main()


