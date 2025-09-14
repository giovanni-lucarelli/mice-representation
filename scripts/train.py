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
from src.model.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train model with YAML configs")
    p.add_argument("--project-config", type=str, default="configs/project.yaml")
    p.add_argument("--config", type=str, required=True, help="Experiment YAML file")
    p.add_argument(
        "--set",
        type=str,
        nargs="*",
        default=[],
        help="Override keys, e.g. train.num_epochs=100 train.optimizer.params.lr=1e-4",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    resolved, flat = load_and_resolve_configs(
        project_yaml=args.project_config,
        experiment_yaml=args.config,
        overrides=args.__dict__.get("set", []),
    )

    dirs = ensure_dirs(resolved)
    flat["run_dir"] = dirs["run_dir"].as_posix()
    # persist resolved config next to run
    save_resolved_config(flat, dirs["run_dir"] / "resolved_config.yaml")

    # determinism and device
    seed_everything(resolved.experiment.device.seed)
    device_str = select_device(resolved.experiment.device)
    device = torch.device(device_str)
    
    # Transformations
    train_transform = mouse_transform(
        img_size = 224,
        blur_sig = 1.76,
        noise_std = 0.25,
        to_gray = resolved.experiment.diet.grayscale,
        apply_blur = resolved.experiment.diet.blur,
        apply_noise = resolved.experiment.diet.noise,
        train = True, # enables augmentation
    )
    
    eval_transform = mouse_transform(
        img_size = 224,
        blur_sig = 1.76,
        noise_std = 0.25,
        to_gray = resolved.experiment.diet.grayscale,
        apply_blur = resolved.experiment.diet.blur,
        apply_noise = resolved.experiment.diet.noise,
        train = False,
    )

    data_cfg = resolved.experiment.data
    train_cfg = resolved.experiment.train
    loss_type = train_cfg.loss.name
    
    # Data manager
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
        train_transform=train_transform,
        eval_transform=eval_transform,
    )
    dm.setup()

    # Model
    # Extract optimizer/scheduler configs (normalized in config loader)
    opt_params = dict(getattr(train_cfg.optimizer, "params", {}) or {})
    scheduler_params = dict(getattr(train_cfg.scheduler, "params", {}) or {})
    loss_params = dict(getattr(train_cfg.loss, "params", {}) or {})

    print(f"Optimizer params: {opt_params}")
    print(f"Scheduler params: {scheduler_params}")
    print(f"Loss params: {loss_params}")

    model = Trainer(
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
    )

    history = model.train()
    model.plot_training_history()
    model.test()


if __name__ == "__main__":
    main()


