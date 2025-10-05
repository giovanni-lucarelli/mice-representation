#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    load_and_resolve_configs,
    ensure_dirs,
    save_resolved_config,
    seed_everything,
    select_device,
)
from src.datasets.DataManager import DataManager
from src.pipeline.mouse_transforms import mouse_transform
from src.model.memory_bank import MemoryBank
from src.model.id_loss import InstanceDiscriminationLoss


def adapt_model_for_ir(model: torch.nn.Module) -> torch.nn.Module:
    try:
        if hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Sequential):
            modules = list(model.classifier.children())
            if len(modules) >= 1 and isinstance(modules[-1], torch.nn.Linear):
                model.classifier = torch.nn.Sequential(*modules[:-1])
    except Exception:
        pass
    return model


@torch.no_grad()
def warmup_memory_bank(model: torch.nn.Module,
                       proj: torch.nn.Module,
                       memory_bank: MemoryBank,
                       loader: DataLoader,
                       device: torch.device,
                       steps: int = 0) -> None:
    if steps <= 0:
        return
    model.eval()
    proj.eval()
    done = 0
    pbar = tqdm(total=steps, desc="Warmup memory bank", leave=True, dynamic_ncols=True)
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            indices, images, _ = batch
        elif isinstance(batch, dict):
            # fallback if transforms return dict
            indices = batch.get('idx')
            images = batch.get('imgs') or batch.get('image')
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        images = images.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True).long()
        outputs = model(images)
        emb = proj(outputs.float())
        # L2-normalize embeddings to match training
        emb = emb / torch.sqrt(torch.sum(emb ** 2, dim=1, keepdim=True)).clamp_min(1e-12)
        memory_bank.update(indices, emb)

        done += 1
        pbar.update(1)
        if done >= steps:
            break
    pbar.close()


@torch.no_grad()
def estimate_z_const(model: torch.nn.Module,
                     proj: torch.nn.Module,
                     memory_bank: MemoryBank,
                     loader: DataLoader,
                     tau: float,
                     num_batches: int = 50,
                     num_negatives: int = 8192,
                     device: torch.device = torch.device("cpu")) -> dict:
    model.eval()
    proj.eval()
    total_samples = 0
    running_sum_z = 0.0

    N = memory_bank.num_entries
    mem = memory_bank.get_memory_bank()

    bs_done = 0
    pbar = tqdm(total=num_batches, desc="Estimating Z_const (stable)", leave=True, dynamic_ncols=True)
    log_m = torch.log(torch.tensor(float(min(num_negatives, N)), device=device))

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            indices, images, _ = batch
        elif isinstance(batch, dict):
            indices = batch.get('idx')
            images = batch.get('imgs') or batch.get('image')
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        images = images.to(device, non_blocking=True)
        outputs = model(images)
        emb = proj(outputs.float())
        emb = emb / torch.sqrt(torch.sum(emb ** 2, dim=1, keepdim=True)).clamp_min(1e-12)

        B = emb.shape[0]
        m = min(num_negatives, N)
        neg_idx = torch.randint(0, N, (m,), device=device).long()
        mem_sample = mem.index_select(0, neg_idx)

        logits = emb @ mem_sample.t()

        # --- INIZIO MODIFICA STABILE ---
        # Calcola log(sum(exp(logits/tau))) in modo stabile
        log_sum_exp = torch.logsumexp(logits / float(tau), dim=1)

        # Calcola log(Z) per ogni campione: log_sum_exp - log(m)
        log_z_samples = log_sum_exp - log_m

        # Riconverti in spazio lineare e somma per il batch
        z_batch_sum = torch.exp(log_z_samples).sum().item()
        # --- FINE MODIFICA STABILE ---

        running_sum_z += z_batch_sum
        total_samples += B

        bs_done += 1
        pbar.update(1)
        if bs_done >= num_batches:
            break
    pbar.close()

    z_const = running_sum_z / max(1, total_samples)
    return {
        "z_const": float(z_const),
        "tau": float(tau),
        "num_batches": int(bs_done),
        "num_negatives": int(min(num_negatives, N)),
        "total_samples": int(total_samples),
        "memory_entries": int(N),
    }


@torch.no_grad()
def estimate_z_const_exact(model: torch.nn.Module,
                           proj: torch.nn.Module,
                           memory_bank: MemoryBank,
                           loader: DataLoader,
                           tau: float,
                           num_batches: int = 10,
                           mem_chunk: int = 8192,
                           device: torch.device = torch.device("cpu")) -> dict:
    """Exact Z via sum over full memory bank in chunks using two-pass log-sum-exp per row."""
    model.eval()
    proj.eval()
    N = memory_bank.num_entries
    mem = memory_bank.get_memory_bank()  # (N, D)

    total_samples = 0
    running_sum = 0.0

    # First pass: per-row max over all memory chunks
    row_max_global = None

    pbar1 = tqdm(total=num_batches, desc="Exact Z (pass 1: max)", leave=True, dynamic_ncols=True)
    bs_done = 0
    cached_emb = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            _, images, _ = batch
        elif isinstance(batch, dict):
            images = batch.get('imgs') or batch.get('image')
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")
        images = images.to(device, non_blocking=True)
        emb = proj(model(images).float())
        emb = emb / torch.sqrt(torch.sum(emb ** 2, dim=1, keepdim=True)).clamp_min(1e-12)
        cached_emb.append(emb.detach())

        # compute row-wise max over chunks
        row_max = torch.full((emb.size(0),), fill_value=-float('inf'), device=device)
        for start in range(0, N, mem_chunk):
            chunk = mem[start:start+mem_chunk]
            logits = (emb @ chunk.t()) / float(tau)
            row_max = torch.maximum(row_max, logits.max(dim=1).values)
        row_max_global = row_max if row_max_global is None else torch.maximum(row_max_global, row_max)

        bs_done += 1
        pbar1.update(1)
        if bs_done >= num_batches:
            break
    pbar1.close()

    # Second pass: sumexp with stabilization using global row max
    pbar2 = tqdm(total=bs_done, desc="Exact Z (pass 2: sumexp)", leave=True, dynamic_ncols=True)
    for emb in cached_emb:
        # recompute stabilized sumexp over all chunks
        sumexp = torch.zeros((emb.size(0),), device=device)
        for start in range(0, N, mem_chunk):
            chunk = mem[start:start+mem_chunk]
            logits = (emb @ chunk.t()) / float(tau)
            sumexp += torch.exp(logits - row_max_global.view(-1, 1)).sum(dim=1)
        # per-row exact sumexp
        z_rows = torch.exp(row_max_global) * sumexp
        running_sum += z_rows.sum().item()
        total_samples += emb.size(0)
        pbar2.update(1)
    pbar2.close()

    z_const = running_sum / max(1, total_samples)
    return {
        "z_const": float(z_const),
        "tau": float(tau),
        "num_batches": int(bs_done),
        "mem_chunk": int(mem_chunk),
        "total_samples": int(total_samples),
        "memory_entries": int(N),
        "exact": True,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Estimate Z_const for NCE on MiniImageNet (or your dataset)")
    p.add_argument("--project-config", type=str, default="configs/project.yaml")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--set", type=str, nargs="*", default=[], help="Overrides like train.loss.params.tau=0.07")
    p.add_argument("--num-batches", type=int, default=50)
    p.add_argument("--num-negatives", type=int, default=8192, help="Used in sampled mode")
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--exact", action="store_true", help="Use exact computation over full memory bank")
    p.add_argument("--mem-chunk", type=int, default=8192, help="Memory-bank chunk size for exact mode")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="artifacts/z_const_estimate.json")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to a model checkpoint to load weights from")
    args = p.parse_args()

    resolved, flat = load_and_resolve_configs(
        project_yaml=args.project_config,
        experiment_yaml=args.config,
        overrides=args.__dict__.get("set", []),
    )

    # dirs (reuse run dir structure)
    dirs = ensure_dirs(resolved)
    save_resolved_config(flat, dirs["run_dir"] / "resolved_config.yaml")

    seed_everything(int(args.seed))
    device_str = select_device(resolved.experiment.device)
    device = torch.device(device_str)

    # transforms: align to eval pipeline to reduce randomness
    eval_transform = mouse_transform(
        img_size=resolved.experiment.data.image_size,
        blur_sig=1.76,
        noise_std=0.25,
        to_gray=resolved.experiment.diet.grayscale,
        apply_blur=resolved.experiment.diet.blur,
        apply_noise=resolved.experiment.diet.noise,
        train=False,
        apply_motion=False,
        self_supervised=resolved.experiment.train.self_supervised,
    )

    # Resolve dataset path (expand ~ and handle relative vs absolute like train.py)
    cfg_data_path = flat["experiment"]["data"]["data_path"]
    cfg_path = Path(os.path.expanduser(str(cfg_data_path)))
    if not cfg_path.is_absolute():
        cfg_path = (resolved.root / Path(cfg_data_path))
    data_path_str = str(cfg_path.expanduser().resolve())

    dm = DataManager(
        data_path=data_path_str,
        batch_size=int(resolved.experiment.data.batch_size),
        num_workers=int(resolved.experiment.data.num_workers),
        train_split=float(resolved.experiment.data.train_split),
        val_split=float(resolved.experiment.data.val_split),
        split_seed=int(resolved.experiment.data.split_seed),
        use_cuda=resolved.experiment.device.use_cuda,
        train_transform=eval_transform,
        eval_transform=eval_transform,
        return_indices=True,
    )
    dm.setup()

    # model and loss params
    train_cfg = resolved.experiment.train
    loss_params = dict(getattr(train_cfg.loss, "params", {}) or {})
    model_output_dim = int(loss_params.get("model_output_dim", 4096))
    embedding_dim = int(loss_params.get("embedding_dim", 128))
    tau = float(loss_params.get("tau", 0.07))
    mode = str(loss_params.get("mode", "dynamic"))

    model = models.alexnet(weights=None, num_classes=dm.num_classes, dropout=float(train_cfg.dropout_rate))
    model = adapt_model_for_ir(model)

    # Memory bank
    memory_bank = MemoryBank(num_entries=len(dm.train_loader.dataset), embedding_dim=embedding_dim, device=device)

    # Projection head from the loss (initialized randomly for estimation)
    loss = InstanceDiscriminationLoss(memory_bank.get_memory_bank(), model_output_dim,
                                      m=int(loss_params.get("m", 4096)),
                                      gamma=float(loss_params.get("gamma", 0.5)),
                                      tau=tau,
                                      embedding_dim=embedding_dim,
                                      mode=mode)
    proj = loss._embedding_func

    if args.checkpoint:
        print(f"Loading weights from checkpoint: {args.checkpoint}")
        cpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(cpt['model_state_dict'])

        # Carica anche la testa di proiezione
        if 'embedding_func' in cpt:
            proj.load_state_dict(cpt['embedding_func'])
        elif 'loss_state_dict' in cpt and '_embedding_func.weight' in cpt['loss_state_dict']:
            proj_state_dict = {k.replace('_embedding_func.', ''): v for k, v in cpt['loss_state_dict'].items() if k.startswith('_embedding_func.')}
            proj.load_state_dict(proj_state_dict)

    model.to(device)
    model.eval()
    proj.to(device)
    proj.eval()

    # warmup memory a bit to better reflect embeddings distribution
    warmup_memory_bank(model, proj, memory_bank, dm.train_loader, device, steps=max(0, int(args.warmup_steps)))
    # ensure loss sees the up-to-date bank
    loss.update_memory_bank(memory_bank.get_memory_bank())

    # estimate
    if args.exact:
        est = estimate_z_const_exact(model, proj, memory_bank, dm.train_loader, tau,
                                     num_batches=max(1, int(args.num_batches)),
                                     mem_chunk=max(1, int(args.mem_chunk)),
                                     device=device)
    else:
        est = estimate_z_const(model, proj, memory_bank, dm.train_loader, tau,
                               num_batches=max(1, int(args.num_batches)),
                               num_negatives=max(1, int(args.num_negatives)),
                               device=device)

    # persist
    out_path = (Path(dirs["run_dir"]) / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(dirs["run_dir"]),
        **est,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()


