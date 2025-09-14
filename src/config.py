import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .model.utils import (
    _deep_merge_dicts,
    _set_by_dots,
    _parse_override,
    load_yaml,
    _dict_to_dataclass,
    coerce_config_scalars,
)

#? --------------- Project Config --------------- #

@dataclass
class PathsConfig:
    log_dir: str = "log"
    checkpoint_base_dir: str = "checkpoints"
    artifacts_dir: str = "artifacts"
    root: Optional[str] = None  # resolved at runtime


@dataclass
class DataSourcesConfig:
    mini_imagenet_path: str = "~/.cache/kagglehub/datasets/arjunashok33/miniimagenet/versions/1"


@dataclass
class AllenDataConfig:
    neuropixels_url: str = "https://mouse-vision-neuraldata.s3.amazonaws.com/mouse_neuropixels_visual_data_with_reliabilities.pkl"
    calcium_url: str = "https://mouse-vision-neuraldata.s3.amazonaws.com/mouse_calcium_visual_data_with_reliabilities.pkl"


@dataclass
class ProjectConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data_sources: DataSourcesConfig = field(default_factory=DataSourcesConfig)
    allen_data: AllenDataConfig = field(default_factory=AllenDataConfig)

#? --------------- Experiment Config --------------- #

@dataclass
class TrainDataConfig:
    data_path: Optional[str] = None
    image_size: int = 224
    batch_size: int = 256
    num_workers: int = 12
    train_split: float = 0.7
    val_split: float = 0.15
    split_seed: int = 42


@dataclass
class TrainOptimizerConfig:
    name: str = "AdamW"
    # torch.optim kwargs, e.g. {"lr": 3e-4, "weight_decay": 1e-4, ...}
    params: Dict[str, Any] = field(
        default_factory=lambda: {
            "lr": 3e-4, 
            "weight_decay": 1e-4
        }
    )


@dataclass
class TrainSchedulerConfig:
    name: str = "ReduceLROnPlateau"
    # torch.optim.lr_scheduler kwargs, e.g. for RLRP {"mode": "min", "factor": 0.5, "patience": 5}
    params: Dict[str, Any] = field(
        default_factory=lambda: {
            "factor": 0.5, 
            "patience": 5, 
            "mode": "min"
        }
    )


@dataclass
class TrainLossConfig:
    name: str = "CrossEntropyLoss"
    params: Dict[str, Any] = field(
        default_factory=lambda: {
            "label_smoothing": 0.1
        }
    )

@dataclass
class TrainCoreConfig:
    num_epochs: int = 100
    dropout_rate: float = 0.3
    early_stopping_patience: int = 15
    checkpoint_sub_dir: str = "default"
    optimizer: TrainOptimizerConfig = field(default_factory=TrainOptimizerConfig)
    scheduler: TrainSchedulerConfig = field(default_factory=TrainSchedulerConfig)
    loss: TrainLossConfig = field(default_factory=TrainLossConfig)
    save_every_n: int = 10


@dataclass
class DeviceConfig:
    device: str = "auto"  # auto|cpu|cuda|cuda:0
    use_cuda: bool = True
    seed: int = 42


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class DietConfig:
    grayscale: bool = False
    blur: bool = False
    noise: bool = False


@dataclass
class ExperimentConfig:
    data: TrainDataConfig = field(default_factory=TrainDataConfig)
    train: TrainCoreConfig = field(default_factory=TrainCoreConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    diet: DietConfig = field(default_factory=DietConfig)


@dataclass
class ResolvedConfig:
    project: ProjectConfig
    experiment: ExperimentConfig
    root: Path
    run_dir: Path

def _resolve_paths(cfg: ProjectConfig, root: Path) -> ProjectConfig:
    # fill root field if missing
    if cfg.paths.root is None:
        cfg.paths.root = root.as_posix()
    return cfg


def load_and_resolve_configs(
    project_yaml: str | Path,
    experiment_yaml: str | Path,
    overrides: Optional[List[str]] = None,
    root: Optional[Path] = None,
) -> tuple[ResolvedConfig, Dict[str, Any]]:
    root_path = root if root is not None else Path(__file__).resolve().parents[2]

    project_dict = load_yaml(project_yaml)
    exp_dict = load_yaml(experiment_yaml)

    # apply CLI overrides
    if overrides:
        for raw in overrides:
            k, v = _parse_override(raw)
            # decide which side to apply based on prefix
            if k.startswith("project."):
                _set_by_dots(project_dict, k[len("project."):], v)
            else:
                _set_by_dots(exp_dict, k, v)

    # Coerce scalar-like strings centrally to avoid type issues downstream
    project_dict = coerce_config_scalars(project_dict)
    exp_dict = coerce_config_scalars(exp_dict)

    project = _dict_to_dataclass(project_dict, ProjectConfig)
    project = _resolve_paths(project, root_path)

    # backfill experiment data paths from project if missing
    if "data" not in exp_dict:
        exp_dict["data"] = {}
    data_block = exp_dict["data"]
    if "data_path" not in data_block or data_block["data_path"] in (None, ""):
        data_block["data_path"] = project.data_sources.mini_imagenet_path

    experiment = _dict_to_dataclass(exp_dict, ExperimentConfig)

    # resolve run dir
    base = Path(os.path.expanduser(project.paths.checkpoint_base_dir))
    if not base.is_absolute():
        base = Path(project.paths.root or root_path.as_posix()) / base
    sub = experiment.train.checkpoint_sub_dir or "default"
    run_dir = (base / sub).resolve()

    resolved = ResolvedConfig(
        project=project,
        experiment=experiment,
        root=root_path,
        run_dir=run_dir,
    )

    flat: Dict[str, Any] = {
        "project": asdict(project),
        "experiment": asdict(experiment),
        "root": root_path.as_posix(),
        "run_dir": run_dir.as_posix(),
    }
    return resolved, flat


def ensure_dirs(resolved: ResolvedConfig) -> Dict[str, Path]:
    from datetime import datetime
    root = resolved.root
    paths = resolved.project.paths
    base_run_dir = resolved.run_dir
    # timestamped subfolder per run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (base_run_dir / ts).resolve()
    # reflect change into resolved object
    resolved.run_dir = run_dir

    ckpt_dir = run_dir
    # keep artifacts colocated with the run for reproducibility
    art_dir = run_dir / "artifacts"

    for p in [run_dir, ckpt_dir, art_dir]:
        p.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "train.log"

    return {"run_dir": run_dir, "log_file": log_file, "checkpoint_dir": ckpt_dir, "artifacts_dir": art_dir}


def save_resolved_config(flat: Dict[str, Any], out_path: Path) -> None:
    # local import to avoid extra dependency here
    import yaml  # type: ignore
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(flat, f, sort_keys=False)


def select_device(cfg: DeviceConfig) -> str:
    if cfg.device != "auto":
        return cfg.device
    try:
        import torch
        if cfg.use_cuda and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
