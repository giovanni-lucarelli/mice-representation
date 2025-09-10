import json
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dicts(out[k], v)  # type: ignore
        else:
            out[k] = v
    return out


def _set_by_dots(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Override must be key=value, got: {raw}")
    k, v = raw.split("=", 1)
    v = v.strip()
    try:
        parsed = json.loads(v)
        return k.strip(), parsed
    except Exception:
        return k.strip(), v


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} does not contain a mapping at root")
    return data


def _dict_to_dataclass(data: Dict[str, Any], cls):
    kwargs = {}
    for field_name, field_def in cls.__dataclass_fields__.items():  # type: ignore[attr-defined]
        if field_name in data:
            val = data[field_name]
            field_type = field_def.type
            try:
                if hasattr(field_type, "__dataclass_fields__"):
                    kwargs[field_name] = _dict_to_dataclass(val, field_type)
                else:
                    kwargs[field_name] = val
            except Exception:
                kwargs[field_name] = val
    return cls(**kwargs)

