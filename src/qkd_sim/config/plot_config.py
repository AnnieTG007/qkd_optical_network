"""统一绘图模型配置加载器。

从 config/defaults/plot_para/model_comparison.yaml 读取模型定义、颜色和分组，
供所有绘图脚本（signal_tx, fwm_noise 等）共用。

Examples
--------
>>> cfg = load_model_config()
>>> specs = load_model_specs("signal_tx")
>>> color = get_color("discrete")
"""

from __future__ import annotations

from pathlib import Path

import yaml

from qkd_sim.physical.signal import SpectrumType

# ---- 路径 ----
_CONFIG_DIR = Path(__file__).resolve().parent
_YAML_PATH = _CONFIG_DIR / "defaults" / "plot_para" / "model_comparison.yaml"

# ---- 缓存（模块级，只加载一次）----
_cache: dict | None = None


def _load_yaml() -> dict:
    if not _YAML_PATH.exists():
        raise FileNotFoundError(f"model_comparison.yaml not found: {_YAML_PATH}")
    with open(_YAML_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_cache() -> dict:
    global _cache
    if _cache is None:
        _cache = _load_yaml()
    return _cache


def load_model_config() -> dict:
    """返回完整配置字典（含 models / colors / model_groups）。"""
    return _get_cache()


def load_model_specs(group: str) -> dict[str, dict]:
    """返回指定分组的完整模型规格字典。

    Parameters
    ----------
    group : str
        分组名（如 "signal_tx"、"fwm_noise"、"noise_spectrum"），
        对应 model_comparison.yaml 中的 model_groups.* 键。

    Returns
    -------
    dict[str, dict]
        以模型 key 为键的规格字典。
        spectrum_type 已转换为 SpectrumType 枚举，
        color 已补全（从 colors 字典读取）。

    Examples
    --------
    >>> specs = load_model_specs("fwm_noise")
    >>> for key, spec in specs.items():
    ...     print(key, spec["label"])
    """
    cfg = _get_cache()
    group_keys: list[str] = cfg["model_groups"][group]
    models: dict[str, dict] = cfg["models"]
    colors: dict[str, str] = cfg["colors"]

    result = {}
    for k in group_keys:
        m = dict(models[k])  # 浅拷贝
        # spectrum_type 字符串 -> SpectrumType 枚举
        stype_str = m["spectrum_type"]
        m["spectrum_type"] = SpectrumType[stype_str]
        # color：从 colors 字典补全（YAML 中已含，可直接用）
        m["color"] = colors[k]
        result[k] = m
    return result


def get_color(key: str) -> str:
    """返回模型对应的绘图颜色。"""
    cfg = _get_cache()
    return cfg["colors"][key]
