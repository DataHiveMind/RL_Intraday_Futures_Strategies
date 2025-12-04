import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
	import yaml
except Exception as e:  # pragma: no cover - environment may vary
	raise ImportError("PyYAML is required to use src.utils.config (install pyyaml)") from e


def load_yaml(path: str) -> Dict[str, Any]:
	"""Load a YAML file and return its contents as a dict.

	Args:
		path: Path to a YAML file.

	Returns:
		Parsed YAML as a dictionary (empty dict for empty files).
	"""
	p = Path(path)
	if not p.exists():
		raise FileNotFoundError(f"Config file not found: {path}")
	with p.open("r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	if not isinstance(data, dict):
		raise ValueError(f"Expected mapping at top-level of YAML file: {path}")
	return data


def _merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
	"""Recursively merge dict `b` into dict `a` and return the result.

	Values in `b` take precedence over values in `a`.
	"""
	result = dict(a)
	for k, v in b.items():
		if k in result and isinstance(result[k], dict) and isinstance(v, dict):
			result[k] = _merge_dicts(result[k], v)
		else:
			result[k] = v
	return result


def load_config(path_or_name: str, config_dir: str = "config") -> Dict[str, Any]:
	"""Load a configuration by file path or by name within the project's `config` folder.

	Behavior:
	- If `path_or_name` points to an existing file, it will be loaded directly.
	- Otherwise, the function will try common locations under `config/`:
	  - `config/{path_or_name}.yaml`
	  - `config/{path_or_name}` (if provided with extension already)

	Args:
		path_or_name: File path or logical config name.
		config_dir: Base directory where configs live (defaults to `config`).

	Returns:
		Merged config dictionary loaded from the YAML file.
	"""
	p = Path(path_or_name)
	if p.exists():
		return load_yaml(str(p))

	base = Path(config_dir)
	candidates = [base / f"{path_or_name}.yaml", base / path_or_name]
	for c in candidates:
		if c.exists():
			return load_yaml(str(c))

	raise FileNotFoundError(
		f"Could not find config '{path_or_name}'. Tried: {p}, {candidates}"
	)


__all__ = ["load_yaml", "load_config", "_merge_dicts"]

