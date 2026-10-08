"""Verify the isolated WSL CUDA stack without task resets or training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

SCRIPT_START = time.perf_counter()
SCRIPT_CPU_START = time.process_time()

import gymnasium
import mujoco
import numpy as np
import stable_baselines3
import torch
import yaml
from dm_control import suite  # noqa: F401 - import check only; never load/reset a task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    assert config["scope"] == "isolated_cuda_environment_and_tiny_tensor_smoke_only"
    assert config["environment"] == {
        "task_environment_imported": False,
        "task_environment_reset": False,
        "task_steps": 0,
        "training_steps": 0,
    }

    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "gymnasium": gymnasium.__version__,
        "dm_control": importlib.metadata.version("dm-control"),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "pyyaml": yaml.__version__,
    }
    expected_versions = {key: value for key, value in config["runtime"].items() if key != "platform"}
    assert config["runtime"]["platform"] == "WSL2"
    assert platform.system() == "Linux" and "microsoft" in platform.release().lower(), (
        f"Expected WSL2, got {platform.system()} {platform.release()}"
    )
    assert versions == expected_versions, f"Pinned runtime mismatch: {versions}"
    assert torch.version.cuda == "11.8", f"Unexpected Torch CUDA build: {torch.version.cuda}"
    assert torch.cuda.is_available(), "CUDA is not visible to PyTorch"

    device = torch.device(config["device"]["expected"])
    left = torch.ones((256, 256), dtype=torch.float32, device=device)
    result = left @ left
    torch.cuda.synchronize(device)
    assert result.device.type == "cuda"
    assert torch.isfinite(result).all().item()
    elapsed = time.perf_counter() - SCRIPT_START
    elapsed_cpu = time.process_time() - SCRIPT_CPU_START
    assert elapsed < float(config["limits"]["max_wall_seconds"])
    assert elapsed_cpu < float(config["limits"]["max_process_cpu_seconds"])

    print(json.dumps({
        "status": "PASS",
        "versions": versions,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_runtime": torch.version.cuda,
        "operation": config["device"]["operation"],
        "wall_seconds": round(elapsed, 6),
        "process_cpu_seconds": round(elapsed_cpu, 6),
        "task_resets": 0,
        "task_steps": 0,
        "training_steps": 0,
        "saved_artifacts": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
