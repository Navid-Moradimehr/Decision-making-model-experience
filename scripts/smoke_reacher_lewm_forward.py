"""Strict CPU forward smoke for the pinned LeWM Reacher checkpoint.

This is an interface/finite-output check only. It does not score candidate
actions, generate planner labels, train a model, or claim task performance.
The isolated venv and pinned source checkout are supplied via environment
variables so the project's default environment stays lean.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

import gymnasium as gym
import numpy as np
import torch
from torchvision.transforms import v2 as transforms


EXPECTED_CHECKPOINT_SHA256 = (
    "eb70b1fd5409f8f81875d62f5ee5a20dd220a3128a477de66b5760f475f0f469"
)
ENV_ID = "swm/ReacherDMControl-v0"
SEED = 61002


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _namespace(name: str, path: Path) -> None:
    """Avoid unrelated eager package initializers; load the pinned modules."""
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(0)
    assert not torch.cuda.is_initialized(), "CUDA must remain unused"

    checkpoint_dir = Path(os.environ["LEWM_REACHER_CHECKPOINT"]).resolve()
    weights_path = checkpoint_dir / "weights.pt"
    config_path = checkpoint_dir / "config.json"
    source_root = Path(os.environ["STABLE_WORLD_MODEL_SOURCE"]).resolve()
    assert weights_path.is_file() and config_path.is_file()
    actual_sha256 = _sha256(weights_path)
    assert actual_sha256 == EXPECTED_CHECKPOINT_SHA256, actual_sha256
    model_config = json.loads(config_path.read_text(encoding="utf-8"))

    # The published stable-pretraining helper is called from its exact pinned
    # source file. Its package initializer eagerly imports training/data extras
    # irrelevant to vit_hf; a namespace package skips that initializer only.
    import stable_pretraining

    sp_root = Path(stable_pretraining.__file__).resolve().parent
    _namespace("stable_pretraining.backbone", sp_root / "backbone")
    from stable_pretraining.backbone.utils import vit_hf

    # Import the exact pinned LeWM classes while bypassing only package-level
    # eager registrations and unrelated model families.
    _namespace("stable_worldmodel", source_root / "stable_worldmodel")
    _namespace("stable_worldmodel.wm", source_root / "stable_worldmodel" / "wm")
    _namespace(
        "stable_worldmodel.wm.lewm",
        source_root / "stable_worldmodel" / "wm" / "lewm",
    )
    from stable_worldmodel.wm.lewm.lewm import LeWM
    from stable_worldmodel.wm.lewm.module import Embedder, MLP, Predictor

    enc_cfg = model_config["encoder"]
    encoder = vit_hf(
        size=enc_cfg["size"],
        patch_size=enc_cfg["patch_size"],
        image_size=enc_cfg["image_size"],
        pretrained=enc_cfg["pretrained"],
        use_mask_token=enc_cfg["use_mask_token"],
    )
    pred_cfg = model_config["predictor"]
    predictor = Predictor(**{k: v for k, v in pred_cfg.items() if k != "_target_"})
    act_cfg = model_config["action_encoder"]
    action_encoder = Embedder(**{k: v for k, v in act_cfg.items() if k != "_target_"})

    def build_projector(cfg: dict) -> MLP:
        # The checkpoint uses BatchNorm1d partials in both projection MLPs.
        return MLP(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            output_dim=cfg["output_dim"],
            norm_fn=torch.nn.BatchNorm1d,
        )

    model = LeWM(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=build_projector(model_config["projector"]),
        pred_proj=build_projector(model_config["pred_proj"]),
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.requires_grad_(False)

    # Same qpos_match wrapper/task used in the preceding environment smoke.
    # Its DMC subpackage initializer imports an unused expert requiring Torch;
    # retain the exact package paths while skipping only that initializer.
    dmcontrol_path = source_root / "stable_worldmodel" / "envs" / "dmcontrol"
    _namespace("stable_worldmodel.envs.dmcontrol", dmcontrol_path)
    import stable_worldmodel.envs  # noqa: F401 - register pinned swm IDs

    env = gym.make(ENV_ID, task="qpos_match")
    started = time.perf_counter()
    try:
        _, reset_info = env.reset(seed=SEED)
        frame = env.unwrapped.render(width=224, height=224)
        assert frame.shape == (224, 224, 3) and frame.dtype == np.uint8
        initial_qpos = np.asarray(reset_info["qpos"]).copy()
        initial_qvel = np.asarray(reset_info["qvel"]).copy()
        goal_qpos = initial_qpos + np.asarray([0.7, -0.6])
        env.unwrapped.set_target_qpos(goal_qpos)
        env.unwrapped.set_state(goal_qpos, np.zeros_like(initial_qvel))
        goal_frame = env.unwrapped.render(width=224, height=224)
        assert goal_frame.shape == (224, 224, 3) and goal_frame.dtype == np.uint8

        # The reference evaluation uses RGB, scale-to-[0,1], ImageNet
        # normalization and a 224 resize. The source frame is already 224².
        preprocess = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                transforms.Resize(size=224),
            ]
        )
        # dm_control's renderer may return a vertically flipped negative
        # stride view; materialize contiguous RGB before ToTensor.
        pixels = preprocess(np.ascontiguousarray(frame)).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        assert pixels.shape == (1, 1, 1, 3, 224, 224)
        goal_pixels = preprocess(np.ascontiguousarray(goal_frame)).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        assert goal_pixels.shape == pixels.shape

        # One real frame is the policy context. Each future planning step is a
        # 5-step sequence of 2-D torque actions flattened to 10 scalars.
        action_blocks = torch.zeros(1, 1, 2, 10, dtype=torch.float32)
        action_blocks[0, 0, 0] = torch.tensor([0.2, -0.2]).repeat(5)
        action_blocks[0, 0, 1] = torch.tensor([-0.2, 0.2]).repeat(5)
        assert action_blocks.shape[-1] == 5 * env.action_space.shape[0]
        assert action_blocks.abs().max().item() <= 1.0

        with torch.inference_mode():
            encoded = model.encode({"pixels": pixels[:, 0]})["emb"]
            goal_encoded = model.encode({"pixels": goal_pixels[:, 0]})["emb"]
            rollout_info = model.rollout(
                {"pixels": pixels}, action_blocks, history_size=3
            )
        predicted = rollout_info["predicted_emb"]
        # Matches the released criterion's final-step squared-L2 goal-embedding
        # cost; this checks tensor compatibility only, not cost validity.
        final_cost = (predicted[:, :, -1] - goal_encoded[:, -1]).square().sum(-1)
        assert encoded.shape == (1, 1, 192), encoded.shape
        assert goal_encoded.shape == encoded.shape
        assert predicted.shape == (1, 1, 3, 192), predicted.shape
        assert torch.isfinite(encoded).all()
        assert torch.isfinite(goal_encoded).all()
        assert torch.isfinite(predicted).all()
        assert torch.isfinite(final_cost).all()
        assert not torch.cuda.is_initialized()

        print(f"Checkpoint SHA-256: {actual_sha256}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Real frame context: {pixels.shape}; encoded: {tuple(encoded.shape)}")
        print(f"Separate target-qpos goal image: {tuple(goal_pixels.shape)}; final latent cost: {final_cost.item():.6f}")
        print(f"Two 5x2 torque blocks -> latent rollout: {tuple(predicted.shape)}")
        print(f"Finite CPU prediction: PASS; CUDA initialized: {torch.cuda.is_initialized()}")
        print(f"Reset qpos: {initial_qpos.tolist()}; goal qpos: {goal_qpos.tolist()}")
        print(f"Forward+render wall time: {time.perf_counter() - started:.3f}s")
        print("Candidate ranking/performance/training: NOT TESTED")
    finally:
        env.close()


if __name__ == "__main__":
    main()
