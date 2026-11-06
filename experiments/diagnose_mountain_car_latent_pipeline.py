import argparse
import importlib.util
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from lzero.model.ensemble_muzero_model import EnsembleMuZeroModelMLP
from lzero.model.utils import renormalize


def _load_config(path: Path) -> Dict:
    spec = importlib.util.spec_from_file_location("diagnostic_run_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.exp_config


def _diversity_stats(values: torch.Tensor) -> Dict[str, float]:
    values = values.detach().reshape(values.shape[0], -1).float()
    centered = values - values.mean(dim=0, keepdim=True)
    feature_std = values.std(dim=0, unbiased=False)
    normalized = torch.nn.functional.normalize(values, dim=-1, eps=1e-8)
    sample_count = int(values.shape[0])
    cosine_sum = normalized.sum(dim=0).square().sum() - sample_count
    pairwise_cosine = cosine_sum / max(sample_count * (sample_count - 1), 1)

    singular_values = torch.linalg.svdvals(centered)
    spectrum = singular_values.square()
    if float(spectrum.sum().item()) > 1e-12:
        probabilities = spectrum / spectrum.sum()
        effective_rank = torch.exp(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        )
    else:
        effective_rank = values.new_tensor(0.0)

    return {
        "feature_std": float(feature_std.mean().item()),
        "effective_rank": float(effective_rank.item()),
        "pairwise_cosine": float(pairwise_cosine.item()),
        "centered_rms": float(centered.square().mean().sqrt().item()),
    }


def _mountain_car_grid(device: torch.device) -> torch.Tensor:
    positions = np.linspace(-1.2, 0.6, 37, dtype=np.float32)
    velocities = np.linspace(-0.07, 0.07, 15, dtype=np.float32)
    observations = np.asarray(
        [(position, velocity) for position in positions for velocity in velocities],
        dtype=np.float32,
    )
    return torch.from_numpy(observations).to(device)


def _member_stages(member, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
    stages = {}
    encoder = member.representation_network
    encoder_raw = encoder.fc_representation(observations)
    stages["encoder_raw"] = encoder_raw
    encoder_final_norm = encoder.norm(encoder_raw)
    stages["encoder_final_ln"] = encoder_final_norm
    root_latent = renormalize(encoder_final_norm) if member.state_norm else encoder_final_norm
    stages["root_after_state_norm"] = root_latent

    prediction_common = member.prediction_network.fc_prediction_common
    first_prediction_linear = prediction_common[0]
    stages["counterfactual_pred_linear_on_encoder_raw"] = first_prediction_linear(encoder_raw)
    stages["counterfactual_pred_linear_on_encoder_ln"] = first_prediction_linear(encoder_final_norm)
    stages["pred_linear_on_state_norm"] = first_prediction_linear(root_latent)

    if member.self_supervised_learning_loss:
        projection = root_latent
        for index, layer in enumerate(member.projection):
            projection = layer(projection)
            stages[f"ssl_projection_{index}_{type(layer).__name__}"] = projection
        ssl_prediction = projection
        for index, layer in enumerate(member.prediction_head):
            ssl_prediction = layer(ssl_prediction)
            stages[f"ssl_predictor_{index}_{type(layer).__name__}"] = ssl_prediction

    prediction = root_latent
    for index, layer in enumerate(prediction_common):
        prediction = layer(prediction)
        stages[f"prediction_common_{index}_{type(layer).__name__}"] = prediction

    value = prediction
    for index, layer in enumerate(member.prediction_network.fc_value_head):
        value = layer(value)
        stages[f"value_head_{index}_{type(layer).__name__}"] = value

    policy = prediction
    for index, layer in enumerate(member.prediction_network.fc_policy_head):
        policy = layer(policy)
        stages[f"policy_head_{index}_{type(layer).__name__}"] = policy
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure state diversity through a MountainCar ensemble MLP checkpoint."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", default="iteration_6000.pth.tar")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = _load_config(args.run_dir / "total_config.py")
    model_config = dict(config["policy"]["model"])
    model = EnsembleMuZeroModelMLP(**model_config).to(args.device)
    checkpoint_path = args.run_dir / "ckpt" / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    observations = _mountain_car_grid(torch.device(args.device))

    print(f"checkpoint: {checkpoint_path}")
    print(f"grid states: {len(observations)}")
    print("metric columns: feature_std | effective_rank | pairwise_cosine | centered_rms")
    with torch.no_grad():
        member_stages = [
            _member_stages(member, observations)
            for member in model.members
        ]
    for stage_name in member_stages[0]:
        stats = [_diversity_stats(stages[stage_name]) for stages in member_stages]
        aggregate = {
            key: float(np.mean([member_stats[key] for member_stats in stats]))
            for key in stats[0]
        }
        print(
            f"{stage_name:42s} "
            f"{aggregate['feature_std']:10.6f} | "
            f"{aggregate['effective_rank']:8.3f} | "
            f"{aggregate['pairwise_cosine']:8.5f} | "
            f"{aggregate['centered_rms']:10.6f}"
        )


if __name__ == "__main__":
    main()
