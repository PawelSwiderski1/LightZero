import argparse

import numpy as np
import pytest
import torch

from experiments.prepare_branching_rooms_macro_augmented_dataset import (
    dataset_split_diagnostics,
)
from experiments.run_branching_rooms_action_novelty import DatasetBundle
from experiments.run_branching_rooms_spatial_exploration_world_model import (
    SpatialExplorationWorldModel,
    _selection_score,
    action_conditioned_predictive_equivalence_loss,
    autoencoder_loss,
    dynamics_loss,
    evaluate_model,
    freeze_observation_model,
    gather_spatial_rollout_batch,
    position_first_visit_rewards,
    stage_two_loss,
)


def _args():
    return argparse.Namespace(
        temperature=0.1,
        direct_reconstruction_weight=2.0,
        direct_contrastive_weight=1.0,
        belief_observation_alignment_weight=1.0,
        pairwise_revisit_weight=1.0,
        pairwise_revisit_negative_samples=8,
        inverse_action_weight=0.1,
        variance_weight=0.1,
        horizon=3,
        latent_dynamics_weight=1.0,
        key_dynamics_weight=1.0,
        observation_key_consistency_weight=0.0,
        rollout_reconstruction_weight=2.0,
        rollout_contrastive_weight=1.0,
        memory_novelty_weight=0.0,
        predictive_equivalence_weight=0.0,
        predictive_equivalence_negative_margin=0.2,
        joint_representation_dynamics=False,
    )


def _batch():
    observations = torch.randint(0, 256, (2, 7, 3, 28, 28), dtype=torch.uint8)
    actions = torch.randint(0, 3, (2, 6))
    state_ids = torch.arange(7).repeat(2, 1)
    return {
        "observations": observations,
        "actions": actions,
        "state_ids": state_ids,
    }


def test_spatial_model_reconstructs_and_rolls_multiple_actions():
    model = SpatialExplorationWorldModel(
        spatial_channels=8, key_dim=16, belief_dim=32, action_dim=4
    )
    batch = _batch()
    encoded = model.encode_trajectory(batch["observations"], batch["actions"])
    gathered = gather_spatial_rollout_batch(
        encoded,
        batch["observations"],
        batch["actions"],
        batch["state_ids"],
        torch.tensor([0, 2]),
        horizon=3,
    )
    rollout = model.rollout(
        gathered["root_latents"],
        gathered["root_beliefs"],
        gathered["suffix_actions"],
    )

    assert encoded.latents.shape == (2, 7, 8, 7, 7)
    assert encoded.keys.shape == (2, 7, 16)
    assert encoded.beliefs.shape == (2, 7, 32)
    assert rollout.latents.shape == (4, 3, 8, 7, 7)
    assert rollout.observation_keys.shape == (4, 3, 16)
    assert rollout.keys.shape == (4, 3, 16)
    assert model.decode_latents(rollout.latents).shape == (4, 3, 3, 28, 28)
    assert torch.allclose(
        rollout.keys.norm(dim=-1), torch.ones(4, 3), atol=1e-5
    )


def test_belief_memory_keys_depend_on_history_for_the_same_observation():
    torch.manual_seed(11)
    model = SpatialExplorationWorldModel(
        spatial_channels=8,
        key_dim=16,
        belief_dim=32,
        action_dim=4,
        memory_key_mode="belief",
    )
    observations = torch.zeros((2, 3, 3, 28, 28), dtype=torch.uint8)
    actions = torch.tensor([[0, 2], [1, 2]])

    encoded = model.encode_trajectory(observations, actions)

    assert torch.allclose(
        encoded.observation_keys[0], encoded.observation_keys[1], atol=1e-6
    )
    assert not torch.allclose(encoded.beliefs[0, -1], encoded.beliefs[1, -1])
    assert not torch.allclose(encoded.keys[0, -1], encoded.keys[1, -1])
    assert model.decode_beliefs(encoded.beliefs).shape == observations.shape

    args = _args()
    args.horizon = 2
    args.joint_representation_dynamics = True
    args.latent_dynamics_weight = 0.0
    loss, _ = stage_two_loss(
        model,
        {
            "observations": observations,
            "actions": actions,
            "state_ids": torch.zeros((2, 3), dtype=torch.long),
        },
        torch.tensor([0]),
        args,
    )
    loss.backward()

    assert model.belief_key_projection[0].weight.grad is not None
    assert model.belief_key_projection[0].weight.grad.abs().sum() > 0
    assert model.belief_decoder_projection.weight.grad is not None
    assert model.belief_decoder_projection.weight.grad.abs().sum() > 0


def test_belief_to_observation_uses_observations_for_memory_and_beliefs_for_queries():
    torch.manual_seed(17)
    model = SpatialExplorationWorldModel(
        spatial_channels=8,
        key_dim=16,
        belief_dim=32,
        action_dim=4,
        memory_key_mode="belief_to_observation",
    )
    observations = torch.zeros((2, 3, 3, 28, 28), dtype=torch.uint8)
    actions = torch.tensor([[0, 2], [1, 2]])
    encoded = model.encode_trajectory(observations, actions)

    assert not torch.allclose(encoded.beliefs[0, -1], encoded.beliefs[1, -1])
    assert torch.allclose(encoded.keys, encoded.observation_keys, atol=1e-6)

    next_latent, _, query_key, next_belief = model.predict_next(
        encoded.latents[:, -1],
        encoded.beliefs[:, -1],
        torch.tensor([0, 0]),
    )
    assert torch.allclose(
        query_key,
        model.keys_from_beliefs(next_belief),
        atol=1e-6,
    )
    assert next_latent.shape == encoded.latents[:, -1].shape

    args = _args()
    args.horizon = 2
    args.joint_representation_dynamics = True
    args.latent_dynamics_weight = 0.0
    loss, components = stage_two_loss(
        model,
        {
            "observations": observations,
            "actions": actions,
            "state_ids": torch.zeros((2, 3), dtype=torch.long),
        },
        torch.tensor([0]),
        args,
    )
    loss.backward()

    assert components["representation_belief_observation_alignment"] > 0.0
    assert model.belief_key_projection[0].weight.grad is not None
    assert model.belief_key_projection[0].weight.grad.abs().sum() > 0.0


def test_belief_revisit_loss_uses_exact_rgb_pairs_and_trains_pair_head():
    torch.manual_seed(23)
    model = SpatialExplorationWorldModel(
        spatial_channels=8,
        key_dim=16,
        belief_dim=32,
        action_dim=4,
        memory_key_mode="belief_revisit",
    )
    observations = torch.zeros((1, 5, 3, 28, 28), dtype=torch.uint8)
    observations[:, 1] = 64
    observations[:, 2] = observations[:, 0]
    observations[:, 3] = 128
    observations[:, 4] = observations[:, 1]
    actions = torch.tensor([[0, 1, 2, 0]])
    state_ids = torch.tensor([[0, 1, 0, 2, 1]])
    args = _args()
    args.horizon = 2
    args.joint_representation_dynamics = True
    args.belief_dynamics_consistency_weight = 1.0
    loss, components, sample_priorities = stage_two_loss(
        model,
        {
            "observations": observations,
            "actions": actions,
            "state_ids": state_ids,
        },
        torch.tensor([0, 1, 2]),
        args,
        return_sample_priorities=True,
    )
    args.prioritized_replay_priority_mode = "revisit_error"
    _, revisit_components, revisit_priorities = stage_two_loss(
        model,
        {
            "observations": observations,
            "actions": actions,
            "state_ids": state_ids,
        },
        torch.tensor([0, 1, 2]),
        args,
        return_sample_priorities=True,
    )
    loss.backward()

    assert components["pairwise_revisit_positive_count"] > 0
    assert components["pairwise_revisit_negative_count"] > 0
    assert components["pairwise_revisit_loss"] > 0
    assert components["belief_dynamics_consistency"] > 0
    assert components["belief_dynamics_consistency_depth_1"] > 0
    assert components["belief_dynamics_consistency_depth_2"] > 0
    assert sample_priorities.shape == (1, 3)
    assert torch.all(sample_priorities[0] > 0)
    assert components["belief_prediction_priority_mean"] == pytest.approx(
        float(sample_priorities.mean())
    )
    assert components["replay_priority_mode_revisit_error"] == 0.0
    assert revisit_components[
        "replay_priority_mode_revisit_error"
    ] == 1.0
    assert revisit_components["replay_priority_mean"] == pytest.approx(
        float(revisit_priorities.mean())
    )
    assert revisit_components[
        "revisit_prediction_priority_mean"
    ] == pytest.approx(float(revisit_priorities.mean()))
    assert not torch.allclose(sample_priorities, revisit_priorities)
    assert components["pairwise_revisit_novelty_bce"] > 0
    assert model.revisit_pair_head[0].weight.grad is not None
    assert model.revisit_pair_head[0].weight.grad.abs().sum() > 0


def test_belief_revisit_aligns_predicted_observation_keys_with_real_future_keys():
    torch.manual_seed(29)
    model = SpatialExplorationWorldModel(
        spatial_channels=8,
        key_dim=16,
        belief_dim=32,
        action_dim=4,
        memory_key_mode="belief_revisit",
    )
    args = _args()
    args.latent_dynamics_weight = 0.0
    args.key_dynamics_weight = 0.0
    args.observation_key_consistency_weight = 1.0
    args.rollout_reconstruction_weight = 0.0
    args.rollout_contrastive_weight = 0.0
    args.memory_novelty_weight = 0.0
    args.pairwise_revisit_weight = 0.0

    loss, components = dynamics_loss(
        model,
        _batch(),
        torch.tensor([0, 2]),
        args,
    )
    loss.backward()

    assert components["observation_key_consistency"] > 0.0
    assert components["observation_key_consistency_depth_1"] > 0.0
    assert components["observation_key_consistency_depth_3"] > 0.0
    assert torch.allclose(
        loss.detach(),
        torch.tensor(components["observation_key_consistency"]),
        atol=1e-6,
    )
    assert model.transition[0].weight.grad is not None
    assert model.transition[0].weight.grad.abs().sum() > 0.0


def test_compressed_key_reconstructs_coarse_observation_and_coarse_identity():
    model = SpatialExplorationWorldModel(
        spatial_channels=8,
        key_dim=16,
        belief_dim=32,
        action_dim=4,
        representation_mode="compressed_key_reconstruction",
        coarse_reconstruction_size=7,
    )
    observations = torch.zeros((1, 2, 3, 28, 28), dtype=torch.uint8)
    observations[0, 0, 0, 0, 1] = 2
    observations[0, 1, 0, 0, 0] = 1
    observations[0, 1, 0, 0, 1] = 1
    full_ids = torch.tensor([[10, 11]])

    latents = model.encode_observations(observations)
    decoded = model.decode_latents(latents)
    targets = model.reconstruction_targets(observations)
    coarse_ids = model.representation_state_ids(observations, full_ids)

    assert decoded.shape == (1, 2, 3, 7, 7)
    assert targets.shape == decoded.shape
    assert coarse_ids[0, 0] == coarse_ids[0, 1]

    batch = {
        "observations": observations,
        "actions": torch.tensor([[1]]),
        "state_ids": full_ids,
    }
    loss, _ = autoencoder_loss(model, batch, _args())
    loss.backward()
    assert model.key_decoder[0].weight.grad is not None
    assert model.encoder[0].weight.grad is not None


def test_spatial_autoencoder_and_dynamics_losses_backpropagate():
    model = SpatialExplorationWorldModel(
        spatial_channels=8, key_dim=16, belief_dim=32, action_dim=4
    )
    batch = _batch()
    first_loss, _ = autoencoder_loss(model, batch, _args())
    first_loss.backward()
    model.zero_grad(set_to_none=True)

    encoded = model.encode_trajectory(batch["observations"], batch["actions"])
    assert encoded.latents.requires_grad
    second_loss, _ = dynamics_loss(
        model, batch, torch.tensor([0, 2]), _args()
    )
    second_loss.backward()

    assert torch.isfinite(first_loss)
    assert torch.isfinite(second_loss)
    assert model.transition[0].weight.grad is not None


def test_inverse_dynamics_embedding_is_the_decoder_free_memory_space():
    model = SpatialExplorationWorldModel(
        spatial_channels=8,
        key_dim=16,
        belief_dim=32,
        action_dim=4,
        representation_mode="inverse_dynamics",
    )
    args = _args()
    args.direct_reconstruction_weight = 0.0
    args.direct_contrastive_weight = 0.0
    args.inverse_action_weight = 1.0
    args.variance_weight = 0.1

    loss, components = autoencoder_loss(model, _batch(), args)
    loss.backward()

    assert not model.supports_reconstruction
    assert components["reconstruction_mae"] == 0.0
    assert components["contrastive"] == 0.0
    assert components["inverse"] > 0.0
    assert model.encoder[0].weight.grad is not None
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert model.key_projection[1].weight.grad is not None
    assert model.key_projection[1].weight.grad.abs().sum() > 0
    with pytest.raises(RuntimeError, match="no observation decoder"):
        model.decode_latents(model.encode_observations(_batch()["observations"]))


def test_key_only_dynamics_excludes_latent_and_pixel_losses():
    model = SpatialExplorationWorldModel(
        spatial_channels=8, key_dim=16, belief_dim=32, action_dim=4
    )
    freeze_observation_model(model)
    args = _args()
    args.latent_dynamics_weight = 0.0
    args.key_dynamics_weight = 1.0
    args.rollout_reconstruction_weight = 0.0
    args.rollout_contrastive_weight = 1.0

    loss, components = dynamics_loss(
        model, _batch(), torch.tensor([0, 2]), args
    )
    loss.backward()

    assert components["latent"] == 0.0
    assert components["rollout_reconstruction_mae"] == 0.0
    assert np.isclose(
        components["loss"], components["key"] + components["contrastive"]
    )
    assert model.transition[0].weight.grad is not None
    assert model.transition[0].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.decoder.parameters())
    assert all(parameter.grad is None for parameter in model.encoder.parameters())


def test_joint_stage_two_loss_updates_representation_and_dynamics():
    model = SpatialExplorationWorldModel(
        spatial_channels=8, key_dim=16, belief_dim=32, action_dim=4
    )
    args = _args()
    args.joint_representation_dynamics = True
    args.latent_dynamics_weight = 0.0
    args.rollout_reconstruction_weight = 0.0

    loss, components = stage_two_loss(
        model, _batch(), torch.tensor([0, 2]), args
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert components["dynamics_loss"] > 0.0
    assert components["representation_loss"] > 0.0
    assert model.transition[0].weight.grad is not None
    assert model.transition[0].weight.grad.abs().sum() > 0
    assert model.encoder[0].weight.grad is not None
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert model.key_projection[1].weight.grad is not None
    assert model.key_projection[1].weight.grad.abs().sum() > 0
    assert model.decoder[-2].weight.grad is not None
    assert model.decoder[-2].weight.grad.abs().sum() > 0


def test_memory_novelty_loss_trains_inverse_future_keys_without_decoder():
    model = SpatialExplorationWorldModel(
        spatial_channels=8,
        key_dim=16,
        belief_dim=32,
        action_dim=4,
        representation_mode="inverse_dynamics",
    )
    freeze_observation_model(model)
    args = _args()
    args.latent_dynamics_weight = 0.0
    args.key_dynamics_weight = 1.0
    args.rollout_reconstruction_weight = 0.0
    args.rollout_contrastive_weight = 0.0
    args.memory_novelty_weight = 10.0

    loss, components = dynamics_loss(
        model, _batch(), torch.tensor([0, 2]), args
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert components["memory_novelty_mae"] > 0.0
    assert model.transition[0].weight.grad is not None
    assert model.transition[0].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.encoder.parameters())


def test_predictive_equivalence_uses_matching_actions_and_real_outcomes():
    root_keys = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]],
        requires_grad=True,
    )
    root_ids = torch.tensor([[7, 7, 7]])
    suffix_actions = torch.tensor([[[0, 1], [0, 1], [0, 1]]])
    target_ids = torch.tensor([[[2, 3], [2, 3], [9, 3]]])

    loss, components = action_conditioned_predictive_equivalence_loss(
        root_keys,
        root_ids,
        suffix_actions,
        target_ids,
        negative_margin=0.2,
    )
    loss.backward()

    assert components["predictive_equivalence_pair_count"] == 3.0
    assert components["predictive_equivalence_positive_count"] == 1.0
    assert components["predictive_equivalence_negative_count"] == 2.0
    assert components["predictive_equivalence_mean_prefix"] == 2.0
    assert loss > 0.0
    assert root_keys.grad is not None
    assert root_keys.grad.abs().sum() > 0.0


def test_predictive_equivalence_ignores_unmatched_actions_and_observations():
    root_keys = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]],
        requires_grad=True,
    )
    root_ids = torch.tensor([[4, 4, 9]])
    suffix_actions = torch.tensor([[[0, 1], [2, 1], [0, 1]]])
    target_ids = torch.tensor([[[2, 3], [2, 3], [2, 3]]])

    loss, components = action_conditioned_predictive_equivalence_loss(
        root_keys,
        root_ids,
        suffix_actions,
        target_ids,
    )

    assert components["predictive_equivalence_pair_count"] == 0.0
    assert loss == 0.0


def test_predictive_equivalence_compares_histories_across_episodes():
    root_keys = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]]],
        requires_grad=True,
    )
    root_ids = torch.tensor([[4], [4]])
    suffix_actions = torch.tensor([[[0, 1]], [[0, 1]]])
    target_ids = torch.tensor([[[2, 3]], [[2, 9]]])

    loss, components = action_conditioned_predictive_equivalence_loss(
        root_keys,
        root_ids,
        suffix_actions,
        target_ids,
    )

    assert components["predictive_equivalence_pair_count"] == 1.0
    assert components["predictive_equivalence_negative_count"] == 1.0
    assert components["predictive_equivalence_active"] == 0.0
    assert loss == 0.0


def test_predictive_equivalence_does_not_apply_positive_only_attraction():
    root_keys = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]],
        requires_grad=True,
    )
    root_ids = torch.tensor([[4, 4]])
    suffix_actions = torch.tensor([[[0, 1], [0, 1]]])
    target_ids = torch.tensor([[[2, 3], [2, 3]]])

    loss, components = action_conditioned_predictive_equivalence_loss(
        root_keys,
        root_ids,
        suffix_actions,
        target_ids,
    )

    assert components["predictive_equivalence_positive_count"] == 1.0
    assert components["predictive_equivalence_negative_count"] == 0.0
    assert components["predictive_equivalence_active"] == 0.0
    assert loss == 0.0


def test_key_novelty_selection_ignores_pixels_and_physical_positions():
    metrics = {
        "by_depth": {
            "1": {
                "rgb_novelty_auc": 0.7,
                "position_novelty_auc": 0.1,
                "changed_rgb_retrieval_top1": 0.2,
                "copy_root_baseline": {"mae": 0.2},
                "rollout_mae_gain_vs_copy": -10.0,
            },
            "2": {
                "rgb_novelty_auc": 0.9,
                "position_novelty_auc": 0.99,
                "changed_rgb_retrieval_top1": 0.4,
                "copy_root_baseline": {"mae": 0.1},
                "rollout_mae_gain_vs_copy": 10.0,
            },
        }
    }

    assert np.isclose(_selection_score(metrics, 2, "key_novelty"), 1.2)

    metrics["by_depth"]["1"]["position_novelty_auc"] = 0.99
    metrics["by_depth"]["1"]["rollout_mae_gain_vs_copy"] = 1000.0
    assert np.isclose(_selection_score(metrics, 2, "key_novelty"), 1.2)


def test_novelty_fidelity_selection_rewards_correlation_and_low_error():
    metrics = {
        "by_depth": {
            "1": {
                "novelty_prediction_correlation": 0.8,
                "novelty_prediction_mae": 0.1,
            },
            "2": {
                "novelty_prediction_correlation": 0.6,
                "novelty_prediction_mae": 0.2,
            },
        }
    }

    assert np.isclose(
        _selection_score(metrics, 2, "novelty_fidelity"),
        0.7 - 0.15,
    )

    metrics["by_depth"]["2"]["novelty_prediction_mae"] = 0.4
    assert np.isclose(
        _selection_score(metrics, 2, "novelty_fidelity"),
        0.7 - 0.25,
    )


def test_position_first_visit_rewards_ignore_orientation_changes():
    poses = np.asarray(
        [
            [1, 1, 0],
            [1, 1, 1],
            [2, 1, 1],
            [2, 1, 2],
            [1, 1, 2],
            [3, 1, 2],
        ]
    )
    np.testing.assert_array_equal(
        position_first_visit_rewards(poses),
        np.asarray([0, 1, 0, 0, 1], dtype=np.float32),
    )


def test_evaluation_can_sample_noncontiguous_roots():
    model = SpatialExplorationWorldModel(
        spatial_channels=8, key_dim=16, belief_dim=32, action_dim=4
    )
    observations = torch.randint(
        0, 256, (1, 11, 3, 28, 28), dtype=torch.uint8
    )
    actions = torch.randint(0, 3, (1, 10))
    state_ids = torch.tensor([[0, 1, 2, 1, 3, 4, 3, 5, 6, 5, 7]])
    poses = torch.tensor(
        [[[index // 2, 0, index % 4] for index in range(11)]]
    )
    novelty_targets = torch.tensor(
        [[[2, 2], [2, 0], [0, 2], [2, 2], [2, 0], [0, 2], [2, 2], [2, 2], [2, 2]]]
    )
    loader = [
        {
            "observations": observations,
            "actions": actions,
            "state_ids": state_ids,
            "poses": poses,
            "novelty_targets": novelty_targets,
        }
    ]

    metrics = evaluate_model(
        model,
        loader,
        torch.device("cpu"),
        horizon=2,
        maximum_roots_per_episode=3,
    )

    assert metrics["by_depth"]["1"]["count"] == 3
    assert metrics["by_depth"]["2"]["count"] == 3

    inverse_model = SpatialExplorationWorldModel(
        spatial_channels=8,
        key_dim=16,
        belief_dim=32,
        action_dim=4,
        representation_mode="inverse_dynamics",
    )
    inverse_metrics = evaluate_model(
        inverse_model,
        loader,
        torch.device("cpu"),
        horizon=2,
        maximum_roots_per_episode=3,
    )

    assert inverse_metrics["by_depth"]["1"]["count"] == 3
    assert inverse_metrics["mean_rollout_mae"] is None
    assert inverse_metrics["by_depth"]["1"]["rollout_reconstruction"] == {
        "mae": None,
        "mse": None,
        "psnr": None,
    }


def test_augmented_dataset_diagnostics_report_policy_mix_and_no_overlap():
    observations = np.zeros((4, 3, 3, 4, 4), dtype=np.uint8)
    actions = np.zeros((4, 2), dtype=np.uint8)
    for episode in range(4):
        observations[episode, :, :, 0, 0] = episode
        actions[episode, 0] = episode % 3
    bundle = DatasetBundle(
        observations=observations,
        actions=actions,
        poses=np.zeros((4, 3, 3), dtype=np.int16),
        observation_targets=np.zeros((4, 1, 2), dtype=np.uint8),
        pose_targets=np.zeros((4, 1, 2), dtype=np.uint8),
        split_ids=np.asarray([0, 0, 1, 3], dtype=np.uint8),
        policy_ids=np.asarray([0, 2, 2, 2], dtype=np.uint8),
        episode_seeds=np.asarray([11, 12, 21, 31], dtype=np.int64),
        metadata={
            "policy_names": ["uniform", "persistent", "macro"],
            "split_names": ["train", "validation", "test", "heldout_policy"],
        },
    )

    diagnostics = dataset_split_diagnostics(bundle)

    assert diagnostics["splits"]["train"]["policy_counts"] == {
        "macro": 1,
        "uniform": 1,
    }
    assert diagnostics["splits"]["heldout_policy"]["policy_counts"] == {
        "macro": 1
    }
    assert diagnostics["globally_unique_episode_seeds"]
    assert not any(diagnostics["cross_split_seed_overlap"].values())
    assert not any(
        diagnostics["cross_split_exact_trajectory_overlap"].values()
    )
