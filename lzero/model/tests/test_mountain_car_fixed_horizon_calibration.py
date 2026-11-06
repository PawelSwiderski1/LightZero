import json

import numpy as np

from experiments.diagnose_mountain_car_fixed_horizon_calibration import (
    OccupancyIndex,
    add_tree_local_path_metrics,
    density_bucket,
    evaluate_reachable_novelty,
    mountain_car_step,
    observation_bin_key,
    simulate_action_sequences,
)


def test_mountain_car_step_matches_classic_control_dynamics():
    position, velocity, reward, terminated = mountain_car_step(-0.5, 0.0, 2)

    expected_velocity = 0.001 - 0.0025 * np.cos(-1.5)
    assert np.isclose(velocity, expected_velocity)
    assert np.isclose(position, -0.5 + expected_velocity)
    assert reward == -1.0
    assert not terminated


def test_sequence_simulation_stops_rewarding_after_goal():
    observations = np.asarray([[0.499, 0.01], [-0.5, 0.0]], dtype=np.float32)
    sequences = np.asarray([[2, 2]], dtype=np.int64)
    result = simulate_action_sequences(observations, sequences, discount=0.9)

    assert bool(result["terminated"][0, 0])
    assert result["executed_steps"][0, 0] == 1
    assert np.isclose(result["reward_returns"][0, 0], -1.0)
    assert not bool(result["terminated"][1, 0])
    assert result["executed_steps"][1, 0] == 2
    assert np.isclose(result["reward_returns"][1, 0], -1.9)


def test_occupancy_index_counts_neighboring_bins():
    center = (-0.5, 0.0)
    same_bin = (-0.499, 0.0002)
    neighboring_bin = (-0.495, 0.001)
    occupancy = OccupancyIndex([center, same_bin, neighboring_bin])

    assert observation_bin_key(*center) == observation_bin_key(*same_bin)
    assert occupancy.count(*center, radius_bins=0) == 2
    assert occupancy.count(*center, radius_bins=1) == 3
    assert density_bucket(0) == "unseen"
    assert density_bucket(5) == "sparse_1_5"
    assert density_bucket(50) == "medium_6_50"
    assert density_bucket(51) == "dense_gt_50"

    positions = np.asarray([center[0], neighboring_bin[0], -1.2], dtype=np.float32)
    velocities = np.asarray([center[1], neighboring_bin[1], 0.0], dtype=np.float32)
    assert np.array_equal(
        occupancy.count_many(positions, velocities, radius_bins=0),
        np.asarray([2, 1, 0]),
    )
    assert np.array_equal(
        occupancy.count_many(positions, velocities, radius_bins=1),
        np.asarray([3, 3, 0]),
    )


def test_reachable_novelty_prefers_unvisited_true_successor():
    observation = np.asarray([[-0.5, 0.0]], dtype=np.float32)
    left_position, left_velocity, _, _ = mountain_car_step(-0.5, 0.0, 0)
    occupancy = OccupancyIndex([(left_position, left_velocity) for _ in range(9)])
    continuations = np.asarray([[0], [2]], dtype=np.int64)

    result = evaluate_reachable_novelty(
        observation,
        continuations,
        occupancy,
        density_radius_bins=0,
        novelty_discount=1.0,
    )

    assert result["best_sequence_index"].tolist() == [1]
    assert np.isclose(result["best_return"][0], 1.0)
    assert result["best_endpoint_density"].tolist() == [0]
    assert result["best_path_min_density"].tolist() == [0]


def test_tree_local_metrics_remove_member_offsets():
    member_offsets = np.asarray([0.0, 100.0, -50.0])
    records = []
    for sequence_index, path_value in enumerate([1.0, 2.0, 3.0]):
        member_values = member_offsets + path_value
        records.append(
            {
                "root_sample_index": 0,
                "true_terminated": 0,
                "sequence": str(sequence_index),
                "imagined_q_members": json.dumps(member_values.tolist()),
                "direct_q_members": json.dumps(member_values.tolist()),
            }
        )

    add_tree_local_path_metrics(records)

    for record in records:
        assert np.isclose(record["tree_centered_imagined_qstd"], 0.0)
        assert np.isclose(record["tree_centered_direct_qstd"], 0.0)
        assert np.isclose(record["best_mean_path_relative_imagined_qstd"], 0.0)
        assert np.isclose(record["mean_pairwise_imagined_qstd"], 0.0)
