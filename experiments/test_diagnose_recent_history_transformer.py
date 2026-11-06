import numpy as np
import pytest
import torch

from experiments.diagnose_recent_history_transformer import (
    binary_auc,
    model_observation_tensor,
    prediction_metrics,
    recent_revisit_ages,
)


def test_recent_revisit_ages_uses_the_nearest_match_in_the_window():
    observations = np.asarray([0, 1, 0, 2, 1, 3], dtype=np.uint8).reshape(
        1, 6, 1, 1, 1
    )
    roots = np.asarray([[1]], dtype=np.int64)

    ages = recent_revisit_ages(
        observations,
        roots,
        horizon=4,
        history_window=4,
    )

    assert ages.tolist() == [[[2, 0, 3, 0]]]


def test_prediction_metrics_reports_ranking_and_calibration():
    targets = np.asarray([0, 0, 1, 1], dtype=np.float32)
    scores = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float32)

    metrics = prediction_metrics(targets, scores)

    assert binary_auc(targets, scores) == pytest.approx(1.0)
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(0.085)
    assert metrics["novel_prediction_mean"] == pytest.approx(0.75)
    assert metrics["revisit_prediction_mean"] == pytest.approx(0.25)


def test_symbolic_model_observations_are_not_rgb_normalized():
    observations = np.full((1, 3, 7, 7), 5, dtype=np.uint8)

    symbolic = model_observation_tensor(
        observations, "symbolic_7x7", torch.device("cpu")
    )
    rgb = model_observation_tensor(
        observations, "rgb_tile_7x7", torch.device("cpu")
    )

    assert symbolic.max().item() == 5.0
    assert rgb.max().item() == pytest.approx(5.0 / 255.0)
