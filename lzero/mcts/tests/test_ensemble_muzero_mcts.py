import numpy as np
import pytest
import torch
from easydict import EasyDict

from lzero.mcts.ensemble_muzero_mcts import (
    DepthNormalizedBackupDisagreementStats,
    EnsembleQMinMaxStats,
    EnsembleSearchNode,
    EnsembleSharedTreeMCTS,
)
from lzero.model.common import MZNetworkOutput


class _IdentityTransform:

    def __call__(self, value):
        return value


class _FakeEnsembleModel:

    def eval(self):
        return self

    def recurrent_inference(self, latent_state, action, member_id=None):
        member_id = torch.as_tensor(member_id, dtype=torch.float32, device=latent_state.device).view(-1, 1)
        action = action.float().view(-1, 1)
        next_latent_state = latent_state + action + member_id
        reward = 1.0 + 2.0 * member_id
        value = torch.zeros_like(reward)
        policy_logits = torch.zeros(latent_state.shape[0], 2, device=latent_state.device)
        return MZNetworkOutput(value, reward, policy_logits, next_latent_state)

    def project(self, latent_state, with_grad=True, member_id=None):
        return latent_state * 2.0 if with_grad else latent_state.detach()

    def intrinsic_reward_from_latent(self, latent_state, member_id=None):
        member_id = torch.as_tensor(member_id, dtype=torch.float32, device=latent_state.device).view(-1, 1)
        return 0.5 + member_id

    def intrinsic_value_from_latent(self, latent_state, member_id=None):
        member_id = torch.as_tensor(member_id, dtype=torch.float32, device=latent_state.device).view(-1, 1)
        return 2.0 + member_id


class _ActionIntrinsicModel:

    def eval(self):
        return self

    def recurrent_inference(self, latent_state, action, member_id=None):
        action = action.float().view(-1, 1)
        next_latent_state = torch.cat([action, latent_state[:, 1:]], dim=1)
        reward = torch.zeros_like(action)
        value = torch.zeros_like(action)
        policy_logits = torch.zeros(latent_state.shape[0], 3, device=latent_state.device)
        return MZNetworkOutput(value, reward, policy_logits, next_latent_state)

    def intrinsic_reward_from_latent(self, latent_state, member_id=None):
        is_noop = (latent_state[:, 0:1] - 1.0).abs() < 1e-6
        return torch.where(is_noop, torch.full_like(latent_state[:, 0:1], 8.0), torch.zeros_like(latent_state[:, 0:1]))

    def intrinsic_value_from_latent(self, latent_state, member_id=None):
        return torch.zeros(latent_state.shape[0], 1, device=latent_state.device)


class _MixedRewardTradeoffModel:

    def eval(self):
        return self

    def recurrent_inference(self, latent_state, action, member_id=None):
        action = action.float().view(-1, 1)
        next_latent_state = torch.cat([action, latent_state[:, 1:]], dim=1)
        reward = torch.where(action < 0.5, torch.full_like(action, 10.0), torch.zeros_like(action))
        value = torch.zeros_like(action)
        policy_logits = torch.zeros(latent_state.shape[0], 2, device=latent_state.device)
        return MZNetworkOutput(value, reward, policy_logits, next_latent_state)

    def intrinsic_reward_from_latent(self, latent_state, member_id=None):
        return torch.where(
            latent_state[:, 0:1] > 0.5,
            torch.ones_like(latent_state[:, 0:1]),
            torch.zeros_like(latent_state[:, 0:1]),
        )

    def intrinsic_value_from_latent(self, latent_state, member_id=None):
        return torch.zeros(latent_state.shape[0], 1, device=latent_state.device)


@pytest.mark.unittest
def test_ensemble_q_min_max_stats_normalizes_mean_and_std_in_same_units():
    stats = EnsembleQMinMaxStats(enabled=True)
    stats.update(np.asarray([-200.0, -150.0, -100.0], dtype=np.float32))

    assert stats.minimum == pytest.approx(-200.0)
    assert stats.maximum == pytest.approx(-100.0)
    assert stats.value_range == pytest.approx(100.0)
    assert stats.normalize_value(-150.0) == pytest.approx(0.5)
    assert stats.normalize_std(25.0) == pytest.approx(0.25)


@pytest.mark.unittest
def test_ensemble_q_min_max_stats_can_preserve_legacy_raw_scoring():
    stats = EnsembleQMinMaxStats(enabled=False)
    stats.update(np.asarray([-200.0, -100.0], dtype=np.float32))

    assert stats.value_range == 0.0
    assert stats.normalize_value(-150.0) == pytest.approx(-150.0)
    assert stats.normalize_std(25.0) == pytest.approx(25.0)


@pytest.mark.unittest
def test_value_calibration_changes_uncertainty_without_changing_mean_q():
    cfg = EasyDict(dict(discount_factor=0.5, pb_c_base=19652, pb_c_init=1.25))
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())
    stats = EnsembleQMinMaxStats(enabled=True)
    root = EnsembleSearchNode(0.0, [0], 2, q_min_max_stats=stats)
    root.expand({0: 1.0})
    child = root.children[0]
    leaf_values = np.asarray([10.0, 20.0], dtype=np.float32)
    centered_member_offsets = np.asarray([-5.0, 5.0], dtype=np.float32)

    mcts._backup(
        root,
        [child],
        leaf_values,
        value_calibration_offsets=centered_member_offsets,
    )

    assert child.q_mean.tolist() == pytest.approx([5.0, 10.0])
    assert child.calibrated_q_mean.tolist() == pytest.approx([7.5, 7.5])
    assert np.mean(child.calibrated_q_mean) == pytest.approx(np.mean(child.q_mean))
    assert child.raw_uncertainty == pytest.approx(2.5)
    assert child.uncertainty == pytest.approx(0.0)

    diagnostics = mcts._root_action_diagnostics(
        root, 'optimistic_shared_tree', 2.0, False, 0.0
    )[0]
    assert diagnostics['mean_q'] == pytest.approx(7.5)
    assert diagnostics['raw_q_std'] == pytest.approx(2.5)
    assert diagnostics['calibrated_q_std'] == pytest.approx(0.0)
    assert diagnostics['q_values'] == pytest.approx([5.0, 10.0])
    assert diagnostics['calibrated_q_values'] == pytest.approx([7.5, 7.5])


@pytest.mark.unittest
def test_value_calibration_offset_is_discounted_from_the_leaf():
    cfg = EasyDict(dict(discount_factor=0.5, pb_c_base=19652, pb_c_init=1.25))
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())
    root = EnsembleSearchNode(0.0, [0], 2)
    first = EnsembleSearchNode(1.0, [0], 2, reward=np.asarray([1.0, 1.0], dtype=np.float32))
    second = EnsembleSearchNode(1.0, [0], 2, reward=np.asarray([1.0, 1.0], dtype=np.float32))
    trace = mcts._backup(
        root,
        [first, second],
        np.asarray([10.0, 20.0], dtype=np.float32),
        value_calibration_offsets=np.asarray([-5.0, 5.0], dtype=np.float32),
    )

    # Two edges discount the leaf correction by gamma^2 = 0.25.
    assert first.q_mean.tolist() == pytest.approx([4.0, 6.5])
    assert first.calibrated_q_mean.tolist() == pytest.approx([5.25, 5.25])
    assert trace['backed_up_q_values'] == pytest.approx([4.0, 6.5])
    assert trace['backed_up_calibrated_q_values'] == pytest.approx([5.25, 5.25])


@pytest.mark.unittest
def test_mixed_reward_backup_is_extrinsic_q_plus_beta_intrinsic_q():
    cfg = EasyDict(
        dict(
            discount_factor=0.5,
            rnd_intrinsic_discount_factor=0.25,
            intrinsic_tree_immediate_reward_weight=1.0,
            intrinsic_tree_leaf_value_weight=1.0,
            pb_c_base=19652,
            pb_c_init=1.25,
        )
    )
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())
    stats = EnsembleQMinMaxStats(enabled=True)
    root = EnsembleSearchNode(0.0, [0], 2, q_min_max_stats=stats)
    root.expand({0: 1.0})
    child = root.children[0]
    child.reward = np.asarray([2.0, 2.0], dtype=np.float32)
    child.intrinsic_reward = 3.0

    trace = mcts._backup(
        root,
        [child],
        np.asarray([10.0, 20.0], dtype=np.float32),
        leaf_intrinsic_value=4.0,
        mixed_reward_backup=True,
        intrinsic_reward_weight=5.0,
    )

    # Extrinsic: 2 + 0.5 * [10, 20] = [7, 12].
    # Intrinsic: 3 + 0.25 * 4 = 4. Mixed: extrinsic + 5 * intrinsic.
    assert child.q_mean.tolist() == pytest.approx([7.0, 12.0])
    assert child.intrinsic_value == pytest.approx(4.0)
    assert child.mixed_q_mean.tolist() == pytest.approx([27.0, 32.0])
    assert trace['backed_up_mixed_q_values'] == pytest.approx([27.0, 32.0])
    assert stats.minimum == pytest.approx(27.0)
    assert stats.maximum == pytest.approx(32.0)


@pytest.mark.unittest
def test_backup_records_root_edge_returns_by_exact_search_depth():
    cfg = EasyDict(dict(discount_factor=0.5, pb_c_base=19652, pb_c_init=1.25))
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())
    root = EnsembleSearchNode(0.0, [0], 2)
    root.expand({0: 1.0})
    child = root.children[0]
    grandchild = EnsembleSearchNode(
        1.0,
        [0],
        2,
        reward=np.zeros(2, dtype=np.float32),
    )

    mcts._backup(
        root,
        [child],
        np.asarray([2.0, 4.0], dtype=np.float32),
        leaf_depth=1,
    )
    mcts._backup(
        root,
        [child, grandchild],
        np.asarray([8.0, 12.0], dtype=np.float32),
        leaf_depth=2,
    )

    assert child.visit_count_by_search_depth == {1: 1, 2: 1}
    assert child.search_depth_q_mean(1).tolist() == pytest.approx([1.0, 2.0])
    assert child.search_depth_q_mean(2).tolist() == pytest.approx([2.0, 3.0])
    assert root.visit_count_by_search_depth == {}
    assert grandchild.visit_count_by_search_depth == {1: 1}
    assert grandchild.search_depth_q_mean(1).tolist() == pytest.approx([4.0, 6.0])


@pytest.mark.unittest
def test_depth_matched_qstd_uses_uniform_common_depths_and_all_within_depth_returns():
    cfg = EasyDict(
        dict(
            discount_factor=0.5,
            pb_c_base=19652,
            pb_c_init=1.25,
            depth_matched_qstd_diagnostics=True,
            depth_matched_qstd_max_depth=3,
        )
    )
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())
    root = EnsembleSearchNode(0.0, [0, 1], 2)
    root.expand({0: 0.5, 1: 0.5})
    left = root.children[0]
    right = root.children[1]

    # The two depth-1 left returns are averaged within that exact-depth bucket.
    left.record_search_depth_return(1, np.asarray([0.0, 2.0]), np.asarray([0.0, 2.0]))
    left.record_search_depth_return(1, np.asarray([2.0, 4.0]), np.asarray([2.0, 4.0]))
    left.record_search_depth_return(2, np.asarray([9.0, 15.0]), np.asarray([9.0, 15.0]))
    left.record_search_depth_return(3, np.asarray([100.0, 200.0]), np.asarray([100.0, 200.0]))

    right.record_search_depth_return(1, np.asarray([2.0, 4.0]), np.asarray([2.0, 4.0]))
    right.record_search_depth_return(2, np.asarray([14.0, 18.0]), np.asarray([14.0, 18.0]))

    diagnostics = mcts._root_depth_matched_q_diagnostics(root)

    # Depth 3 is excluded from the mixture because right has no depth-3 return.
    assert diagnostics[0]['depth_matched_common_depths'] == [1, 2]
    assert diagnostics[0]['depth_1_return_count'] == 2
    assert diagnostics[1]['depth_1_return_count'] == 1
    assert diagnostics[0]['depth_1_q_values'] == pytest.approx([1.0, 3.0])
    assert diagnostics[0]['depth_3_q_values'] == pytest.approx([100.0, 200.0])

    # Uniform depth mixture: ([1, 3] + [9, 15]) / 2 = [5, 9].
    assert diagnostics[0]['depth_matched_q_values'] == pytest.approx([5.0, 9.0])
    assert diagnostics[0]['depth_matched_raw_q_std'] == pytest.approx(2.0)
    # Uniform depth mixture: ([2, 4] + [14, 18]) / 2 = [8, 11].
    assert diagnostics[1]['depth_matched_q_values'] == pytest.approx([8.0, 11.0])
    assert diagnostics[1]['depth_matched_raw_q_std'] == pytest.approx(1.5)


@pytest.mark.unittest
def test_depth_corrected_qstd_preserves_old_aggregate_cancellation():
    cfg = EasyDict(
        dict(
            num_simulations=2,
            pb_c_base=19652,
            pb_c_init=1.25,
            depth_normalized_backup_disagreement=True,
            depth_disagreement_max_depth=2,
        )
    )
    normalizer = DepthNormalizedBackupDisagreementStats(cfg)
    normalizer.load_state_dict(
        {
            'raw_ratio': {1: 1.0, 2: 2.0},
            'calibrated_ratio': {1: 1.0, 2: 2.0},
            'raw_baseline': {1: 1.0, 2: 2.0},
            'calibrated_baseline': {1: 1.0, 2: 2.0},
            'total_group_count': {1: 100, 2: 100},
        }
    )
    mcts = EnsembleSharedTreeMCTS(
        cfg, _IdentityTransform(), _IdentityTransform(), depth_disagreement_stats=normalizer
    )
    parent = EnsembleSearchNode(0.0, [0], 2)
    parent.visit_count = 10
    child = EnsembleSearchNode(1.0, [0], 2)
    returns = [
        np.asarray([0.0, 2.0], dtype=np.float32),
        np.asarray([2.0, 0.0], dtype=np.float32),
    ]
    for values in returns:
        child.q_sum += values
        child.calibrated_q_sum += values
        child.has_calibrated_q = True
        child.record_search_depth_return(2, values, values, normalizer.scale(2))
        child.visit_count += 1

    components = mcts._score_child_components(
        parent, child, 'optimistic_shared_tree', 1.0, False, 0.0
    )

    assert child.q_mean.tolist() == pytest.approx([1.0, 1.0])
    assert components['aggregate_calibrated_uncertainty'] == pytest.approx(0.0)
    assert child.calibrated_backup_disagreement_mean == pytest.approx(1.0)
    assert child.depth_normalized_backup_disagreement_mean == pytest.approx(0.5)
    assert child.depth_corrected_aggregate_uncertainty(
        calibrated=True, cap_at_aggregate=False
    ) == pytest.approx(0.0)
    assert components['uncertainty'] == pytest.approx(0.0)
    assert components['uncertainty_metric'] == 'depth_corrected_aggregate_q_std'


@pytest.mark.unittest
def test_depth_corrected_qstd_equals_old_qstd_when_all_depth_scales_are_one():
    child = EnsembleSearchNode(1.0, [0], 3)
    observations = [
        (1, np.asarray([-4.0, 2.0, 5.0], dtype=np.float32)),
        (2, np.asarray([3.0, -1.0, 7.0], dtype=np.float32)),
        (2, np.asarray([6.0, 8.0, -2.0], dtype=np.float32)),
    ]
    for depth, values in observations:
        child.q_sum += values
        child.calibrated_q_sum += values
        child.has_calibrated_q = True
        child.record_search_depth_return(depth, values, values, 1.0)
        child.visit_count += 1

    assert child.depth_corrected_aggregate_uncertainty(
        calibrated=True, cap_at_aggregate=False
    ) == pytest.approx(child.uncertainty)


@pytest.mark.unittest
def test_depth_corrected_qstd_scales_residuals_and_conservative_cap_prevents_amplification():
    cfg = EasyDict(
        dict(
            num_simulations=2,
            pb_c_base=19652,
            pb_c_init=1.25,
            depth_normalized_backup_disagreement=True,
            depth_disagreement_max_depth=2,
            depth_disagreement_conservative_cap=True,
        )
    )
    normalizer = DepthNormalizedBackupDisagreementStats(cfg)
    normalizer.load_state_dict(
        {
            'raw_ratio': {1: 1.0, 2: 2.0},
            'calibrated_ratio': {1: 1.0, 2: 2.0},
            'total_group_count': {1: 100, 2: 100},
        }
    )
    mcts = EnsembleSharedTreeMCTS(
        cfg, _IdentityTransform(), _IdentityTransform(), depth_disagreement_stats=normalizer
    )
    parent = EnsembleSearchNode(0.0, [0], 2)
    parent.visit_count = 10
    child = EnsembleSearchNode(1.0, [0], 2)

    # Without correction the two exact-depth member residuals cancel completely.
    for depth, values in [
            (1, np.asarray([0.0, 4.0], dtype=np.float32)),
            (2, np.asarray([4.0, 0.0], dtype=np.float32)),
    ]:
        child.q_sum += values
        child.calibrated_q_sum += values
        child.has_calibrated_q = True
        child.record_search_depth_return(depth, values, values, normalizer.scale(depth))
        child.visit_count += 1

    uncapped = child.depth_corrected_aggregate_uncertainty(
        calibrated=True, cap_at_aggregate=False
    )
    components = mcts._score_child_components(
        parent, child, 'optimistic_shared_tree', 1.0, False, 0.0
    )

    assert child.uncertainty == pytest.approx(0.0)
    assert uncapped == pytest.approx(0.5)
    assert components['depth_corrected_uncertainty_uncapped'] == pytest.approx(0.5)
    assert components['depth_corrected_uncertainty'] == pytest.approx(0.0)
    assert components['depth_correction_cap_applied'] == 1.0
    assert components['uncertainty'] == pytest.approx(0.0)


@pytest.mark.unittest
def test_depth_disagreement_phase_stats_use_one_mean_per_node_action_depth_group():
    cfg = EasyDict(
        dict(
            num_simulations=4,
            depth_normalized_backup_disagreement=True,
            depth_disagreement_max_depth=3,
            depth_disagreement_ema_momentum=0.5,
            depth_disagreement_trim_fraction=0.0,
            depth_disagreement_min_groups=2,
            depth_disagreement_min_scale=0.1,
        )
    )
    normalizer = DepthNormalizedBackupDisagreementStats(cfg)
    root = EnsembleSearchNode(0.0, [0, 1], 2)
    root.expand({0: 0.5, 1: 0.5})

    # Each node/action/depth group contributes std of its member-mean return.
    # For action 0 at depth 2, pathwise stds are 3 and 5, but the member-mean
    # return [5, 3] has std 1 after the two paths partially cancel.
    observations = {
        0: {1: [[0.0, 2.0]], 2: [[0.0, 6.0], [10.0, 0.0]]},
        1: {1: [[0.0, 4.0]], 2: [[0.0, 12.0]]},
    }
    for action, depths in observations.items():
        child = root.children[action]
        for depth, return_values in depths.items():
            for values in return_values:
                values = np.asarray(values, dtype=np.float32)
                child.record_search_depth_return(depth, values, values, 1.0)
                child.visit_count += 1

    normalizer.observe_roots([root])
    report = {row['depth']: row for row in normalizer.finalize_phase(0)}

    # Depth-1 group Qstds are 1 and 2; depth-2 group Qstds are 1 and 6.
    assert report[1]['phase_group_count'] == 2
    assert report[1]['phase_calibrated_trimmed_mean'] == pytest.approx(1.5)
    assert report[2]['phase_group_count'] == 2
    assert report[2]['phase_calibrated_trimmed_mean'] == pytest.approx(3.5)
    assert report[2]['phase_calibrated_ratio_to_depth_1'] == pytest.approx(3.5 / 1.5)
    assert report[2]['ema_calibrated_ratio_to_depth_1'] == pytest.approx(3.5 / 1.5)
    assert report[2]['depth_scale_next_phase'] == pytest.approx(1.5 / 3.5)
    assert normalizer.scale(2) == pytest.approx(1.5 / 3.5)
    assert report[3]['baseline_updated'] == 0.0

    # The next phase has a much larger absolute scale but no depth inflation.
    # EMA is applied to the within-phase ratio, not to the absolute baselines.
    next_root = EnsembleSearchNode(0.0, [0, 1], 2)
    next_root.expand({0: 0.5, 1: 0.5})
    for action in [0, 1]:
        child = next_root.children[action]
        for depth in [1, 2]:
            values = np.asarray([0.0, 20.0], dtype=np.float32)
            child.record_search_depth_return(depth, values, values, normalizer.scale(depth))
            child.visit_count += 1
    normalizer.observe_roots([next_root])
    next_report = {row['depth']: row for row in normalizer.finalize_phase(1)}
    expected_ratio = 0.5 * (3.5 / 1.5) + 0.5 * 1.0
    assert next_report[2]['phase_calibrated_ratio_to_depth_1'] == pytest.approx(1.0)
    assert next_report[2]['ema_calibrated_ratio_to_depth_1'] == pytest.approx(expected_ratio)
    assert normalizer.scale(2) == pytest.approx(1.0 / expected_ratio)

    restored = DepthNormalizedBackupDisagreementStats(cfg)
    restored.load_state_dict(normalizer.state_dict())
    assert restored.scale(2) == pytest.approx(1.0 / expected_ratio)


@pytest.mark.unittest
def test_bounded_ensemble_scores_clip_to_member_envelopes_before_normalization():
    cfg = EasyDict(dict(pb_c_base=19652, pb_c_init=1.25))
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())
    stats = EnsembleQMinMaxStats(enabled=True)
    q_values = np.asarray([-200.0, -150.0, -100.0], dtype=np.float32)
    stats.update(q_values)
    parent = EnsembleSearchNode(0.0, [0], 3, q_min_max_stats=stats)
    parent.visit_count = 10
    child = EnsembleSearchNode(0.0, [0], 3, q_min_max_stats=stats)
    child.visit_count = 4
    child.q_sum = q_values * child.visit_count

    optimistic = mcts._score_child_components(
        parent, child, 'bounded_optimistic_shared_tree', 2.0, False, 0.0
    )
    pessimistic = mcts._score_child_components(
        parent, child, 'bounded_pessimistic_shared_tree', 2.0, False, 0.0
    )

    assert optimistic['unbounded_ensemble_q'] > -100.0
    assert optimistic['bounded_ensemble_q'] == pytest.approx(-100.0)
    assert optimistic['normalized_ensemble_q'] == pytest.approx(1.0)
    assert optimistic['ensemble_envelope_member_id'] == 2
    assert optimistic['ensemble_envelope_cap_applied'] == 1.0
    assert optimistic['uncertainty_score'] == pytest.approx(0.5)

    assert pessimistic['unbounded_ensemble_q'] < -200.0
    assert pessimistic['bounded_ensemble_q'] == pytest.approx(-200.0)
    assert pessimistic['normalized_ensemble_q'] == pytest.approx(0.0)
    assert pessimistic['ensemble_envelope_member_id'] == 0
    assert pessimistic['ensemble_envelope_cap_applied'] == 1.0
    assert pessimistic['uncertainty_score'] == pytest.approx(-0.5)


@pytest.mark.unittest
def test_bounded_root_diagnostics_compare_envelope_and_mean_argmax():
    cfg = EasyDict(dict(pb_c_base=19652, pb_c_init=1.25))
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())
    stats = EnsembleQMinMaxStats(enabled=True)
    root = EnsembleSearchNode(0.0, [0, 1], 3, q_min_max_stats=stats)
    root.expand({0: 0.5, 1: 0.5})
    root.visit_count = 10
    action_q_values = {
        0: np.asarray([-100.0, -100.0, -100.0], dtype=np.float32),
        1: np.asarray([-200.0, -200.0, 0.0], dtype=np.float32),
    }
    for action, q_values in action_q_values.items():
        child = root.children[action]
        child.visit_count = 5
        child.q_sum = q_values * child.visit_count
        stats.update(q_values)

    diagnostics = mcts._root_action_diagnostics(
        root, 'bounded_optimistic_shared_tree', 2.0, False, 0.0
    )

    assert {item['mean_score_best_action'] for item in diagnostics} == {0}
    assert {item['bounded_score_best_action'] for item in diagnostics} == {1}
    assert {item['bounded_changed_mean_score_argmax'] for item in diagnostics} == {1.0}
    assert {item['same_envelope_member_all_actions'] for item in diagnostics} == {0.0}


@pytest.mark.unittest
def test_ensemble_shared_tree_search_tracks_vector_disagreement():
    cfg = EasyDict(
        dict(
            num_simulations=3,
            pb_c_base=19652,
            pb_c_init=1.25,
            discount_factor=0.9,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
            ensemble_ucb_beta=1.0,
            ensemble_uncertainty_decay=False,
        )
    )
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())
    root_latent_states = torch.zeros(1, 2, 4)
    root_policy_logits = torch.zeros(1, 2, 1)
    root_values = torch.zeros(1, 2)

    output = mcts.search(
        model=_FakeEnsembleModel(),
        root_latent_states=root_latent_states,
        root_policy_logits=root_policy_logits,
        root_values=root_values,
        legal_actions=[[0]],
        add_exploration_noise=False,
        mode='optimistic_shared_tree',
    )

    assert output.distributions == [[3]]
    assert output.values[0] > 0
    assert output.root_uncertainties[0] > 0
    assert output.predicted_values == [0.0]
    assert len(output.root_action_diagnostics) == 1
    assert output.root_action_diagnostics[0][0]['action'] == 0
    for key in [
            'visit_count', 'prior', 'mean_q', 'normalized_mean_q', 'q_std', 'normalized_q_std',
            'unbounded_ensemble_q', 'bounded_ensemble_q', 'normalized_ensemble_q',
            'ensemble_envelope_q', 'ensemble_envelope_gap_from_mean', 'ensemble_envelope_member_id',
            'ensemble_envelope_cap_applied', 'ensemble_envelope_cap_adjustment', 'bounded_ensemble_mode',
            'bounded_score_best_action', 'mean_score_best_action', 'bounded_changed_mean_score_argmax',
            'same_envelope_member_all_actions',
            'q_min', 'q_max', 'q_range', 'predicted_reward_return_mean',
            'predicted_reward_return_std', 'leaf_value_contribution_mean',
            'leaf_value_contribution_std', 'prior_score', 'uncertainty_score',
            'unbounded_uncertainty_score',
            'intrinsic_score', 'intrinsic_value', 'intrinsic_reward_return_mean',
            'intrinsic_leaf_value_contribution_mean', 'normalized_intrinsic_value',
            'dynamics_disagreement_latent_reward',
            'dynamics_disagreement_projection_reward',
            'dynamics_disagreement_prediction_projection_reward',
            'total_score', 'q_values', 'predicted_reward_return_values',
            'leaf_value_contribution_values'
    ]:
        assert key in output.root_action_diagnostics[0][0]
    diagnostics = output.root_action_diagnostics[0][0]
    q_values = torch.tensor(diagnostics['q_values'])
    reward_return_values = torch.tensor(diagnostics['predicted_reward_return_values'])
    leaf_value_values = torch.tensor(diagnostics['leaf_value_contribution_values'])
    assert torch.allclose(q_values, reward_return_values + leaf_value_values)
    assert diagnostics['dynamics_disagreement_latent_reward'] > 0.0
    assert diagnostics['dynamics_disagreement_projection_reward'] > 0.0
    assert diagnostics['dynamics_disagreement_prediction_projection_reward'] > 0.0
    assert diagnostics['q_range'] > 0.0
    assert 0.0 <= diagnostics['normalized_mean_q'] <= 1.0
    assert diagnostics['normalized_q_std'] == pytest.approx(diagnostics['q_std'] / diagnostics['q_range'])
    assert diagnostics['uncertainty_score'] == pytest.approx(diagnostics['normalized_q_std'])


@pytest.mark.unittest
def test_ensemble_shared_tree_search_bootstraps_learned_intrinsic_value():
    cfg = EasyDict(
        dict(
            num_simulations=1,
            pb_c_base=19652,
            pb_c_init=1.25,
            discount_factor=0.9,
            rnd_intrinsic_discount_factor=0.5,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
            ensemble_ucb_beta=0.0,
            ensemble_uncertainty_decay=False,
            rnd_intrinsic_reward_weight=1.0,
        )
    )
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())
    root_latent_states = torch.zeros(1, 2, 4)
    root_policy_logits = torch.zeros(1, 2, 1)
    root_values = torch.zeros(1, 2)

    output = mcts.search(
        model=_FakeEnsembleModel(),
        root_latent_states=root_latent_states,
        root_policy_logits=root_policy_logits,
        root_values=root_values,
        legal_actions=[[0]],
        add_exploration_noise=False,
        mode='optimistic_shared_tree',
        use_model_intrinsic=True,
    )

    # mean intrinsic reward over members: 1.0, mean intrinsic leaf value: 2.5.
    assert output.intrinsic_values == [pytest.approx(2.25)]


@pytest.mark.unittest
def test_qstd_source_uses_learned_prefix_reward_and_leaf_value():
    cfg = EasyDict(
        dict(
            num_simulations=1,
            pb_c_base=19652,
            pb_c_init=1.25,
            discount_factor=0.9,
            rnd_intrinsic_discount_factor=0.5,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
            ensemble_ucb_beta=0.0,
            ensemble_uncertainty_decay=False,
            qstd_intrinsic_reward_weight=1.0,
            intrinsic_tree_leaf_value_source='default',
        )
    )
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())

    output = mcts.search(
        model=_FakeEnsembleModel(),
        root_latent_states=torch.zeros(1, 2, 4),
        root_policy_logits=torch.zeros(1, 2, 1),
        root_values=torch.zeros(1, 2),
        legal_actions=[[0]],
        add_exploration_noise=False,
        mode='mixed_reward_shared_tree',
        use_model_intrinsic=True,
        intrinsic_reward_source='qstd',
        intrinsic_reward_weight=1.0,
    )

    # Mean learned reward is 1.0 and mean learned leaf value is 2.5.
    assert output.intrinsic_values == [pytest.approx(2.25)]


@pytest.mark.unittest
def test_mixed_reward_tree_selects_using_joint_q():
    cfg = EasyDict(
        dict(
            num_simulations=20,
            pb_c_base=19652,
            pb_c_init=1.25,
            discount_factor=0.9,
            rnd_intrinsic_discount_factor=0.9,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
            ensemble_ucb_beta=0.0,
            ensemble_uncertainty_decay=False,
            ensemble_normalize_q=True,
            rnd_intrinsic_reward_weight=20.0,
        )
    )
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())

    output = mcts.search(
        model=_MixedRewardTradeoffModel(),
        root_latent_states=torch.zeros(1, 1, 2),
        root_policy_logits=torch.zeros(1, 1, 2),
        root_values=torch.zeros(1, 1),
        legal_actions=[[0, 1]],
        add_exploration_noise=False,
        mode='mixed_reward_shared_tree',
        use_model_intrinsic=True,
        intrinsic_reward_weight=20.0,
        force_root_action_coverage=True,
    )

    assert output.distributions[0][1] > output.distributions[0][0]
    diagnostics = {item['action']: item for item in output.root_action_diagnostics[0]}
    assert diagnostics[0]['mean_q'] > diagnostics[1]['mean_q']
    assert diagnostics[1]['mixed_mean_q'] > diagnostics[0]['mixed_mean_q']
    assert diagnostics[1]['score_mean_q'] == pytest.approx(diagnostics[1]['mixed_mean_q'])


@pytest.mark.unittest
def test_intrinsic_path_reward_callback_overrides_only_immediate_intrinsic_reward():
    cfg = EasyDict(
        dict(
            num_simulations=4,
            pb_c_base=19652,
            pb_c_init=1.25,
            discount_factor=0.9,
            rnd_intrinsic_discount_factor=0.9,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
            ensemble_ucb_beta=0.0,
            ensemble_uncertainty_decay=False,
            ensemble_normalize_q=True,
            rnd_intrinsic_reward_weight=1.0,
            intrinsic_tree_leaf_value_source='none',
            intrinsic_tree_normalize_q=True,
            intrinsic_tree_immediate_reward_weight=1.0,
            intrinsic_tree_leaf_value_weight=1.0,
        )
    )
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())
    observed_paths = []
    observed_latent_shapes = []

    def path_reward(root_indices, action_paths, imagined_latents):
        observed_paths.extend((int(root), list(path)) for root, path in zip(root_indices, action_paths))
        observed_latent_shapes.append(tuple(imagined_latents.shape))
        return np.asarray([10.0 + float(path[-1]) for path in action_paths], dtype=np.float32)

    output = mcts.search(
        model=_ActionIntrinsicModel(),
        root_latent_states=torch.zeros(1, 1, 2),
        root_policy_logits=torch.zeros(1, 1, 3),
        root_values=torch.zeros(1, 1),
        legal_actions=[[0, 1, 2]],
        add_exploration_noise=False,
        mode='intrinsic_shared_tree',
        use_model_intrinsic=True,
        intrinsic_reward_source='rnd',
        force_root_action_coverage=True,
        intrinsic_path_reward_fn=path_reward,
    )

    observed_path_set = {tuple(path) for _, path in observed_paths}
    assert {(0,), (1,), (2,)}.issubset(observed_path_set)
    assert any(len(path) == 2 for path in observed_path_set)
    assert {root for root, _ in observed_paths} == {0}
    assert all(shape[1:] == (1, 2) for shape in observed_latent_shapes)
    diagnostics = {item['action']: item for item in output.root_action_diagnostics[0]}
    for action in [0, 1, 2]:
        assert diagnostics[action]['intrinsic_immediate_reward'] == pytest.approx(10.0 + action)
        assert diagnostics[action]['intrinsic_landscape_query_reward'] == pytest.approx(10.0 + action)
        assert diagnostics[action]['intrinsic_leaf_value_raw_contribution_mean'] == pytest.approx(0.0)


@pytest.mark.unittest
def test_intrinsic_path_reward_transform_replaces_model_reward_before_modifier():
    cfg = EasyDict(
        dict(
            num_simulations=3,
            pb_c_base=19652,
            pb_c_init=1.25,
            discount_factor=0.9,
            rnd_intrinsic_discount_factor=0.9,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
            ensemble_ucb_beta=0.0,
            ensemble_uncertainty_decay=False,
            ensemble_normalize_q=True,
            rnd_intrinsic_reward_weight=1.0,
            intrinsic_tree_leaf_value_source='none',
            intrinsic_tree_normalize_q=True,
            intrinsic_tree_immediate_reward_weight=1.0,
            intrinsic_tree_leaf_value_weight=0.0,
        )
    )
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())

    def path_transform(_root_indices, action_paths, _imagined_latents, base_rewards):
        transformed = np.asarray(base_rewards, dtype=np.float32).copy()
        for row, path in enumerate(action_paths):
            if int(path[-1]) == 1:
                transformed[row] = 3.5
        return transformed

    output = mcts.search(
        model=_ActionIntrinsicModel(),
        root_latent_states=torch.zeros(1, 1, 2),
        root_policy_logits=torch.zeros(1, 1, 3),
        root_values=torch.zeros(1, 1),
        legal_actions=[[0, 1, 2]],
        add_exploration_noise=False,
        mode='intrinsic_shared_tree',
        use_model_intrinsic=True,
        intrinsic_reward_source='rnd',
        force_root_action_coverage=True,
        intrinsic_path_reward_transform_fn=path_transform,
    )

    diagnostics = {item['action']: item for item in output.root_action_diagnostics[0]}
    assert diagnostics[1]['intrinsic_reward_before_path_transform'] == pytest.approx(8.0)
    assert diagnostics[1]['intrinsic_reward_before_path_modifier'] == pytest.approx(3.5)
    assert diagnostics[1]['intrinsic_immediate_reward'] == pytest.approx(3.5)


@pytest.mark.unittest
def test_intrinsic_path_reward_modifier_multiplies_model_reward_without_replacing_it():
    cfg = EasyDict(
        dict(
            num_simulations=3,
            pb_c_base=19652,
            pb_c_init=1.25,
            discount_factor=0.9,
            rnd_intrinsic_discount_factor=0.9,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
            ensemble_ucb_beta=0.0,
            ensemble_uncertainty_decay=False,
            ensemble_normalize_q=True,
            rnd_intrinsic_reward_weight=1.0,
            intrinsic_tree_leaf_value_source='none',
            intrinsic_tree_normalize_q=True,
            intrinsic_tree_immediate_reward_weight=1.0,
            intrinsic_tree_leaf_value_weight=0.0,
        )
    )
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())

    def path_modifier(_root_indices, action_paths, _imagined_latents):
        return np.asarray(
            [0.25 if int(path[-1]) == 1 else 0.5 for path in action_paths],
            dtype=np.float32,
        )

    output = mcts.search(
        model=_ActionIntrinsicModel(),
        root_latent_states=torch.zeros(1, 1, 2),
        root_policy_logits=torch.zeros(1, 1, 3),
        root_values=torch.zeros(1, 1),
        legal_actions=[[0, 1, 2]],
        add_exploration_noise=False,
        mode='intrinsic_shared_tree',
        use_model_intrinsic=True,
        intrinsic_reward_source='rnd',
        force_root_action_coverage=True,
        intrinsic_path_reward_modifier_fn=path_modifier,
    )

    diagnostics = {item['action']: item for item in output.root_action_diagnostics[0]}
    assert diagnostics[1]['intrinsic_reward_before_path_modifier'] == pytest.approx(8.0)
    assert diagnostics[1]['intrinsic_path_reward_modifier'] == pytest.approx(0.25)
    assert diagnostics[1]['intrinsic_immediate_reward'] == pytest.approx(2.0)
    assert diagnostics[0]['intrinsic_reward_before_path_modifier'] == pytest.approx(0.0)
    assert diagnostics[0]['intrinsic_immediate_reward'] == pytest.approx(0.0)


@pytest.mark.unittest
def test_intrinsic_path_reward_bonus_adds_to_model_reward_after_modifier():
    cfg = EasyDict(
        dict(
            num_simulations=3,
            pb_c_base=19652,
            pb_c_init=1.25,
            discount_factor=0.9,
            rnd_intrinsic_discount_factor=0.9,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
            ensemble_ucb_beta=0.0,
            ensemble_uncertainty_decay=False,
            ensemble_normalize_q=True,
            rnd_intrinsic_reward_weight=1.0,
            intrinsic_tree_leaf_value_source='none',
            intrinsic_tree_normalize_q=True,
            intrinsic_tree_immediate_reward_weight=1.0,
            intrinsic_tree_leaf_value_weight=0.0,
        )
    )
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())

    def path_modifier(_root_indices, _action_paths, _imagined_latents):
        return np.full(len(_action_paths), 0.5, dtype=np.float32)

    def path_bonus(_root_indices, action_paths, _imagined_latents):
        return np.asarray(
            [1.5 if int(path[-1]) == 1 else 0.25 for path in action_paths],
            dtype=np.float32,
        )

    output = mcts.search(
        model=_ActionIntrinsicModel(),
        root_latent_states=torch.zeros(1, 1, 2),
        root_policy_logits=torch.zeros(1, 1, 3),
        root_values=torch.zeros(1, 1),
        legal_actions=[[0, 1, 2]],
        add_exploration_noise=False,
        mode='intrinsic_shared_tree',
        use_model_intrinsic=True,
        intrinsic_reward_source='rnd',
        force_root_action_coverage=True,
        intrinsic_path_reward_modifier_fn=path_modifier,
        intrinsic_path_reward_bonus_fn=path_bonus,
    )

    diagnostics = {item['action']: item for item in output.root_action_diagnostics[0]}
    assert diagnostics[1]['intrinsic_reward_before_path_modifier'] == pytest.approx(8.0)
    assert diagnostics[1]['intrinsic_reward_before_path_bonus'] == pytest.approx(4.0)
    assert diagnostics[1]['intrinsic_path_reward_bonus'] == pytest.approx(1.5)
    assert diagnostics[1]['intrinsic_immediate_reward'] == pytest.approx(5.5)
    assert diagnostics[0]['intrinsic_reward_before_path_bonus'] == pytest.approx(0.0)
    assert diagnostics[0]['intrinsic_immediate_reward'] == pytest.approx(0.25)


@pytest.mark.unittest
def test_intrinsic_leaf_value_modifier_multiplies_only_the_leaf_bootstrap():
    cfg = EasyDict(
        dict(
            num_simulations=3,
            pb_c_base=19652,
            pb_c_init=1.25,
            discount_factor=0.9,
            rnd_intrinsic_discount_factor=0.9,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
            ensemble_ucb_beta=0.0,
            ensemble_uncertainty_decay=False,
            ensemble_normalize_q=True,
            rnd_intrinsic_reward_weight=1.0,
            intrinsic_tree_leaf_value_source='intrinsic_value',
            intrinsic_tree_normalize_q=True,
            intrinsic_tree_immediate_reward_weight=1.0,
            intrinsic_tree_leaf_value_weight=1.0,
        )
    )
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())

    def leaf_modifier(_root_indices, action_paths, _imagined_latents):
        return np.full(len(action_paths), 0.25, dtype=np.float32)

    output = mcts.search(
        model=_FakeEnsembleModel(),
        root_latent_states=torch.zeros(1, 2, 2),
        root_policy_logits=torch.zeros(1, 2, 2),
        root_values=torch.zeros(1, 2),
        legal_actions=[[0, 1]],
        add_exploration_noise=False,
        mode='intrinsic_shared_tree',
        use_model_intrinsic=True,
        intrinsic_reward_source='rnd',
        force_root_action_coverage=True,
        intrinsic_leaf_value_modifier_fn=leaf_modifier,
    )

    diagnostics = output.root_action_diagnostics[0]
    visited = [row for row in diagnostics if row['visit_count'] > 0]
    assert visited
    for row in visited:
        assert row['intrinsic_leaf_value_modifier_mean'] == pytest.approx(0.25)
        assert row[
            'intrinsic_leaf_value_raw_contribution_mean'
        ] == pytest.approx(
            0.25
            * row['intrinsic_leaf_value_before_modifier_contribution_mean']
        )


@pytest.mark.unittest
def test_intrinsic_shared_tree_mode_uses_intrinsic_stats():
    cfg = EasyDict(
        dict(
            num_simulations=4,
            pb_c_base=19652,
            pb_c_init=1.25,
            discount_factor=0.9,
            rnd_intrinsic_discount_factor=0.5,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
            ensemble_ucb_beta=100.0,
            ensemble_uncertainty_decay=False,
            rnd_intrinsic_reward_weight=0.05,
        )
    )
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())
    root_latent_states = torch.zeros(1, 2, 4)
    root_policy_logits = torch.zeros(1, 2, 2)
    root_values = torch.zeros(1, 2)

    output = mcts.search(
        model=_FakeEnsembleModel(),
        root_latent_states=root_latent_states,
        root_policy_logits=root_policy_logits,
        root_values=root_values,
        legal_actions=[[0, 1]],
        add_exploration_noise=False,
        mode='intrinsic_shared_tree',
        use_model_intrinsic=True,
        intrinsic_reward_weight=1.0,
        return_tree_diagnostics=True,
        tree_diagnostics_max_depth=3,
        tree_diagnostics_max_nodes=32,
    )

    diagnostics = output.root_action_diagnostics[0]
    assert {item['search_mode'] for item in diagnostics} == {'intrinsic_shared_tree'}
    assert all(item['uncertainty_score'] == 0.0 for item in diagnostics)
    assert max(item['intrinsic_value'] for item in diagnostics) > 0.0
    assert sum(output.distributions[0]) == 4
    assert output.tree_diagnostics is not None
    tree = output.tree_diagnostics[0]
    assert tree['num_simulations_recorded'] == 4
    assert tree['nodes_recorded'] > 1
    assert tree['simulation_traces'][0]['path_actions']
    assert 'backed_up_intrinsic_reward_return' in tree['simulation_traces'][0]
    assert 'backed_up_intrinsic_leaf_value_contribution' in tree['simulation_traces'][0]
    assert 'intrinsic_reward_return_mean' in tree['nodes'][0]
    assert 'intrinsic_leaf_value_contribution_mean' in tree['nodes'][0]


@pytest.mark.unittest
def test_intrinsic_shared_tree_can_use_dynamics_disagreement_as_reward():
    cfg = EasyDict(
        dict(
            num_simulations=2,
            pb_c_base=19652,
            pb_c_init=1.25,
            discount_factor=0.9,
            rnd_intrinsic_discount_factor=0.99,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
            ensemble_ucb_beta=0.0,
            ensemble_uncertainty_decay=False,
            rnd_intrinsic_reward_weight=0.0,
            intrinsic_tree_normalize_q=True,
            dynamics_disagreement_normalize_latent=False,
        )
    )
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())
    root_latent_states = torch.zeros(1, 2, 4)
    root_policy_logits = torch.zeros(1, 2, 1)
    root_values = torch.zeros(1, 2)

    output = mcts.search(
        model=_FakeEnsembleModel(),
        root_latent_states=root_latent_states,
        root_policy_logits=root_policy_logits,
        root_values=root_values,
        legal_actions=[[0]],
        add_exploration_noise=False,
        mode='intrinsic_shared_tree',
        use_model_intrinsic=False,
        intrinsic_reward_source='dynamics_disagreement',
        intrinsic_reward_weight=1.0,
    )

    assert output.intrinsic_values[0] > 0.0
    diagnostics = output.root_action_diagnostics[0][0]
    assert diagnostics['intrinsic_reward_return_mean'] > 0.0
    assert diagnostics['intrinsic_leaf_value_contribution_mean'] == 0.0


@pytest.mark.unittest
def test_dynamics_disagreement_reward_supports_l2_metric():
    cfg = EasyDict(
        dict(
            dynamics_disagreement_normalize_latent=False,
            dynamics_disagreement_distance_metric='l2',
            dynamics_disagreement_reward_clip=None,
        )
    )
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())
    flat_latents = torch.tensor([[0.0, 0.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]])

    l2_reward = mcts._dynamics_disagreement_reward_np(flat_latents, batch_size=1, ensemble_size=2)
    assert l2_reward.tolist() == pytest.approx([5.0])

    cfg.dynamics_disagreement_distance_metric = 'mse'
    mse_reward = mcts._dynamics_disagreement_reward_np(flat_latents, batch_size=1, ensemble_size=2)
    assert mse_reward.tolist() == pytest.approx([6.25])


@pytest.mark.unittest
def test_intrinsic_shared_tree_normalized_q_keeps_root_exploration_alive():
    cfg = EasyDict(
        dict(
            num_simulations=25,
            pb_c_base=19652,
            pb_c_init=1.25,
            discount_factor=0.9,
            rnd_intrinsic_discount_factor=0.99,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
            ensemble_ucb_beta=0.0,
            ensemble_uncertainty_decay=False,
            rnd_intrinsic_reward_weight=1.0,
            intrinsic_tree_normalize_q=True,
        )
    )
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())
    root_latent_states = torch.zeros(1, 1, 2)
    root_policy_logits = torch.tensor([[[0.0, 0.35, 0.0]]])
    root_values = torch.zeros(1, 1)

    output = mcts.search(
        model=_ActionIntrinsicModel(),
        root_latent_states=root_latent_states,
        root_policy_logits=root_policy_logits,
        root_values=root_values,
        legal_actions=[[0, 1, 2]],
        add_exploration_noise=False,
        mode='intrinsic_shared_tree',
        use_model_intrinsic=True,
        intrinsic_reward_weight=1.0,
    )

    distribution = output.distributions[0]
    assert distribution[1] < 25
    assert distribution[0] + distribution[2] > 0
    diagnostics_by_action = {
        item['action']: item
        for item in output.root_action_diagnostics[0]
    }
    assert diagnostics_by_action[1]['intrinsic_value'] > 1.0
    assert 0.0 <= diagnostics_by_action[1]['normalized_intrinsic_value'] <= 1.0


@pytest.mark.unittest
def test_intrinsic_tree_uniform_priors_and_forced_root_coverage_visit_every_action():
    cfg = EasyDict(
        dict(
            num_simulations=3,
            pb_c_base=19652,
            pb_c_init=1.25,
            discount_factor=0.9,
            rnd_intrinsic_discount_factor=0.99,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
            ensemble_ucb_beta=0.0,
            ensemble_uncertainty_decay=False,
            rnd_intrinsic_reward_weight=1.0,
            intrinsic_tree_normalize_q=True,
        )
    )
    mcts = EnsembleSharedTreeMCTS(cfg, _IdentityTransform(), _IdentityTransform())
    root_latent_states = torch.zeros(1, 1, 2)
    root_policy_logits = torch.tensor([[[-20.0, 20.0, -20.0]]])
    root_values = torch.zeros(1, 1)

    output = mcts.search(
        model=_ActionIntrinsicModel(),
        root_latent_states=root_latent_states,
        root_policy_logits=root_policy_logits,
        root_values=root_values,
        legal_actions=[[0, 1, 2]],
        add_exploration_noise=False,
        mode='intrinsic_shared_tree',
        use_model_intrinsic=True,
        intrinsic_reward_weight=1.0,
        policy_prior_mode='uniform',
        force_root_action_coverage=True,
    )

    assert output.distributions[0] == [1, 1, 1]
    for action_terms in output.root_action_diagnostics[0]:
        assert action_terms['prior'] == pytest.approx(1.0 / 3.0)
        assert action_terms['policy_prior_mode'] == 'uniform'
        assert action_terms['force_root_action_coverage'] == 1.0
        assert action_terms['root_forced_coverage_complete'] == 1.0
