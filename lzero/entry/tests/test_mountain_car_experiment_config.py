import sys
from pathlib import Path


EXPERIMENTS_DIR = Path(__file__).resolve().parents[3] / 'experiments'
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from run_mountain_car_ensemble import build_config, build_parser  # noqa: E402
from run_mountain_car_full_intrinsic_rnd import (  # noqa: E402
    build_parser as build_full_intrinsic_parser,
    build_preset_overrides,
)
from run_mountain_car_rnd_landscape_query import (  # noqa: E402
    build_parser as build_landscape_query_parser,
    build_preset_overrides as build_landscape_query_overrides,
)
from run_mountain_car_rnd_landscape_query_centered import (  # noqa: E402
    build_parser as build_centered_query_parser,
    build_preset_overrides as build_centered_query_overrides,
)
from run_mountain_car_rnd_landscape_query_no_diversity import (  # noqa: E402
    build_parser as build_no_diversity_query_parser,
    build_preset_overrides as build_no_diversity_query_overrides,
)
from run_mountain_car_rnd_landscape_query_no_context import (  # noqa: E402
    build_parser as build_no_context_query_parser,
    build_preset_overrides as build_no_context_query_overrides,
)
from run_mountain_car_rnd_landscape_query_detached_world_model import (  # noqa: E402
    build_parser as build_detached_query_parser,
    build_preset_overrides as build_detached_query_overrides,
)
from run_mountain_car_rnd_query_auxiliary_only import (  # noqa: E402
    build_parser as build_auxiliary_only_parser,
    build_preset_overrides as build_auxiliary_only_overrides,
)


def test_full_intrinsic_preset_enables_post_update_rnd_snapshot_refresh():
    wrapper_args = build_full_intrinsic_parser().parse_args([])
    overrides = build_preset_overrides(wrapper_args)
    assert overrides['rnd_target_snapshot_refresh_after_fresh_update'] is True
    assert overrides['rnd_initialization_seed'] == 100000

    args = build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    cfg, _ = build_config(args)

    assert cfg.policy.rnd_target_snapshot_refresh_per_collect is True
    assert cfg.policy.rnd_target_snapshot_refresh_after_fresh_update is True


def test_rnd_landscape_query_preset_is_separate_and_uses_live_snapshot_queries():
    wrapper_args = build_landscape_query_parser().parse_args([])
    overrides = build_landscape_query_overrides(wrapper_args)

    assert overrides['intrinsic_only_training'] is True
    assert overrides['rnd_landscape_query'] is True
    assert overrides['rnd_landscape_query_count'] == 16
    assert overrides['rnd_landscape_query_loss_weight'] == 1.0
    assert overrides['rnd_landscape_query_min_snapshots'] == 2
    assert overrides['rnd_landscape_query_diversity_weight'] == 0.01
    assert overrides['rnd_landscape_query_centered_prediction_weight'] == 0.0
    assert overrides['rnd_target_snapshot_refresh_after_fresh_update'] is True
    assert overrides['rnd_initialization_seed'] == 100000
    assert overrides['rnd_landscape_query_initialization_seed'] == 200000
    assert overrides['rnd_landscape_query_reader_use_context'] is True
    assert overrides['rnd_landscape_query_train_world_model'] is True

    args = build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    cfg, _ = build_config(args)

    assert cfg.policy.rnd_landscape_query is True
    assert cfg.policy.rnd_landscape_query_loss_weight == 1.0
    assert cfg.policy.collection_policy_mode == 'intrinsic_tree_only'
    assert cfg.policy.intrinsic_tree_leaf_value_source == 'intrinsic_value'


def test_rnd_landscape_query_ablation_presets_are_isolated():
    no_diversity_args = build_no_diversity_query_parser().parse_args([])
    no_diversity = build_no_diversity_query_overrides(no_diversity_args)
    assert no_diversity['rnd_landscape_query_diversity_weight'] == 0.0
    assert no_diversity['rnd_landscape_query_centered_prediction_weight'] == 0.0

    centered_args = build_centered_query_parser().parse_args([])
    centered = build_centered_query_overrides(centered_args)
    assert centered['rnd_landscape_query_diversity_weight'] == 0.0
    assert centered['rnd_landscape_query_centered_prediction_weight'] == 1.0

    no_context_args = build_no_context_query_parser().parse_args([])
    no_context = build_no_context_query_overrides(no_context_args)
    assert no_context['rnd_landscape_query_reader_use_context'] is False
    assert no_context['rnd_landscape_query_train_world_model'] is True
    assert no_context['rnd_landscape_query_diversity_weight'] == 0.0

    detached_args = build_detached_query_parser().parse_args([])
    detached = build_detached_query_overrides(detached_args)
    assert detached['rnd_landscape_query_reader_use_context'] is True
    assert detached['rnd_landscape_query_train_world_model'] is False


def test_auxiliary_reader_factorial_flags_only_change_decoder_and_target():
    parser = build_auxiliary_only_parser()
    shared_log = build_auxiliary_only_overrides(parser.parse_args([]))
    member_raw = build_auxiliary_only_overrides(
        parser.parse_args([
            '--rnd-landscape-query-member-specific-reader',
            '--rnd-landscape-query-target-transform',
            'raw',
        ])
    )
    shared_normalized_raw = build_auxiliary_only_overrides(
        parser.parse_args([
            '--rnd-landscape-query-target-transform',
            'normalized_raw',
            '--rnd-landscape-query-normalized-raw-reference-size',
            '4096',
            '--rnd-landscape-query-normalized-raw-clip',
            '8',
            '--rnd-landscape-query-gradient-diagnostics-interval',
            '100',
        ])
    )

    assert shared_log['rnd_landscape_query_member_specific_reader'] is False
    assert shared_log['rnd_landscape_query_target_transform'] == 'log'
    assert member_raw['rnd_landscape_query_member_specific_reader'] is True
    assert member_raw['rnd_landscape_query_target_transform'] == 'raw'
    assert shared_normalized_raw['rnd_landscape_query_target_transform'] == (
        'normalized_raw'
    )
    assert shared_normalized_raw[
        'rnd_landscape_query_normalized_raw_reference_size'
    ] == 4096
    assert shared_normalized_raw['rnd_landscape_query_normalized_raw_clip'] == 8.0
    assert shared_normalized_raw[
        'rnd_landscape_query_gradient_diagnostics_interval'
    ] == 100
    for overrides in [shared_log, member_raw, shared_normalized_raw]:
        assert overrides['rnd_landscape_query_reader_use_context'] is False
        assert overrides['rnd_landscape_query_train_world_model'] is True
        assert overrides['rnd_landscape_query_use_tree_rewards'] is False
        assert overrides['rnd_landscape_query_diversity_weight'] == 0.0

    args = build_parser().parse_args([])
    for key, value in shared_normalized_raw.items():
        setattr(args, key, value)
    cfg, _ = build_config(args)
    assert cfg.policy.rnd_landscape_query_member_specific_reader is False
    assert cfg.policy.rnd_landscape_query_target_transform == 'normalized_raw'
    assert cfg.policy.rnd_landscape_query_normalized_raw_reference_size == 4096
    assert cfg.policy.rnd_landscape_query_normalized_raw_clip == 8.0
    assert cfg.policy.rnd_landscape_query_gradient_diagnostics_interval == 100
