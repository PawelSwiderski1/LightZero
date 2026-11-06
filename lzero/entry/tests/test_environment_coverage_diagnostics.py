import numpy as np
from easydict import EasyDict

from lzero.entry.train_ensemble_muzero import (
    EnvironmentCoverageTracker,
    _write_rnd_episodic_memory_diagnostics,
)


class _Segment:
    def __init__(self, initial_state, states):
        self.environment_initial_state = initial_state
        self.environment_state_segment = np.asarray(states, dtype=np.int64)

    def __len__(self):
        return len(self.environment_state_segment)


def _collection(segment):
    return [[segment], [{'segment_length': len(segment)}]]


def _tracker():
    return EnvironmentCoverageTracker({
        'environment_coverage_diagnostics': True,
        'environment_coverage_cell_diagnostics': True,
        'environment_reachable_position_count': 10,
        'environment_room_count': 2,
        'environment_dead_end_room_ids': ((0, 1),),
    })


def test_environment_coverage_tracker_persists_exact_phase_cells():
    tracker = _tracker()
    phase_zero = _Segment(
        (1, 1, 0, 0, 0),
        [
            (2, 1, 0, 0, 0),
            (2, 1, 1, 0, 0),
            (1, 1, 2, 0, 0),
        ],
    )

    stats = tracker.annotate_collection(_collection(phase_zero), collection_phase=0)
    rows = {(row['x'], row['y']): row for row in tracker.latest_cell_rows()}

    assert stats['phase_position_count'] == 2
    assert stats['phase_pose_count'] == 4
    assert stats['new_global_position_count'] == 2
    assert rows[(1, 1)]['phase_visit_count'] == 2
    assert rows[(1, 1)]['new_global_position'] == 1
    assert rows[(1, 1)]['first_visit_phase'] == 0
    assert rows[(1, 1)]['phase_heading_mask'] == 0b0101
    assert rows[(2, 1)]['phase_heading_mask'] == 0b0011
    assert rows[(2, 1)]['phase_heading_0_visit_count'] == 1
    assert rows[(2, 1)]['phase_heading_1_visit_count'] == 1


def test_environment_coverage_tracker_retains_cumulative_cell_state():
    tracker = _tracker()
    tracker.annotate_collection(
        _collection(_Segment(
            (1, 1, 0, 0, 0),
            [(2, 1, 0, 0, 0), (2, 1, 1, 0, 0)],
        )),
        collection_phase=0,
    )
    phase_one = _Segment(
        (2, 1, 1, 0, 0),
        [(3, 1, 1, 0, 1), (3, 1, 1, 0, 1)],
    )

    stats = tracker.annotate_collection(_collection(phase_one), collection_phase=1)
    rows = {(row['x'], row['y']): row for row in tracker.latest_cell_rows()}

    assert stats['phase_position_count'] == 2
    assert stats['new_global_position_count'] == 1
    assert stats['global_position_count'] == 3
    assert stats['new_global_room_count'] == 1
    assert stats['global_dead_end_room_count'] == 1
    assert rows[(2, 1)]['new_global_position'] == 0
    assert rows[(2, 1)]['first_visit_phase'] == 0
    assert rows[(2, 1)]['cumulative_visit_count'] == 3
    assert rows[(2, 1)]['new_global_heading_mask'] == 0
    assert rows[(3, 1)]['new_global_position'] == 1
    assert rows[(3, 1)]['first_visit_phase'] == 1
    assert rows[(3, 1)]['room_col'] == 1


def test_environment_coverage_tracker_skips_cell_rows_when_disabled():
    tracker = EnvironmentCoverageTracker({
        'environment_coverage_diagnostics': True,
        'environment_coverage_cell_diagnostics': False,
    })
    segment = _Segment((1, 1, 0, 0, 0), [(2, 1, 0, 0, 0)])

    tracker.annotate_collection(_collection(segment), collection_phase=0)

    assert tracker.latest_cell_rows() == []


def test_episodic_memory_summary_uses_selected_action_for_one_step_knn(tmp_path):
    class _Policy:
        @staticmethod
        def pop_rnd_episodic_memory_diagnostics():
            return [
                {
                    'selected': 1,
                    'one_step_reward_factor': 0.5,
                    'tree_path_match_fraction': 0.0,
                    'one_step_pseudo_count': 2.0,
                    'one_step_normalized_knn_distance_mean': 0.25,
                    'one_step_knn_warmup': 1.0,
                },
                {
                    'selected': 0,
                    'one_step_reward_factor': 1.0,
                    'tree_path_match_fraction': 0.0,
                    'one_step_pseudo_count': 0.0,
                    'one_step_normalized_knn_distance_mean': np.nan,
                    'one_step_knn_warmup': 0.0,
                },
            ]

    class _Logger:
        def __init__(self):
            self.scalars = {}

        def add_scalar(self, name, value, step):
            self.scalars[name] = (value, step)

    cfg = EasyDict(
        exp_name='unused',
        policy=EasyDict(
            rnd_episodic_memory=True,
            diagnostics_dir=str(tmp_path),
        ),
    )
    logger = _Logger()

    _write_rnd_episodic_memory_diagnostics(
        cfg, _Policy(), logger, collection_phase=3
    )

    assert logger.scalars[
        'rnd_episodic_memory/one_step_pseudo_count_mean'
    ] == (2.0, 3)
    assert logger.scalars[
        'rnd_episodic_memory/one_step_normalized_knn_distance_mean'
    ] == (0.25, 3)
    assert logger.scalars[
        'rnd_episodic_memory/one_step_knn_warmup_fraction'
    ] == (1.0, 3)
