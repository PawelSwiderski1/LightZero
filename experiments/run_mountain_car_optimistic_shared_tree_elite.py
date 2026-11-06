import argparse

from run_mountain_car_optimistic_shared_tree import build_wrapper_parser, run


def build_elite_parser() -> argparse.ArgumentParser:
    parser = build_wrapper_parser()
    parser.description = (
        "Reproduce the strong optimistic shared-tree MountainCar run with elite Monte Carlo replay."
    )
    parser.set_defaults(
        exp_name="data_ensemble_muzero/mountain_car_optimistic_shared_tree_elite_stratified_seed0",
        root_dirichlet_alpha=0.3,
        record_collect_video=True,
    )
    parser.add_argument("--elite-buffer-size", type=int, default=20)
    parser.add_argument("--elite-sample-ratio", type=float, default=0.25)
    parser.add_argument("--elite-min-return", type=float, default=-199.0)
    parser.add_argument(
        "--elite-start-bin-edges",
        type=str,
        default="-0.55,-0.50,-0.45",
        help="Comma-separated MountainCar start-position boundaries for stratified elite replay.",
    )
    return parser


def elite_preset_overrides(args: argparse.Namespace) -> dict:
    return dict(
        elite_replay=True,
        elite_buffer_size=args.elite_buffer_size,
        elite_sample_ratio=args.elite_sample_ratio,
        elite_min_return=args.elite_min_return,
        elite_mc_value_target=True,
        elite_start_stratification=True,
        elite_start_observation_index=0,
        elite_start_bin_edges=args.elite_start_bin_edges,
        save_elite_trajectories=True,
        ensemble_value_calibration=False,
        eval_freq=1000,
    )


def main() -> None:
    args = build_elite_parser().parse_args()
    run(args, preset_overrides=elite_preset_overrides(args))


if __name__ == "__main__":
    main()
