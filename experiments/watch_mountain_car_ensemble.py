import argparse
import copy
import time
import warnings
from pathlib import Path


def quiet_gym_video_warnings() -> None:
    warnings.simplefilter("ignore", DeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"gym(\.|$)")
    warnings.filterwarnings("ignore", message=r".*Box bound precision lowered.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=r".*Overwriting existing videos.*", category=UserWarning)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record MountainCar videos for an ensemble MuZero checkpoint.")
    parser.add_argument("--model-path", type=str, default=None, help="Checkpoint to load. Omit to watch an untrained policy.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--video-dir", type=str, default="./video/mountain_car_ensemble")
    parser.add_argument("--exp-name", type=str, default="lz_result/eval/mountain_car_ensemble_video")
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--num-simulations", type=int, default=25)
    parser.add_argument("--random-prior-scale", type=float, default=0.0)
    parser.add_argument(
        "--ensemble-eval-search-mode",
        choices=[
            "member0", "mean_shared_tree", "optimistic_shared_tree", "pessimistic_shared_tree",
            "bounded_optimistic_shared_tree", "bounded_pessimistic_shared_tree",
        ],
        default="member0",
        help="Eval-time action selection used for the recorded policy.",
    )
    parser.add_argument(
        "--ensemble-eval-ucb-beta",
        type=float,
        default=None,
        help="Eval-time ensemble uncertainty weight. Defaults to the config value when omitted.",
    )
    parser.add_argument(
        "--ensemble-eval-uncertainty-decay",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Eval-time uncertainty decay. Defaults to the config value when omitted.",
    )
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--clean-video-dir",
        action="store_true",
        help="Delete existing mp4/meta files in --video-dir before recording.",
    )
    return parser


def clean_video_dir(video_dir: str) -> None:
    path = Path(video_dir)
    path.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.mp4", "*.meta.json"):
        for file_path in path.glob(pattern):
            file_path.unlink()


def build_config(args: argparse.Namespace):
    from zoo.classic_control.mountain_car.config.mountain_car_ensemble_muzero_config import create_config, main_config

    cfg = copy.deepcopy(main_config)
    create_cfg = copy.deepcopy(create_config)

    create_cfg.env_manager.type = "base"

    cfg.exp_name = args.exp_name
    cfg.env.evaluator_env_num = 1
    cfg.env.n_evaluator_episode = args.episodes
    cfg.env.replay_path = args.video_dir

    cfg.policy.evaluator_env_num = 1
    cfg.policy.cuda = args.cuda
    cfg.policy.num_simulations = args.num_simulations
    cfg.policy.thompson_sampling = False
    cfg.policy.ensemble_search_mode = "none"
    cfg.policy.ensemble_eval_search_mode = args.ensemble_eval_search_mode
    cfg.policy.ensemble_eval_ucb_beta = args.ensemble_eval_ucb_beta
    cfg.policy.ensemble_eval_uncertainty_decay = args.ensemble_eval_uncertainty_decay
    cfg.policy.model.ensemble_size = args.ensemble_size
    cfg.policy.model.random_prior_scale = args.random_prior_scale

    return cfg, create_cfg


def main() -> None:
    quiet_gym_video_warnings()
    args = build_parser().parse_args()
    if args.clean_video_dir:
        clean_video_dir(args.video_dir)
    cfg, create_cfg = build_config(args)
    run_start_time = time.time()

    print("MountainCar ensemble video eval")
    print(f"  model_path: {args.model_path}")
    print(f"  seed: {args.seed}")
    print(f"  episodes: {args.episodes}")
    print(f"  video_dir: {args.video_dir}")
    print(f"  ensemble_size: {cfg.policy.model.ensemble_size}")
    print(f"  random_prior_scale: {cfg.policy.model.random_prior_scale}")
    print(f"  ensemble_eval_search_mode: {cfg.policy.ensemble_eval_search_mode}")
    print(f"  ensemble_eval_ucb_beta: {cfg.policy.ensemble_eval_ucb_beta}")
    print(f"  ensemble_eval_uncertainty_decay: {cfg.policy.ensemble_eval_uncertainty_decay}")
    print(f"  num_simulations: {cfg.policy.num_simulations}")
    print(f"  cuda: {cfg.policy.cuda}")
    print("  note: eval uses the policy's eval mode, not training-time exploration mode.")

    from lzero.entry.eval_muzero import eval_muzero

    quiet_gym_video_warnings()
    returns_mean, returns = eval_muzero(
        [cfg, create_cfg],
        seed=args.seed,
        model_path=args.model_path,
        num_episodes_each_seed=1,
        print_seed_details=True,
    )

    print("=" * 20)
    print(f"Mean return: {returns_mean}")
    print(f"Returns: {returns}")
    print(f"Videos written under: {args.video_dir}")
    skipped_tiny_files = 0
    for path in sorted(Path(args.video_dir).glob("*.mp4")):
        if path.stat().st_mtime < run_start_time - 1:
            continue
        if path.stat().st_size <= 5000:
            skipped_tiny_files += 1
            continue
        print(f"  {path.name}: {path.stat().st_size} bytes")
    if skipped_tiny_files:
        print(f"  skipped {skipped_tiny_files} tiny shutdown video file(s)")
    print("=" * 20)


if __name__ == "__main__":
    main()
