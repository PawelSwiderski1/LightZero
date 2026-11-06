import argparse
import copy
import time
import warnings
from functools import partial
from pathlib import Path


def quiet_gym_video_warnings() -> None:
    warnings.simplefilter("ignore", DeprecationWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"gym(\.|$)")
    warnings.filterwarnings("ignore", message=r".*Box bound precision lowered.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=r".*Overwriting existing videos.*", category=UserWarning)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record actual MountainCar collect-mode episodes.")
    parser.add_argument("--model-path", type=str, default=None, help="Checkpoint to load. Omit to collect with an untrained policy.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--video-dir", type=str, default="./video/mountain_car_collect")
    parser.add_argument("--exp-name", type=str, default="lz_result/collect/mountain_car_ensemble_video")
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--num-simulations", type=int, default=25)
    parser.add_argument("--random-prior-scale", type=float, default=0.0)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ensemble-search-mode",
        choices=[
            "none", "thompson", "optimistic_shared_tree", "pessimistic_shared_tree",
            "bounded_optimistic_shared_tree", "bounded_pessimistic_shared_tree",
        ],
        default="optimistic_shared_tree",
    )
    parser.add_argument("--ensemble-ucb-beta", type=float, default=1.0)
    parser.add_argument("--ensemble-uncertainty-decay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--epsilon", type=float, default=0.0)
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
    cfg.env.collector_env_num = 1
    cfg.env.evaluator_env_num = 1
    cfg.env.n_evaluator_episode = 1

    cfg.policy.collector_env_num = 1
    cfg.policy.evaluator_env_num = 1
    cfg.policy.n_episode = args.episodes
    cfg.policy.cuda = args.cuda
    cfg.policy.num_simulations = args.num_simulations
    cfg.policy.model.ensemble_size = args.ensemble_size
    cfg.policy.model.random_prior_scale = args.random_prior_scale
    cfg.policy.ensemble_search_mode = args.ensemble_search_mode
    cfg.policy.ensemble_ucb_beta = args.ensemble_ucb_beta
    cfg.policy.ensemble_uncertainty_decay = args.ensemble_uncertainty_decay
    cfg.policy.thompson_sampling = args.ensemble_search_mode == "thompson"
    cfg.policy.use_wandb = False

    return cfg, create_cfg


def print_video_summary(video_dir: str, run_start_time: float) -> None:
    skipped_tiny_files = 0
    for path in sorted(Path(video_dir).glob("*.mp4")):
        if path.stat().st_mtime < run_start_time - 1:
            continue
        if path.stat().st_size <= 5000:
            skipped_tiny_files += 1
            continue
        print(f"  {path.name}: {path.stat().st_size} bytes")
    if skipped_tiny_files:
        print(f"  skipped {skipped_tiny_files} tiny shutdown video file(s)")


def main() -> None:
    quiet_gym_video_warnings()
    args = build_parser().parse_args()
    if args.clean_video_dir:
        clean_video_dir(args.video_dir)
    cfg, create_cfg = build_config(args)
    run_start_time = time.time()

    print("MountainCar ensemble collect-mode video")
    print(f"  model_path: {args.model_path}")
    print(f"  seed: {args.seed}")
    print(f"  episodes: {args.episodes}")
    print(f"  video_dir: {args.video_dir}")
    print(f"  ensemble_size: {cfg.policy.model.ensemble_size}")
    print(f"  random_prior_scale: {cfg.policy.model.random_prior_scale}")
    print(f"  ensemble_search_mode: {cfg.policy.ensemble_search_mode}")
    print(f"  ensemble_ucb_beta: {cfg.policy.ensemble_ucb_beta}")
    print(f"  ensemble_uncertainty_decay: {cfg.policy.ensemble_uncertainty_decay}")
    print(f"  temperature: {args.temperature}")
    print(f"  epsilon: {args.epsilon}")
    print(f"  num_simulations: {cfg.policy.num_simulations}")
    print(f"  cuda: {cfg.policy.cuda}")

    import torch
    from ding.config import compile_config
    from ding.envs import create_env_manager, get_vec_env_setting
    from ding.policy import create_policy
    from ding.utils import set_pkg_seed
    from lzero.worker import MuZeroCollector

    if cfg.policy.cuda and torch.cuda.is_available():
        cfg.policy.device = "cuda"
    else:
        cfg.policy.device = "cpu"

    cfg = compile_config(cfg, seed=args.seed, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)
    env_fn, collector_env_cfg, _ = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    collector_env.enable_save_replay(args.video_dir)
    collector_env.seed(cfg.seed)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    policy = create_policy(cfg.policy, model=None, enable_field=["learn", "collect"])
    if args.model_path is not None:
        state_dict = torch.load(args.model_path, map_location=cfg.policy.device)
        policy.learn_mode.load_state_dict(state_dict)

    collector = MuZeroCollector(
        collect_print_freq=int(1e9),
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=None,
        exp_name=cfg.exp_name,
        policy_config=cfg.policy,
    )

    try:
        collector.collect(
            n_episode=args.episodes,
            train_iter=0,
            policy_kwargs={"temperature": args.temperature, "epsilon": args.epsilon},
        )
        rewards = [episode_info["reward"] for episode_info in collector._episode_info]
    finally:
        collector.close()

    print("=" * 20)
    print(f"Collected returns: {rewards}")
    print(f"Mean return: {sum(rewards) / len(rewards) if rewards else float('nan')}")
    print(f"Videos written under: {args.video_dir}")
    print_video_summary(args.video_dir, run_start_time)
    print("=" * 20)


if __name__ == "__main__":
    main()
