import argparse
import copy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a MountainCar EfficientZero baseline.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--max-env-step", type=int, default=200000)
    parser.add_argument("--max-train-iter", type=int, default=3000)
    parser.add_argument("--collector-env-num", type=int, default=8)
    parser.add_argument("--evaluator-env-num", type=int, default=8)
    parser.add_argument("--n-episode", type=int, default=15)
    parser.add_argument("--n-evaluator-episode", type=int, default=8)
    parser.add_argument("--num-simulations", type=int, default=25)
    parser.add_argument("--update-per-collect", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-freq", type=int, default=200)
    parser.add_argument("--td-steps", type=int, default=20)
    parser.add_argument("--game-segment-length", type=int, default=200)
    parser.add_argument("--reanalyze-ratio", type=float, default=0.0)
    parser.add_argument("--use-priority", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--priority-prob-alpha", type=float, default=0.6)
    parser.add_argument("--priority-prob-beta", type=float, default=0.4)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--latent-state-dim", type=int, default=64)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved run settings without training.")
    return parser


def default_exp_name(args: argparse.Namespace) -> str:
    priority_suffix = "per" if args.use_priority else "noprior"
    return f"data_ez/mountain_car_efficientzero_{priority_suffix}_seed{args.seed}"


def build_config(args: argparse.Namespace):
    from zoo.classic_control.mountain_car.config.mountain_car_efficientzero_config import create_config, main_config

    cfg = copy.deepcopy(main_config)
    create_cfg = copy.deepcopy(create_config)

    cfg.exp_name = args.exp_name or default_exp_name(args)

    cfg.env.collector_env_num = args.collector_env_num
    cfg.env.evaluator_env_num = args.evaluator_env_num
    cfg.env.n_evaluator_episode = args.n_evaluator_episode

    cfg.policy.collector_env_num = args.collector_env_num
    cfg.policy.evaluator_env_num = args.evaluator_env_num
    cfg.policy.n_episode = args.n_episode
    cfg.policy.num_simulations = args.num_simulations
    cfg.policy.update_per_collect = args.update_per_collect
    cfg.policy.batch_size = args.batch_size
    cfg.policy.eval_freq = args.eval_freq
    cfg.policy.td_steps = args.td_steps
    cfg.policy.game_segment_length = args.game_segment_length
    cfg.policy.cuda = args.cuda
    cfg.policy.reanalyze_ratio = args.reanalyze_ratio
    cfg.policy.use_priority = args.use_priority
    cfg.policy.priority_prob_alpha = args.priority_prob_alpha
    cfg.policy.priority_prob_beta = args.priority_prob_beta
    cfg.policy.learning_rate = args.learning_rate
    cfg.policy.use_wandb = False
    cfg.policy.model.latent_state_dim = args.latent_state_dim
    cfg.policy.model.lstm_hidden_size = args.lstm_hidden_size

    return cfg, create_cfg


def print_run_summary(cfg, args: argparse.Namespace) -> None:
    print("MountainCar EfficientZero baseline run")
    print(f"  exp_name: {cfg.exp_name}")
    print(f"  seed: {args.seed}")
    print(f"  num_simulations: {cfg.policy.num_simulations}")
    print(f"  collector_env_num: {cfg.env.collector_env_num}")
    print(f"  evaluator_env_num: {cfg.env.evaluator_env_num}")
    print(f"  n_episode: {cfg.policy.n_episode}")
    print(f"  n_evaluator_episode: {cfg.env.n_evaluator_episode}")
    print(f"  update_per_collect: {cfg.policy.update_per_collect}")
    print(f"  batch_size: {cfg.policy.batch_size}")
    print(f"  eval_freq: {cfg.policy.eval_freq}")
    print(f"  td_steps: {cfg.policy.td_steps}")
    print(f"  game_segment_length: {cfg.policy.game_segment_length}")
    print(f"  reanalyze_ratio: {cfg.policy.reanalyze_ratio}")
    print(f"  use_priority: {cfg.policy.use_priority}")
    print(f"  priority_prob_alpha: {cfg.policy.priority_prob_alpha}")
    print(f"  priority_prob_beta: {cfg.policy.priority_prob_beta}")
    print(f"  learning_rate: {cfg.policy.learning_rate}")
    print(f"  latent_state_dim: {cfg.policy.model.latent_state_dim}")
    print(f"  lstm_hidden_size: {cfg.policy.model.lstm_hidden_size}")
    print(f"  stop_value: {cfg.env.stop_value}")
    print(f"  max_env_step: {args.max_env_step}")
    print(f"  max_train_iter: {args.max_train_iter}")
    print(f"  cuda: {cfg.policy.cuda}")


def main() -> None:
    args = build_parser().parse_args()
    cfg, create_cfg = build_config(args)
    print_run_summary(cfg, args)

    if args.dry_run:
        return

    from lzero.entry import train_muzero

    train_muzero(
        [cfg, create_cfg],
        seed=args.seed,
        model_path=cfg.policy.model_path,
        max_env_step=args.max_env_step,
        max_train_iter=args.max_train_iter,
    )


if __name__ == "__main__":
    main()
