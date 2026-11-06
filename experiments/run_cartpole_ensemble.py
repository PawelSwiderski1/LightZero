import argparse
import copy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CartPole with the ensemble MuZero entry point.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--max-env-step", type=int, default=50000)
    parser.add_argument("--max-train-iter", type=int, default=1000)
    parser.add_argument("--collector-env-num", type=int, default=8)
    parser.add_argument("--evaluator-env-num", type=int, default=8)
    parser.add_argument("--n-episode", type=int, default=8)
    parser.add_argument("--n-evaluator-episode", type=int, default=8)
    parser.add_argument("--num-simulations", type=int, default=25)
    parser.add_argument("--update-per-collect", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-freq", type=int, default=100)
    parser.add_argument("--reanalyze-ratio", type=float, default=0.0)
    parser.add_argument("--random-prior-scale", type=float, default=0.0)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--thompson-sampling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--ensemble-search-mode",
        choices=[
            "none", "thompson", "optimistic_shared_tree", "pessimistic_shared_tree",
            "bounded_optimistic_shared_tree", "bounded_pessimistic_shared_tree",
        ],
        default=None,
    )
    parser.add_argument("--ensemble-ucb-beta", type=float, default=0.0)
    parser.add_argument("--ensemble-uncertainty-decay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--ensemble-eval-search-mode",
        choices=[
            "member0", "mean_shared_tree", "optimistic_shared_tree", "pessimistic_shared_tree",
            "bounded_optimistic_shared_tree", "bounded_pessimistic_shared_tree",
        ],
        default=None,
        help="Eval-time action selection. member0 preserves the old single-member MuZero eval.",
    )
    parser.add_argument(
        "--ensemble-eval-ucb-beta",
        type=float,
        default=None,
        help="Eval-time ensemble uncertainty weight. Defaults to --ensemble-ucb-beta when omitted.",
    )
    parser.add_argument(
        "--ensemble-eval-uncertainty-decay",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Eval-time uncertainty decay. Defaults to --ensemble-uncertainty-decay when omitted.",
    )
    parser.add_argument("--profile-timing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--profile-timing-freq", type=int, default=1)
    parser.add_argument("--profile-learn-timing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved run settings without training.")
    return parser


def default_exp_name(args: argparse.Namespace, thompson_sampling: bool) -> str:
    run_kind = "ensemble" if args.ensemble_size > 1 else "control"
    thompson_suffix = "ts" if thompson_sampling else "nots"
    return f"data_ensemble_muzero/cartpole_{run_kind}_ens{args.ensemble_size}_{thompson_suffix}_seed{args.seed}"


def build_config(args: argparse.Namespace):
    from zoo.classic_control.cartpole.config.cartpole_ensemble_muzero_config import create_config, main_config

    cfg = copy.deepcopy(main_config)
    create_cfg = copy.deepcopy(create_config)

    thompson_sampling = args.thompson_sampling
    if thompson_sampling is None:
        thompson_sampling = args.ensemble_size > 1
    ensemble_search_mode = args.ensemble_search_mode
    if ensemble_search_mode is None:
        ensemble_search_mode = "thompson" if thompson_sampling else "none"
    elif ensemble_search_mode == "thompson":
        thompson_sampling = True
    else:
        thompson_sampling = False
    profile_learn_timing = args.profile_learn_timing
    if profile_learn_timing is None:
        profile_learn_timing = args.profile_timing

    cfg.exp_name = args.exp_name or default_exp_name(args, thompson_sampling)

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
    cfg.policy.cuda = args.cuda
    cfg.policy.reanalyze_ratio = args.reanalyze_ratio
    cfg.policy.thompson_sampling = thompson_sampling
    cfg.policy.ensemble_search_mode = ensemble_search_mode
    cfg.policy.ensemble_ucb_beta = args.ensemble_ucb_beta
    cfg.policy.ensemble_uncertainty_decay = args.ensemble_uncertainty_decay
    if args.ensemble_eval_search_mode is not None:
        cfg.policy.ensemble_eval_search_mode = args.ensemble_eval_search_mode
    cfg.policy.ensemble_eval_ucb_beta = args.ensemble_eval_ucb_beta
    cfg.policy.ensemble_eval_uncertainty_decay = args.ensemble_eval_uncertainty_decay
    cfg.policy.use_wandb = False
    cfg.policy.profile_train_loop_timing = args.profile_timing
    cfg.policy.profile_train_loop_timing_freq = args.profile_timing_freq
    cfg.policy.profile_learn_timing = profile_learn_timing

    cfg.policy.model.ensemble_size = args.ensemble_size
    cfg.policy.model.random_prior_scale = args.random_prior_scale

    return cfg, create_cfg


def print_run_summary(cfg, args: argparse.Namespace) -> None:
    print("CartPole ensemble MuZero run")
    print(f"  exp_name: {cfg.exp_name}")
    print(f"  seed: {args.seed}")
    print(f"  ensemble_size: {cfg.policy.model.ensemble_size}")
    print(f"  random_prior_scale: {cfg.policy.model.random_prior_scale}")
    print(f"  thompson_sampling: {cfg.policy.thompson_sampling}")
    print(f"  ensemble_search_mode: {cfg.policy.ensemble_search_mode}")
    print(f"  ensemble_ucb_beta: {cfg.policy.ensemble_ucb_beta}")
    print(f"  ensemble_uncertainty_decay: {cfg.policy.ensemble_uncertainty_decay}")
    print(f"  ensemble_eval_search_mode: {cfg.policy.ensemble_eval_search_mode}")
    print(f"  ensemble_eval_ucb_beta: {cfg.policy.ensemble_eval_ucb_beta}")
    print(f"  ensemble_eval_uncertainty_decay: {cfg.policy.ensemble_eval_uncertainty_decay}")
    print(f"  num_simulations: {cfg.policy.num_simulations}")
    print(f"  collector_env_num: {cfg.env.collector_env_num}")
    print(f"  evaluator_env_num: {cfg.env.evaluator_env_num}")
    print(f"  n_episode: {cfg.policy.n_episode}")
    print(f"  n_evaluator_episode: {cfg.env.n_evaluator_episode}")
    print(f"  update_per_collect: {cfg.policy.update_per_collect}")
    print(f"  batch_size: {cfg.policy.batch_size}")
    print(f"  eval_freq: {cfg.policy.eval_freq}")
    print(f"  max_env_step: {args.max_env_step}")
    print(f"  max_train_iter: {args.max_train_iter}")
    print(f"  cuda: {cfg.policy.cuda}")
    print(f"  profile_timing: {cfg.policy.profile_train_loop_timing}")
    print(f"  profile_learn_timing: {cfg.policy.profile_learn_timing}")


def main() -> None:
    args = build_parser().parse_args()
    cfg, create_cfg = build_config(args)
    print_run_summary(cfg, args)

    if args.dry_run:
        return

    from lzero.entry.train_ensemble_muzero import train_ensemble_muzero

    train_ensemble_muzero(
        [cfg, create_cfg],
        seed=args.seed,
        model_path=cfg.policy.model_path,
        max_env_step=args.max_env_step,
        max_train_iter=args.max_train_iter,
    )


if __name__ == "__main__":
    main()
