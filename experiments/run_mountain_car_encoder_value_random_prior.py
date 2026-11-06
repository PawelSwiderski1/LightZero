import argparse

from run_mountain_car_encoder_random_prior import (
    build_parser as build_encoder_prior_parser,
    build_preset_overrides as build_encoder_prior_overrides,
)
from run_mountain_car_optimistic_shared_tree import run


def build_parser() -> argparse.ArgumentParser:
    parser = build_encoder_prior_parser()
    parser.description = (
        "MountainCar ensemble with member-specific encoder priors and frozen latent-input "
        "additive scalar value priors whose gradients may shape the learned representation and dynamics."
    )
    parser.add_argument(
        "--value-random-prior-scale",
        type=float,
        default=5.0,
        help="Maximum frozen member-specific additive prior in original scalar value units.",
    )
    parser.add_argument(
        "--value-prior-backprop-to-latent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Backpropagate the frozen value prior through its latent input while keeping its "
            "parameters frozen."
        ),
    )
    parser.add_argument(
        "--value-prior-scalar-loss-weight",
        type=float,
        default=1.0,
        help="Weight of the transformed scalar consistency loss used with the residual categorical target.",
    )
    parser.set_defaults(
        exp_name="data_ensemble_muzero/mountain_car_local_encoder_value_residual_prior_seed0",
        use_qstd_for_collection=True,
    )
    return parser


def build_preset_overrides(args: argparse.Namespace) -> dict:
    overrides = build_encoder_prior_overrides(args)
    overrides.update(
        random_prior_scale=max(0.0, float(args.value_random_prior_scale)),
        value_prior_backprop_to_latent=bool(args.value_prior_backprop_to_latent),
        value_prior_scalar_loss_weight=max(0.0, float(args.value_prior_scalar_loss_weight)),
    )
    return overrides


def main() -> None:
    args = build_parser().parse_args()
    run(args, preset_overrides=build_preset_overrides(args))


if __name__ == "__main__":
    main()
