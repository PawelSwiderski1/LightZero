import argparse
import copy
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MountainCar with the ensemble MuZero entry point.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--max-env-step", type=int, default=200000)
    parser.add_argument("--max-train-iter", type=int, default=3000)
    parser.add_argument("--collector-env-num", type=int, default=8)
    parser.add_argument("--evaluator-env-num", type=int, default=8)
    parser.add_argument("--n-episode", type=int, default=8)
    parser.add_argument("--n-evaluator-episode", type=int, default=8)
    parser.add_argument("--num-simulations", type=int, default=25)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--root-noise-weight", type=float, default=0.25)
    parser.add_argument("--update-per-collect", type=int, default=100)
    parser.add_argument("--replay-ratio", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Override the main MuZero optimizer weight decay.",
    )
    parser.add_argument(
        "--representation-prediction-weight-decay",
        type=float,
        default=None,
        help="Override weight decay only for MCTS representation and prediction networks.",
    )
    parser.add_argument("--latent-variance-loss-weight", type=float, default=0.0)
    parser.add_argument("--latent-covariance-loss-weight", type=float, default=0.0)
    parser.add_argument("--latent-variance-target", type=float, default=1.0)
    parser.add_argument("--latent-regularization-eps", type=float, default=1e-4)
    parser.add_argument("--eval-freq", type=int, default=200)
    parser.add_argument(
        "--stop-value",
        type=float,
        default=None,
        help="Override env.stop_value. For MountainCar, use 0 to avoid early stopping at the default -110.",
    )
    parser.add_argument("--td-steps", type=int, default=None)
    parser.add_argument("--num-unroll-steps", type=int, default=None)
    parser.add_argument("--game-segment-length", type=int, default=None)
    parser.add_argument(
        "--include-remaining-time",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append normalized remaining episode time to MountainCar observations.",
    )
    parser.add_argument(
        "--bootstrap-at-time-limit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bootstrap value targets at a time-limit truncation instead of treating it as a finite-horizon terminal.",
    )
    parser.add_argument("--ssl-loss-weight", type=float, default=None)
    parser.add_argument("--raw-latent-ssl-loss-weight", type=float, default=0.0)
    parser.add_argument("--reward-loss-weight", type=float, default=None)
    parser.add_argument("--value-loss-weight", type=float, default=None)
    parser.add_argument("--policy-loss-weight", type=float, default=None)
    parser.add_argument(
        "--state-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Min-max normalize every encoded and recurrent latent state to [0, 1].",
    )
    parser.add_argument(
        "--representation-last-linear-layer-init-zero",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override zero initialization of the representation network's final linear layer.",
    )
    parser.add_argument(
        "--value-last-linear-layer-init-zero",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override zero initialization of the value head's final layer without changing reward or policy heads."
        ),
    )
    parser.add_argument(
        "--norm-type",
        choices=["BN", "LN"],
        default="BN",
        help="Normalization used by the MCTS-facing MLP backbones.",
    )
    parser.add_argument(
        "--ssl-norm-type",
        choices=["BN", "LN"],
        default=None,
        help="Normalization used by SSL projection heads; defaults to --norm-type.",
    )
    parser.add_argument("--reanalyze-ratio", type=float, default=0.0)
    parser.add_argument("--reanalyze-use-search-value-target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--value-target-mode", choices=["auto", "td", "search_bootstrap", "mixed_sve"], default="auto")
    parser.add_argument("--sve-start-train-iter", type=int, default=0)
    parser.add_argument("--sve-recent-window", type=int, default=0)
    parser.add_argument("--sve-real-reward-prefix-steps", type=int, default=0)
    parser.add_argument("--sve-search-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ensemble-member-bootstrap-targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-priority", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--priority-prob-alpha", type=float, default=0.6)
    parser.add_argument("--priority-prob-beta", type=float, default=0.4)
    parser.add_argument(
        "--priority-initial-mode",
        type=str,
        default="bellman_max",
        choices=["bellman_max", "pred_vs_search"],
    )
    parser.add_argument("--priority-new-data-use-max", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-prior-scale", type=float, default=0.0)
    parser.add_argument(
        "--value-prior-backprop-to-latent",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow the frozen value-prior output to backpropagate through its latent input into "
            "the representation and dynamics networks. Prior parameters remain frozen."
        ),
    )
    parser.add_argument(
        "--value-prior-latent-normalization",
        choices=["none", "layer_norm", "l2"],
        default="layer_norm",
        help="Fixed normalization applied before evaluating the frozen scalar value prior.",
    )
    parser.add_argument(
        "--value-prior-scalar-loss-weight",
        type=float,
        default=1.0,
        help="Weight of the transformed scalar loss for decoded base value plus frozen prior.",
    )
    parser.add_argument("--encoder-random-prior-scale", type=float, default=0.0)
    parser.add_argument(
        "--encoder-random-prior-mode",
        choices=["smooth_mlp", "local_grid"],
        default="smooth_mlp",
    )
    parser.add_argument("--encoder-random-prior-hidden-dim", type=int, default=64)
    parser.add_argument("--encoder-random-prior-cell-size", type=float, default=0.25)
    parser.add_argument("--encoder-random-prior-hash-size", type=int, default=16384)
    parser.add_argument(
        "--encoder-random-prior-input-mean",
        type=float,
        nargs=2,
        default=(-0.3, 0.0),
    )
    parser.add_argument(
        "--encoder-random-prior-input-scale",
        type=float,
        nargs=2,
        default=(0.9, 0.07),
    )
    parser.add_argument("--encoder-random-prior-output-norm", type=float, default=1.0)
    parser.add_argument(
        "--encoder-prior-composed-latent-normalization",
        choices=["none", "l2"],
        default="none",
        help="Optionally normalize the learned-encoder plus frozen-prior composition.",
    )
    parser.add_argument(
        "--encoder-prior-composed-latent-norm-scale",
        type=float,
        default=1.0,
        help="Final L2 norm after composing the learned encoder latent and frozen prior.",
    )
    parser.add_argument("--dynamics-random-prior-scale", type=float, default=0.0)
    parser.add_argument("--dynamics-random-prior-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--dynamics-random-prior-action-mode",
        choices=["one_hot", "per_action"],
        default="one_hot",
        help="Use one shared prior over [latent, action] or a separate frozen prior network per discrete action.",
    )
    parser.add_argument(
        "--dynamics-encoder-latent-normalization",
        choices=["none", "l2"],
        default="none",
        help="Normalize encoder/root latents before prediction and dynamics.",
    )
    parser.add_argument(
        "--dynamics-encoder-latent-norm-scale",
        type=float,
        default=1.0,
        help="Target L2 norm for encoder/root latents when --dynamics-encoder-latent-normalization=l2.",
    )
    parser.add_argument(
        "--dynamics-trainable-latent-normalization",
        choices=["none", "l2", "rms", "l2_cap"],
        default="none",
        help=(
            "Normalize the learned recurrent latent before adding the random dynamics prior. "
            "'l2' sets a fixed vector L2 norm; 'rms' sets a fixed per-dimension RMS; "
            "'l2_cap' smoothly caps the vector L2 norm."
        ),
    )
    parser.add_argument(
        "--dynamics-trainable-latent-norm-scale",
        type=float,
        default=1.0,
        help="Target L2 norm for l2 mode, target per-dimension RMS for rms mode, or cap for l2_cap mode.",
    )
    parser.add_argument(
        "--dynamics-trainable-latent-norm-eps",
        type=float,
        default=1e-3,
        help="Numerical epsilon for trainable dynamics latent normalization.",
    )
    parser.add_argument(
        "--dynamics-prior-latent-norm-cap",
        type=float,
        default=0.0,
        help="Smoothly cap the final scaled dynamics prior L2 norm. Use 0 to disable.",
    )
    parser.add_argument("--shared-representation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shared-prediction", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shared-projection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--shared-intrinsic-reward-head",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--rnd-exploration", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rnd-intrinsic-reward-weight", type=float, default=0.0)
    parser.add_argument("--rnd-learning-rate", type=float, default=1e-4)
    parser.add_argument("--rnd-hidden-dim", type=int, default=128)
    parser.add_argument("--rnd-feature-dim", type=int, default=64)
    parser.add_argument(
        "--rnd-initialization-seed",
        type=int,
        default=None,
        help="Optional isolated RNG seed for RND target and predictor initialization.",
    )
    parser.add_argument("--rnd-train-batch-size", type=int, default=512)
    parser.add_argument("--rnd-predictor-update-proportion", type=float, default=0.25)
    parser.add_argument("--rnd-train-on-new-data-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--rnd-train-from-fresh-collection-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Update the RND predictor only from observations in the latest collection phase.",
    )
    parser.add_argument("--rnd-fresh-update-batch-size", type=int, default=64)
    parser.add_argument("--rnd-train-observation-mode", choices=["next", "all"], default="next")
    parser.add_argument(
        "--rnd-observation-dim",
        type=int,
        default=None,
        help="Use the first N observation features for RND. MountainCar defaults to 2 physical features.",
    )
    parser.add_argument(
        "--rnd-observation-indices",
        type=str,
        default=None,
        help="Comma-separated observation feature indices for RND. Overrides --rnd-observation-dim.",
    )
    parser.add_argument(
        "--rnd-observation-low",
        type=str,
        default=None,
        help="Comma-separated fixed lower bounds for the selected RND observation features.",
    )
    parser.add_argument(
        "--rnd-observation-high",
        type=str,
        default=None,
        help="Comma-separated fixed upper bounds for the selected RND observation features.",
    )
    parser.add_argument("--rnd-ignore-elite-replay-for-predictor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rnd-bonus-normalization", choices=["zscore", "raw"], default="zscore")
    parser.add_argument("--rnd-bonus-clip", type=float, default=5.0)
    parser.add_argument("--rnd-store-collection-rewards", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rnd-update-stats-on-collection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rnd-use-stored-collection-rewards", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rnd-intrinsic-target-mode", choices=["td", "monte_carlo"], default="td")
    parser.add_argument("--rnd-use-learned-intrinsic-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rnd-intrinsic-discount-factor", type=float, default=0.99)
    parser.add_argument("--rnd-intrinsic-td-steps", type=int, default=None)
    parser.add_argument("--rnd-intrinsic-bootstrap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rnd-target-snapshot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rnd-target-snapshot-update-interval", type=int, default=None)
    parser.add_argument(
        "--rnd-target-snapshot-refresh-per-collect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze one current RND predictor snapshot at the start of every collection phase.",
    )
    parser.add_argument(
        "--rnd-target-snapshot-refresh-after-fresh-update",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After fresh-collection RND optimizer steps, refresh the frozen target "
            "before replay targets and intrinsic-model learning are computed."
        ),
    )
    parser.add_argument(
        "--rnd-landscape-query",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use active RND landscape queries for intrinsic-tree edge rewards and "
            "train the query reader jointly on every world-model replay minibatch."
        ),
    )
    parser.add_argument("--rnd-landscape-query-count", type=int, default=16)
    parser.add_argument("--rnd-landscape-query-hidden-dim", type=int, default=64)
    parser.add_argument("--rnd-landscape-query-learning-rate", type=float, default=3e-4)
    parser.add_argument("--rnd-landscape-query-weight-decay", type=float, default=0.0)
    parser.add_argument("--rnd-landscape-query-loss-weight", type=float, default=1.0)
    parser.add_argument("--rnd-landscape-query-max-horizon", type=int, default=8)
    parser.add_argument("--rnd-landscape-query-min-snapshots", type=int, default=2)
    parser.add_argument("--rnd-landscape-query-diversity-weight", type=float, default=0.01)
    parser.add_argument(
        "--rnd-landscape-query-centered-prediction-weight", type=float, default=0.0
    )
    parser.add_argument("--rnd-landscape-query-grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--rnd-landscape-query-reader-use-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let the latent reader use queried state latents and measured query errors.",
    )
    parser.add_argument(
        "--rnd-landscape-query-member-specific-reader",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use an independent auxiliary RND decoder for each ensemble member.",
    )
    parser.add_argument(
        "--rnd-landscape-query-reader-architecture",
        choices=("landscape", "ordinary"),
        default="landscape",
        help="Use the original auxiliary reader or an ordinary reward-head-shaped decoder.",
    )
    parser.add_argument(
        "--rnd-landscape-query-reader-hidden-dim", type=int, default=32
    )
    parser.add_argument(
        "--rnd-landscape-query-reader-norm-type",
        choices=("BN", "LN"),
        default="BN",
    )
    parser.add_argument(
        "--rnd-landscape-query-reader-last-linear-layer-init-zero",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--rnd-landscape-query-target-transform",
        choices=("log", "raw", "normalized_raw"),
        default="log",
        help="Prediction space used by the auxiliary RND decoder.",
    )
    parser.add_argument(
        "--rnd-landscape-query-target-log-reference", type=float, default=1.0
    )
    parser.add_argument(
        "--rnd-landscape-query-error-epsilon", type=float, default=1e-10
    )
    parser.add_argument(
        "--rnd-landscape-query-max-raw-error", type=float, default=None
    )
    parser.add_argument(
        "--rnd-landscape-query-normalized-raw-reference-size",
        type=int,
        default=8192,
        help="Replay observations used to fix the normalized-raw scale per phase.",
    )
    parser.add_argument(
        "--rnd-landscape-query-normalized-raw-clip",
        type=float,
        default=10.0,
        help="Upper bound on normalized raw RND targets.",
    )
    parser.add_argument(
        "--rnd-landscape-query-normalized-raw-min-scale",
        type=float,
        default=1e-8,
        help="Lower bound for the phase-fixed mean RND error scale.",
    )
    parser.add_argument(
        "--rnd-landscape-query-gradient-diagnostics-interval",
        type=int,
        default=0,
        help="Measure auxiliary-only gradient norms every N learner updates; 0 disables it.",
    )
    parser.add_argument(
        "--rnd-landscape-query-train-world-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Backpropagate the query-reader replay loss into MuZero's encoder and dynamics.",
    )
    parser.add_argument(
        "--rnd-landscape-query-use-tree-rewards",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use query-reader predictions instead of the intrinsic reward head inside MCTS.",
    )
    parser.add_argument(
        "--rnd-landscape-query-initialization-seed",
        type=int,
        default=None,
        help="Optional isolated RNG seed for query-policy and reader initialization.",
    )
    parser.add_argument("--rnd-aux-confirmatory-condition", type=str, default="none")
    parser.add_argument("--rnd-aux-confirmatory-plan-version", type=int, default=0)
    parser.add_argument(
        "--intrinsic-reward-source",
        choices=[
            "rnd", "dynamics_disagreement", "qstd", "state_value_disagreement",
            "episodic_novelty", "state_density",
        ],
        default="rnd",
    )
    parser.add_argument(
        "--episodic-novelty-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Store real episodic kNN novelty and train its immediate reward head for diagnostics only.",
    )
    parser.add_argument("--episodic-novelty-encoder-member", type=int, default=0)
    parser.add_argument("--episodic-novelty-knn-k", type=int, default=10)
    parser.add_argument("--episodic-novelty-kernel-cluster-distance", type=float, default=0.008)
    parser.add_argument("--episodic-novelty-kernel-epsilon", type=float, default=1e-4)
    parser.add_argument("--episodic-novelty-pseudo-count-constant", type=float, default=0.001)
    parser.add_argument("--episodic-novelty-maximum-similarity", type=float, default=8.0)
    parser.add_argument(
        "--episodic-novelty-zero-until-knn-full",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--episodic-novelty-reward-scale", type=float, default=1.0)
    parser.add_argument("--episodic-novelty-reward-clip", type=float, default=None)
    parser.add_argument(
        "--episodic-novelty-shadow-predictions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--episodic-novelty-history-shadow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train and log a recurrent episode-history-conditioned novelty reward model.",
    )
    parser.add_argument("--episodic-novelty-history-dim", type=int, default=64)
    parser.add_argument("--episodic-novelty-history-hidden-dim", type=int, default=64)
    parser.add_argument("--episodic-novelty-history-learning-rate", type=float, default=3e-4)
    parser.add_argument("--episodic-novelty-history-train-epochs", type=int, default=1)
    parser.add_argument("--episodic-novelty-history-train-batch-size", type=int, default=5)
    parser.add_argument("--episodic-novelty-history-updates-per-collect", type=int, default=60)
    parser.add_argument("--episodic-novelty-history-replay-capacity", type=int, default=1000)
    parser.add_argument("--episodic-novelty-history-recent-phase-count", type=int, default=2)
    parser.add_argument("--episodic-novelty-history-recent-sample-ratio", type=float, default=0.5)
    parser.add_argument("--episodic-novelty-history-grad-clip", type=float, default=5.0)
    parser.add_argument("--state-value-disagreement-reward-scale", type=float, default=1.0)
    parser.add_argument("--state-value-disagreement-reward-clip", type=float, default=None)
    parser.add_argument("--state-value-disagreement-inference-batch-size", type=int, default=1024)
    parser.add_argument(
        "--state-value-disagreement-store-collection-rewards",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Store real next-state value disagreement for episode diagnostics and optional training targets.",
    )
    parser.add_argument(
        "--state-value-disagreement-use-stored-collection-rewards",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train intrinsic heads on the disagreement captured when each transition was collected.",
    )
    parser.add_argument(
        "--state-value-disagreement-intrinsic-target-mode",
        choices=["td", "monte_carlo"],
        default="td",
    )
    parser.add_argument("--state-density-position-bin-width", type=float, default=0.02)
    parser.add_argument("--state-density-velocity-bin-width", type=float, default=0.002)
    parser.add_argument("--state-density-neighbor-radius", type=int, default=1)
    parser.add_argument("--state-density-snapshot-update-interval", type=int, default=1)
    parser.add_argument("--state-density-reward-scale", type=float, default=1.0)
    parser.add_argument("--state-density-reward-clip", type=float, default=1.0)
    parser.add_argument(
        "--state-density-intrinsic-target-mode",
        choices=["td", "monte_carlo"],
        default="td",
    )
    parser.add_argument(
        "--state-density-train-on-intrinsic-replay-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Train state-density intrinsic heads only on samples selected from the "
            "intrinsic episode replay. By default all replay samples are eligible and "
            "their rewards are recomputed from the current density snapshot."
        ),
    )
    parser.add_argument("--qstd-exploration", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qstd-intrinsic-reward-weight", type=float, default=0.0)
    parser.add_argument("--qstd-store-collection-rewards", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--qstd-reward-aggregation",
        choices=["max", "mean", "mean_topk"],
        default="max",
        help="How root action qstd values are collapsed into the state intrinsic reward.",
    )
    parser.add_argument("--qstd-reward-top-k", type=int, default=1)
    parser.add_argument("--qstd-reward-scale", type=float, default=1.0)
    parser.add_argument("--qstd-reward-clip", type=float, default=None)
    parser.add_argument("--qstd-reward-num-simulations", type=int, default=None)
    parser.add_argument(
        "--qstd-reward-search-mode",
        choices=[
            "mean_shared_tree", "optimistic_shared_tree", "pessimistic_shared_tree",
            "bounded_optimistic_shared_tree", "bounded_pessimistic_shared_tree",
        ],
        default="mean_shared_tree",
        help="Search mode used only to compute qstd rewards for real next observations.",
    )
    parser.add_argument(
        "--qstd-recompute-replay-rewards",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Recompute the replay reward for each real (state, action) as the ensemble "
            "standard deviation of predicted one-step extrinsic Q."
        ),
    )
    parser.add_argument(
        "--qstd-one-step-include-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include each member's predicted immediate extrinsic reward in one-step Qstd.",
    )
    parser.add_argument(
        "--qstd-one-step-use-value-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Subtract frozen per-member value offsets before computing one-step Qstd.",
    )
    parser.add_argument("--qstd-target-snapshot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--qstd-target-snapshot-refresh-per-collect",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze one extrinsic ensemble and Qstd reward cache for each learning phase.",
    )
    parser.add_argument("--qstd-target-inference-batch-size", type=int, default=1024)
    parser.add_argument(
        "--qstd-intrinsic-target-mode", choices=["td", "monte_carlo"], default="td"
    )
    parser.add_argument("--dynamics-disagreement-exploration", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dynamics-disagreement-intrinsic-reward-weight", type=float, default=0.0)
    parser.add_argument("--dynamics-disagreement-shared-root-latent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamics-disagreement-normalize-latent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dynamics-disagreement-space",
        choices=["latent", "projection", "prediction_projection"],
        default="latent",
        help="Feature space used for dynamics-disagreement intrinsic rewards.",
    )
    parser.add_argument(
        "--dynamics-disagreement-distance-metric",
        choices=["mse", "l2"],
        default="mse",
        help="Pairwise ensemble latent disagreement metric. 'mse' is legacy per-dimension squared distance; "
        "'l2' keeps bounded random-prior differences visible.",
    )
    parser.add_argument("--dynamics-disagreement-reward-clip", type=float, default=None)
    parser.add_argument(
        "--dynamics-disagreement-train-intrinsic-value",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train the intrinsic value head from selected dynamics-disagreement rewards stored during collection.",
    )
    parser.add_argument("--dynamics-prior-alignment-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--dynamics-prior-alignment-target-normalization",
        choices=["same_as_trainable", "none", "l2", "rms"],
        default="none",
    )
    parser.add_argument("--separate-intrinsic-tree", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--mixed-reward-tree",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Back up and jointly normalize extrinsic Q plus the weighted intrinsic Q "
            "inside the main MCTS tree."
        ),
    )
    parser.add_argument(
        "--main-tree-policy-prior-mode",
        choices=["learned", "uniform"],
        default="learned",
    )
    parser.add_argument(
        "--main-tree-force-root-action-coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--collection-policy-mode",
        choices=["normal", "intrinsic_tree_only"],
        default="normal",
        help="Use normal collection action selection or use only the separate intrinsic tree policy.",
    )
    parser.add_argument("--intrinsic-tree-num-simulations", type=int, default=None)
    parser.add_argument("--intrinsic-tree-add-exploration-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--intrinsic-tree-prior-mode",
        choices=["learned", "uniform"],
        default="learned",
    )
    parser.add_argument(
        "--intrinsic-tree-force-root-action-coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--intrinsic-tree-pure-exploration-episodes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--intrinsic-value-train-on-pure-exploration-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--intrinsic-tree-pure-episode-store-intrinsic-policy",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--intrinsic-tree-final-score-temperature", type=float, default=1.0)
    parser.add_argument("--intrinsic-tree-store-final-policy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--intrinsic-tree-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--intrinsic-only-training", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--intrinsic-tree-normalize-q", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--intrinsic-tree-immediate-reward-weight", type=float, default=1.0)
    parser.add_argument(
        "--intrinsic-tree-leaf-value-source",
        choices=["default", "none", "rnd", "intrinsic_value"],
        default="default",
        help=(
            "Leaf bootstrap source for the intrinsic tree. 'default' keeps old behavior: RND source uses "
            "learned RND value; qstd source uses the intrinsic value head; dynamics-disagreement source uses "
            "the intrinsic value head only when --dynamics-disagreement-train-intrinsic-value is enabled."
        ),
    )
    parser.add_argument("--intrinsic-tree-leaf-value-weight", type=float, default=1.0)
    parser.add_argument("--intrinsic-tree-leaf-value-clip", type=float, default=None)
    parser.add_argument("--intrinsic-value-bootstrap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--collect-random-action-epsilon",
        type=float,
        default=0.0,
        help=(
            "Post-search collection epsilon. With this probability, replace the action sampled from MCTS "
            "with a uniform random legal action while preserving the original MCTS diagnostics."
        ),
    )
    parser.add_argument("--main-tree-trace-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--main-tree-trace-min-train-iter", type=int, default=0)
    parser.add_argument("--main-tree-trace-max-per-train-iter", type=int, default=4)
    parser.add_argument("--main-tree-trace-max-depth", type=int, default=4)
    parser.add_argument("--main-tree-trace-max-nodes", type=int, default=128)
    parser.add_argument(
        "--main-tree-trace-min-q-std",
        type=float,
        default=5.0,
        help="Only write main-tree traces when the selected root action's q_std is at least this value.",
    )
    parser.add_argument(
        "--main-tree-trace-selected-action",
        type=int,
        default=-1,
        help="Only write main-tree traces when this action is selected. Use -1 for any selected action.",
    )
    parser.add_argument("--main-tree-trace-min-position", type=float, default=None)
    parser.add_argument("--main-tree-trace-max-position", type=float, default=None)
    parser.add_argument("--main-tree-trace-min-velocity", type=float, default=None)
    parser.add_argument("--main-tree-trace-max-velocity", type=float, default=None)
    parser.add_argument("--intrinsic-tree-trace-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--intrinsic-tree-trace-best-action",
        type=int,
        default=1,
        help="Only write intrinsic tree traces when this root action is best. Use -1 to trace any best action.",
    )
    parser.add_argument("--intrinsic-tree-trace-min-train-iter", type=int, default=0)
    parser.add_argument("--intrinsic-tree-trace-max-per-train-iter", type=int, default=4)
    parser.add_argument("--intrinsic-tree-trace-max-depth", type=int, default=4)
    parser.add_argument("--intrinsic-tree-trace-max-nodes", type=int, default=128)
    parser.add_argument("--intrinsic-tree-trace-min-position", type=float, default=None)
    parser.add_argument("--intrinsic-tree-trace-max-position", type=float, default=None)
    parser.add_argument("--intrinsic-tree-trace-min-velocity", type=float, default=None)
    parser.add_argument("--intrinsic-tree-trace-max-velocity", type=float, default=None)
    parser.add_argument("--intrinsic-model-loss-weight", type=float, default=1.0)
    parser.add_argument("--intrinsic-reward-loss-weight", type=float, default=1.0)
    parser.add_argument("--intrinsic-value-loss-weight", type=float, default=1.0)
    parser.add_argument("--intrinsic-train-value-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--intrinsic-learning-rate",
        type=float,
        default=3e-4,
        help=(
            "Learning rate for intrinsic reward/value heads; shared encoder and "
            "dynamics parameters use the primary MuZero learning rate."
        ),
    )
    parser.add_argument("--intrinsic-head-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--intrinsic-loss-type", choices=["huber", "mse"], default="huber")
    parser.add_argument("--intrinsic-huber-delta", type=float, default=1.0)
    parser.add_argument("--intrinsic-reward-target-clip", type=float, default=5.0)
    parser.add_argument("--intrinsic-value-target-clip", type=float, default=50.0)
    parser.add_argument(
        "--intrinsic-reward-target-transform",
        choices=["raw", "log"],
        default="raw",
    )
    parser.add_argument("--intrinsic-reward-log-reference", type=float, default=1e-2)
    parser.add_argument("--intrinsic-reward-log-epsilon", type=float, default=1e-8)
    parser.add_argument("--intrinsic-reward-log-max-raw", type=float, default=5.0)
    parser.add_argument("--latent-novelty-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--latent-novelty-dim", type=int, default=32)
    parser.add_argument("--latent-novelty-hidden-dim", type=int, default=64)
    parser.add_argument("--latent-novelty-prior-scale", type=float, default=0.1)
    parser.add_argument("--latent-novelty-learning-rate", type=float, default=3e-4)
    parser.add_argument("--latent-novelty-train-batch-size", type=int, default=512)
    parser.add_argument("--latent-novelty-temperature", type=float, default=0.1)
    parser.add_argument("--latent-novelty-alignment-loss-weight", type=float, default=0.1)
    parser.add_argument("--elite-replay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--elite-buffer-size", type=int, default=20)
    parser.add_argument("--elite-sample-ratio", type=float, default=0.25)
    parser.add_argument("--elite-min-return", type=float, default=None)
    parser.add_argument("--elite-mc-value-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--elite-start-stratification", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--elite-start-observation-index", type=int, default=0)
    parser.add_argument(
        "--elite-start-bin-edges",
        type=str,
        default=None,
        help=(
            "Comma-separated boundaries used to stratify elite episodes by their initial observation. "
            "For MountainCar, '-0.55,-0.50,-0.45' creates four start-position bins."
        ),
    )
    parser.add_argument("--intrinsic-episode-replay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--intrinsic-episode-buffer-size", type=int, default=20)
    parser.add_argument("--intrinsic-episode-sample-ratio", type=float, default=0.25)
    parser.add_argument("--intrinsic-episode-min-score", type=float, default=0.0)
    parser.add_argument("--intrinsic-episode-score-mode", choices=["sum", "max"], default="sum")
    parser.add_argument("--intrinsic-episode-peak-window-before", type=int, default=120)
    parser.add_argument("--intrinsic-episode-peak-window-after", type=int, default=5)
    parser.add_argument(
        "--intrinsic-episode-max-age-phases",
        type=int,
        default=0,
        help="Evict novelty episodes this many phases after collection; 0 keeps them indefinitely.",
    )
    parser.add_argument("--save-intrinsic-episode-trajectories", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--intrinsic-episode-save-dir", type=str, default=None)
    parser.add_argument("--collection-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--root-action-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--collection-ssl-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--collection-ssl-diagnostics-batch-size", type=int, default=512)
    parser.add_argument("--encoder-prior-grid-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--encoder-prior-grid-position-min", type=float, default=-1.2)
    parser.add_argument("--encoder-prior-grid-position-max", type=float, default=0.6)
    parser.add_argument("--encoder-prior-grid-position-count", type=int, default=37)
    parser.add_argument("--encoder-prior-grid-velocity-min", type=float, default=-0.07)
    parser.add_argument("--encoder-prior-grid-velocity-max", type=float, default=0.07)
    parser.add_argument("--encoder-prior-grid-velocity-count", type=int, default=29)
    parser.add_argument(
        "--encoder-prior-imagined-novelty-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--encoder-prior-imagined-novelty-root-count", type=int, default=32)
    parser.add_argument("--encoder-prior-imagined-novelty-bank-size", type=int, default=2048)
    parser.add_argument("--encoder-prior-imagined-novelty-history-limit", type=int, default=100000)
    parser.add_argument("--encoder-prior-imagined-novelty-min-bank-size", type=int, default=32)
    parser.add_argument("--encoder-prior-imagined-novelty-depth", type=int, default=5)
    parser.add_argument("--encoder-prior-imagined-novelty-max-sequences", type=int, default=0)
    parser.add_argument("--encoder-prior-imagined-novelty-knn-k", type=int, default=10)
    parser.add_argument("--encoder-prior-imagined-novelty-batch-size", type=int, default=256)
    parser.add_argument("--encoder-prior-imagined-novelty-feature-std-eps", type=float, default=1e-3)
    parser.add_argument("--encoder-prior-imagined-novelty-position-scale", type=float, default=0.05)
    parser.add_argument("--encoder-prior-imagined-novelty-velocity-scale", type=float, default=0.005)
    parser.add_argument(
        "--normalization-mode-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compare recurrent latent/value/SSL consistency in model.train() and model.eval() on fresh data.",
    )
    parser.add_argument("--normalization-mode-diagnostics-depth", type=int, default=5)
    parser.add_argument("--normalization-mode-diagnostics-sample-size", type=int, default=256)
    parser.add_argument(
        "--prediction-gradient-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Log first prediction-trunk weight norms and separate value/policy gradient norms. "
            "This requires two additional backward traversals per learner update."
        ),
    )
    parser.add_argument("--fixed-sequence-intrinsic-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fixed-sequence-intrinsic-max-states", type=int, default=64)
    parser.add_argument(
        "--fixed-sequence-intrinsic-sequences",
        type=str,
        default=None,
        help=(
            "Optional semicolon-separated fixed action sequences for intrinsic diagnostics, e.g. "
            "'left5:0x5;right5:2x5;dither:1,1,0,2,1'."
        ),
    )
    parser.add_argument("--diagnostics-dir", type=str, default=None)
    parser.add_argument("--save-elite-trajectories", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--elite-save-dir", type=str, default=None)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--thompson-sampling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--ensemble-search-mode",
        choices=[
            "none",
            "thompson",
            "mean_shared_tree",
            "mean_tree_final_qstd",
            "optimistic_shared_tree",
            "pessimistic_shared_tree",
            "bounded_optimistic_shared_tree",
            "bounded_pessimistic_shared_tree",
        ],
        default=None,
    )
    parser.add_argument("--ensemble-ucb-beta", type=float, default=0.0)
    parser.add_argument("--ensemble-normalize-q", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ensemble-uncertainty-decay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--depth-matched-qstd-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log exact-depth and uniformly depth-matched root-action ensemble Q disagreement.",
    )
    parser.add_argument("--depth-matched-qstd-max-depth", type=int, default=5)
    parser.add_argument(
        "--depth-normalized-backup-disagreement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use depth-corrected aggregate edge Qstd as the MCTS uncertainty term."
        ),
    )
    parser.add_argument("--depth-disagreement-ema-momentum", type=float, default=0.5)
    parser.add_argument("--depth-disagreement-trim-fraction", type=float, default=0.1)
    parser.add_argument("--depth-disagreement-min-groups", type=int, default=100)
    parser.add_argument("--depth-disagreement-min-scale", type=float, default=0.5)
    parser.add_argument(
        "--depth-disagreement-max-depth",
        type=int,
        default=0,
        help="Maximum exact backup depth tracked; 0 uses --num-simulations.",
    )
    parser.add_argument(
        "--depth-disagreement-conservative-cap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prevent depth correction from increasing Qstd above the original aggregate edge Qstd.",
    )
    parser.add_argument("--ensemble-value-calibration", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ensemble-value-calibration-start-phase", type=int, default=10)
    parser.add_argument("--ensemble-value-calibration-sample-size", type=int, default=4096)
    parser.add_argument("--ensemble-value-calibration-inference-batch-size", type=int, default=1024)
    parser.add_argument(
        "--final-qstd-beta",
        type=float,
        default=0.0,
        help="Root-only qstd logit weight used when --ensemble-search-mode mean_tree_final_qstd.",
    )
    parser.add_argument(
        "--final-qstd-temperature",
        type=float,
        default=None,
        help="Temperature for root visit-count logits in mean_tree_final_qstd. Defaults to collect MCTS temperature.",
    )
    parser.add_argument(
        "--final-qstd-store-final-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store qstd-adjusted final root policy as the policy target in mean_tree_final_qstd mode.",
    )
    parser.add_argument("--separate-exploration-tree", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--separate-exploration-tree-horizon", type=int, default=3)
    parser.add_argument("--separate-exploration-tree-top-k", type=int, default=3)
    parser.add_argument("--separate-exploration-tree-beta", type=float, default=0.0)
    parser.add_argument(
        "--separate-exploration-tree-temperature",
        type=float,
        default=None,
        help="Temperature for root visit-count logits when merging the separate exploration tree.",
    )
    parser.add_argument(
        "--separate-exploration-tree-score-normalization",
        choices=["none", "center", "zscore", "minmax"],
        default="minmax",
    )
    parser.add_argument("--separate-exploration-tree-max-sequences", type=int, default=256)
    parser.add_argument(
        "--separate-exploration-tree-sequence-score-mode",
        choices=["qstd", "mean_q_plus_qstd", "bounded_mean_q_plus_qstd"],
        default="qstd",
        help=(
            "Score used to select fixed-horizon sequences in pure exploration episodes. "
            "The bounded mode caps mean_Q + beta * Qstd at the most optimistic member Q."
        ),
    )
    parser.add_argument("--separate-exploration-tree-sequence-qstd-beta", type=float, default=1.0)
    parser.add_argument(
        "--separate-exploration-tree-sequence-qstd-mode",
        choices=["raw", "member_centered"],
        default="raw",
        help="Use raw member return std or remove each member's across-sequence baseline first.",
    )
    parser.add_argument(
        "--separate-exploration-tree-store-final-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pure-exploration-episode-fraction", type=float, default=0.0)
    parser.add_argument("--pure-exploration-commit-steps", type=int, default=3)
    parser.add_argument("--pure-exploration-sequence-temperature", type=float, default=0.0)
    parser.add_argument(
        "--pure-exploration-store-exploration-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    parser.add_argument(
        "--record-collect-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record videos from the actual collector env used during training.",
    )
    parser.add_argument(
        "--collect-video-dir",
        type=str,
        default=None,
        help="Directory for collector videos. Defaults to ./video/<exp_name>_collect.",
    )
    parser.add_argument(
        "--clean-collect-video-dir",
        action="store_true",
        help="Delete existing mp4/meta files in the collector video directory before training.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved run settings without training.")
    return parser


def default_exp_name(args: argparse.Namespace, thompson_sampling: bool) -> str:
    run_kind = "ensemble" if args.ensemble_size > 1 else "control"
    thompson_suffix = "ts" if thompson_sampling else "nots"
    return f"data_ensemble_muzero/mountain_car_{run_kind}_ens{args.ensemble_size}_{thompson_suffix}_seed{args.seed}"


def default_collect_video_dir(exp_name: str) -> str:
    safe_exp_name = exp_name.strip("./").replace("/", "_")
    return f"./video/{safe_exp_name}_collect"


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
    if args.record_collect_video:
        create_cfg.env_manager.type = "base"

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
    collect_video_dir = args.collect_video_dir or default_collect_video_dir(cfg.exp_name)

    cfg.env.collector_env_num = args.collector_env_num
    cfg.env.evaluator_env_num = args.evaluator_env_num
    cfg.env.n_evaluator_episode = args.n_evaluator_episode
    cfg.env.include_remaining_time = args.include_remaining_time
    if args.stop_value is not None:
        cfg.env.stop_value = args.stop_value

    cfg.policy.collector_env_num = args.collector_env_num
    cfg.policy.evaluator_env_num = args.evaluator_env_num
    cfg.policy.n_episode = args.n_episode
    cfg.policy.num_simulations = args.num_simulations
    cfg.policy.root_dirichlet_alpha = args.root_dirichlet_alpha
    cfg.policy.root_noise_weight = args.root_noise_weight
    cfg.policy.update_per_collect = args.update_per_collect
    cfg.policy.replay_ratio = args.replay_ratio
    cfg.policy.batch_size = args.batch_size
    if args.weight_decay is not None:
        cfg.policy.weight_decay = args.weight_decay
    cfg.policy.representation_prediction_weight_decay = args.representation_prediction_weight_decay
    cfg.policy.latent_variance_loss_weight = args.latent_variance_loss_weight
    cfg.policy.latent_covariance_loss_weight = args.latent_covariance_loss_weight
    cfg.policy.latent_variance_target = args.latent_variance_target
    cfg.policy.latent_regularization_eps = args.latent_regularization_eps
    cfg.policy.eval_freq = args.eval_freq
    if args.td_steps is not None:
        cfg.policy.td_steps = args.td_steps
    if args.num_unroll_steps is not None:
        cfg.policy.num_unroll_steps = args.num_unroll_steps
    if args.game_segment_length is not None:
        cfg.policy.game_segment_length = args.game_segment_length
    if args.ssl_loss_weight is not None:
        cfg.policy.ssl_loss_weight = args.ssl_loss_weight
    cfg.policy.raw_latent_ssl_loss_weight = args.raw_latent_ssl_loss_weight
    if args.reward_loss_weight is not None:
        cfg.policy.reward_loss_weight = args.reward_loss_weight
    if args.value_loss_weight is not None:
        cfg.policy.value_loss_weight = args.value_loss_weight
    if args.policy_loss_weight is not None:
        cfg.policy.policy_loss_weight = args.policy_loss_weight
    cfg.policy.cuda = args.cuda
    cfg.policy.reanalyze_ratio = args.reanalyze_ratio
    cfg.policy.reanalyze_use_search_value_target = args.reanalyze_use_search_value_target
    cfg.policy.value_target_mode = args.value_target_mode
    cfg.policy.bootstrap_at_time_limit = args.bootstrap_at_time_limit
    cfg.policy.sve_start_train_iter = args.sve_start_train_iter
    cfg.policy.sve_recent_window = args.sve_recent_window
    cfg.policy.sve_real_reward_prefix_steps = args.sve_real_reward_prefix_steps
    cfg.policy.sve_search_noise = args.sve_search_noise
    cfg.policy.ensemble_member_bootstrap_targets = args.ensemble_member_bootstrap_targets
    cfg.policy.use_priority = args.use_priority
    cfg.policy.priority_prob_alpha = args.priority_prob_alpha
    cfg.policy.priority_prob_beta = args.priority_prob_beta
    cfg.policy.priority_initial_mode = args.priority_initial_mode
    cfg.policy.priority_new_data_use_max = args.priority_new_data_use_max
    cfg.policy.thompson_sampling = thompson_sampling
    cfg.policy.ensemble_search_mode = ensemble_search_mode
    cfg.policy.ensemble_ucb_beta = args.ensemble_ucb_beta
    cfg.policy.ensemble_normalize_q = args.ensemble_normalize_q
    cfg.policy.ensemble_uncertainty_decay = args.ensemble_uncertainty_decay
    cfg.policy.depth_matched_qstd_diagnostics = args.depth_matched_qstd_diagnostics
    cfg.policy.depth_matched_qstd_max_depth = args.depth_matched_qstd_max_depth
    cfg.policy.depth_normalized_backup_disagreement = args.depth_normalized_backup_disagreement
    cfg.policy.depth_disagreement_ema_momentum = args.depth_disagreement_ema_momentum
    cfg.policy.depth_disagreement_trim_fraction = args.depth_disagreement_trim_fraction
    cfg.policy.depth_disagreement_min_groups = args.depth_disagreement_min_groups
    cfg.policy.depth_disagreement_min_scale = args.depth_disagreement_min_scale
    cfg.policy.depth_disagreement_max_depth = args.depth_disagreement_max_depth
    cfg.policy.depth_disagreement_conservative_cap = args.depth_disagreement_conservative_cap
    cfg.policy.ensemble_value_calibration = args.ensemble_value_calibration
    cfg.policy.ensemble_value_calibration_start_phase = args.ensemble_value_calibration_start_phase
    cfg.policy.ensemble_value_calibration_sample_size = args.ensemble_value_calibration_sample_size
    cfg.policy.ensemble_value_calibration_inference_batch_size = (
        args.ensemble_value_calibration_inference_batch_size
    )
    cfg.policy.final_qstd_beta = args.final_qstd_beta
    cfg.policy.final_qstd_temperature = args.final_qstd_temperature
    cfg.policy.final_qstd_store_final_policy = args.final_qstd_store_final_policy
    cfg.policy.separate_exploration_tree = args.separate_exploration_tree
    cfg.policy.separate_exploration_tree_horizon = args.separate_exploration_tree_horizon
    cfg.policy.separate_exploration_tree_top_k = args.separate_exploration_tree_top_k
    cfg.policy.separate_exploration_tree_beta = args.separate_exploration_tree_beta
    cfg.policy.separate_exploration_tree_temperature = args.separate_exploration_tree_temperature
    cfg.policy.separate_exploration_tree_score_normalization = args.separate_exploration_tree_score_normalization
    cfg.policy.separate_exploration_tree_max_sequences = args.separate_exploration_tree_max_sequences
    cfg.policy.separate_exploration_tree_store_final_policy = args.separate_exploration_tree_store_final_policy
    cfg.policy.separate_exploration_tree_sequence_score_mode = args.separate_exploration_tree_sequence_score_mode
    cfg.policy.separate_exploration_tree_sequence_qstd_beta = args.separate_exploration_tree_sequence_qstd_beta
    cfg.policy.separate_exploration_tree_sequence_qstd_mode = args.separate_exploration_tree_sequence_qstd_mode
    cfg.policy.pure_exploration_episode_fraction = args.pure_exploration_episode_fraction
    cfg.policy.pure_exploration_commit_steps = args.pure_exploration_commit_steps
    cfg.policy.pure_exploration_sequence_temperature = args.pure_exploration_sequence_temperature
    cfg.policy.pure_exploration_store_exploration_policy = args.pure_exploration_store_exploration_policy
    if args.ensemble_eval_search_mode is not None:
        cfg.policy.ensemble_eval_search_mode = args.ensemble_eval_search_mode
    cfg.policy.ensemble_eval_ucb_beta = args.ensemble_eval_ucb_beta
    cfg.policy.ensemble_eval_uncertainty_decay = args.ensemble_eval_uncertainty_decay
    cfg.policy.use_wandb = False
    cfg.policy.profile_train_loop_timing = args.profile_timing
    cfg.policy.profile_train_loop_timing_freq = args.profile_timing_freq
    cfg.policy.profile_learn_timing = profile_learn_timing
    cfg.policy.record_collect_video = args.record_collect_video
    cfg.policy.collect_video_dir = collect_video_dir

    cfg.policy.model.ensemble_size = args.ensemble_size
    cfg.policy.model.observation_shape = 3 if args.include_remaining_time else 2
    cfg.policy.model.state_norm = args.state_norm
    if args.representation_last_linear_layer_init_zero is not None:
        cfg.policy.model.representation_last_linear_layer_init_zero = (
            args.representation_last_linear_layer_init_zero
        )
    if args.value_last_linear_layer_init_zero is not None:
        cfg.policy.model.value_last_linear_layer_init_zero = args.value_last_linear_layer_init_zero
    cfg.policy.model.norm_type = args.norm_type
    cfg.policy.model.ssl_norm_type = args.ssl_norm_type
    cfg.policy.model.random_prior_scale = args.random_prior_scale
    cfg.policy.model.value_prior_backprop_to_latent = args.value_prior_backprop_to_latent
    cfg.policy.model.value_prior_latent_normalization = args.value_prior_latent_normalization
    cfg.policy.model.value_prior_scalar_loss_weight = args.value_prior_scalar_loss_weight
    cfg.policy.model.encoder_random_prior_scale = args.encoder_random_prior_scale
    cfg.policy.model.encoder_random_prior_mode = args.encoder_random_prior_mode
    cfg.policy.model.encoder_random_prior_hidden_dim = args.encoder_random_prior_hidden_dim
    cfg.policy.model.encoder_random_prior_cell_size = args.encoder_random_prior_cell_size
    cfg.policy.model.encoder_random_prior_hash_size = args.encoder_random_prior_hash_size
    encoder_prior_input_mean = list(args.encoder_random_prior_input_mean)
    encoder_prior_input_scale = list(args.encoder_random_prior_input_scale)
    if args.include_remaining_time:
        encoder_prior_input_mean.append(0.5)
        encoder_prior_input_scale.append(0.5)
    cfg.policy.model.encoder_random_prior_input_mean = encoder_prior_input_mean
    cfg.policy.model.encoder_random_prior_input_scale = encoder_prior_input_scale
    cfg.policy.model.encoder_random_prior_output_norm = args.encoder_random_prior_output_norm
    cfg.policy.model.encoder_prior_composed_latent_normalization = (
        args.encoder_prior_composed_latent_normalization
    )
    cfg.policy.model.encoder_prior_composed_latent_norm_scale = (
        args.encoder_prior_composed_latent_norm_scale
    )
    cfg.policy.model.dynamics_random_prior_scale = args.dynamics_random_prior_scale
    cfg.policy.model.dynamics_random_prior_hidden_dim = args.dynamics_random_prior_hidden_dim
    cfg.policy.model.dynamics_random_prior_action_mode = args.dynamics_random_prior_action_mode
    cfg.policy.model.dynamics_encoder_latent_normalization = args.dynamics_encoder_latent_normalization
    cfg.policy.model.dynamics_encoder_latent_norm_scale = args.dynamics_encoder_latent_norm_scale
    cfg.policy.model.dynamics_trainable_latent_normalization = args.dynamics_trainable_latent_normalization
    cfg.policy.model.dynamics_trainable_latent_norm_scale = args.dynamics_trainable_latent_norm_scale
    cfg.policy.model.dynamics_trainable_latent_norm_eps = args.dynamics_trainable_latent_norm_eps
    cfg.policy.model.dynamics_prior_latent_norm_cap = args.dynamics_prior_latent_norm_cap
    cfg.policy.model.shared_representation = args.shared_representation
    cfg.policy.model.shared_prediction = args.shared_prediction
    cfg.policy.model.shared_projection = args.shared_projection
    cfg.policy.model.shared_intrinsic_reward_head = args.shared_intrinsic_reward_head
    cfg.policy.model.intrinsic_reward_target_transform = (
        args.intrinsic_reward_target_transform
    )
    cfg.policy.model.intrinsic_reward_log_reference = max(
        1e-12, args.intrinsic_reward_log_reference
    )
    cfg.policy.model.intrinsic_reward_log_epsilon = max(
        1e-12, args.intrinsic_reward_log_epsilon
    )
    cfg.policy.model.intrinsic_reward_log_max_raw = max(
        cfg.policy.model.intrinsic_reward_log_epsilon,
        args.intrinsic_reward_log_max_raw,
    )
    cfg.policy.rnd_exploration = args.rnd_exploration
    cfg.policy.rnd_intrinsic_reward_weight = args.rnd_intrinsic_reward_weight
    cfg.policy.rnd_learning_rate = args.rnd_learning_rate
    cfg.policy.rnd_hidden_dim = args.rnd_hidden_dim
    cfg.policy.rnd_feature_dim = args.rnd_feature_dim
    cfg.policy.rnd_train_batch_size = args.rnd_train_batch_size
    cfg.policy.rnd_predictor_update_proportion = args.rnd_predictor_update_proportion
    cfg.policy.rnd_train_on_new_data_only = args.rnd_train_on_new_data_only
    cfg.policy.rnd_train_from_fresh_collection_only = args.rnd_train_from_fresh_collection_only
    cfg.policy.rnd_fresh_update_batch_size = args.rnd_fresh_update_batch_size
    cfg.policy.rnd_train_observation_mode = args.rnd_train_observation_mode
    if args.rnd_observation_dim is not None:
        cfg.policy.rnd_observation_dim = args.rnd_observation_dim
    if args.rnd_observation_indices is not None:
        cfg.policy.rnd_observation_indices = [
            int(index.strip())
            for index in args.rnd_observation_indices.split(',')
            if index.strip()
        ]
    if args.rnd_observation_low is not None:
        cfg.policy.rnd_observation_low = [
            float(value.strip())
            for value in args.rnd_observation_low.split(',')
            if value.strip()
        ]
    if args.rnd_observation_high is not None:
        cfg.policy.rnd_observation_high = [
            float(value.strip())
            for value in args.rnd_observation_high.split(',')
            if value.strip()
        ]
    cfg.policy.rnd_ignore_elite_replay_for_predictor = args.rnd_ignore_elite_replay_for_predictor
    cfg.policy.rnd_bonus_normalization = args.rnd_bonus_normalization
    cfg.policy.rnd_bonus_clip = args.rnd_bonus_clip
    cfg.policy.rnd_store_collection_rewards = args.rnd_store_collection_rewards
    cfg.policy.rnd_update_stats_on_collection = args.rnd_update_stats_on_collection
    cfg.policy.rnd_use_stored_collection_rewards = args.rnd_use_stored_collection_rewards
    cfg.policy.rnd_intrinsic_target_mode = args.rnd_intrinsic_target_mode
    cfg.policy.rnd_use_learned_intrinsic_model = args.rnd_use_learned_intrinsic_model
    cfg.policy.rnd_intrinsic_discount_factor = args.rnd_intrinsic_discount_factor
    cfg.policy.rnd_intrinsic_td_steps = args.rnd_intrinsic_td_steps
    cfg.policy.rnd_intrinsic_bootstrap = args.rnd_intrinsic_bootstrap
    cfg.policy.rnd_target_snapshot = args.rnd_target_snapshot
    cfg.policy.rnd_target_snapshot_update_interval = args.rnd_target_snapshot_update_interval
    cfg.policy.rnd_target_snapshot_refresh_per_collect = args.rnd_target_snapshot_refresh_per_collect
    cfg.policy.rnd_target_snapshot_refresh_after_fresh_update = (
        args.rnd_target_snapshot_refresh_after_fresh_update
    )
    cfg.policy.rnd_initialization_seed = args.rnd_initialization_seed
    cfg.policy.rnd_landscape_query = args.rnd_landscape_query
    cfg.policy.rnd_landscape_query_count = args.rnd_landscape_query_count
    cfg.policy.rnd_landscape_query_hidden_dim = args.rnd_landscape_query_hidden_dim
    cfg.policy.rnd_landscape_query_learning_rate = args.rnd_landscape_query_learning_rate
    cfg.policy.rnd_landscape_query_weight_decay = max(
        0.0, args.rnd_landscape_query_weight_decay
    )
    cfg.policy.rnd_landscape_query_loss_weight = args.rnd_landscape_query_loss_weight
    cfg.policy.rnd_landscape_query_max_horizon = args.rnd_landscape_query_max_horizon
    cfg.policy.rnd_landscape_query_min_snapshots = args.rnd_landscape_query_min_snapshots
    cfg.policy.rnd_landscape_query_diversity_weight = args.rnd_landscape_query_diversity_weight
    cfg.policy.rnd_landscape_query_centered_prediction_weight = (
        args.rnd_landscape_query_centered_prediction_weight
    )
    cfg.policy.rnd_landscape_query_grad_clip = args.rnd_landscape_query_grad_clip
    cfg.policy.rnd_landscape_query_reader_use_context = (
        args.rnd_landscape_query_reader_use_context
    )
    cfg.policy.rnd_landscape_query_member_specific_reader = (
        args.rnd_landscape_query_member_specific_reader
    )
    cfg.policy.rnd_landscape_query_reader_architecture = (
        args.rnd_landscape_query_reader_architecture
    )
    cfg.policy.rnd_landscape_query_reader_hidden_channels = (
        max(1, args.rnd_landscape_query_reader_hidden_dim),
    )
    cfg.policy.rnd_landscape_query_reader_norm_type = (
        args.rnd_landscape_query_reader_norm_type
    )
    cfg.policy.rnd_landscape_query_reader_last_linear_layer_init_zero = (
        args.rnd_landscape_query_reader_last_linear_layer_init_zero
    )
    cfg.policy.rnd_landscape_query_target_transform = (
        args.rnd_landscape_query_target_transform
    )
    cfg.policy.rnd_landscape_query_target_log_reference = max(
        1e-12, args.rnd_landscape_query_target_log_reference
    )
    cfg.policy.rnd_landscape_query_error_epsilon = max(
        1e-12, args.rnd_landscape_query_error_epsilon
    )
    cfg.policy.rnd_landscape_query_max_raw_error = (
        max(1e-12, args.rnd_landscape_query_max_raw_error)
        if args.rnd_landscape_query_max_raw_error is not None else None
    )
    cfg.policy.rnd_landscape_query_normalized_raw_reference_size = max(
        1, args.rnd_landscape_query_normalized_raw_reference_size
    )
    cfg.policy.rnd_landscape_query_normalized_raw_clip = max(
        1e-6, args.rnd_landscape_query_normalized_raw_clip
    )
    cfg.policy.rnd_landscape_query_normalized_raw_min_scale = max(
        1e-12, args.rnd_landscape_query_normalized_raw_min_scale
    )
    cfg.policy.rnd_landscape_query_gradient_diagnostics_interval = max(
        0, args.rnd_landscape_query_gradient_diagnostics_interval
    )
    cfg.policy.rnd_landscape_query_train_world_model = (
        args.rnd_landscape_query_train_world_model
    )
    cfg.policy.rnd_landscape_query_use_tree_rewards = (
        args.rnd_landscape_query_use_tree_rewards
    )
    cfg.policy.rnd_landscape_query_initialization_seed = (
        args.rnd_landscape_query_initialization_seed
    )
    cfg.policy.rnd_aux_confirmatory_condition = args.rnd_aux_confirmatory_condition
    cfg.policy.rnd_aux_confirmatory_plan_version = (
        args.rnd_aux_confirmatory_plan_version
    )
    cfg.policy.intrinsic_reward_source = args.intrinsic_reward_source
    cfg.policy.episodic_novelty_diagnostics = args.episodic_novelty_diagnostics
    cfg.policy.episodic_novelty_encoder_member = args.episodic_novelty_encoder_member
    cfg.policy.episodic_novelty_knn_k = args.episodic_novelty_knn_k
    cfg.policy.episodic_novelty_kernel_cluster_distance = (
        args.episodic_novelty_kernel_cluster_distance
    )
    cfg.policy.episodic_novelty_kernel_epsilon = args.episodic_novelty_kernel_epsilon
    cfg.policy.episodic_novelty_pseudo_count_constant = (
        args.episodic_novelty_pseudo_count_constant
    )
    cfg.policy.episodic_novelty_maximum_similarity = (
        args.episodic_novelty_maximum_similarity
    )
    cfg.policy.episodic_novelty_zero_until_knn_full = (
        args.episodic_novelty_zero_until_knn_full
    )
    cfg.policy.episodic_novelty_reward_scale = args.episodic_novelty_reward_scale
    cfg.policy.episodic_novelty_reward_clip = args.episodic_novelty_reward_clip
    cfg.policy.episodic_novelty_shadow_predictions = args.episodic_novelty_shadow_predictions
    cfg.policy.episodic_novelty_history_shadow = args.episodic_novelty_history_shadow
    cfg.policy.episodic_novelty_history_learning_rate = args.episodic_novelty_history_learning_rate
    cfg.policy.episodic_novelty_history_train_epochs = args.episodic_novelty_history_train_epochs
    cfg.policy.episodic_novelty_history_train_batch_size = (
        args.episodic_novelty_history_train_batch_size
    )
    cfg.policy.episodic_novelty_history_updates_per_collect = (
        args.episodic_novelty_history_updates_per_collect
    )
    cfg.policy.episodic_novelty_history_replay_capacity = (
        args.episodic_novelty_history_replay_capacity
    )
    cfg.policy.episodic_novelty_history_recent_phase_count = (
        args.episodic_novelty_history_recent_phase_count
    )
    cfg.policy.episodic_novelty_history_recent_sample_ratio = (
        args.episodic_novelty_history_recent_sample_ratio
    )
    cfg.policy.episodic_novelty_history_grad_clip = args.episodic_novelty_history_grad_clip
    cfg.policy.model.episodic_history_dim = (
        args.episodic_novelty_history_dim if args.episodic_novelty_history_shadow else 0
    )
    cfg.policy.model.episodic_history_hidden_dim = args.episodic_novelty_history_hidden_dim
    cfg.policy.state_value_disagreement_reward_scale = args.state_value_disagreement_reward_scale
    cfg.policy.state_value_disagreement_reward_clip = args.state_value_disagreement_reward_clip
    cfg.policy.state_value_disagreement_inference_batch_size = (
        args.state_value_disagreement_inference_batch_size
    )
    cfg.policy.state_value_disagreement_store_collection_rewards = (
        args.state_value_disagreement_store_collection_rewards
    )
    cfg.policy.state_value_disagreement_use_stored_collection_rewards = (
        args.state_value_disagreement_use_stored_collection_rewards
    )
    cfg.policy.state_value_disagreement_intrinsic_target_mode = (
        args.state_value_disagreement_intrinsic_target_mode
    )
    cfg.policy.state_density_position_bin_width = args.state_density_position_bin_width
    cfg.policy.state_density_velocity_bin_width = args.state_density_velocity_bin_width
    cfg.policy.state_density_neighbor_radius = args.state_density_neighbor_radius
    cfg.policy.state_density_snapshot_update_interval = args.state_density_snapshot_update_interval
    cfg.policy.state_density_reward_scale = args.state_density_reward_scale
    cfg.policy.state_density_reward_clip = args.state_density_reward_clip
    cfg.policy.state_density_intrinsic_target_mode = args.state_density_intrinsic_target_mode
    cfg.policy.state_density_train_on_intrinsic_replay_only = (
        args.state_density_train_on_intrinsic_replay_only
    )
    cfg.policy.qstd_exploration = args.qstd_exploration
    cfg.policy.qstd_intrinsic_reward_weight = args.qstd_intrinsic_reward_weight
    cfg.policy.qstd_store_collection_rewards = args.qstd_store_collection_rewards
    cfg.policy.qstd_reward_aggregation = args.qstd_reward_aggregation
    cfg.policy.qstd_reward_top_k = args.qstd_reward_top_k
    cfg.policy.qstd_reward_scale = args.qstd_reward_scale
    cfg.policy.qstd_reward_clip = args.qstd_reward_clip
    cfg.policy.qstd_reward_num_simulations = args.qstd_reward_num_simulations
    cfg.policy.qstd_reward_search_mode = args.qstd_reward_search_mode
    cfg.policy.qstd_recompute_replay_rewards = args.qstd_recompute_replay_rewards
    cfg.policy.qstd_one_step_include_reward = args.qstd_one_step_include_reward
    cfg.policy.qstd_one_step_use_value_calibration = args.qstd_one_step_use_value_calibration
    cfg.policy.qstd_target_snapshot = args.qstd_target_snapshot
    cfg.policy.qstd_target_snapshot_refresh_per_collect = (
        args.qstd_target_snapshot_refresh_per_collect
    )
    cfg.policy.qstd_target_inference_batch_size = args.qstd_target_inference_batch_size
    cfg.policy.qstd_intrinsic_target_mode = args.qstd_intrinsic_target_mode
    cfg.policy.dynamics_disagreement_exploration = args.dynamics_disagreement_exploration
    cfg.policy.dynamics_disagreement_intrinsic_reward_weight = args.dynamics_disagreement_intrinsic_reward_weight
    cfg.policy.dynamics_disagreement_shared_root_latent = args.dynamics_disagreement_shared_root_latent
    cfg.policy.dynamics_disagreement_normalize_latent = args.dynamics_disagreement_normalize_latent
    cfg.policy.dynamics_disagreement_space = args.dynamics_disagreement_space
    cfg.policy.dynamics_disagreement_distance_metric = args.dynamics_disagreement_distance_metric
    cfg.policy.dynamics_disagreement_reward_clip = args.dynamics_disagreement_reward_clip
    cfg.policy.dynamics_disagreement_train_intrinsic_value = args.dynamics_disagreement_train_intrinsic_value
    cfg.policy.dynamics_prior_alignment_loss_weight = args.dynamics_prior_alignment_loss_weight
    cfg.policy.dynamics_prior_alignment_target_normalization = args.dynamics_prior_alignment_target_normalization
    cfg.policy.store_selected_intrinsic_rewards = bool(
        args.dynamics_disagreement_train_intrinsic_value
        or args.rnd_store_collection_rewards
        or args.rnd_use_stored_collection_rewards
        or args.qstd_store_collection_rewards
        or args.state_value_disagreement_store_collection_rewards
        or args.episodic_novelty_diagnostics
        or args.intrinsic_reward_source == "state_density"
        or (
            args.intrinsic_reward_source == "qstd"
            and not args.qstd_recompute_replay_rewards
        )
        or args.intrinsic_episode_replay
        or (
            args.intrinsic_reward_source == "dynamics_disagreement"
            and args.intrinsic_tree_leaf_value_source == "intrinsic_value"
        )
    )
    cfg.policy.separate_intrinsic_tree = args.separate_intrinsic_tree
    cfg.policy.mixed_reward_tree = args.mixed_reward_tree
    cfg.policy.main_tree_policy_prior_mode = args.main_tree_policy_prior_mode
    cfg.policy.main_tree_force_root_action_coverage = (
        args.main_tree_force_root_action_coverage
    )
    cfg.policy.collection_policy_mode = args.collection_policy_mode
    cfg.policy.intrinsic_tree_num_simulations = args.intrinsic_tree_num_simulations
    cfg.policy.intrinsic_tree_add_exploration_noise = args.intrinsic_tree_add_exploration_noise
    cfg.policy.intrinsic_tree_prior_mode = args.intrinsic_tree_prior_mode
    cfg.policy.intrinsic_tree_force_root_action_coverage = args.intrinsic_tree_force_root_action_coverage
    cfg.policy.intrinsic_tree_pure_exploration_episodes = args.intrinsic_tree_pure_exploration_episodes
    cfg.policy.intrinsic_value_train_on_pure_exploration_only = (
        args.intrinsic_value_train_on_pure_exploration_only
    )
    cfg.policy.intrinsic_tree_pure_episode_store_intrinsic_policy = (
        args.intrinsic_tree_pure_episode_store_intrinsic_policy
    )
    cfg.policy.intrinsic_tree_final_score_temperature = args.intrinsic_tree_final_score_temperature
    cfg.policy.intrinsic_tree_store_final_policy = args.intrinsic_tree_store_final_policy
    cfg.policy.intrinsic_tree_eval = args.intrinsic_tree_eval
    cfg.policy.intrinsic_only_training = args.intrinsic_only_training
    cfg.policy.intrinsic_tree_normalize_q = args.intrinsic_tree_normalize_q
    cfg.policy.intrinsic_tree_immediate_reward_weight = args.intrinsic_tree_immediate_reward_weight
    cfg.policy.intrinsic_tree_leaf_value_source = args.intrinsic_tree_leaf_value_source
    cfg.policy.intrinsic_tree_leaf_value_weight = args.intrinsic_tree_leaf_value_weight
    cfg.policy.intrinsic_tree_leaf_value_clip = args.intrinsic_tree_leaf_value_clip
    cfg.policy.intrinsic_value_bootstrap = args.intrinsic_value_bootstrap
    cfg.policy.collect_random_action_epsilon = args.collect_random_action_epsilon
    cfg.policy.main_tree_trace_diagnostics = args.main_tree_trace_diagnostics
    cfg.policy.main_tree_trace_min_train_iter = args.main_tree_trace_min_train_iter
    cfg.policy.main_tree_trace_max_per_train_iter = args.main_tree_trace_max_per_train_iter
    cfg.policy.main_tree_trace_max_depth = args.main_tree_trace_max_depth
    cfg.policy.main_tree_trace_max_nodes = args.main_tree_trace_max_nodes
    cfg.policy.main_tree_trace_min_q_std = args.main_tree_trace_min_q_std
    cfg.policy.main_tree_trace_selected_action = args.main_tree_trace_selected_action
    cfg.policy.main_tree_trace_min_position = args.main_tree_trace_min_position
    cfg.policy.main_tree_trace_max_position = args.main_tree_trace_max_position
    cfg.policy.main_tree_trace_min_velocity = args.main_tree_trace_min_velocity
    cfg.policy.main_tree_trace_max_velocity = args.main_tree_trace_max_velocity
    cfg.policy.intrinsic_tree_trace_diagnostics = args.intrinsic_tree_trace_diagnostics
    cfg.policy.intrinsic_tree_trace_best_action = args.intrinsic_tree_trace_best_action
    cfg.policy.intrinsic_tree_trace_min_train_iter = args.intrinsic_tree_trace_min_train_iter
    cfg.policy.intrinsic_tree_trace_max_per_train_iter = args.intrinsic_tree_trace_max_per_train_iter
    cfg.policy.intrinsic_tree_trace_max_depth = args.intrinsic_tree_trace_max_depth
    cfg.policy.intrinsic_tree_trace_max_nodes = args.intrinsic_tree_trace_max_nodes
    cfg.policy.intrinsic_tree_trace_min_position = args.intrinsic_tree_trace_min_position
    cfg.policy.intrinsic_tree_trace_max_position = args.intrinsic_tree_trace_max_position
    cfg.policy.intrinsic_tree_trace_min_velocity = args.intrinsic_tree_trace_min_velocity
    cfg.policy.intrinsic_tree_trace_max_velocity = args.intrinsic_tree_trace_max_velocity
    cfg.policy.intrinsic_model_loss_weight = args.intrinsic_model_loss_weight
    cfg.policy.intrinsic_reward_loss_weight = args.intrinsic_reward_loss_weight
    cfg.policy.intrinsic_value_loss_weight = args.intrinsic_value_loss_weight
    cfg.policy.intrinsic_train_value_head = args.intrinsic_train_value_head
    cfg.policy.intrinsic_learning_rate = args.intrinsic_learning_rate
    cfg.policy.intrinsic_head_only = args.intrinsic_head_only
    cfg.policy.intrinsic_loss_type = args.intrinsic_loss_type
    cfg.policy.intrinsic_huber_delta = args.intrinsic_huber_delta
    cfg.policy.intrinsic_reward_target_clip = args.intrinsic_reward_target_clip
    cfg.policy.intrinsic_value_target_clip = args.intrinsic_value_target_clip
    cfg.policy.latent_novelty_diagnostics = args.latent_novelty_diagnostics
    cfg.policy.latent_novelty_learning_rate = args.latent_novelty_learning_rate
    cfg.policy.latent_novelty_train_batch_size = args.latent_novelty_train_batch_size
    cfg.policy.latent_novelty_temperature = args.latent_novelty_temperature
    cfg.policy.latent_novelty_alignment_loss_weight = args.latent_novelty_alignment_loss_weight
    cfg.policy.model.latent_novelty_dim = args.latent_novelty_dim if args.latent_novelty_diagnostics else 0
    cfg.policy.model.latent_novelty_hidden_dim = args.latent_novelty_hidden_dim
    cfg.policy.model.latent_novelty_prior_scale = args.latent_novelty_prior_scale
    cfg.policy.elite_episode_replay = args.elite_replay
    cfg.policy.elite_episode_buffer_size = args.elite_buffer_size
    cfg.policy.elite_episode_sample_ratio = args.elite_sample_ratio
    cfg.policy.elite_episode_min_return = args.elite_min_return
    cfg.policy.elite_episode_monte_carlo_value_target = args.elite_mc_value_target
    cfg.policy.elite_episode_start_stratification = args.elite_start_stratification
    cfg.policy.elite_episode_start_observation_index = args.elite_start_observation_index
    cfg.policy.elite_episode_start_bin_edges = (
        [float(value.strip()) for value in args.elite_start_bin_edges.split(',') if value.strip()]
        if isinstance(args.elite_start_bin_edges, str)
        else args.elite_start_bin_edges
    )
    cfg.policy.intrinsic_episode_replay = args.intrinsic_episode_replay
    cfg.policy.intrinsic_episode_buffer_size = args.intrinsic_episode_buffer_size
    cfg.policy.intrinsic_episode_sample_ratio = args.intrinsic_episode_sample_ratio
    cfg.policy.intrinsic_episode_min_score = args.intrinsic_episode_min_score
    cfg.policy.intrinsic_episode_score_mode = args.intrinsic_episode_score_mode
    cfg.policy.intrinsic_episode_peak_window_before = args.intrinsic_episode_peak_window_before
    cfg.policy.intrinsic_episode_peak_window_after = args.intrinsic_episode_peak_window_after
    cfg.policy.intrinsic_episode_max_age_phases = args.intrinsic_episode_max_age_phases
    cfg.policy.save_intrinsic_episode_trajectories = args.save_intrinsic_episode_trajectories
    cfg.policy.intrinsic_episode_save_dir = args.intrinsic_episode_save_dir
    cfg.policy.collection_diagnostics = args.collection_diagnostics
    cfg.policy.collection_root_action_diagnostics = args.root_action_diagnostics
    cfg.policy.collection_ssl_diagnostics = args.collection_ssl_diagnostics
    cfg.policy.collection_ssl_diagnostics_batch_size = args.collection_ssl_diagnostics_batch_size
    cfg.policy.encoder_prior_grid_diagnostics = args.encoder_prior_grid_diagnostics
    cfg.policy.encoder_prior_grid_position_min = args.encoder_prior_grid_position_min
    cfg.policy.encoder_prior_grid_position_max = args.encoder_prior_grid_position_max
    cfg.policy.encoder_prior_grid_position_count = args.encoder_prior_grid_position_count
    cfg.policy.encoder_prior_grid_velocity_min = args.encoder_prior_grid_velocity_min
    cfg.policy.encoder_prior_grid_velocity_max = args.encoder_prior_grid_velocity_max
    cfg.policy.encoder_prior_grid_velocity_count = args.encoder_prior_grid_velocity_count
    cfg.policy.encoder_prior_imagined_novelty_diagnostics = (
        args.encoder_prior_imagined_novelty_diagnostics
    )
    cfg.policy.encoder_prior_imagined_novelty_root_count = (
        args.encoder_prior_imagined_novelty_root_count
    )
    cfg.policy.encoder_prior_imagined_novelty_bank_size = (
        args.encoder_prior_imagined_novelty_bank_size
    )
    cfg.policy.encoder_prior_imagined_novelty_history_limit = (
        args.encoder_prior_imagined_novelty_history_limit
    )
    cfg.policy.encoder_prior_imagined_novelty_min_bank_size = (
        args.encoder_prior_imagined_novelty_min_bank_size
    )
    cfg.policy.encoder_prior_imagined_novelty_depth = args.encoder_prior_imagined_novelty_depth
    cfg.policy.encoder_prior_imagined_novelty_max_sequences = (
        args.encoder_prior_imagined_novelty_max_sequences
    )
    cfg.policy.encoder_prior_imagined_novelty_knn_k = args.encoder_prior_imagined_novelty_knn_k
    cfg.policy.encoder_prior_imagined_novelty_batch_size = (
        args.encoder_prior_imagined_novelty_batch_size
    )
    cfg.policy.encoder_prior_imagined_novelty_feature_std_eps = (
        args.encoder_prior_imagined_novelty_feature_std_eps
    )
    cfg.policy.encoder_prior_imagined_novelty_position_scale = (
        args.encoder_prior_imagined_novelty_position_scale
    )
    cfg.policy.encoder_prior_imagined_novelty_velocity_scale = (
        args.encoder_prior_imagined_novelty_velocity_scale
    )
    cfg.policy.normalization_mode_diagnostics = args.normalization_mode_diagnostics
    cfg.policy.normalization_mode_diagnostics_depth = args.normalization_mode_diagnostics_depth
    cfg.policy.normalization_mode_diagnostics_sample_size = args.normalization_mode_diagnostics_sample_size
    cfg.policy.prediction_gradient_diagnostics = args.prediction_gradient_diagnostics
    cfg.policy.fixed_sequence_intrinsic_diagnostics = args.fixed_sequence_intrinsic_diagnostics
    cfg.policy.fixed_sequence_intrinsic_max_states = args.fixed_sequence_intrinsic_max_states
    cfg.policy.fixed_sequence_intrinsic_sequences = args.fixed_sequence_intrinsic_sequences
    cfg.policy.diagnostics_dir = args.diagnostics_dir
    cfg.policy.save_elite_episode_trajectories = args.save_elite_trajectories
    cfg.policy.elite_episode_save_dir = args.elite_save_dir

    return cfg, create_cfg


def print_run_summary(cfg, create_cfg, args: argparse.Namespace) -> None:
    print("MountainCar ensemble MuZero run")
    print(f"  exp_name: {cfg.exp_name}")
    print(f"  seed: {args.seed}")
    print(f"  ensemble_size: {cfg.policy.model.ensemble_size}")
    print(f"  weight_decay: {cfg.policy.get('weight_decay', None)}")
    print(
        "  representation_prediction_weight_decay: "
        f"{cfg.policy.representation_prediction_weight_decay}"
    )
    print(f"  latent_variance_loss_weight: {cfg.policy.latent_variance_loss_weight}")
    print(f"  latent_covariance_loss_weight: {cfg.policy.latent_covariance_loss_weight}")
    print(f"  latent_variance_target: {cfg.policy.latent_variance_target}")
    print(f"  latent_regularization_eps: {cfg.policy.latent_regularization_eps}")
    print(f"  prediction_gradient_diagnostics: {cfg.policy.prediction_gradient_diagnostics}")
    print(f"  root_dirichlet_alpha: {cfg.policy.root_dirichlet_alpha}")
    print(f"  root_noise_weight: {cfg.policy.root_noise_weight}")
    print(f"  random_prior_scale: {cfg.policy.model.random_prior_scale}")
    print(f"  value_prior_backprop_to_latent: {cfg.policy.model.value_prior_backprop_to_latent}")
    print(f"  value_prior_latent_normalization: {cfg.policy.model.value_prior_latent_normalization}")
    print(f"  value_prior_scalar_loss_weight: {cfg.policy.model.value_prior_scalar_loss_weight}")
    print(f"  encoder_random_prior_scale: {cfg.policy.model.encoder_random_prior_scale}")
    print(f"  encoder_random_prior_mode: {cfg.policy.model.encoder_random_prior_mode}")
    print(f"  encoder_random_prior_hidden_dim: {cfg.policy.model.encoder_random_prior_hidden_dim}")
    print(f"  encoder_random_prior_cell_size: {cfg.policy.model.encoder_random_prior_cell_size}")
    print(f"  encoder_random_prior_hash_size: {cfg.policy.model.encoder_random_prior_hash_size}")
    print(f"  encoder_random_prior_input_mean: {cfg.policy.model.encoder_random_prior_input_mean}")
    print(f"  encoder_random_prior_input_scale: {cfg.policy.model.encoder_random_prior_input_scale}")
    print(f"  encoder_random_prior_output_norm: {cfg.policy.model.encoder_random_prior_output_norm}")
    print(
        "  encoder_prior_composed_latent_normalization: "
        f"{cfg.policy.model.encoder_prior_composed_latent_normalization}"
    )
    print(
        "  encoder_prior_composed_latent_norm_scale: "
        f"{cfg.policy.model.encoder_prior_composed_latent_norm_scale}"
    )
    print(f"  dynamics_random_prior_scale: {cfg.policy.model.dynamics_random_prior_scale}")
    print(f"  dynamics_random_prior_hidden_dim: {cfg.policy.model.dynamics_random_prior_hidden_dim}")
    print(f"  dynamics_random_prior_action_mode: {cfg.policy.model.dynamics_random_prior_action_mode}")
    print(f"  dynamics_encoder_latent_normalization: {cfg.policy.model.dynamics_encoder_latent_normalization}")
    print(f"  dynamics_encoder_latent_norm_scale: {cfg.policy.model.dynamics_encoder_latent_norm_scale}")
    print(f"  dynamics_trainable_latent_normalization: {cfg.policy.model.dynamics_trainable_latent_normalization}")
    print(f"  dynamics_trainable_latent_norm_scale: {cfg.policy.model.dynamics_trainable_latent_norm_scale}")
    print(f"  dynamics_trainable_latent_norm_eps: {cfg.policy.model.dynamics_trainable_latent_norm_eps}")
    print(f"  dynamics_prior_latent_norm_cap: {cfg.policy.model.dynamics_prior_latent_norm_cap}")
    print(f"  shared_representation: {cfg.policy.model.shared_representation}")
    print(f"  shared_prediction: {cfg.policy.model.shared_prediction}")
    print(f"  shared_projection: {cfg.policy.model.shared_projection}")
    print(
        "  shared_intrinsic_reward_head: "
        f"{cfg.policy.model.shared_intrinsic_reward_head}"
    )
    print(
        "  intrinsic_reward_target_transform: "
        f"{cfg.policy.model.intrinsic_reward_target_transform}"
    )
    print(
        "  intrinsic_reward_log_reference: "
        f"{cfg.policy.model.intrinsic_reward_log_reference}"
    )
    print(f"  rnd_exploration: {cfg.policy.rnd_exploration}")
    print(f"  rnd_intrinsic_reward_weight: {cfg.policy.rnd_intrinsic_reward_weight}")
    print(f"  rnd_learning_rate: {cfg.policy.rnd_learning_rate}")
    print(f"  rnd_hidden_dim: {cfg.policy.rnd_hidden_dim}")
    print(f"  rnd_feature_dim: {cfg.policy.rnd_feature_dim}")
    print(f"  rnd_initialization_seed: {cfg.policy.rnd_initialization_seed}")
    print(f"  rnd_train_batch_size: {cfg.policy.rnd_train_batch_size}")
    print(f"  rnd_predictor_update_proportion: {cfg.policy.rnd_predictor_update_proportion}")
    print(f"  rnd_train_on_new_data_only: {cfg.policy.rnd_train_on_new_data_only}")
    print(
        "  rnd_train_from_fresh_collection_only: "
        f"{cfg.policy.rnd_train_from_fresh_collection_only}"
    )
    print(f"  rnd_fresh_update_batch_size: {cfg.policy.rnd_fresh_update_batch_size}")
    print(f"  rnd_train_observation_mode: {cfg.policy.rnd_train_observation_mode}")
    print(f"  rnd_observation_dim: {cfg.policy.get('rnd_observation_dim', None)}")
    print(f"  rnd_observation_indices: {cfg.policy.get('rnd_observation_indices', None)}")
    print(f"  rnd_observation_low: {cfg.policy.get('rnd_observation_low', None)}")
    print(f"  rnd_observation_high: {cfg.policy.get('rnd_observation_high', None)}")
    print(f"  rnd_ignore_elite_replay_for_predictor: {cfg.policy.rnd_ignore_elite_replay_for_predictor}")
    print(f"  rnd_bonus_normalization: {cfg.policy.rnd_bonus_normalization}")
    print(f"  rnd_bonus_clip: {cfg.policy.rnd_bonus_clip}")
    print(f"  rnd_store_collection_rewards: {cfg.policy.rnd_store_collection_rewards}")
    print(f"  rnd_update_stats_on_collection: {cfg.policy.rnd_update_stats_on_collection}")
    print(f"  rnd_use_stored_collection_rewards: {cfg.policy.rnd_use_stored_collection_rewards}")
    print(f"  rnd_intrinsic_target_mode: {cfg.policy.rnd_intrinsic_target_mode}")
    print(f"  rnd_use_learned_intrinsic_model: {cfg.policy.rnd_use_learned_intrinsic_model}")
    print(f"  rnd_intrinsic_discount_factor: {cfg.policy.rnd_intrinsic_discount_factor}")
    print(f"  rnd_intrinsic_td_steps: {cfg.policy.rnd_intrinsic_td_steps}")
    print(f"  rnd_intrinsic_bootstrap: {cfg.policy.rnd_intrinsic_bootstrap}")
    print(f"  rnd_target_snapshot: {cfg.policy.rnd_target_snapshot}")
    print(
        "  rnd_target_snapshot_refresh_per_collect: "
        f"{cfg.policy.rnd_target_snapshot_refresh_per_collect}"
    )
    print(
        "  rnd_target_snapshot_refresh_after_fresh_update: "
        f"{cfg.policy.rnd_target_snapshot_refresh_after_fresh_update}"
    )
    print(f"  rnd_target_snapshot_update_interval: {cfg.policy.rnd_target_snapshot_update_interval}")
    print(f"  rnd_landscape_query: {cfg.policy.rnd_landscape_query}")
    print(f"  rnd_landscape_query_count: {cfg.policy.rnd_landscape_query_count}")
    print(f"  rnd_landscape_query_min_snapshots: {cfg.policy.rnd_landscape_query_min_snapshots}")
    print(
        "  rnd_landscape_query_reader_use_context: "
        f"{cfg.policy.rnd_landscape_query_reader_use_context}"
    )
    print(
        "  rnd_landscape_query_weight_decay: "
        f"{cfg.policy.rnd_landscape_query_weight_decay}"
    )
    print(
        "  rnd_landscape_query_member_specific_reader: "
        f"{cfg.policy.rnd_landscape_query_member_specific_reader}"
    )
    print(
        "  rnd_landscape_query_reader_architecture: "
        f"{cfg.policy.rnd_landscape_query_reader_architecture}"
    )
    print(
        "  rnd_landscape_query_reader_hidden_channels: "
        f"{cfg.policy.rnd_landscape_query_reader_hidden_channels}"
    )
    print(
        "  rnd_landscape_query_reader_norm_type: "
        f"{cfg.policy.rnd_landscape_query_reader_norm_type}"
    )
    print(
        "  rnd_landscape_query_reader_last_linear_layer_init_zero: "
        f"{cfg.policy.rnd_landscape_query_reader_last_linear_layer_init_zero}"
    )
    print(
        "  rnd_landscape_query_target_transform: "
        f"{cfg.policy.rnd_landscape_query_target_transform}"
    )
    print(
        "  rnd_landscape_query_target_log_reference: "
        f"{cfg.policy.rnd_landscape_query_target_log_reference}"
    )
    print(
        "  rnd_landscape_query_error_epsilon: "
        f"{cfg.policy.rnd_landscape_query_error_epsilon}"
    )
    print(
        "  rnd_landscape_query_max_raw_error: "
        f"{cfg.policy.rnd_landscape_query_max_raw_error}"
    )
    print(
        "  rnd_landscape_query_normalized_raw_reference_size: "
        f"{cfg.policy.rnd_landscape_query_normalized_raw_reference_size}"
    )
    print(
        "  rnd_landscape_query_normalized_raw_clip: "
        f"{cfg.policy.rnd_landscape_query_normalized_raw_clip}"
    )
    print(
        "  rnd_landscape_query_normalized_raw_min_scale: "
        f"{cfg.policy.rnd_landscape_query_normalized_raw_min_scale}"
    )
    print(
        "  rnd_landscape_query_gradient_diagnostics_interval: "
        f"{cfg.policy.rnd_landscape_query_gradient_diagnostics_interval}"
    )
    print(
        "  rnd_landscape_query_train_world_model: "
        f"{cfg.policy.rnd_landscape_query_train_world_model}"
    )
    print(
        "  rnd_landscape_query_use_tree_rewards: "
        f"{cfg.policy.rnd_landscape_query_use_tree_rewards}"
    )
    print(
        "  rnd_aux_confirmatory_condition: "
        f"{cfg.policy.rnd_aux_confirmatory_condition}"
    )
    print(
        "  rnd_aux_confirmatory_plan_version: "
        f"{cfg.policy.rnd_aux_confirmatory_plan_version}"
    )
    print(
        "  rnd_landscape_query_initialization_seed: "
        f"{cfg.policy.rnd_landscape_query_initialization_seed}"
    )
    print(f"  intrinsic_reward_source: {cfg.policy.intrinsic_reward_source}")
    print(f"  episodic_novelty_diagnostics: {cfg.policy.episodic_novelty_diagnostics}")
    print(f"  episodic_novelty_encoder_member: {cfg.policy.episodic_novelty_encoder_member}")
    print(f"  episodic_novelty_knn_k: {cfg.policy.episodic_novelty_knn_k}")
    print(
        "  episodic_novelty_kernel: "
        f"cluster_distance={cfg.policy.episodic_novelty_kernel_cluster_distance}, "
        f"epsilon={cfg.policy.episodic_novelty_kernel_epsilon}, "
        f"c={cfg.policy.episodic_novelty_pseudo_count_constant}, "
        f"maximum_similarity={cfg.policy.episodic_novelty_maximum_similarity}"
    )
    print(
        "  episodic_novelty_zero_until_knn_full: "
        f"{cfg.policy.episodic_novelty_zero_until_knn_full}"
    )
    print(f"  episodic_novelty_reward_scale: {cfg.policy.episodic_novelty_reward_scale}")
    print(f"  episodic_novelty_reward_clip: {cfg.policy.episodic_novelty_reward_clip}")
    print(
        "  episodic_novelty_shadow_predictions: "
        f"{cfg.policy.episodic_novelty_shadow_predictions}"
    )
    print(f"  episodic_novelty_history_shadow: {cfg.policy.episodic_novelty_history_shadow}")
    print(f"  episodic_novelty_history_dim: {cfg.policy.model.episodic_history_dim}")
    print(
        "  episodic_novelty_history_learning_rate: "
        f"{cfg.policy.episodic_novelty_history_learning_rate}"
    )
    print(
        "  episodic_novelty_history_training: "
        f"updates={cfg.policy.episodic_novelty_history_updates_per_collect}, "
        f"batch_episodes={cfg.policy.episodic_novelty_history_train_batch_size}, "
        f"replay_capacity={cfg.policy.episodic_novelty_history_replay_capacity}, "
        f"recent_phases={cfg.policy.episodic_novelty_history_recent_phase_count}, "
        f"recent_ratio={cfg.policy.episodic_novelty_history_recent_sample_ratio}"
    )
    print(
        "  state_value_disagreement_reward_scale: "
        f"{cfg.policy.state_value_disagreement_reward_scale}"
    )
    print(
        "  state_value_disagreement_reward_clip: "
        f"{cfg.policy.state_value_disagreement_reward_clip}"
    )
    print(
        "  state_value_disagreement_inference_batch_size: "
        f"{cfg.policy.state_value_disagreement_inference_batch_size}"
    )
    print(
        "  state_value_disagreement_store_collection_rewards: "
        f"{cfg.policy.state_value_disagreement_store_collection_rewards}"
    )
    print(
        "  state_value_disagreement_use_stored_collection_rewards: "
        f"{cfg.policy.state_value_disagreement_use_stored_collection_rewards}"
    )
    print(
        "  state_value_disagreement_intrinsic_target_mode: "
        f"{cfg.policy.state_value_disagreement_intrinsic_target_mode}"
    )
    print(
        "  state_density_bins: "
        f"position={cfg.policy.state_density_position_bin_width}, "
        f"velocity={cfg.policy.state_density_velocity_bin_width}, "
        f"neighbor_radius={cfg.policy.state_density_neighbor_radius}"
    )
    print(
        "  state_density_snapshot_update_interval: "
        f"{cfg.policy.state_density_snapshot_update_interval}"
    )
    print(
        "  state_density_reward: "
        f"scale={cfg.policy.state_density_reward_scale}, "
        f"clip={cfg.policy.state_density_reward_clip}"
    )
    print(
        "  state_density_intrinsic_target_mode: "
        f"{cfg.policy.state_density_intrinsic_target_mode}"
    )
    print(
        "  state_density_train_on_intrinsic_replay_only: "
        f"{cfg.policy.state_density_train_on_intrinsic_replay_only}"
    )
    print(f"  qstd_exploration: {cfg.policy.qstd_exploration}")
    print(f"  qstd_intrinsic_reward_weight: {cfg.policy.qstd_intrinsic_reward_weight}")
    print(f"  qstd_store_collection_rewards: {cfg.policy.qstd_store_collection_rewards}")
    print(f"  qstd_reward_aggregation: {cfg.policy.qstd_reward_aggregation}")
    print(f"  qstd_reward_top_k: {cfg.policy.qstd_reward_top_k}")
    print(f"  qstd_reward_scale: {cfg.policy.qstd_reward_scale}")
    print(f"  qstd_reward_clip: {cfg.policy.qstd_reward_clip}")
    print(f"  qstd_reward_num_simulations: {cfg.policy.qstd_reward_num_simulations}")
    print(f"  qstd_reward_search_mode: {cfg.policy.qstd_reward_search_mode}")
    print(f"  qstd_recompute_replay_rewards: {cfg.policy.qstd_recompute_replay_rewards}")
    print(f"  qstd_one_step_include_reward: {cfg.policy.qstd_one_step_include_reward}")
    print(
        "  qstd_one_step_use_value_calibration: "
        f"{cfg.policy.qstd_one_step_use_value_calibration}"
    )
    print(f"  qstd_target_snapshot: {cfg.policy.qstd_target_snapshot}")
    print(
        "  qstd_target_snapshot_refresh_per_collect: "
        f"{cfg.policy.qstd_target_snapshot_refresh_per_collect}"
    )
    print(f"  qstd_target_inference_batch_size: {cfg.policy.qstd_target_inference_batch_size}")
    print(f"  qstd_intrinsic_target_mode: {cfg.policy.qstd_intrinsic_target_mode}")
    print(f"  dynamics_disagreement_exploration: {cfg.policy.dynamics_disagreement_exploration}")
    print(
        "  dynamics_disagreement_intrinsic_reward_weight: "
        f"{cfg.policy.dynamics_disagreement_intrinsic_reward_weight}"
    )
    print(f"  dynamics_disagreement_shared_root_latent: {cfg.policy.dynamics_disagreement_shared_root_latent}")
    print(f"  dynamics_disagreement_normalize_latent: {cfg.policy.dynamics_disagreement_normalize_latent}")
    print(f"  dynamics_disagreement_space: {cfg.policy.dynamics_disagreement_space}")
    print(f"  dynamics_disagreement_distance_metric: {cfg.policy.dynamics_disagreement_distance_metric}")
    print(f"  dynamics_disagreement_reward_clip: {cfg.policy.dynamics_disagreement_reward_clip}")
    print(f"  dynamics_disagreement_train_intrinsic_value: {cfg.policy.dynamics_disagreement_train_intrinsic_value}")
    print(f"  store_selected_intrinsic_rewards: {cfg.policy.store_selected_intrinsic_rewards}")
    print(f"  mixed_reward_tree: {cfg.policy.mixed_reward_tree}")
    print(f"  main_tree_policy_prior_mode: {cfg.policy.main_tree_policy_prior_mode}")
    print(
        "  main_tree_force_root_action_coverage: "
        f"{cfg.policy.main_tree_force_root_action_coverage}"
    )
    print(f"  dynamics_prior_alignment_loss_weight: {cfg.policy.dynamics_prior_alignment_loss_weight}")
    print(
        "  dynamics_prior_alignment_target_normalization: "
        f"{cfg.policy.dynamics_prior_alignment_target_normalization}"
    )
    print(f"  separate_intrinsic_tree: {cfg.policy.separate_intrinsic_tree}")
    print(f"  collection_policy_mode: {cfg.policy.collection_policy_mode}")
    print(f"  intrinsic_tree_num_simulations: {cfg.policy.intrinsic_tree_num_simulations}")
    print(f"  intrinsic_tree_add_exploration_noise: {cfg.policy.intrinsic_tree_add_exploration_noise}")
    print(f"  intrinsic_tree_prior_mode: {cfg.policy.intrinsic_tree_prior_mode}")
    print(
        "  intrinsic_tree_force_root_action_coverage: "
        f"{cfg.policy.intrinsic_tree_force_root_action_coverage}"
    )
    print(
        "  intrinsic_tree_pure_exploration_episodes: "
        f"{cfg.policy.intrinsic_tree_pure_exploration_episodes}"
    )
    print(
        "  intrinsic_tree_pure_episode_store_intrinsic_policy: "
        f"{cfg.policy.intrinsic_tree_pure_episode_store_intrinsic_policy}"
    )
    print(f"  intrinsic_tree_final_score_temperature: {cfg.policy.intrinsic_tree_final_score_temperature}")
    print(f"  intrinsic_tree_store_final_policy: {cfg.policy.intrinsic_tree_store_final_policy}")
    print(f"  intrinsic_tree_eval: {cfg.policy.intrinsic_tree_eval}")
    print(f"  intrinsic_only_training: {cfg.policy.intrinsic_only_training}")
    print(f"  intrinsic_tree_normalize_q: {cfg.policy.intrinsic_tree_normalize_q}")
    print(f"  intrinsic_tree_immediate_reward_weight: {cfg.policy.intrinsic_tree_immediate_reward_weight}")
    print(f"  intrinsic_tree_leaf_value_source: {cfg.policy.intrinsic_tree_leaf_value_source}")
    print(f"  intrinsic_tree_leaf_value_weight: {cfg.policy.intrinsic_tree_leaf_value_weight}")
    print(f"  intrinsic_tree_leaf_value_clip: {cfg.policy.intrinsic_tree_leaf_value_clip}")
    print(f"  intrinsic_value_bootstrap: {cfg.policy.intrinsic_value_bootstrap}")
    print(f"  collect_random_action_epsilon: {cfg.policy.collect_random_action_epsilon}")
    print(f"  main_tree_trace_diagnostics: {cfg.policy.main_tree_trace_diagnostics}")
    print(f"  main_tree_trace_min_train_iter: {cfg.policy.main_tree_trace_min_train_iter}")
    print(f"  main_tree_trace_max_per_train_iter: {cfg.policy.main_tree_trace_max_per_train_iter}")
    print(f"  main_tree_trace_max_depth: {cfg.policy.main_tree_trace_max_depth}")
    print(f"  main_tree_trace_max_nodes: {cfg.policy.main_tree_trace_max_nodes}")
    print(f"  main_tree_trace_min_q_std: {cfg.policy.main_tree_trace_min_q_std}")
    print(f"  main_tree_trace_selected_action: {cfg.policy.main_tree_trace_selected_action}")
    print(f"  main_tree_trace_min_position: {cfg.policy.main_tree_trace_min_position}")
    print(f"  main_tree_trace_max_position: {cfg.policy.main_tree_trace_max_position}")
    print(f"  main_tree_trace_min_velocity: {cfg.policy.main_tree_trace_min_velocity}")
    print(f"  main_tree_trace_max_velocity: {cfg.policy.main_tree_trace_max_velocity}")
    print(f"  intrinsic_tree_trace_diagnostics: {cfg.policy.intrinsic_tree_trace_diagnostics}")
    print(f"  intrinsic_tree_trace_best_action: {cfg.policy.intrinsic_tree_trace_best_action}")
    print(f"  intrinsic_tree_trace_min_train_iter: {cfg.policy.intrinsic_tree_trace_min_train_iter}")
    print(f"  intrinsic_tree_trace_max_per_train_iter: {cfg.policy.intrinsic_tree_trace_max_per_train_iter}")
    print(f"  intrinsic_tree_trace_max_depth: {cfg.policy.intrinsic_tree_trace_max_depth}")
    print(f"  intrinsic_tree_trace_max_nodes: {cfg.policy.intrinsic_tree_trace_max_nodes}")
    print(f"  intrinsic_tree_trace_min_position: {cfg.policy.intrinsic_tree_trace_min_position}")
    print(f"  intrinsic_tree_trace_max_position: {cfg.policy.intrinsic_tree_trace_max_position}")
    print(f"  intrinsic_tree_trace_min_velocity: {cfg.policy.intrinsic_tree_trace_min_velocity}")
    print(f"  intrinsic_tree_trace_max_velocity: {cfg.policy.intrinsic_tree_trace_max_velocity}")
    print(f"  intrinsic_model_loss_weight: {cfg.policy.intrinsic_model_loss_weight}")
    print(f"  intrinsic_reward_loss_weight: {cfg.policy.intrinsic_reward_loss_weight}")
    print(f"  intrinsic_value_loss_weight: {cfg.policy.intrinsic_value_loss_weight}")
    print(f"  intrinsic_train_value_head: {cfg.policy.intrinsic_train_value_head}")
    print(
        "  intrinsic_value_train_on_pure_exploration_only: "
        f"{cfg.policy.intrinsic_value_train_on_pure_exploration_only}"
    )
    print(f"  intrinsic_learning_rate: {cfg.policy.intrinsic_learning_rate}")
    print(f"  intrinsic_head_only: {cfg.policy.intrinsic_head_only}")
    print(f"  intrinsic_loss_type: {cfg.policy.intrinsic_loss_type}")
    print(f"  intrinsic_huber_delta: {cfg.policy.intrinsic_huber_delta}")
    print(f"  intrinsic_reward_target_clip: {cfg.policy.intrinsic_reward_target_clip}")
    print(f"  intrinsic_value_target_clip: {cfg.policy.intrinsic_value_target_clip}")
    print(f"  latent_novelty_diagnostics: {cfg.policy.latent_novelty_diagnostics}")
    print(f"  latent_novelty_dim: {cfg.policy.model.latent_novelty_dim}")
    print(f"  latent_novelty_hidden_dim: {cfg.policy.model.latent_novelty_hidden_dim}")
    print(f"  latent_novelty_prior_scale: {cfg.policy.model.latent_novelty_prior_scale}")
    print(f"  latent_novelty_learning_rate: {cfg.policy.latent_novelty_learning_rate}")
    print(f"  latent_novelty_train_batch_size: {cfg.policy.latent_novelty_train_batch_size}")
    print(f"  latent_novelty_temperature: {cfg.policy.latent_novelty_temperature}")
    print(f"  latent_novelty_alignment_loss_weight: {cfg.policy.latent_novelty_alignment_loss_weight}")
    print(f"  elite_episode_replay: {cfg.policy.elite_episode_replay}")
    print(f"  elite_episode_buffer_size: {cfg.policy.elite_episode_buffer_size}")
    print(f"  elite_episode_sample_ratio: {cfg.policy.elite_episode_sample_ratio}")
    print(f"  elite_episode_min_return: {cfg.policy.elite_episode_min_return}")
    print(f"  elite_episode_monte_carlo_value_target: {cfg.policy.elite_episode_monte_carlo_value_target}")
    print(f"  elite_episode_start_stratification: {cfg.policy.elite_episode_start_stratification}")
    print(f"  elite_episode_start_observation_index: {cfg.policy.elite_episode_start_observation_index}")
    print(f"  elite_episode_start_bin_edges: {cfg.policy.elite_episode_start_bin_edges}")
    print(f"  intrinsic_episode_replay: {cfg.policy.intrinsic_episode_replay}")
    print(f"  intrinsic_episode_max_age_phases: {cfg.policy.intrinsic_episode_max_age_phases}")
    print(f"  intrinsic_episode_buffer_size: {cfg.policy.intrinsic_episode_buffer_size}")
    print(f"  intrinsic_episode_sample_ratio: {cfg.policy.intrinsic_episode_sample_ratio}")
    print(f"  intrinsic_episode_min_score: {cfg.policy.intrinsic_episode_min_score}")
    print(f"  intrinsic_episode_score_mode: {cfg.policy.intrinsic_episode_score_mode}")
    print(f"  intrinsic_episode_peak_window_before: {cfg.policy.intrinsic_episode_peak_window_before}")
    print(f"  intrinsic_episode_peak_window_after: {cfg.policy.intrinsic_episode_peak_window_after}")
    print(f"  save_intrinsic_episode_trajectories: {cfg.policy.save_intrinsic_episode_trajectories}")
    print(f"  intrinsic_episode_save_dir: {cfg.policy.intrinsic_episode_save_dir}")
    print(f"  collection_diagnostics: {cfg.policy.collection_diagnostics}")
    print(f"  collection_root_action_diagnostics: {cfg.policy.collection_root_action_diagnostics}")
    print(f"  collection_ssl_diagnostics: {cfg.policy.collection_ssl_diagnostics}")
    print(f"  collection_ssl_diagnostics_batch_size: {cfg.policy.collection_ssl_diagnostics_batch_size}")
    print(f"  encoder_prior_grid_diagnostics: {cfg.policy.encoder_prior_grid_diagnostics}")
    print(
        "  encoder_prior_grid_shape: "
        f"{cfg.policy.encoder_prior_grid_position_count}x{cfg.policy.encoder_prior_grid_velocity_count}"
    )
    print(
        "  encoder_prior_imagined_novelty_diagnostics: "
        f"{cfg.policy.encoder_prior_imagined_novelty_diagnostics}"
    )
    print(
        "  encoder_prior_imagined_novelty_probe: "
        f"roots={cfg.policy.encoder_prior_imagined_novelty_root_count}, "
        f"bank={cfg.policy.encoder_prior_imagined_novelty_bank_size}, "
        f"depth={cfg.policy.encoder_prior_imagined_novelty_depth}, "
        f"knn={cfg.policy.encoder_prior_imagined_novelty_knn_k}"
    )
    print(f"  normalization_mode_diagnostics: {cfg.policy.normalization_mode_diagnostics}")
    print(f"  normalization_mode_diagnostics_depth: {cfg.policy.normalization_mode_diagnostics_depth}")
    print(f"  normalization_mode_diagnostics_sample_size: {cfg.policy.normalization_mode_diagnostics_sample_size}")
    print(f"  fixed_sequence_intrinsic_diagnostics: {cfg.policy.fixed_sequence_intrinsic_diagnostics}")
    print(f"  fixed_sequence_intrinsic_max_states: {cfg.policy.fixed_sequence_intrinsic_max_states}")
    print(f"  fixed_sequence_intrinsic_sequences: {cfg.policy.fixed_sequence_intrinsic_sequences}")
    print(f"  diagnostics_dir: {cfg.policy.diagnostics_dir}")
    print(f"  save_elite_episode_trajectories: {cfg.policy.save_elite_episode_trajectories}")
    print(f"  elite_episode_save_dir: {cfg.policy.elite_episode_save_dir}")
    print(f"  use_priority: {cfg.policy.use_priority}")
    print(f"  priority_prob_alpha: {cfg.policy.priority_prob_alpha}")
    print(f"  priority_prob_beta: {cfg.policy.priority_prob_beta}")
    print(f"  priority_initial_mode: {cfg.policy.priority_initial_mode}")
    print(f"  priority_new_data_use_max: {cfg.policy.priority_new_data_use_max}")
    print(f"  thompson_sampling: {cfg.policy.thompson_sampling}")
    print(f"  ensemble_search_mode: {cfg.policy.ensemble_search_mode}")
    print(f"  ensemble_ucb_beta: {cfg.policy.ensemble_ucb_beta}")
    print(f"  ensemble_normalize_q: {cfg.policy.ensemble_normalize_q}")
    print(f"  ensemble_uncertainty_decay: {cfg.policy.ensemble_uncertainty_decay}")
    print(f"  depth_matched_qstd_diagnostics: {cfg.policy.depth_matched_qstd_diagnostics}")
    print(f"  depth_matched_qstd_max_depth: {cfg.policy.depth_matched_qstd_max_depth}")
    print(
        "  depth_normalized_backup_disagreement: "
        f"{cfg.policy.depth_normalized_backup_disagreement}"
    )
    print(f"  depth_disagreement_ema_momentum: {cfg.policy.depth_disagreement_ema_momentum}")
    print(f"  depth_disagreement_trim_fraction: {cfg.policy.depth_disagreement_trim_fraction}")
    print(f"  depth_disagreement_min_groups: {cfg.policy.depth_disagreement_min_groups}")
    print(f"  depth_disagreement_min_scale: {cfg.policy.depth_disagreement_min_scale}")
    print(f"  depth_disagreement_max_depth: {cfg.policy.depth_disagreement_max_depth}")
    print(
        "  depth_disagreement_conservative_cap: "
        f"{cfg.policy.depth_disagreement_conservative_cap}"
    )
    print(f"  ensemble_value_calibration: {cfg.policy.ensemble_value_calibration}")
    print(f"  ensemble_value_calibration_start_phase: {cfg.policy.ensemble_value_calibration_start_phase}")
    print(f"  ensemble_value_calibration_sample_size: {cfg.policy.ensemble_value_calibration_sample_size}")
    print(f"  final_qstd_beta: {cfg.policy.final_qstd_beta}")
    print(f"  final_qstd_temperature: {cfg.policy.final_qstd_temperature}")
    print(f"  final_qstd_store_final_policy: {cfg.policy.final_qstd_store_final_policy}")
    print(f"  separate_exploration_tree: {cfg.policy.separate_exploration_tree}")
    print(f"  separate_exploration_tree_horizon: {cfg.policy.separate_exploration_tree_horizon}")
    print(f"  separate_exploration_tree_top_k: {cfg.policy.separate_exploration_tree_top_k}")
    print(f"  separate_exploration_tree_beta: {cfg.policy.separate_exploration_tree_beta}")
    print(f"  separate_exploration_tree_temperature: {cfg.policy.separate_exploration_tree_temperature}")
    print(
        "  separate_exploration_tree_score_normalization: "
        f"{cfg.policy.separate_exploration_tree_score_normalization}"
    )
    print(f"  separate_exploration_tree_max_sequences: {cfg.policy.separate_exploration_tree_max_sequences}")
    print(
        "  separate_exploration_tree_sequence_score_mode: "
        f"{cfg.policy.separate_exploration_tree_sequence_score_mode}"
    )
    print(
        "  separate_exploration_tree_sequence_qstd_beta: "
        f"{cfg.policy.separate_exploration_tree_sequence_qstd_beta}"
    )
    print(
        "  separate_exploration_tree_sequence_qstd_mode: "
        f"{cfg.policy.separate_exploration_tree_sequence_qstd_mode}"
    )
    print(
        "  separate_exploration_tree_store_final_policy: "
        f"{cfg.policy.separate_exploration_tree_store_final_policy}"
    )
    print(f"  pure_exploration_episode_fraction: {cfg.policy.pure_exploration_episode_fraction}")
    print(f"  pure_exploration_commit_steps: {cfg.policy.pure_exploration_commit_steps}")
    print(f"  pure_exploration_sequence_temperature: {cfg.policy.pure_exploration_sequence_temperature}")
    print(
        "  pure_exploration_store_exploration_policy: "
        f"{cfg.policy.pure_exploration_store_exploration_policy}"
    )
    print(f"  ensemble_eval_search_mode: {cfg.policy.ensemble_eval_search_mode}")
    print(f"  ensemble_eval_ucb_beta: {cfg.policy.ensemble_eval_ucb_beta}")
    print(f"  ensemble_eval_uncertainty_decay: {cfg.policy.ensemble_eval_uncertainty_decay}")
    print(f"  num_simulations: {cfg.policy.num_simulations}")
    print(f"  collector_env_num: {cfg.env.collector_env_num}")
    print(f"  evaluator_env_num: {cfg.env.evaluator_env_num}")
    print(f"  env_manager_type: {create_cfg.env_manager.type}")
    print(f"  n_episode: {cfg.policy.n_episode}")
    print(f"  n_evaluator_episode: {cfg.env.n_evaluator_episode}")
    print(f"  update_per_collect: {cfg.policy.update_per_collect}")
    print(f"  replay_ratio: {cfg.policy.replay_ratio}")
    print(f"  batch_size: {cfg.policy.batch_size}")
    print(f"  eval_freq: {cfg.policy.eval_freq}")
    print(f"  td_steps: {cfg.policy.td_steps}")
    print(f"  num_unroll_steps: {cfg.policy.get('num_unroll_steps', None)}")
    print(f"  game_segment_length: {cfg.policy.game_segment_length}")
    print(f"  observation_shape: {cfg.policy.model.observation_shape}")
    print(f"  state_norm: {cfg.policy.model.state_norm}")
    print(
        "  representation_last_linear_layer_init_zero: "
        f"{cfg.policy.model.get('representation_last_linear_layer_init_zero', None)}"
    )
    print(
        "  value_last_linear_layer_init_zero: "
        f"{cfg.policy.model.get('value_last_linear_layer_init_zero', None)}"
    )
    print(f"  norm_type: {cfg.policy.model.norm_type}")
    print(f"  ssl_norm_type: {cfg.policy.model.ssl_norm_type or cfg.policy.model.norm_type}")
    print(f"  include_remaining_time: {cfg.env.include_remaining_time}")
    print(f"  bootstrap_at_time_limit: {cfg.policy.bootstrap_at_time_limit}")
    print(f"  ssl_loss_weight: {cfg.policy.ssl_loss_weight}")
    print(f"  raw_latent_ssl_loss_weight: {cfg.policy.raw_latent_ssl_loss_weight}")
    print(f"  reward_loss_weight: {cfg.policy.reward_loss_weight}")
    print(f"  value_loss_weight: {cfg.policy.value_loss_weight}")
    print(f"  policy_loss_weight: {cfg.policy.policy_loss_weight}")
    print(f"  reanalyze_ratio: {cfg.policy.reanalyze_ratio}")
    print(f"  reanalyze_use_search_value_target: {cfg.policy.reanalyze_use_search_value_target}")
    print(f"  value_target_mode: {cfg.policy.value_target_mode}")
    print(f"  sve_start_train_iter: {cfg.policy.sve_start_train_iter}")
    print(f"  sve_recent_window: {cfg.policy.sve_recent_window}")
    print(f"  sve_real_reward_prefix_steps: {cfg.policy.sve_real_reward_prefix_steps}")
    print(f"  sve_search_noise: {cfg.policy.sve_search_noise}")
    print(f"  ensemble_member_bootstrap_targets: {cfg.policy.ensemble_member_bootstrap_targets}")
    print(f"  stop_value: {cfg.env.stop_value}")
    print(f"  max_env_step: {args.max_env_step}")
    print(f"  max_train_iter: {args.max_train_iter}")
    print(f"  cuda: {cfg.policy.cuda}")
    print(f"  profile_timing: {cfg.policy.profile_train_loop_timing}")
    print(f"  profile_learn_timing: {cfg.policy.profile_learn_timing}")
    print(f"  record_collect_video: {cfg.policy.record_collect_video}")
    if cfg.policy.record_collect_video:
        print(f"  collect_video_dir: {cfg.policy.collect_video_dir}")


def main() -> None:
    args = build_parser().parse_args()
    cfg, create_cfg = build_config(args)
    if cfg.policy.record_collect_video and args.clean_collect_video_dir:
        clean_video_dir(cfg.policy.collect_video_dir)
    print_run_summary(cfg, create_cfg, args)

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
