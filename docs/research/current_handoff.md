# Current research handoff

Prepared on 2026-11-06 for continuing work from another machine. No commit or
push is performed as part of this handoff; the repository owner will review and
create the Git commit.

## Active research question

The current focus is episode-local novelty prediction in the fixed Branching
Rooms MiniGrid layout. RND supplies phase-level novelty, but it cannot identify
short-term revisits while its target snapshot is frozen. The auxiliary model is
intended to predict whether an imagined action endpoint will produce an exact
RGB observation already encountered recently in the episode.

## Current candidate: recent-history transformer

The active model is `RecentHistoryTransformerNovelty` in
`lzero/policy/recurrent_sequence_novelty.py`.

Its input consists of:

1. Up to 40 recent real observations and their executed actions.
2. A candidate action-only suffix of up to five actions from the MCTS root.

It predicts a novelty probability at every candidate depth. The model does not
receive imagined RGB observations or predicted beliefs. The intended behavior
is to learn enough relative position, orientation, movement, wall collision,
and loop information from the real history to decide whether each candidate
endpoint is a revisit.

The online condition is
`shared_log_auxiliary_recent_transformer_novelty` in
`experiments/run_branching_rooms_intrinsic_rnd.py`. The cluster entry point is:

```bash
cluster/submit_branching_rooms_recent_transformer_novelty_long_online_delftblue.sh
```

The standard preset uses one seed, a 40-step history, a five-step candidate
horizon, 64 novelty updates per collection phase, and cached suffix inference.

## Inference optimization

The initial implementation repeatedly processed complete history-plus-suffix
sequences during MCTS and made collection prohibitively slow. The current
`cached_suffix` mode processes the real history once for a root and reuses that
state while scoring candidate action suffixes. Predictions are also stored for
already evaluated tree nodes.

The optimization reduced novelty inference accounting to roughly one second
per collection phase in the observed cluster runs. It preserves the model's
full history input; it only avoids repeating identical computation.

The main online run with cached suffix inference was DelftBlue job 1063625. By
phases 15 through 17, root-action selected AUC was approximately 0.75 to 0.78,
and fixed-layout position coverage reached 635 of 693 positions. This was a
clear improvement over the early phases, but not strong enough to treat the
predictor as solved.

## Why the earlier belief comparator was deprioritized

The preceding approach predicted future beliefs and compared them with stored
real and imagined belief states. It eventually achieved strong aggregate AUC
on the training layout, but corridor diagnostics exposed weaker transfer to
unseen corridor trajectories. Belief-dynamics consistency and replay priority
based on belief or revisit error did not remove that failure reliably.

One training bug was also corrected: revisit memory had been truncated to the
last 128 steps rather than representing the current episode. The corrected
episode-history run improved late aggregate metrics but did not establish
trajectory generalization. This led to the simpler history-transformer test,
which predicts revisit directly from real observation/action history and the
candidate actions.

The belief implementation and its diagnostics remain in the repository for
comparison. They are not deleted or treated as invalid; they are simply not the
current preferred direction.

## Observation-encoder diagnostic

The current transformer originally used a convolutional RGB encoder ending in
adaptive 2 by 2 pooling. Two alternatives were added:

- `rgb_tile_7x7`: preserves the rendered MiniGrid tile grid.
- `symbolic_7x7`: embeds native MiniGrid object, color, and state identifiers.

The symbolic encoder still sees only the partial egocentric 7 by 7 view. It
does not receive the agent's global position, the global map, cells behind the
agent, or cells hidden behind walls.

Three seeds were trained for 1024 updates. The full reports and aggregate table
are under:

```text
docs/research/results/recent_transformer_encoder_comparison_1024/
```

At 1024 updates, held-out macro-policy AUC was:

| Encoder | Mean AUC | Standard deviation |
|---|---:|---:|
| pooled RGB 2x2 | 0.597 | 0.008 |
| tile-aligned RGB 7x7 | 0.592 | 0.014 |
| symbolic 7x7 | 0.632 | 0.012 |

The symbolic model beat pooled RGB in every seed. Tile-aligned RGB did not.
The result identifies visual semantic extraction as one bottleneck but does not
show that symbolic input solves movement or loop generalization.

Important limitation: this is a held-out-policy test on the same map, not an
unseen-layout test. No claim of cross-layout generalization is supported yet.

## Current implementation boundary

All three encoders are implemented and tested in the standalone diagnostic.
The online experiment still uses the pixel model by default. Native symbolic
observations have not been wired through the collector into the online novelty
module, and doing so would be an explicit simulator-side-channel design choice.

The cleaner pixel-only follow-up is to train the tile RGB encoder with auxiliary
per-cell object, color, and state targets during training, while continuing to
provide only RGB at inference. That experiment has not yet been implemented.

## Recommended next experiment

Keep the main MuZero agent pixel-based and run a controlled local diagnostic:

1. Add object, color, and state prediction heads to the tile-aligned RGB
   encoder.
2. Use native symbolic cells only as training targets for those auxiliary
   heads.
3. Remove those targets at inference so the novelty predictor remains
   pixel-only.
4. Compare pooled RGB, tile RGB, tile RGB with semantic supervision, and native
   symbolic input with three seeds and 1024 updates.
5. If semantic supervision closes a meaningful part of the symbolic gap, wire
   that pixel-only model into the online run before spending cluster capacity.

Only after the fixed-layout predictor is dependable should the project add
multi-layout training and a genuinely unseen-layout evaluation.

## Reproducing the latest local comparison

Create the deterministic dataset:

```bash
experiments/prepare_recent_transformer_encoder_dataset_local.sh
```

Run the comparison:

```bash
experiments/run_recent_transformer_encoder_comparison_local.sh
```

The generated dataset and outputs stay under `.local_runs` and are intentionally
ignored by Git. The symbolic cache is reconstructed by replaying each stored
action sequence from its episode seed and validating every replayed pose.

## Verification targets

The focused test set for this line is:

```bash
.venv/bin/python -m pytest \
  lzero/policy/tests/test_recurrent_sequence_novelty.py \
  experiments/test_diagnose_recent_history_transformer.py \
  experiments/test_branching_rooms_recurrent_sequence_novelty.py
```

The two reproduction shell scripts support `DRY_RUN=1` for checking commands
without starting training.

## Moving to another machine

After the owner commits and pushes the `ensemble-muzero` branch, clone it with:

```bash
git clone git@github.com:PawelSwiderski1/LightZero.git
cd LightZero
git switch ensemble-muzero
```

The cluster environment remains reproducible through `cluster/setup_hpc_env.sh`.
For local development, create a Python 3.10 environment, install the repository
and test requirements, and build the Cython extensions using the normal
LightZero installation procedure. The current verified local interpreter is
Python 3.10.

Before creating the travel snapshot, review and run:

```bash
git status --short
git add -A
git diff --cached --stat
git diff --cached --check
git commit -m "feat: add ensemble exploration and Branching Rooms research"
git push -u origin ensemble-muzero
```

`git add -A` is safe only after confirming that the generated directories are
shown as ignored and not as candidate files. The repository-level `.gitignore`
contains those exclusions.

## What is intentionally not in Git

- `.local_runs` and `.hpc_runs`
- fetched `hpc_results`
- model checkpoints and replay buffers
- generated videos and images
- Slurm output and Python caches

These artifacts occupy more than 100 GB locally. They are not needed to clone,
inspect, test, or continue the current source implementation. Curated scalar
reports needed to justify the latest decision are tracked under
`docs/research/results`.
