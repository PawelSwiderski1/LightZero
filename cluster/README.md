# LightZero on one Slurm A100 with Apptainer

This setup transfers the current working tree, pulls a pinned CUDA/PyTorch
Apptainer image on the cluster, creates a persistent Python environment in
scratch, and submits one-GPU experiments. Generated checkpoints, videos, local
virtual environments, container images, and compiled extensions are not
transferred.

## 1. Sync the current source

From the local LightZero repository:

For DelftBlue, this defaults to `~/LightZero` on the cluster:

```bash
cluster/sync_to_cluster.sh pmswiderski@login.delftblue.tudelft.nl
```

Preview the file count and total transfer size without copying anything:

```bash
DRY_RUN=1 cluster/sync_to_cluster.sh \
  pmswiderski@login.delftblue.tudelft.nl
```

The sync excludes local environments, all `data_*` directories at any depth,
`tmp`, videos, result directories, documentation assets, caches, checkpoints,
and compiled extensions. The expected transfer is source-sized, not gigabytes.

To choose another destination:

```bash
cluster/sync_to_cluster.sh \
  pmswiderski@login.delftblue.tudelft.nl \
  /absolute/path/on/cluster/LightZero
```

This uses the current files rather than `git clone`, so uncommitted experiment
changes are included.

## 2. Create the Apptainer environment

On the cluster:

```bash
ssh pmswiderski@login.delftblue.tudelft.nl
cd ~/LightZero
cluster/setup_hpc_env.sh
```

This downloads the image directly to
`/scratch/$USER/lightzero_apptainer/images`, creates the environment under
`/scratch/$USER/lightzero_apptainer/venvs`, installs the dependencies, and
builds the Cython extensions inside the container. The default image is:

```text
docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```

The image is several GB but is downloaded once by the cluster; it is not copied
from the local machine. Override the source, image, or storage root with:

```text
LIGHTZERO_APPTAINER_SOURCE
LIGHTZERO_APPTAINER_IMAGE
LIGHTZERO_APPTAINER_ROOT
```

The CUDA check can report `False` on a login node. The Slurm job enters the same
image with `apptainer exec --nv` and checks CUDA again on the allocated A100.

The job creates `~/LightZero/.hpc_runs` as a symlink to the scratch result root.
This keeps LightZero components that build paths as `./<exp_name>` from placing
logs or checkpoints in the repository or home filesystem.

Do not use the old repository-local `.venv-hpc`: it may have been created with
the login node's Python 3.6 and is not used by this workflow.

## 3. Submit the state-density diagnostic

The DelftBlue wrapper defaults to partition `gpu-a100`, one A100, eight CPUs,
2 GB per CPU, and a four-hour limit:

```bash
cd ~/LightZero
cluster/submit_delftblue.sh
```

The wrapper follows the same environment-variable override pattern as other
DelftBlue jobs:

```bash
PARTITION=gpu-a100 \
TIME_LIMIT=04:00:00 \
CPUS_PER_TASK=8 \
MEM_PER_CPU=2GB \
GPUS_PER_TASK=1 \
  cluster/submit_delftblue.sh
```

Set `SLURM_ACCOUNT=ACCOUNT` as well when an account argument is required.

The default output location is `$SCRATCH/lightzero_runs`. Override it with:

```bash
LIGHTZERO_RUN_ROOT=/path/to/scratch/runs cluster/submit_delftblue.sh
```

## Other presets and overrides

The first job argument is an experiment script. Further arguments are passed to
that experiment:

```bash
cluster/submit_delftblue.sh \
  experiments/run_mountain_car_state_density_intrinsic_diagnostic.py \
  --replay-ratio 0.02 \
  --max-env-step 120000
```

Environment variables supported by the job include:

```text
LIGHTZERO_REPO_ROOT
LIGHTZERO_APPTAINER_ROOT
LIGHTZERO_APPTAINER_IMAGE
LIGHTZERO_VENV
LIGHTZERO_RUN_ROOT
RUN_LABEL
SEED
```

Training videos are disabled in the Slurm job. Copy a checkpoint back and render
videos locally to avoid using GPU allocation time and shared-filesystem bandwidth
for MP4 encoding.

## Corrected RND diagnostic

The corrected-RND reward/value propagation experiment has a short wrapper:

```bash
cd ~/LightZero
cluster/submit_corrected_rnd_diagnostic.sh
```

It uses the four-hour A100 defaults and accepts the same resource environment
variables and experiment overrides as `submit_delftblue.sh`. For example:

```bash
TIME_LIMIT=06:00:00 cluster/submit_corrected_rnd_diagnostic.sh \
  --max-env-step 120000
```

The RND signal and its reward/value heads are trained and diagnosed, but their
action-selection weight is zero. The run writes true-versus-predicted RND
sequence diagnostics to `diagnostics/fixed_sequence_intrinsic_returns.csv`.

## Controlled RND-query matrix

Submit the pure-intrinsic baseline, latent-only reader, detached-world-model
reader, and full joint-query conditions for three matched seeds:

```bash
cd ~/LightZero
SEEDS="0 1 2" cluster/submit_rnd_query_matrix_delftblue.sh
```

The presets isolate RND initialization from optional query modules. For a given
seed, all four conditions therefore start with the same shared MuZero model and
RND networks. Override the run length or resources with the normal environment
variables, for example:

```bash
SEEDS="0 1 2" MAX_ENV_STEP=45000 MAX_TRAIN_ITER=15000 \
TIME_LIMIT=04:00:00 cluster/submit_rnd_query_matrix_delftblue.sh
```

After the initial matrix, complete the latent-reader/world-model 2x2 factorial
with the two missing conditions across the same three seeds:

```bash
cd ~/LightZero
SEEDS="0 1 2" cluster/submit_rnd_query_factorial_followup_delftblue.sh
```

`no_context_detached` uses the latent-only reader in MCTS without propagating
its loss into the encoder or dynamics. `auxiliary_only` propagates that loss
into the world model but keeps the original intrinsic reward head inside MCTS.
Together with the existing baseline and `no_context` runs, these isolate tree
reward replacement from auxiliary world-model training.

## Mixed-reward RND tree

Run one shared MCTS tree on the weighted sum of extrinsic and RND returns:

```bash
cd ~/LightZero
cluster/submit_corrected_rnd_mixed_reward_tree.sh \
  --mixed-reward-beta 50
```

This preset keeps separate reward and value heads for the two objectives, but
jointly normalizes `Q_extrinsic + beta * Q_intrinsic` for main-tree PUCT. The
separate intrinsic tree and pure intrinsic episodes are disabled.

## One-step Qstd reward/value tree

Run one shared MCTS tree with an intrinsic reward learned from direct replay
edges. For each real `(state, action)`, the target is the ensemble standard
deviation of `predicted_reward + discount * predicted_next_value`. A separate
intrinsic value head learns the discounted return of those rewards:

```bash
cd ~/LightZero
cluster/submit_qstd_reward_value_mixed_tree.sh \
  --mixed-reward-beta 1.0
```

The Qstd target ensemble is frozen once per collection/learning phase and
cached by replay transition. Raw Qstd is never evaluated on imagined tree
latents; MCTS uses the learned reward and value heads there.

## Encoder random-prior diagnostic

Measure member-specific frozen observation-to-latent priors without using Qstd
for collection:

```bash
cd ~/LightZero
cluster/submit_encoder_random_prior.sh
```

To let optimistic Qstd influence collection in the same preset:

```bash
cluster/submit_encoder_random_prior.sh --use-qstd-for-collection \
  --ensemble-ucb-beta 2.0
```

The default prior contribution has L2 norm `1.0`. Sweep it with
`--encoder-random-prior-scale`. The run writes the fixed true-state grid to
`diagnostics/encoder_prior_state_grid.csv`; it contains pre/post-learn latent
disagreement, value spread, imagined one-step Qstd, exact-physics one-step
Qstd, recurrent error, and cumulative visit counts.

## DAIC A40

DAIC uses the persistent Linux home directory rather than DelftBlue `/scratch`.
For `pmswiderski`, `$HOME` is `/home/nfs/pmswiderski`. Sync the source from
the local repository:

```bash
cluster/sync_to_cluster.sh \
  pmswiderski@login.daic.tudelft.nl \
  LightZero
```

Initialize the DAIC Apptainer environment once. This command is run from your
local terminal and will prompt for the SSH password:

```bash
ssh -t pmswiderski@login.daic.tudelft.nl \
  'cd ~/LightZero && cluster/setup_daic_env.sh'
```

For the Branching Rooms runs, use the smaller runtime dependency set. It omits
unrelated Atari, language-model, and development packages:

```bash
LIGHTZERO_REQUIREMENTS_FILE="$HOME/LightZero/cluster/requirements-branching-rooms-daic.txt" \
  cluster/setup_daic_env.sh
```

Then submit the passive encoder-prior diagnostic:

```bash
ssh -t pmswiderski@login.daic.tudelft.nl \
  'cd ~/LightZero && cluster/submit_encoder_random_prior_daic.sh'
```

The DAIC defaults match account `ewi-insy-prb`, partition `insy`, one A40,
one CPU, 4 GB host memory, and four hours. Override resources when submitting,
for example:

```bash
ssh -t pmswiderski@login.daic.tudelft.nl \
  'cd ~/LightZero && CPUS_PER_TASK=4 MEMORY=8GB \
   cluster/submit_encoder_random_prior_daic.sh'
```

Submit the long Branching Rooms compact-query experiments with resources sized
for their vectorized environments:

```bash
ssh -t pmswiderski@login.daic.tudelft.nl \
  'cd ~/LightZero && CONDITIONS="compact_query_projected_memory" SEEDS="0" \
   cluster/submit_branching_rooms_compact_query_daic.sh'
```

This wrapper defaults to eight CPUs, 16 GB host memory, and an eight-hour time
limit because an A40 is slower than the DelftBlue A100 used for the original
runs. Override `CONDITIONS`, `SEEDS`, or `TIME_LIMIT` to submit only missing
runs or adjust the allocation.

To let Qstd affect collection, append
`--use-qstd-for-collection --ensemble-ucb-beta 2.0`. Results are stored under
`~/lightzero_runs/run_mountain_car_encoder_random_prior_seed0_JOB_ID`. Set
`DAIC_STORAGE_ROOT` to a project-storage path if the Apptainer image and run
outputs should not consume home-directory quota.

## Monitoring and downloading results

```bash
squeue -u "$USER"
tail -f slurm-lightzero-mcar-JOB_ID.out
```

Download a completed run from the local machine with `rsync`:

```bash
rsync -az USER@CLUSTER:/path/to/scratch/lightzero_runs/RUN_DIRECTORY/ \
  ./hpc_results/RUN_DIRECTORY/
```
