# Experiment Registry

**Cutoff:** 2026-09-04  
**Purpose:** provenance index for the detailed evidence record in
[experimental_history.md](experimental_history.md) and the meeting-oriented
[research narrative](research_narrative.md).

## Status convention

- **Valid, replicated:** controlled three-seed family used for comparative
  evidence.
- **Valid, exploratory:** useful one-seed or short-run evidence; no final
  performance claim.
- **Mechanical only:** smoke test, implementation check, or diagnostic sanity
  test.
- **Invalid for comparison:** a bug, wrong target definition, failed job, or
  incomplete transfer prevents the intended inference.
- **Pending:** implemented/submitted but no result available at the cutoff.

Run names are preserved exactly because many configurations evolved between
similarly named experiments. The configuration stored inside each run is the
source of truth.

## 1. Core experiment entry points

| Purpose | Entry point |
|---|---|
| Ensemble CartPole | `experiments/run_cartpole_ensemble.py` |
| Main MountainCar ensemble | `experiments/run_mountain_car_ensemble.py` |
| Optimistic shared-tree base | `experiments/run_mountain_car_optimistic_shared_tree.py` |
| Full intrinsic RND | `experiments/run_mountain_car_full_intrinsic_rnd.py` |
| Physical RND landscape query | `experiments/run_mountain_car_rnd_landscape_query.py` |
| Query factorial controls | `cluster/submit_rnd_query_factorial_followup_delftblue.sh` plus the two query condition scripts |
| Auxiliary decoder architecture matrix | `cluster/submit_rnd_query_auxiliary_decoder_ablation_delftblue.sh` with `experiments/run_mountain_car_rnd_query_auxiliary_only.py` |
| Ordinary intrinsic reward-head controls | `experiments/run_mountain_car_intrinsic_reward_head_ablation.py` |
| Pre-registered confirmatory controls | `experiments/run_mountain_car_rnd_aux_confirmatory_ablation.py` |
| Branching Rooms experiments | `experiments/run_branching_rooms_intrinsic_rnd.py` |
| Branching Rooms visualizer | `experiments/visualize_branching_rooms.py` |

## 2. CartPole and ensemble validation

| Run | Status | Role/result |
|---|---|---|
| `data_ensemble_muzero/cartpole_control_ens1_nots_seed0` | Mechanical only | Single-model control; first score 200 at iteration 400 |
| `data_ensemble_muzero/cartpole_ensemble_ens3_ts_seed0_260708_201304` | Mechanical only | Three-member Thompson ensemble; first score 200 at iteration 600 |
| `data_ensemble_muzero/smoke_*`, `cartpole_diag_*`, `profile_*` | Mechanical only | Construction, multi-environment, training-all-members, and runtime checks |

The utility `experiments/summarize_cartpole_runs.py` reproduces the reported
evaluation milestones.

## 3. Initial MountainCar ensemble and optimism

| Run/artifact | Status | Role/result |
|---|---|---|
| `data_ensemble_muzero/mountain_car_ablation_single_model_seed0` | Valid, exploratory | Did not solve; final/best evaluation -200 |
| `data_ensemble_muzero/mountain_car_ablation_ensemble_beta0_seed0` | Valid, exploratory | Ensemble beta 0; best evaluation -104 |
| `data_ensemble_muzero/mountain_car_ablation_beta0_pess_eval_seed0_fixed` | Valid, exploratory | Pessimistic evaluation; best -106 |
| `data_ensemble_muzero/mountain_car_best_search_value_seed0` | Valid, exploratory | Optimistic collection; best -101.25 |
| `data_ensemble_muzero/ablation_comparison_plots/` | Derived evidence | Phase-by-phase evaluation and collection comparison |
| `data_ensemble_muzero/uncertainty_review_main_method/` | Derived evidence | Ten-phase Q/reward/leaf disagreement decomposition |

Earlier `mountain_car_smoke`, `mountain_car_ensemble_*`, shared-tree, random-prior,
video, and timing directories were iterative implementation runs. They establish
development chronology but are not independent scientific comparisons.

## 4. Disagreement-driven exploration probes

These are mostly one-seed experiments with changing budgets. They should be
read as mechanism tests.

| Run | Status | Main observation |
|---|---|---|
| `mountain_car_intrinsic_tree_only_seed0` | Valid, exploratory | No goal in 12 phases |
| `mountain_car_intrinsic_tree_only_desc_trace_seed0` | Valid, exploratory | No goal in 10 phases |
| `mountain_car_dynamics_disagreement_l2_prior05_seed0` | Valid, exploratory | No goal in 34 phases |
| `mountain_car_dynamics_disagreement_l2_prior05_align5_seed0` | Valid, exploratory | No goal in 8 phases |
| `mountain_car_dynamics_disagreement_l2_prior05_align50_seed0` | Valid, exploratory | No goal in 13 phases |
| `mountain_car_dynamics_disagreement_noprior_seed0` | Valid, exploratory | No goal in 18 phases |
| `mountain_car_dynamics_disagreement_sslproj_seed0` | Valid, exploratory | Eval best -125.5; no collection goal |
| `mountain_car_q_disagreement_fixed_explore_seed0` | Valid, exploratory | First goal phase 7; 104 successful segments |
| `mountain_car_q_disagreement_no_elite_h5_seed0` | Valid, exploratory | No goal in 16 phases |
| `mountain_car_q_disagreement_pure_explore_h5_seed0` | Valid, exploratory | First goal phase 9; only 2 successful segments |
| `mountain_car_qstd_intrinsic_tree_only_seed0_260804_123055` | Valid, exploratory | No goal in short four-phase run |

## 5. Fixed-horizon, calibration, replay, and representation probes

| Run | Status | Main observation |
|---|---|---|
| `mountain_car_fixed_horizon_bounded_optimism_seed0` | Valid, exploratory | First goal phase 18; best eval -115.75 |
| `mountain_car_fixed_horizon_bounded_envelopes_seed0` | Valid, exploratory | First goal phase 13; best eval -136.13 |
| `mountain_car_fixed_horizon_uncapped_beta2_seed0` | Valid, exploratory | First goal phase 20; best eval -105.38 |
| `mountain_car_fixed_horizon_value_calibrated_seed0` | Valid, exploratory | First goal phase 6 |
| `mountain_car_optimistic_shared_tree_elite_stratified_seed0` | Valid, exploratory | First goal phase 6; 703 successful segments |
| `mountain_car_optimistic_shared_tree_elite_finite_horizon_seed0` | Valid, exploratory | No collection goal; best eval -116.88 |
| `mountain_car_optimistic_shared_tree_mixed_sve_seed0` | Valid, exploratory | First goal phase 8; best eval -120.75 |
| `mountain_car_mixed_sve_per_fixed_seed0` | Valid, exploratory | First goal phase 6; best eval -114.88 |
| `mountain_car_depth_matched_qstd_diag_seed0` | Diagnostic | Tested common-depth Q comparisons |
| `mountain_car_depth_normalized_backup_qstd_seed0` | Diagnostic | Tested depth-wise disagreement normalization |
| `mountain_car_state_value_disagreement_diagnostic_seed0` | Diagnostic | Measured value disagreement directly over physical-state grids |
| `mountain_car_state_value_disagreement_full_seed0` | Valid, exploratory | Longer state-value disagreement probe |
| `mountain_car_episodic_novelty_diagnostic_seed0` | Valid, exploratory | Early nearest-history novelty probe; no goal |
| `mountain_car_episodic_novelty_history_diagnostic_seed0` | Valid, exploratory | History-conditioned novelty prediction; no goal |
| `mountain_car_ngu_episodic_novelty_history_diagnostic_seed0_260816_191631` | Valid, exploratory | Longest NGU-style probe; max position 0.226, no goal |
| `mountain_car_state_norm_*`, `mountain_car_hybrid_no_state_norm_short_seed0` | Exploratory short runs | All remained at -200; insufficient budgets for ranking normalization schemes |
| `mountain_car_hybrid_vicreg_short_seed0` | Exploratory short run | Best -198.25 |
| `mountain_car_hybrid_vicreg_random_value_init_short_seed0` | Valid, exploratory | First goal phase 6; 13 successful segments |
| `mountain_car_hybrid_vicreg_random_value_init_mixed_sve_seed0` | Valid, exploratory | Best eval -101.63; bundled intervention |
| `mountain_car_hybrid_vicreg_random_value_init_mixed_sve_td5_seed0` | Valid, exploratory | Did not solve; showed target sensitivity |
| `mountain_car_hybrid_vicreg_random_value_init_efficientzero_seed0` | Valid, exploratory | Best eval -172; no collection goal |
| `mountain_car_qstd_reward_value_mixed_tree_local_seed0_260822_020904` | Valid, exploratory | Combined Qstd with learned intrinsic reward/value before the prior experiments |

## 6. Density and corrected-RND evidence

| Artifact/run | Status | Main use |
|---|---|---|
| `hpc_results/run_mountain_car_state_density_intrinsic_diagnostic_seed0_10675632` | Valid, exploratory | Exact density target, value propagation, and source trajectories |
| `.../rnd_signal_diagnostic/summary.json` | Valid, five diagnostic seeds | RND error versus density, state distance, and target smoothness |
| `.../rnd_signal_gain_sweep/` | Diagnostic | Target gain sensitivity |
| `.../rnd_signal_lr1e3/`, `.../rnd_signal_lr1e4/` | Diagnostic | Predictor learning-rate sensitivity |
| `hpc_results/run_mountain_car_corrected_rnd_intrinsic_diagnostic_seed0_10676135` | Valid, exploratory | Corrected RND heads disconnected from action selection |
| `hpc_results/run_mountain_car_corrected_rnd_intrinsic_diagnostic_seed0_10676689` | Valid, exploratory | Longer diagnostic under normal optimistic/extrinsic collection; not pure intrinsic |
| `hpc_results/run_mountain_car_corrected_rnd_intrinsic_tree_seed0_10677264` | Valid, exploratory | Separate intrinsic episodes with early tree design |
| `hpc_results/run_mountain_car_corrected_rnd_intrinsic_tree_seed0_10677845` | Valid, exploratory | Longer early separate-tree design; no goal |
| `hpc_results/run_mountain_car_corrected_rnd_mixed_reward_tree_seed0_10680888` | Valid, exploratory | One tree combining extrinsic and current RND returns |
| `hpc_results/intrinsic_world_model_seed0_10680015` | Valid, exploratory | Joint intrinsic world-model update probe |
| `data_ensemble_muzero/mountain_car_full_intrinsic_rnd_postupdate_v2_seed0` | Valid, exploratory | First complete intrinsic-only post-update run; first goal phase 8 |

The first state-density run `..._10674448` and pre-correction RND smoke runs are
retained for debugging history but should not be used as evidence about RND
quality.

## 7. Encoder and value-prior program

| Run | Status | Main observation |
|---|---|---|
| `mountain_car_encoder_random_prior_seed0` | Valid, exploratory | Direct latent reflected prior-induced novelty; recurrent transfer weak |
| `mountain_car_encoder_random_prior_raw_ssl_seed0` | Valid, exploratory | Introduced raw-latent consistency |
| `mountain_car_encoder_random_prior_raw_ssl_probe_seed0` | Valid, diagnostic | Full grid/sequence novelty probe; physical-to-direct 0.841, physical-to-imagined 0.051 at phase 3 |
| `mountain_car_encoder_value_random_prior_grad_seed0` | Invalid early value-prior design | Exposed scalar/categorical translation issue |
| `mountain_car_encoder_value_scalar_prior_grad_seed0` | Mechanical/diagnostic | Scalar prior correction path |
| `mountain_car_encoder_value_residual_prior_seed0` | Valid, exploratory | Residual cancellation probe |
| `mountain_car_local_encoder_value_residual_prior_seed0` | Valid, exploratory | Local hash/grid prior geometry |
| `mountain_car_fixed_norm_local_encoder_value_residual_prior_seed0` | Valid, exploratory | Prevented magnitude swamping; did not establish novelty-calibrated Qstd |

Primary files within the probe runs are:

- `diagnostics/encoder_prior_state_grid.csv`
- `diagnostics/encoder_prior_imagined_novelty.csv`
- `diagnostics/encoder_prior_observation_history.npz`

## 8. MountainCar query matrix

Root directory: `hpc_results/rnd_query_matrix_20260826/`

| Condition | Runs | Status |
|---|---|---|
| Baseline | `mcar_rnd_query_baseline_seed{0,1,2}_{8176,8226,8227}` | Valid, replicated |
| No query context | `mcar_rnd_query_no_context_seed{0,1,2}_{8228,8229,8230}` | Valid, replicated |
| Detached world model | `mcar_rnd_query_detached_world_seed{0,1,2}_{8231,8232,8233}` | Valid, replicated |
| Joint query | `mcar_rnd_query_joint_query_seed{0,1,2}_{8234,8235,8236}` | Valid, replicated |

The initial one-seed `mountain_car_rnd_landscape_query*`, no-diversity,
centered-loss, and transition-coupled directories document the design evolution.
They were superseded by the replay-integrated matrix.

## 9. Query factorial follow-up

Canonical copied family:
`hpc_results/rnd_query_factorial_followup_complete_20260826/`

| Condition | Runs | Status |
|---|---|---|
| No context, detached | `mcar_rnd_query_factorial_no_context_detached_seed{0,1,2}_{13168,13169,13170}` | Valid, replicated |
| Auxiliary only | `mcar_rnd_query_factorial_auxiliary_only_seed{0,1,2}_{13171,13172,13173}` | Valid, replicated |

The earlier jobs `9265-9270` include timeouts and exit-code-135 failures caused
during the DelftBlue storage incident. They were rerun and should not be mixed
with the completed `13168-13173` family.

## 10. Auxiliary decoder architecture matrix

Canonical root: `hpc_results/rnd_aux_decoder_ablation_20260827/`

| Condition | Job IDs | Status |
|---|---|---|
| Shared log | 15026-15028 | Valid, replicated |
| Member log | 15029-15031 | Valid, replicated |
| Shared raw | 15032, 18842, 18843 | Valid, replicated after reruns |
| Member raw | 18844-18846 | Valid, replicated after reruns |
| Shared normalized raw | 19244-19246 | Valid, replicated; copied through intermittent I/O failures |
| Member normalized raw | 19247-19249 | Valid, replicated; copied through intermittent I/O failures |

`hpc_results/rnd_aux_decoder_normalized_raw_recovery_20260828/` is a recovery
copy of the normalized-raw artifacts, not an independent replication.

## 11. Ordinary intrinsic reward-head ablation

Root: `hpc_results/intrinsic_reward_head_ablation_20260828/`

| Condition | Job IDs | Status |
|---|---|---|
| Double loss | 25673-25675 | Valid, replicated |
| Log target | 25676-25678 | Valid, replicated |
| Shared raw head | 27184-27186 | Valid, replicated |
| Shared log head | 28555-28557 | Valid, replicated |

These runs do not instantiate the auxiliary reader. They test whether loss
weight, target transform, or an operational shared reward head alone explains
the auxiliary result.

## 12. Pre-registered confirmatory MountainCar family

Root: `hpc_results/rnd_aux_confirmatory_20260830/`

| Condition | Job IDs | Status |
|---|---|---|
| Base | 30218-30220 | Valid, replicated |
| Joint train-only | 30221, 30222, 30224 | Valid, replicated |
| Detached train-only | 30226-30228 | Valid, replicated |
| Ordinary matched train-only | 30229-30231 | Valid, replicated |
| Ordinary exact train-only | 30232-30234 | Valid, replicated |
| Joint tree | 30235-30237 | Valid, replicated |
| Joint tree without private reward loss | 30238-30240 | Valid, replicated |

The plan and intended causal contrasts are recorded at the top of
`experiments/run_mountain_car_rnd_aux_confirmatory_ablation.py`.

## 13. Branching Rooms environment and visuals

| Artifact | Purpose |
|---|---|
| `zoo/minigrid/envs/branching_rooms_env.py` | Environment implementation and map topology |
| `zoo/minigrid/config/branching_rooms_ensemble_muzero_config.py` | Pixel MuZero/RND configuration |
| `zoo/minigrid/envs/test_branching_rooms_env.py` | Environment invariants |
| `zoo/minigrid/config/test_branching_rooms_intrinsic_rnd_config.py` | Experiment-condition invariants |
| `video/branching_rooms_visuals/branching_rooms_map.png` | Exact logical map |
| `video/branching_rooms_visuals/branching_rooms_walkthrough.gif` | Scripted route through the map |
| `video/branching_rooms_long_bootstrap/` | Phase/cumulative baseline versus auxiliary coverage |
| `video/branching_rooms_compact_query/` | Compact-query coverage comparisons |

The first submitted jobs failed on an MLP tuple/list mismatch and a terminal
`to_play` indexing bug. They are mechanical failures and are excluded.

## 14. Branching Rooms run families

### Pilot

Root: `hpc_results/branching_rooms_rnd_pilot_20260831/`

| Runs | Status | Result |
|---|---|---|
| baseline seed 0 job 48844; auxiliary seed 0 job 48845 | Mechanical/one-phase | Identical initial coverage |

### Three-phase comparison

Roots:

- `hpc_results/branching_rooms_rnd_three_phase_20260831/` for seed 0;
- `hpc_results/branching_rooms_rnd_three_phase_replications_20260902/` for seeds
  1 and 2.

Jobs 49867-49868 and 72907-72910 form the valid three-seed comparison.

### Finite-horizon 20-step family

Root: `hpc_results/branching_rooms_rnd_finite_horizon_20step_20260903/`

Jobs 78468-78473 completed, but the family is **invalid for the intended TD
comparison** because continuation value was zero rather than bootstrapped.

### Correct six-phase bootstrapped family

Root: `hpc_results/branching_rooms_long_bootstrap_20260903/`

| Condition | Job IDs | Status |
|---|---|---|
| Baseline | 87101-87103 | Valid, replicated |
| Shared log auxiliary | 87104-87106 | Valid, replicated |

### Compact-query family

Root: `hpc_results/branching_rooms_compact_query_20260904/`

| Condition | Job IDs | Status |
|---|---|---|
| Compact query with context | 87483-87485 | Valid, replicated |
| Compact query without context | 87486-87488 | Seeds 0-1 complete; seed 2 timed out after phase 4 |

Job 87488 reached the learner after collecting phase 4 and was cancelled at the
four-hour limit. It is not evidence of a model crash, but its final coverage is
not directly comparable to six-phase runs.

Compact-query geometry is stored in each
`diagnostics/compact_query_phase_*.npz`; phase coverage is in
`diagnostics/environment_coverage.csv` and
`diagnostics/environment_coverage_cells.csv`.

## 15. Episodic-memory family

Conditions implemented in `experiments/run_branching_rooms_intrinsic_rnd.py`:

| Condition | Status | Required comparison |
|---|---|---|
| `compact_query_projected_memory` | Complete, three rerun seeds: 911678-911680 | Memory mostly inactive; negligible mean pose-coverage change |
| `compact_query_raw_ssl` | Complete: 911681, 911682, 97867 | No-memory control; highest mean position/pose coverage in this family |
| `compact_query_raw_memory` | Complete: 97888-97890 | Strong depth-dependent suppression and worse coverage in all paired seeds |

The memory implementation is in `lzero/policy/episodic_latent_memory.py` and its
tests in `lzero/policy/tests/test_episodic_latent_memory.py`.

The first projected-memory attempts, 97862 and 97863, and the interrupted jobs
97864-97866 were infrastructure failures and are excluded. Complete artifacts
are stored under
`hpc_results/branching_rooms_episodic_memory_20260905`. The matched projected
no-memory control is stored under
`hpc_results/branching_rooms_compact_query_20260904` as jobs 87483-87485.

## 16. Operational provenance

### DelftBlue migration

The cluster moved to AlmaLinux 9, Slurm 26.05, and a 2026 module stack. Job
scripts were adapted to use Bash and the new GPU stack. GPU resource names became
`nvidia_a100` and `tesla_v100s`.

### Storage incident

BeeGFS produced intermittent remote-I/O errors while creating Apptainer
containers, reading completed artifacts, and transferring multi-gigabyte
diagnostic files. Relevant safeguards were:

- validate the SIF with `apptainer sif list` before submitting;
- retain a working environment until no dependent jobs remain;
- copy diagnostics in resumable chunks;
- verify expected files locally before deleting remote runs; and
- clean `/scratch/$USER/lightzero_runs` promptly after verification.

Failed jobs 9265-9270 and partially failed first attempts 15033-15037 were
replaced by completed reruns. Recovery directories and duplicate result trees
must not be counted as additional seeds.
