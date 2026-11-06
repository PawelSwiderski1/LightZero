# Recent-history transformer encoder comparison

## Question

Does the revisit predictor's generalization improve when its observation
encoder preserves the MiniGrid tile structure or receives native symbolic
MiniGrid cells?

## Controlled setup

- One fixed Branching Rooms layout.
- 528 independently seeded trajectories, each with 128 actions.
- Train, validation, and test contain uniform, persistent, and macro policies.
- The held-out split contains fresh macro-policy trajectories only.
- The held-out split tests new trajectories and action patterns on the same
  layout. It is not an unseen-layout test.
- The prediction history contains the previous 40 real observation/action
  steps, followed by an action-only candidate suffix of up to five steps.
- The target is exact-RGB novelty within the recent history window.
- A score of 1 means novel and a score of 0 means revisit.
- Each condition uses 32 sampled roots per gradient update, batch size 1, two
  transformer layers, width 128, four attention heads, and embedding width 64.

The three encoders are:

- `pooled_2x2`: the original RGB convolutional encoder with adaptive 2 by 2
  pooling.
- `rgb_tile_7x7`: an RGB encoder that divides the rendered observation into the
  native 7 by 7 tile grid before spatial processing.
- `symbolic_7x7`: native MiniGrid object, color, and state identifiers for the
  same partial egocentric 7 by 7 view. It receives no global position, global
  map, or cells hidden behind walls.

## AUC results

Values are the mean and population standard deviation across seeds 0, 1, and
2. Higher is better.

| Updates | Encoder | Train | Validation | Test | Held-out macro |
|---:|---|---:|---:|---:|---:|
| 256 | pooled 2x2 | 0.763 +/- 0.005 | 0.722 +/- 0.005 | 0.736 +/- 0.008 | 0.543 +/- 0.017 |
| 256 | RGB tile 7x7 | 0.767 +/- 0.007 | 0.727 +/- 0.010 | 0.739 +/- 0.008 | 0.553 +/- 0.022 |
| 256 | symbolic 7x7 | 0.774 +/- 0.011 | 0.729 +/- 0.017 | 0.742 +/- 0.013 | 0.549 +/- 0.037 |
| 512 | pooled 2x2 | 0.795 +/- 0.005 | 0.747 +/- 0.007 | 0.759 +/- 0.005 | 0.582 +/- 0.012 |
| 512 | RGB tile 7x7 | 0.802 +/- 0.003 | 0.754 +/- 0.005 | 0.768 +/- 0.005 | 0.583 +/- 0.007 |
| 512 | symbolic 7x7 | 0.814 +/- 0.006 | 0.766 +/- 0.003 | 0.777 +/- 0.005 | 0.606 +/- 0.001 |
| 1024 | pooled 2x2 | 0.822 +/- 0.009 | 0.762 +/- 0.009 | 0.781 +/- 0.002 | 0.597 +/- 0.008 |
| 1024 | RGB tile 7x7 | 0.828 +/- 0.005 | 0.762 +/- 0.009 | 0.783 +/- 0.004 | 0.592 +/- 0.014 |
| 1024 | symbolic 7x7 | 0.854 +/- 0.004 | 0.787 +/- 0.005 | 0.805 +/- 0.004 | 0.632 +/- 0.012 |

At 1024 updates, symbolic held-out AUC was 0.637, 0.643, and 0.616. The
matched pooled-RGB values were 0.603, 0.601, and 0.586. Symbolic therefore won
in every matched seed by 0.034, 0.042, and 0.030.

The tile-aligned RGB and symbolic encoders have similar parameter counts, but
only symbolic improved held-out performance. The symbolic gain is therefore
not explained solely by a larger encoder. It shows that visual perception is
one limitation, while the remaining held-out gap shows that temporal and
movement generalization are still incomplete.

## Reproduction

Build the deterministic trajectory dataset:

```bash
experiments/prepare_recent_transformer_encoder_dataset_local.sh
```

Run all nine conditions:

```bash
experiments/run_recent_transformer_encoder_comparison_local.sh
```

Useful overrides include `PYTHON_BIN`, `BUDGETS`, `SEEDS`, `ENCODERS`,
`DATASET`, and `OUTPUT_DIR`. A command preview is available with `DRY_RUN=1`.
