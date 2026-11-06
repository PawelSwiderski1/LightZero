# Experimental History: Uncertainty-Guided Exploration in Ensemble MuZero

**Documentation cutoff:** 2026-09-04  
**Status:** working research record; completed results are separated from pending
experiments and mechanically invalid runs.  
**Meeting-oriented companion:** [research narrative](research_narrative.md)  
**Companion:** [experiment registry](experiment_registry.md)

## 1. Purpose and scope

This document reconstructs the development of the project from its initial
ensemble-MuZero implementation through the current Branching Rooms experiments.
It is written as a research narrative rather than a terminal log. Each stage
records:

1. the question being tested;
2. the intervention that was implemented;
3. the evidence that was collected;
4. what the result does and does not establish; and
5. why the next experiment followed.

The central problem is **directed exploration with a learned MuZero world
model**. The desired agent should use planning to prefer action sequences whose
consequences are epistemically uncertain or novel, including consequences that
have not yet been experienced. A useful signal must therefore satisfy more than
"different networks output different numbers." It should be:

- high for genuinely unsupported consequences;
- lower for familiar consequences;
- coherent across nearby physical states;
- available on imagined MCTS states, not only real observations;
- robust to representation drift; and
- useful enough to change behavior and increase state-space coverage.

## 2. Executive summary

The project progressed through four conceptual stages.

First, an ensemble MuZero implementation was built and validated. Ensemble
disagreement could accelerate MountainCar learning, but the initial uncertainty
signal was dominated by disagreement in leaf/value predictions and was not
automatically a reliable measure of state novelty.

Second, several ways of making disagreement itself drive exploration were
tested: dynamics disagreement, Q disagreement, fixed-horizon optimistic
planning, calibration, random priors, latent regularization, and episodic
novelty. Some variants reached the MountainCar goal, but the behavior was often
unstable, strongly affected by member offsets or search depth, and difficult to
interpret as novelty-directed exploration.

Third, the project moved to an explicit novelty target: RND evaluated on real
states. After correcting RND input normalization and target timing, RND did
track state-density novelty. However, the contrast was moderate rather than
binary: across phases 1-6, mean RND error in unvisited regions was roughly
`1.09x` to `1.37x` the error in dense regions. A fully intrinsic agent was then
built so collection, policy, reward, value, encoder, and dynamics were all
trained from intrinsic experience rather than a mixture of intrinsic and
extrinsic policies.

Fourth, an auxiliary task was added to make imagined latents expose current RND
novelty. In MountainCar this produced the strongest and most repeatable change
in exploration. The decisive ablation was not whether synthetic query values
were available to MCTS; it was whether the auxiliary RND-prediction loss could
backpropagate into the encoder and dynamics. A detached auxiliary branch was
behaviorally identical to baseline, while a joint train-only branch increased
successful goal-reaching trajectory segments from `13.3 +/- 10.3` to
`54.3 +/- 32.7` per seed. An ordinary auxiliary decoder with matched training
could also help. Therefore the supported conclusion is:

> Explicit, depth-wise prediction of current RND from imagined latents can shape
> the learned world model into a representation that supports substantially
> better exploration. MountainCar does not show that learned landscape queries
> themselves are necessary.

Branching Rooms was introduced to test the mechanism in a larger pixel-based,
branched environment. Over six phases, the shared log-RND auxiliary model had
similar raw position coverage to baseline but reached more topologically distant
regions: `13.67` versus `13.00` rooms and `3.00` versus `2.00` dead-end rooms on
average over three seeds. The effect is promising but seed-variable. Compact
query context improved topological coverage relative to the no-context variant,
but the generated query codes occupied latent holes and were almost independent
of the root. This prevents a clean claim that the network learned valid local
state queries.

The final experiment addressed a remaining problem in Branching Rooms: RND is
frozen within each collection phase, so returning to a state visited earlier in
the same episode can remain attractive. Two latent episodic-memory variants
were tested, one in projected SSL space and one under raw-latent SSL. Complete
reruns showed that projected memory was mostly inactive, while raw memory
over-suppressed deeper imagined paths and reduced exploration.

### 2.1 Development timeline

| Period | Main research activity |
|---|---|
| 7-8 July | Ensemble wrapper, member training, Thompson sampling, CartPole controls |
| 8-24 July | MountainCar shared-tree optimism, ablations, diagnostics, replay and search-value work |
| 28 July-18 August | Intrinsic trees, dynamics/Q disagreement, fixed-horizon calibration, latent regularization, episodic novelty |
| 18-24 August | State-density and corrected-RND targets, encoder/value priors, full intrinsic agent |
| 25-30 August | RND landscape queries, replay-integrated auxiliary loss, multi-seed causal ablations |
| 31 August-4 September | Branching Rooms, pixel RND, bootstrapped comparisons, compact queries, episodic latent memory |

### 2.2 Hypothesis status at the cutoff

| Hypothesis | Status | Evidence summary |
|---|---|---|
| Raw ensemble Q disagreement identifies novel consequences | Not supported as stated | Qstd was dominated by leaf/value disagreement and affected by depth and member offsets |
| Direct dynamics disagreement is sufficient for exploration | Rejected in tested forms | Multiple one-seed variants failed to produce reliable goal discovery |
| Correctly normalized real-state RND tracks visitation novelty | Supported | Five diagnostic seeds showed moderate density/distance correlation and higher error outside dense support |
| Forcing recurrent latents to predict current RND improves pure exploration | Supported in MountainCar; tentative transfer | Replicated detached-versus-joint ablation; modest topological gain in Branching Rooms |
| Learned query context is the cause of the MountainCar gain | Rejected | No-context and query-intervention controls retained the behavior |
| Compact query context helps in a branched pixel environment | Tentative | Better room/dead-end means, but one comparison seed is short and generated queries occupy likely latent holes |
| Latent episodic memory can suppress same-episode revisits safely | Rejected in tested forms | Projected memory barely changed root rewards; raw memory strongly suppressed deep paths and reduced coverage in every seed |

## 3. System developed for the experiments

### 3.1 Ensemble MuZero

The base implementation was extended from a single MuZero model to an ensemble
of member models. Each member can have its own representation, dynamics,
prediction, reward, value, and intrinsic heads, subject to configurable sharing.
The implementation supports:

- independent and shared-tree inference;
- Thompson-style member selection;
- mean, pessimistic, and optimistic ensemble aggregation;
- Q-standard-deviation bonuses;
- member-specific frozen priors;
- separate intrinsic and extrinsic trees;
- recurrent-depth diagnostics; and
- member-wise decomposition of reward and leaf-value contributions.

The main implementation is in:

- `lzero/model/ensemble_muzero_model.py`
- `lzero/policy/ensemble_muzero.py`
- `lzero/mcts/ensemble_muzero_mcts.py`
- `lzero/entry/train_ensemble_muzero.py`

### 3.2 Experimental cycle

Most later experiments use a phase-based cycle:

```text
collect trajectories with the current policy
update the RND predictor on fresh observations
freeze a current RND target snapshot
train MuZero and intrinsic modules from replay
write phase-level diagnostics
```

This makes the novelty function stationary during each learner phase and keeps
targets internally consistent. It also creates the within-phase revisitation
problem discussed in Section 12.

### 3.3 Primary metrics

MountainCar is a continuous state space with position and velocity. Exact states
almost never repeat, so "unseen" is defined through grids, nearest-neighbor
distance, density, or RND error rather than exact equality. Metrics include:

- maximum and minimum position;
- maximum absolute velocity;
- coverage of position-velocity bins;
- first phase reaching the goal;
- count of successful 200-step trajectory segments;
- RND error versus density and nearest visited-state distance;
- ensemble Q standard deviation and its reward/value decomposition; and
- direct-versus-imagined latent alignment and novelty ranking.

Latent novelty in the encoder-prior probes was measured as k-nearest-neighbor
distance to a bank of latents encoded from previously collected real states.
The physical counterpart was k-nearest-neighbor distance in normalized
position-velocity space. "Physical-to-imagined correlation" is the correlation
between those two novelty scores over simulated action-sequence endpoints.
"Within-root ranking" asks whether, among sequences starting from the same real
root, the physically more novel endpoint also has the larger latent novelty.
This separates global scale effects from the action ranking that MCTS needs.

In the pure-exploration runs, goal-reaching counts are **diagnostic events, not
the objective**. A novelty-seeking policy should not retain the extrinsically
optimal trajectory after it becomes familiar.

Branching Rooms metrics include:

- unique grid positions out of `693` reachable positions;
- unique position-orientation poses out of `2772`;
- rooms reached out of `16`; and
- designated dead-end rooms reached out of `5`.

Room and dead-end counts measure topological depth. Raw cell coverage alone can
hide the distinction between thoroughly revisiting nearby areas and reaching a
small number of distant branches.

### 3.4 Evidence labels used here

- **Mechanical:** verifies that code paths, gradients, or target construction
  behave as intended.
- **Exploratory:** usually one seed, unequal budgets, or a diagnostic run. Useful
  for generating or rejecting mechanisms, not for a final performance claim.
- **Replicated:** three seeds with a controlled condition family.
- **Invalid for the intended question:** the run may have completed, but a bug or
  wrong target definition means it cannot answer the planned comparison.

## 4. Stage I: ensemble mechanics and control experiments

### 4.1 CartPole validation

The first stage established that ensemble construction, training, checkpointing,
and Thompson collection worked end to end. CartPole was deliberately used as an
easy mechanical control.

| Condition | Evaluations | First score >=150 | First score 200 | Final/best | Wall time |
|---|---:|---:|---:|---:|---:|
| Single model | 5 | iter 100, env step 620 | iter 400, env step 2799 | 200 / 200 | 1.88 min |
| 3-member Thompson ensemble | 7 | iter 300, env step 1889 | iter 600, env step 4111 | 200 / 200 | 6.28 min |

The ensemble was mechanically correct, but it was neither faster nor more
sample-efficient on this easy task. CartPole therefore served as an
implementation check, not evidence that the ensemble exploration method was
better.

### 4.2 Initial MountainCar ablation

The first meaningful comparison asked whether the ensemble and an optimistic
Q-standard-deviation bonus helped on sparse-reward MountainCar.

| Condition, seed 0 | Final evaluation | Best evaluation | Main observation |
|---|---:|---:|---|
| Single model | -200.0 | -200.0 | Did not solve within the run |
| Ensemble, beta 0, mean evaluation | -108.1 | -104.0 | Ensemble alone eventually learned |
| Ensemble, beta 0, pessimistic evaluation | -106.0 | -106.0 | Similar late result |
| Optimistic collection, beta 2 | -115.5 | -101.25 | Improved much earlier, then remained variable |

The beta-2 run improved at iteration 1000, whereas the beta-0 mean run first
improved at iteration 1200 and reached comparable late returns much later. This
supported continued work on uncertainty-guided collection, but it was a
single-seed, partially confounded comparison. It did not prove that Q
disagreement represented novelty.

## 5. Stage II: what the ensemble disagreement represented

### 5.1 Decomposing Q disagreement

The initial working assumption was that independently initialized members would
generalize differently outside their training support, producing epistemic
uncertainty. Diagnostics showed that this was only partly true.

For the main optimistic MountainCar run, ten-phase chunks showed:

| Phases | Collection return | Evaluation return | Q std | Leaf std | Reward std | Future-goal rate |
|---|---:|---:|---:|---:|---:|---:|
| 0-9 | -199.0 | -199.9 | 2.80 | 2.77 | 0.059 | 0.055 |
| 10-19 | -156.2 | -122.1 | 5.60 | 5.52 | 0.136 | 0.582 |
| 20-29 | -123.0 | -107.0 | 9.06 | 8.96 | 0.153 | 0.967 |
| 30-39 | -125.5 | -105.3 | 7.26 | 7.18 | 0.130 | 0.777 |
| 40-49 | -114.9 | -107.7 | 6.41 | 6.39 | 0.073 | 0.872 |
| 50-55 | -128.5 | -108.0 | 6.02 | 5.98 | 0.083 | 0.862 |

Nearly all Q disagreement came from the backed-up leaf/value term, not from
immediate reward disagreement. High Q disagreement coexisted with useful goal
discovery, but it also reflected member-specific value offsets and tree-depth
effects. It could not be interpreted directly as "this action reaches a novel
state."

### 5.2 Direct intrinsic-tree and dynamics-disagreement tests

Several one-seed exploratory variants tested whether disagreement could become
the sole exploration objective.

| Representative condition | Phases | Goal evidence | Outcome |
|---|---:|---:|---|
| Intrinsic tree only | 12 | none | No directed exploration |
| Dynamics disagreement, L2 prior 0.5 | 34 | none | Reached position 0.153 but not goal |
| Dynamics disagreement with projected SSL | 16 | none in collection | Eval improved to -138.9, but exploration remained weak |
| Q disagreement, fixed exploration | 16 | first success phase 7, 104 successful segments | Stronger than dynamics disagreement |
| Q disagreement, pure exploration horizon 5 | 29 | first success phase 9, 2 segments | Goal discovery was possible but not stable |
| Qstd intrinsic tree only | 4 | none | Insufficient directional signal |

These runs rejected the simple proposition that arbitrary latent or dynamics
disagreement is enough. Disagreement can be large for reasons unrelated to
physical novelty, and learned dynamics can map unsupported futures into latent
regions where members nevertheless agree.

### 5.3 Fixed-horizon optimism, calibration, and elite replay

The next family tried to make Q disagreement more comparable across actions and
depths. Work included:

- fixed action-sequence horizons;
- bounded optimistic scores and member envelopes;
- member-wise value calibration;
- depth-matched and depth-normalized disagreement;
- mixed search-value targets; and
- elite replay to retain rare goal-reaching trajectories.

Representative exploratory results were:

| Condition, seed 0 | Phases | First goal phase | Successful segments | Best evaluation |
|---|---:|---:|---:|---:|
| Fixed-horizon bounded optimism | 34 | 18 | 202 | -115.75 |
| Fixed-horizon bounded envelopes | 29 | 13 | 190 | -136.13 |
| Fixed-horizon uncapped beta 2 | 41 | 20 | 217 | -105.38 |
| Fixed-horizon value calibrated | 15 | 6 | 116 | -134.75 |
| Optimistic shared tree, elite stratified | 56 | 6 | 703 | -102.63 |
| Mixed search-value estimate | 23 | 8 | 171 | -120.75 |

The experiments established that careful calibration and replay can make
optimistic control effective. They also exposed a conceptual mismatch with the
research goal. Elite replay and extrinsic-value learning retain "good"
goal-directed trajectories, while a pure exploration agent should keep seeking
new regions. Moreover, the intrinsic model was being trained on a policy partly
determined by extrinsic values. This motivated separating intrinsic collection
and training entirely.

### 5.4 Early episodic novelty and NGU-style probes

Before moving fully to RND, episodic novelty was also estimated from recent
state/latent history. The diagnostic and NGU-style variants tested whether
nearest-neighbor familiarity and a learned intrinsic model could suppress
revisits and favor new regions.

The representative history diagnostic did not reach the goal. The longest
NGU-style run lasted 16 phases and reached maximum position `0.226`, again
without reaching the goal. These runs showed that an episodic signal could move
the frontier beyond the earliest disagreement-only failures, but its learned
predictions were not reliable enough to support planning. They are important
precursors to the current episodic memory, which differs by applying a direct
distance-based correction to each imagined edge rather than asking a value head
to learn the whole correction from replay.

### 5.5 Qstd as a learned intrinsic reward

A later one-seed bridge experiment used Q disagreement itself as the immediate
intrinsic reward, learned reward and value heads for it, and combined that return
with the ordinary tree. It first reached the goal in phase 4 and produced 192
successful trajectory segments over 19 phases. This showed that a propagated
Qstd signal could be behaviorally powerful. It did not solve the semantic
problem: Qstd still included value-head generalization, member offsets, and
depth effects, so strong exploration did not establish that the reward meant
physical novelty. That distinction led back to RND as an externally measurable
target.

## 6. Stage III: representation experiments and random priors

### 6.1 Normalization and latent regularization

Batch normalization, layer normalization, hybrid state normalization, VICReg,
random value initialization, raw-latent consistency, and EfficientZero-style
variants were tested. Most short normalization runs remained at `-200` and were
not long enough for comparative claims. Two observations mattered:

- VICReg plus random value initialization produced goal discovery where nearby
  short controls did not.
- A long VICReg/random-initialization/mixed-target run reached a best evaluation
  of `-101.63`, but its multiple simultaneous changes prevented attribution.

These were hypothesis-generating results: representation geometry and initial
member diversity clearly mattered, but generic regularization did not by itself
define the desired novelty semantics.

### 6.2 Encoder priors and the imagined-latent novelty hypothesis

An earlier proposal placed a frozen member-specific prior on a pair consisting
of a real root state and the action sequence leading to an imagined node. This
would make nearby sequences, such as four versus five repeated right actions,
receive related offsets while keeping distant sequences different. It avoided
feeding a frozen prior an arbitrarily drifting latent, but it introduced a more
fundamental problem: uncertainty was indexed by state-action history rather than
physical state. Two sequences reaching the same known state could remain
artificially different. The idea was useful for clarifying the desired
correlation structure but was not adopted as the final state-novelty mechanism.

The next hypothesis was explicit:

```text
direct latent = trainable encoder output + frozen member-specific state prior
```

Novel real states would initially receive unfamiliar, member-specific latent
offsets. If SSL made recurrent dynamics match direct encodings, imagined states
leading to novel physical states might also become geometrically novel and
produce higher value disagreement.

A phase-3 diagnostic showed that the first half worked but the second did not:

- physical novelty to direct-latent novelty correlation: `0.841`;
- physical novelty to imagined-latent novelty correlation: `0.051`;
- direct to imagined novelty correlation: `0.213`;
- within-root `novel > familiar` ranking: direct `1.000`, imagined `0.469`;
- median imagined/direct alignment relative to direct density: `12.23`;
- median retrieved physical-state error from imagined latent: `0.674`.

The ordinary projected SSL objective could map substantially different raw
latents to similar projection outputs. It therefore did not transfer the
direct-latent novelty geometry into recurrent latent space.

BatchNorm exposed an additional measurement problem. In training mode, batch
statistics made projected imagined and direct latents appear close; in evaluation
mode, fixed running statistics exposed a much larger distribution mismatch. The
projection and prediction networks could satisfy normalized-output consistency
without making raw recurrent states meaningfully comparable to direct encoder
states. This did not make projected SSL useless, but it made it inadequate for
the specific claim that raw latent distance represented novelty.

### 6.3 Raw-latent SSL

Raw-latent consistency was introduced to remove that loophole. This made
recurrent states closely match direct encodings at depth 5 in raw space. The
important negative result was that alignment alone still did not guarantee a
useful uncertainty separation. The model could learn a representation in which
both familiar and unsupported states were handled similarly.

### 6.4 Value priors and cancellation

Frozen random priors were also attached to value predictions. A categorical
support bug was found: adding a scalar in logit space did not produce a fixed
scalar shift in decoded value. The implementation was changed so the decoded
total value changes by exactly the prior scalar, regardless of distribution
sharpness. In the regression sanity check, the **total** predictions converged
to approximately `-100` for priors `-2`, `0`, and `+2`; correspondingly, the
trainable components converged to about `-98`, `-100`, and `-102`.

This verified prior cancellation mechanically. It did not establish useful
epistemic behavior. A further failure mode was then measured: the trainable
encoder produced latent norms around `38-40` while the prior contribution had
norm `1`, making the prior negligible. Fixed-norm composition was implemented
so the encoder could no longer suppress the prior merely by scaling its own
output.

### 6.5 Result of the prior program

The prior experiments clarified the requirements for a useful random prior:

- nearby physical states should receive similar offsets;
- unsupported regions should not be predictable from supported regions;
- the prior-to-learned magnitude ratio must remain controlled; and
- the effect must survive the encoder-dynamics consistency map.

The smooth learned-input prior generalized too broadly, while a local
hash/grid-style prior improved locality but did not yield a clean, sustained
seen/unseen Q-disagreement separation. Fixed-norm composition removed one
escape route, but the overall mechanism remained unresolved. This line was
therefore paused rather than reported as a successful exploration method.

## 7. Stage IV: explicit state novelty with density and RND

### 7.1 State-density diagnostic

An exact discretized state-density reward provided a ground-truth diagnostic in
MountainCar. In phase 0 there was no prior count snapshot, so all 227 occupied
bins received reward `1`. Later phases produced graded novelty:

| Phase | Live unique bins | Reward mean | Reward std | Reward max |
|---|---:|---:|---:|---:|
| 0 | 227 | 1.000 | 0.000 | 1.000 |
| 1 | 336 | 0.120 | 0.152 | 1.000 |
| 2 | 406 | 0.096 | 0.104 | 1.000 |

Value propagation was initially poor under full Monte Carlo targets:

| Phase | Target mean | Prediction mean | MAE | Correlation |
|---|---:|---:|---:|---:|
| 0 | 57.13 | 7.13 | 50.23 | 0.317 |
| 1 | 7.39 | 7.74 | 4.99 | 0.305 |
| 2 | 6.45 | 6.97 | 3.87 | 0.382 |

This explained why early leaf values could be wrong even with a correct reward
definition: the agent had not trained on many relevant states, and the very
large phase-0 return scale made value fitting difficult.

### 7.2 Correcting RND

The first RND implementation had a normalization problem that collapsed many
continuous states to effectively repeated inputs. Fixed physical bounds were
introduced for MountainCar position and velocity. On a 6461-state grid, the
corrected preprocessing retained all 6461 distinct inputs; the old
layer-normalized path retained only 143 rounded unique inputs.

The corrected random target was strongly smooth in physical space. Across five
target seeds, physical-to-target-distance Spearman correlation was
`0.962-0.992`, and far/near target-distance ratios were `17.8-20.0`. This was the
desired local generalization property.

### 7.3 Does RND actually track novelty?

The standalone five-seed diagnostic compared RND error with density and distance
to previous experience. For phases 1-6:

- RND-density Spearman correlation was about `0.46-0.62`;
- nearest-prior-state-distance Spearman correlation rose from `0.25` to `0.63`;
- unvisited/dense mean-error ratio ranged from `1.087` to `1.370`; and
- episode RND-return versus density-novelty correlation was usually high,
  ranging from roughly `0.70` to `0.95` after phase 0.

RND was therefore a meaningful but smooth novelty signal. A geometrically
unvisited bin close to familiar bins could have only modestly higher RND error.
It was incorrect to expect every unvisited bin to be strongly attractive.

### 7.4 Fresh-update target timing

The RND predictor is updated on newly collected observations. If intrinsic
reward targets are captured before that update, the replay target rewards the
novelty that the predictor has just consumed. The target snapshot was therefore
refreshed **after** the fresh RND update. Later logs explicitly report both the
pre/post update error and `target_snapshot_refreshed=1`.

This change aligns the learned reward/value model with the novelty landscape
that will actually govern the next collection phase.

### 7.5 Mixed reward tree versus full intrinsic training

Two architectures were explored.

The mixed tree combined extrinsic and intrinsic returns in one search. It could
propagate novelty toward the start state, but collection remained influenced by
incorrect extrinsic values. Consequently, the intrinsic value function was
trained under a diluted mixed policy rather than the policy it was meant to
evaluate.

The full intrinsic architecture removed that mismatch:

- every collection action came from an intrinsic tree;
- the tree used a learned intrinsic policy prior;
- all replay was generated by that policy;
- extrinsic reward and value losses were zero;
- encoder, dynamics, intrinsic reward, intrinsic value, and policy were trained
  jointly; and
- RND targets were recomputed from the current post-update snapshot.

The completed local post-update run reached the MountainCar goal first in phase
8 and accumulated 16 successful trajectory segments over 16 phases. This was
not the strongest MountainCar count obtained later, but it established that a
fully intrinsic MuZero agent could discover the goal without extrinsic reward
guiding either collection or training.

## 8. Stage V: predicting the current novelty landscape

### 8.1 Motivation

A normal reward head receives a latent state and learns RND reward only where
replay provides targets. To predict novelty before visiting a region, the model
must infer both:

1. what physical consequence an imagined latent represents; and
2. how novel that consequence is under the **current** RND predictor.

The landscape-query idea introduced an external source of current information.
For each real root state, a query policy selected physical states, evaluated
their exact current RND errors, encoded the `(state, error)` pairs, and supplied
that context to a reader. The reader predicted RND for each recurrent latent in
the same MuZero replay unroll. Exact current RND at the real future observations
provided the targets.

No environment dynamics were used to generate supervision. The only supervised
future states were those already present in ordinary replay trajectories.

### 8.2 Evolution of the training design

The first standalone query-training design sampled roots independently and used
counterfactual environment successors. It was rejected because it was detached
from actual collection and relied on known environment dynamics.

An early diversity loss pushed query locations apart. It was removed because
the desired probes may legitimately cluster near a root or frontier; forced
global separation could oppose useful local measurement. A within-root centered
or pairwise ranking loss was also considered, but it was weak when only the one
sequence actually executed after a root had a real future target. This helped
motivate training on all valid depths of ordinary replay unrolls instead of a
separate query-training phase.

The replacement integrated the query loss into every ordinary MuZero learner
minibatch:

```text
replay root and action sequence
    -> encoder and recurrent dynamics produce z1 ... zK
current query policy(root)
    -> queried states and exact current RND values
reader(query context, zk)
    -> predicted current RND at each depth
current RND(real replay future state)
    -> target at each depth
```

The RND model was frozen with respect to this loss. A dual-optimizer bug was
also fixed so the same world-model parameters were not independently stepped by
two optimizers.

### 8.3 Why the auxiliary loss differs from the ordinary reward loss

The ordinary intrinsic reward head and auxiliary decoder both predict RND, so
their redundancy was explicitly investigated. Their initial implementations
differed in sharing, architecture, target transform, optimizer settings, and
whether predictions were used by MCTS. Later ablations removed these bundled
differences.

The remaining conceptual distinction is:

- the ordinary reward prediction is an operational part of each member's
  planning model;
- the auxiliary branch can impose a separate, explicitly weighted depth-wise
  representation objective without replacing the planner's reward channel; and
- when shared, it requires member latents to expose RND in a common decodable
  form.

The experiments ultimately showed that a special landscape-reader architecture
is **not required** for the main MountainCar improvement. The important factor
was an additional current-RND prediction gradient into the shared world model.

## 9. MountainCar query and auxiliary ablations

### 9.1 Initial three-seed query matrix

All conditions used pure intrinsic collection. Successful segments are reported
as a measure of how often exploration reached the distant right-hand region,
not as a return objective.

| Condition | First goal phase by seed | Successful segments, mean +/- sd | Mean max position | Mean max speed |
|---|---|---:|---:|---:|
| Baseline | 8, 8, 13 | 13.3 +/- 10.3 | 0.527 | 0.0611 |
| Query, world model detached | none, 8, 8 | 14.3 +/- 12.4 | 0.443 | 0.0647 |
| Joint query/world-model training | 4, 4, 6 | 49.7 +/- 17.5 | 0.531 | 0.0654 |
| Joint training, no query context | 4, 5, 8 | 60.7 +/- 29.7 | 0.531 | 0.0672 |

The query-enabled joint models reached the goal earlier and much more often.
However, removing query context did not hurt and often helped. This was the
first evidence that the improvement came from the auxiliary objective rather
than information contained in selected queries.

### 9.2 Direct query-context interventions

Post-hoc interventions measured whether collection decisions depended on the
actual query content:

- reversing errors among query locations left `97.8%` of preferred actions
  unchanged;
- replacing all query errors by their mean left `98.7%` unchanged;
- supplying another root's query context left `97.2%` unchanged; and
- zeroing query latents and errors still left `86%` unchanged, while the
  velocity-aligned preference remained on `83%` of states.

The specific query measurements were therefore not the main causal source of
MountainCar behavior. The model had learned a simple, useful momentum-aligned
regularity from the auxiliary future-RND task.

### 9.3 Factorial follow-up

Two controls sharpened the result:

| Condition | First goal phase by seed | Successful segments, mean +/- sd |
|---|---|---:|
| Auxiliary only, no query context used in tree | 5, 6, 7 | 54.3 +/- 32.7 |
| No context and world model detached | none, 9, 9 | 16.7 +/- 24.7 |

The behavior changed only when the auxiliary loss could shape the world model.

### 9.4 Reader sharing and target-transform matrix

The decoder was varied between shared/member-specific and raw/log/normalized
raw RND targets.

| Decoder | Successful segments, mean +/- sd | Comment |
|---|---:|---|
| Shared, log | 54.3 +/- 32.7 | Stable selected configuration |
| Member, log | 38.7 +/- 43.7 | High seed variance |
| Shared, raw | 34.0 +/- 21.9 | Weaker than shared log |
| Member, raw | 33.3 +/- 29.5 | One seed did not reach goal |
| Shared, normalized raw | 43.0 +/- 38.7 | High variance |
| Member, normalized raw | 74.0 +/- 64.7 | Highest mean but extremely variable and unequal phase counts |

There was no clean independent main effect of sharing or target transform.
Shared log was retained because it was stable and conceptually simple, not
because every alternative was inferior.

### 9.5 Ordinary reward-head controls

The ordinary intrinsic reward head was then modified without creating an
auxiliary branch:

| Ordinary reward-head condition | Successful segments, mean +/- sd |
|---|---:|
| Double reward-loss weight | 20.0 +/- 27.1 |
| Log target | 59.7 +/- 33.1 |
| Shared head, raw target | 10.0 +/- 8.2 |
| Shared head, log target | 22.7 +/- 19.6 |

Log targets alone could improve the ordinary head, but a shared log operational
reward head did not reproduce the auxiliary result. Replacing the member reward
heads changes both the learning objective and the reward model used inside
MCTS, so it is not equivalent to adding a separate auxiliary constraint.

### 9.6 Pre-registered confirmatory ablation

The final MountainCar family fixed initialization and predeclared six causal
contrasts. All conditions retained private raw-RND reward heads in MCTS unless
the condition explicitly removed their loss.

| Condition | What changed | Successful segments, mean +/- sd |
|---|---|---:|
| Base | No auxiliary branch | 13.3 +/- 10.3 |
| Detached train-only | Auxiliary learns, no gradient to world model | 13.3 +/- 10.3 |
| Joint train-only | Auxiliary gradients enter encoder/dynamics; prediction discarded in MCTS | 54.3 +/- 32.7 |
| Joint tree | Same branch also supplies tree edge reward | 60.7 +/- 29.7 |
| Joint tree, no private reward loss | Remove private reward training | 28.0 +/- 28.2 |
| Ordinary exact train-only | Auxiliary copy of ordinary architecture/settings | 45.3 +/- 33.0 |
| Ordinary matched train-only | Ordinary-size decoder with matched log target/optimizer | 76.0 +/- 44.9 |

The strongest defensible conclusions are:

1. `base` and `detached_train_only` were behaviorally identical. Merely fitting
   another decoder did nothing.
2. Allowing the loss into encoder/dynamics produced a large change. The causal
   mechanism is auxiliary representation/world-model training.
3. Deploying the auxiliary prediction as the MCTS reward was not necessary in
   MountainCar; train-only retained most of the benefit.
4. A specialized query reader was not necessary. An ordinary decoder with a
   well-weighted, current log-RND objective could provide the same kind of
   useful gradient.
5. Private operational reward-head training remained useful when the auxiliary
   prediction was used in the tree.

What this family **does not** establish is that two identical losses are always
better than one. The auxiliary path had a distinct optimization role and did not
replace the planning reward head. The result calls for a cleaner eventual
architecture, but it is not evidence for a uniquely necessary query module.

### 9.7 Prospective novelty before direct visitation

Trajectory and tree diagnostics showed action preferences toward the distant
MountainCar region before replay contained direct RND rewards from that region.
This is genuine extrapolation, but not evidence that the model reconstructed the
entire current RND field. MountainCar supplies a much simpler regularity: states
farther along the current velocity/momentum direction tend to be outside recent
support. Depth-wise RND supervision can make recurrent latents expose that
regularity, after which a reader or reward head can extrapolate it.

The context interventions show what did **not** cause the extrapolation: the
agent was mostly not consulting root-specific queried RND values. In a more
branched environment, knowing current local novelty should be more useful, which
was one reason for introducing Branching Rooms.

## 10. Stage VI: Branching Rooms

### 10.1 Why a new environment was needed

MountainCar is a poor final test of local novelty-landscape reasoning. Much of
its exploration reduces to maintaining momentum left or right. A model can
generalize this regularity without querying a nuanced local novelty field.

Branching Rooms was created as a more demanding controlled environment:

- `31 x 31` logical grid;
- `693` reachable cells;
- `2772` position-orientation poses;
- `16` rooms connected by corridors;
- `5` designated dead ends;
- three actions: turn left, turn right, move forward;
- partial `84 x 84` RGB observations; and
- 1200 environment steps per episode, collected as 200-step replay segments.

The map and scripted walkthrough are available in
`video/branching_rooms_visuals/`. The environment preserves the interpretability
of a grid while requiring orientation-aware navigation and choices among
branches.

![Branching Rooms map](../../video/branching_rooms_visuals/branching_rooms_map.png)

### 10.2 Mechanical failures fixed before valid runs

Two initial cluster jobs failed before producing valid comparisons:

- a tuple was passed where the MLP builder expected a list; and
- a terminal replay segment exposed an out-of-bounds `to_play` bootstrap index.

Both were corrected before the reported pilot and replicated runs. Those failed
jobs are not treated as experiments.

### 10.3 Pilot and three-phase replication

The one-phase pilot produced identical initial coverage for baseline and
auxiliary: 207 positions, 738 poses, 8 rooms, and no dead ends. This was expected
because phase 0 occurs before the learned auxiliary representation can alter
behavior.

After three phases over three seeds:

| Condition | Positions | Poses | Rooms | Dead ends |
|---|---:|---:|---:|---:|
| Baseline | 392.7 | 1452.0 | 11.00 | 1.00 |
| Shared log auxiliary | 383.3 | 1432.3 | 11.67 | 2.33 |

Raw coverage was essentially unchanged, while the auxiliary agent reached more
rooms and dead ends. This suggested that it explored farther branches rather
than simply covering more nearby cells, but three phases were too short for a
stable conclusion.

### 10.4 Target-construction mistake: finite horizon without bootstrap

A planned "20-step TD" comparison was accidentally configured as 20 rewards
followed by zero continuation value. That is a finite-horizon return, not a
20-step bootstrapped TD target. The runs completed, but they do not answer the
intended question and are scientifically secondary.

Their close results are retained only as a record:

| Condition | Position coverage | Pose coverage | Rooms | Dead ends |
|---|---:|---:|---:|---:|
| Baseline | 0.567 | 0.525 | 11.00 | 1.67 |
| Auxiliary | 0.573 | 0.532 | 11.67 | 1.67 |

This mistake led to explicit configuration and logging of both the TD horizon
and the bootstrap flag in subsequent runs.

### 10.5 Correct long bootstrapped comparison

The corrected experiment used 20-step intrinsic targets with a target-network
value bootstrap and ran for six collection phases.

| Condition | Position coverage | Pose coverage | Rooms | Dead ends |
|---|---:|---:|---:|---:|
| Baseline, 3 seeds | 0.725 | 0.695 | 13.00 | 2.00 |
| Shared log auxiliary, 3 seeds | 0.747 | 0.704 | 13.67 | 3.00 |

Per-seed final room/dead-end results were:

```text
baseline: 13/2, 13/2, 13/2
auxiliary: 12/2, 15/4, 14/3
```

The mean raw-coverage gain was modest: about 2.2 percentage points for positions
and 0.9 points for poses. The larger difference was topological: two auxiliary
seeds reached branches missed by every baseline seed, while one auxiliary seed
underperformed. This supports the hypothesis that the auxiliary signal can
sustain discovery of distant branches, but the variance prevents a definitive
performance claim.

The phrase "sustain exploration longer" must be used carefully. New-position
counts remained high late in auxiliary seeds 1 and 2, but not seed 0. It is a
seed-dependent pattern, not a universal property established by these runs.

### 10.6 Behavior and visualization diagnostics

The environment now records phase-level and cumulative cell/pose visitation.
This supports:

- per-phase heatmaps;
- cumulative coverage maps;
- baseline-versus-auxiliary comparisons; and
- inspection of revisitation versus frontier discovery.

Generated comparisons are stored in:

- `video/branching_rooms_long_bootstrap/`
- `video/branching_rooms_compact_query/`

These maps showed substantial revisitation. The agent does not move monotonically
through new cells; it repeatedly traverses corridors and already visited rooms,
with occasional frontier expansions. This is partly unavoidable because distant
branches require shared corridors, but some immediate backtracking is an
artifact of stale within-phase novelty.

## 11. Compact learned-state queries in Branching Rooms

### 11.1 Design

Querying arbitrary physical states is feasible in two-dimensional MountainCar
but not in high-dimensional observations. The Branching Rooms query model
therefore learned a compact image autoencoder:

```text
observation -> 32-dimensional query code -> reconstructed observation
```

For each root, a recurrent query policy emitted eight compact codes. Each code
was decoded to an image, evaluated by the current image RND model, and supplied
to the latent-conditioned reader. A context-free condition retained the same
auxiliary objective while withholding query context from the reader.

Before this design, a set-valued inverse of the MuZero encoder was considered:
given a MuZero latent, model the distribution of real states compatible with it,
rather than reconstructing one state. With gradients stopped at the MuZero
latent, such an inverse would preserve a many-to-one representation. The idea
was not implemented because the immediate need was the opposite operation: let
the query policy choose candidates in a compact space and map them to concrete
observations on which exact current RND can be evaluated. A separately learned
query-space autoencoder provided that operation without requiring the MuZero
latent itself to be invertible.

### 11.2 Coverage results

| Condition | Seeds/phases | Position coverage | Pose coverage | Rooms | Dead ends |
|---|---|---:|---:|---:|---:|
| Compact query with context | 3 seeds, 6 phases | 0.760 | 0.733 | 14.00 | 3.33 |
| Compact query without context | 2 full seeds + 1 five-phase seed | 0.751 | 0.707 | 13.67 | 2.67 |

The context condition reached 15 rooms and 4 dead ends in two of three seeds.
The no-context comparison is imperfect because seed 2 timed out during phase 5;
its last complete coverage snapshot is phase 4. Unlike MountainCar, Branching
Rooms therefore gives some evidence that context can help topological reach,
but the evidence is modest rather than conclusive.

### 11.3 Prediction quality

At the final learner report, the compact query reader's replay-target
correlations were:

```text
with context:    0.765, 0.794, 0.778
without context: 0.584, 0.917, 0.751 (third seed ended early)
```

The shared auxiliary decoder's RND correlation generally decreased with
recurrent depth and varied strongly by seed. For example, context seed 1 fell
from `0.400` at depth 1 to `0.192` at depth 5, while context seed 2 fell from
`0.917` to `0.641`. Thus the models can predict meaningful RND variation on
replay-supported imagined states, but prediction is neither uniformly accurate
nor specifically validated on never-visited states.

Absolute MAE becomes numerically tiny late in training because raw image-RND
errors are themselves around `1e-6`. Correlation and ranking are more informative
than rounded raw MAE in this regime.

### 11.4 Latent-hole diagnosis

The query generator did not learn root-local valid-state probes. At phase 5:

- the same query slot changed by only `0.02-0.10` in code distance across roots;
- different slots within a root were separated by about `0.24` without context
  and up to `1.61-1.64` with context; and
- query codes were roughly `2.3-4.9` away from the nearest sampled real-root
  code.

The network had learned approximately the same global set of eight synthetic
probes for every root. The decoder produced images on which RND could be
evaluated, but the codes were not constrained to the empirical valid-state
manifold. This is exactly the "latent holes" concern: good task loss does not
guarantee that a decoded query corresponds to a physically reachable state.
The diagnostic used eight sampled real roots per phase rather than an exhaustive
code-space bank, so it is strong evidence of this failure mode rather than a
formal proof that every query lies off-manifold.

Consequently, the current coverage result cannot be attributed cleanly to
meaningful local novelty-field sampling. As in MountainCar, some or all of the
benefit may come from the auxiliary gradient imposed on imagined latents.

## 12. Current stage: within-episode episodic novelty

### 12.1 Remaining failure mode

RND is refreshed between phases, not after each action. In a room, the agent can
move from `A` to `B` and immediately back to `A`. If `A` was novel at the start
of the phase, the phase-frozen RND model still assigns it high reward on return.
The tree can therefore value backtracking even though `A` was already visited in
the current episode.

Updating the full RND predictor after every action was deliberately avoided. It
would be expensive, make planning targets nonstationary, and complicate replay.
Instead, the current experiment multiplies predicted RND by a fast episodic
familiarity factor.

### 12.2 Memory semantics

Each environment has an episode-local memory containing the direct latent of
every real state actually encountered. During MCTS, an imagined candidate is
compared with:

- all direct latent keys in the real episode memory; and
- all proper-prefix imagined states already on the current search path.

The smallest distance determines the reward factor. The memory also estimates a
typical one-step latent distance from consecutive real states and uses that only
as a scale normalization.

For `A -> B -> A`, the final imagined `A` is compared directly with the first
stored `A`. Its minimum distance is therefore near zero and its reward is reduced
to about 5% of the predicted RND. The large `B -> A` transition distance is
irrelevant; the method is not checking only the previous state.

The operational rule is:

```text
final imagined edge reward
    = predicted phase-current RND
    * episodic novelty factor in [0.05, 1.0]
```

The intrinsic leaf value is currently left unchanged. This isolates the effect
to immediate imagined-path rewards.

### 12.3 Completed conditions

The following Branching Rooms conditions completed with three valid seeds:

| Condition | Latent comparison space | Memory used? | Purpose |
|---|---|---:|---|
| Compact query projected-memory | normalized projection/prediction space | yes | Use the standard SSL comparison space |
| Compact query raw-SSL | raw latent with raw consistency loss | no | Control for changing SSL alone |
| Compact query raw-memory | raw latent with raw consistency loss | yes | Test direct raw comparability for episodic memory |

The projected variant compares direct projection keys with recurrent
projection-plus-prediction keys. The raw variant explicitly trains recurrent raw
latents to match direct raw latents. The no-memory raw-SSL control is necessary
because changing the SSL objective can itself alter exploration.

### 12.4 Results and interpretation

The projected-memory gate was too weak. Its mean one-step reward factor was
`0.997`, with no meaningful difference between selected and unselected root
actions. The mean factor was still `0.883` at depth five. Against the existing
projected compact-query control, final mean position coverage changed from
`0.761` to `0.777` and pose coverage from `0.733` to `0.734`. The position
change was positive in one seed, zero in one, and slightly negative in one.
Therefore the projected comparison space did not reliably identify recurrent
imagined revisits as the same states stored from real interaction.

Raw memory was active but harmful. Mean reward factors decreased with search
depth: `0.912`, `0.616`, `0.493`, `0.403`, and `0.357` from depths one through
five. At depth five, about 80% of imagined edges were suppressed below a factor
of `0.5`. Yet selected and unselected one-step factors remained almost equal.
Relative to raw SSL without memory, final mean coverage changed as follows:

| Metric | Raw SSL, no memory | Raw memory | Paired mean change |
|---|---:|---:|---:|
| Positions | 0.808 | 0.681 | -0.127 |
| Poses | 0.775 | 0.659 | -0.116 |
| Rooms | 0.896 | 0.771 | -0.125 |
| Dead-end rooms | 0.667 | 0.467 | -0.200 |

Position coverage decreased in all three paired seeds. It also nearly plateaued
after phase three under raw memory, while the no-memory control continued to
expand. The control establishes that raw SSL itself was not responsible for
the degradation; the reward gate was.

These diagnostics do not map every imagined edge back to an exact physical
state, so they cannot directly report the true-positive rate for `A -> B -> A`.
They do establish the operational failure: projected memory was too weak to
implement the intended correction, and raw memory suppressed a broad set of
deep paths rather than a demonstrably precise set of revisits.

## 13. Consolidated findings

### 13.1 Findings supported by the evidence

1. **Ensemble disagreement is not automatically epistemic novelty.** In the
   original MountainCar method, Q standard deviation was dominated by leaf/value
   disagreement and was sensitive to member offsets and search depth.
2. **Naive dynamics disagreement was insufficient.** Large latent disagreement
   did not reliably produce directed exploration or goal discovery.
3. **RND on correctly normalized real states is a meaningful novelty target.**
   It correlates with state density and distance to previous experience, but its
   seen/unseen contrast is smooth and often modest.
4. **A fully intrinsic MuZero training loop is viable.** MountainCar goal regions
   were discovered with no extrinsic reward or value objective controlling
   collection or training.
5. **Explicit current-RND prediction from recurrent latents improves
   MountainCar exploration when it trains the world model.** Detaching the world
   model removes the effect.
6. **MountainCar does not require query context for that improvement.** Query
   interventions and no-context controls show that the auxiliary representation
   objective, not queried content, is the principal mechanism there.
7. **The auxiliary mechanism transfers partially to Branching Rooms.** Average
   raw coverage changes little, but room and dead-end discovery improve in two
   of three long-run seeds.
8. **Compact query context may matter in a branched pixel environment, but the
   learned queries are not yet valid local-state queries.** Their codes occupy
   latent holes and are nearly root-independent.
9. **Phase-frozen RND creates a real episodic revisitation problem, but latent
   proximity did not solve it safely.** Projected memory was mostly inactive;
   raw memory strongly reduced deep-path rewards and exploration.

### 13.2 Claims not supported yet

The current evidence does **not** justify claiming that:

- the ensemble's raw Q standard deviation is a calibrated epistemic uncertainty
  estimate;
- encoder or value random priors produce reliable unseen-state uncertainty;
- the query policy has learned where to measure a local RND landscape;
- the auxiliary decoder is uniquely better than an ordinary auxiliary head;
- Branching Rooms improvement is statistically decisive with three seeds;
- the reader accurately predicts novelty for truly unsupported imagined states;
  or
- latent episodic distance is a valid physical revisit detector.

### 13.3 Strongest thesis-level contribution so far

The strongest mechanistic result is not "queries solve exploration." It is:

> Training MuZero's encoder and recurrent dynamics to expose a nonstationary,
> current novelty target at every unrolled depth can materially improve pure
> intrinsic exploration, even when that auxiliary prediction is discarded by
> the planner.

This result is supported by the replicated detached-versus-joint ablation and by
the weaker but directionally consistent Branching Rooms topological-coverage
result. It shifts the research focus from manufacturing disagreement to making
the world model represent information needed to predict current novelty.

## 14. Threats to validity

### 14.1 Seed count and variance

Most early experiments used one seed and should be treated as mechanism probes.
Even the later three-seed results have high variance. MountainCar successful
segment counts sometimes have a standard deviation close to the mean, and one
Branching auxiliary seed underperformed baseline.

### 14.2 Pure-exploration evaluation

Extrinsic return is not an appropriate sole metric after the goal is found. The
agent should leave familiar goal trajectories. Coverage, frontier discovery,
room reach, dead ends, and novelty of newly collected states are the primary
metrics. Goal-reaching remains useful only as evidence that a distant region was
discovered.

### 14.3 Current-RND drift

RND targets change after each collection phase. Reward/value metrics from
different phases are not directly comparable unless evaluated against the same
snapshot. This motivated phase-frozen target snapshots and post-update refresh.

### 14.4 Intrinsic value-target choice

The replicated MountainCar query and auxiliary families inherited full Monte
Carlo intrinsic-return targets from the full-intrinsic prototype. This avoids a
bootstrap feedback loop but produces high-variance, policy-dependent targets and
was not considered a satisfactory final design. The intended follow-up is a
bootstrapped-target replication. Branching Rooms therefore uses the explicit
20-step bootstrapped mode in its valid long comparisons. The accidentally
non-bootstrapped finite-horizon family is kept separate.

### 14.5 Representation drift

Any method that stores or queries learned latent states must account for the
encoder changing during training. The episodic memory avoids cross-update stale
keys by retaining observations and re-encoding where required. The compact query
autoencoder still has no guarantee that arbitrary emitted codes remain on a
valid observation manifold.

### 14.6 Operational failures

DelftBlue migrated its operating system, software stack, and scheduler during
the project. Later runs used the 2026 GPU stack and an Apptainer environment.
Intermittent BeeGFS remote-I/O failures caused failed jobs and incomplete rsync
copies. Results were only analyzed after expected files were copied and checked;
failed or partial jobs are marked explicitly in the registry.

## 15. Recommended next decisions

1. Do not continue tuning the current latent-distance gate from aggregate
   coverage alone. First label imagined comparisons by physical identity and
   measure exact-revisit, nearby-new-pose, and genuinely-new-state distance
   distributions.
2. Compare future methods with trajectory-aware metrics, not just cumulative coverage:
   revisit fraction, time since last novel cell, new cells per 1000 steps,
   corridor retraversal, and first phase reaching each room/dead end.
3. Treat compact-query validity as a separate research problem. Before claiming
   local landscape querying, constrain or retrieve queries from a valid-state
   model and verify root dependence and physical reachability.
4. If the auxiliary branch remains beneficial, simplify it deliberately. The
   confirmatory ablation suggests that a compact current-RND auxiliary decoder
   may retain the effect without pretending to be a query mechanism.
5. Increase seed count only after a replacement episodic mechanism passes the
   geometric and behavioral checks above.

## 16. Reproducibility and artifact policy

Every retained run should contain, where available:

- `total_config.py` and `formatted_total_config.py`;
- learner, collector, evaluator, and serial logs;
- collection and coverage diagnostics;
- model-specific diagnostic arrays; and
- the final required checkpoint when checkpoint-level analysis is planned.

The exact run-family index is in [experiment_registry.md](experiment_registry.md).
Cluster output under `/scratch/$USER/lightzero_runs` should be deleted after the
local copy is verified, because the shared filesystem has repeatedly experienced
capacity-related remote-I/O failures.
