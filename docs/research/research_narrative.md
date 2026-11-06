# Research Narrative: From Ensemble Disagreement to Novelty-Aware World Models

**Purpose:** meeting-oriented account for explaining the project to a supervisor  
**Evidence cutoff:** 2026-09-04, with the subsequently observed BeeGFS failures
treated as operational failures rather than experimental outcomes  
**Detailed evidence:** [experimental history](experimental_history.md)  
**Exact run provenance:** [experiment registry](experiment_registry.md)

## Roadmap of the ideas

1. **Ensemble all MuZero components as alternative learned worlds.**
   **Why:** coherent full-model hypotheses should extrapolate differently
   outside data. **Result:** behavior improved initially, but remained unstable,
   while Qstd grew and persisted even in densely visited regions instead of
   contracting with familiarity.
2. **Make disagreement the exploration objective.** **Why:** removing external
   reward might expose the ensemble's epistemic signal. **Result:** some
   variants explored well, but neither Q nor latent disagreement acquired
   reliable novelty semantics.
3. **Stabilize disagreement with calibration and replay.** **Why:** member
   offsets, unequal search depths, and rare successful trajectories distorted
   learning. **Result:** MountainCar control improved, but external-value elite
   replay was not a clean pure-exploration mechanism.
4. **Create unsupported-state uncertainty with representation and value
   priors.** **Why:** consistency can transport a signal but cannot create
   uncertainty; frozen member-specific priors were intended to supply it.
   **Result:** direct encoder novelty appeared, but it did not survive reliably
   through recurrent planning or produce locality-calibrated value
   disagreement.
5. **Define real-state novelty explicitly.** **Why:** density and RND provide a
   target with independently testable semantics. **Result:** correctly
   normalized, post-update RND became the reference novelty signal.
6. **Train a fully intrinsic MuZero agent.** **Why:** replay must come from the
   same intrinsic policy being learned. **Result:** the agent discovered the
   distant MountainCar region without external reward directing collection.
7. **Actively query the novelty landscape.** **Why:** current RND is
   nonstationary and replay does not label unseen imagined futures. **Result:**
   MountainCar improved, but query-context interventions showed that the
   measurements themselves were not causing the gain.
8. **Train imagined latents to expose current novelty.** **Why:** the query
   model also imposed an additional RND-prediction gradient on the world model.
   **Result:** joint training helped, detachment removed the effect, and using
   the auxiliary prediction in MCTS was unnecessary.
9. **Test transfer in Branching Rooms.** **Why:** MountainCar momentum may make
   novelty extrapolation artificially easy. **Result:** raw coverage remained
   similar, while the auxiliary agent showed tentative improvement in distant
   room and dead-end discovery.
10. **Query through a compact observation space.** **Why:** direct state queries
    do not scale to images. **Result:** context looked modestly useful, but the
    query policy exploited off-manifold autoencoder holes and emitted nearly
    root-independent probes.
11. **Suppress same-episode revisits with latent memory.** **Why:** phase-frozen
    RND still rewards `A -> B -> A` within one episode. **Status:** implemented,
    but complete scientific results were still pending after filesystem
    failures interrupted the scheduled runs.

## The short version

The project began with a simple idea: an ensemble of MuZero models should
disagree when planning into parts of the environment that it does not
understand, and MCTS could use that disagreement to explore. The ensemble was
implemented successfully, and optimistic disagreement bonuses sometimes helped
MountainCar substantially. The fact that most Q disagreement came from leaf
values was expected because MountainCar's immediate reward is almost always
the same `-1`. The actual failure was that leaf disagreement did not reliably
fall as states became familiar. It grew while the agent learned, remained large
after extensive visitation, and sometimes ranked dense regions as more
uncertain than unseen ones. Persistent member-wide value offsets and
depth-dependent backup effects explained part of this behavior. A large Qstd
therefore did not reliably mean that the imagined physical state was novel.

We then tried to make uncertainty more principled through dynamics
disagreement, calibrated fixed-horizon planning, representation regularization,
and frozen random priors. These experiments clarified several failure modes but
did not produce a reliable novelty estimate. In particular, a prior could make
a directly encoded novel state geometrically unusual without making the
corresponding recurrently imagined state unusual. Projected self-supervision
could hide a mismatch between raw encoder and dynamics latents, while raw
latent matching fixed the mismatch without making ensemble disagreement
novelty-calibrated. Random priors also introduced avoidable escape routes: the
encoder could overwhelm a prior by increasing its norm, learn broad
compensation, or preserve uncertainty over histories rather than physical
states.

This led to a change in strategy. Instead of hoping that disagreement would
acquire the desired meaning, we introduced an explicit and measurable novelty
target: Random Network Distillation (RND) on real states. After correcting a
serious input-normalization problem, RND behaved as intended. It was higher away
from familiar support and lower in dense regions, although smoothly rather than
as a binary seen/unseen indicator. We then built a fully intrinsic MuZero loop
in which collection, policy, reward, value, encoder, and dynamics were trained
from intrinsic experience alone. That agent could discover the distant
MountainCar goal without any extrinsic objective steering it.

The remaining difficulty was planning before visitation. A reward head learns
RND targets from replay, but the interesting question is whether an imagined
future can be recognized as novel before the agent has collected a reward
there. We therefore built a query-and-reader system intended to sample the
current RND landscape and use those samples to predict RND at recurrent MuZero
latents. The system improved MountainCar exploration dramatically, but the
critical ablations overturned our initial explanation. Removing or corrupting
the query context barely changed decisions. Detaching the world model removed
the improvement. Keeping the auxiliary RND-prediction loss while discarding its
prediction during MCTS retained the improvement. The main mechanism was
therefore not active querying; it was the additional current-RND gradient into
the encoder and dynamics across the recurrent unroll.

That is the strongest result of the project. A replicated confirmatory family
showed that a detached auxiliary decoder behaved like baseline, whereas the
same loss, when allowed to train the world model, produced much more frequent and
earlier discovery of the distant region. A specialized query reader was not
necessary: a suitably trained ordinary auxiliary decoder could provide a
similar representation-shaping signal. The result is scientifically useful,
but it also means the original query architecture was more complicated than
the mechanism required in MountainCar.

MountainCar is too simple to establish that the model learned a general novelty
landscape. Momentum gives it a strong regularity: continuing in the current
direction often reaches less familiar states. We therefore introduced
Branching Rooms, a pixel-based environment with orientation, corridors,
branches, and dead ends. There, the auxiliary objective produced only a modest
change in total cell coverage but more often reached topologically distant
rooms and dead ends. This is promising transfer evidence, not a decisive
performance result.

We also revisited querying in a compact learned observation space. Query
context appeared somewhat more useful in Branching Rooms than in MountainCar,
but geometric diagnostics showed that the query network emitted almost the
same global probes for every root and placed them far from codes of real
observations. It had learned to exploit holes in the autoencoder latent space,
not to ask meaningful local questions about reachable states. Consequently,
we still cannot claim a successful general state-query mechanism.

The project therefore moved from the original claim,

> ensemble disagreement can directly guide exploration,

to the narrower and better-supported claim,

> explicitly training MuZero's recurrent representation to expose current
> novelty can improve planning-based pure exploration.

The next unresolved issue is within-episode familiarity. RND is frozen during a
collection phase, so a state that was novel at the start of an episode can
remain rewarding when the agent immediately revisits it. The current episodic
latent-memory experiments were designed to suppress those revisits without
updating RND after every action. They were pending at the evidence cutoff, and
several later attempts were interrupted by confirmed BeeGFS remote-I/O failures
rather than model failures.

## Research question

The broad goal was not merely to solve MountainCar or maximize sparse external
reward. The goal was to build a MuZero agent that could use its learned model to
direct exploration toward consequences it did not yet understand.

That requirement is stronger than ordinary novelty detection. A real-state RND
model can tell the agent that the state it has just reached is novel. A planning
agent should ideally recognize, while still at the root of an MCTS tree, that
one action sequence leads toward familiar consequences and another leads toward
unsupported ones. The signal must therefore survive the transition from real
observations to encoded latents and then through several recurrent dynamics
steps.

The intended signal also has to generalize at the right scale. Nearby physical
states should share information; otherwise a continuous environment makes
every exact state appear new. At the same time, learning in one region should
not erase uncertainty everywhere. Finally, two different action histories that
reach the same physical state should eventually be treated as the same state,
not as permanently different sources of novelty.

Those requirements explain why many methods that produced interesting behavior
were nevertheless rejected. Reaching the MountainCar goal was evidence that an
exploration signal could be behaviorally effective. It was not enough to show
that the signal meant what we wanted it to mean.

## Idea 1: Ensemble all MuZero components as alternative learned worlds

**What went wrong.** The optimistic ensemble produced an encouraging early
behavioral effect, but neither behavior nor uncertainty became stable as
experience accumulated.

- The main MountainCar run first produced successful collection segments in
  phase 5, lost them entirely in phases 6 and 7, reached a `100%` segment
  success rate in phases 13 and 14, and then fell to `20%` in phase 15 before
  recovering again. Later collection remained variable despite good evaluation
  performance. Finding the goal once did not turn into monotonic improvement.
- Qstd did not contract as the explored support grew. Its ten-phase mean rose
  from `2.80` in phases 0-9 to `5.60` in phases 10-19 and `9.06` in phases
  20-29. It later declined, but remained around `6.02` in phases 50-55 after
  more than one hundred thousand environment steps.
- A fixed-horizon follow-up made the mismatch with novelty explicit. At the
  checkpoint, raw imagined Qstd averaged `17.63` for densely visited endpoints
  but only `3.34` for unseen endpoints. Raw directly evaluated Qstd showed the
  same inversion, `22.12` in dense regions versus `0.53` in unseen regions.
  Raw imagined Qstd was positively correlated with visitation density
  (Spearman about `0.66`), the opposite of the desired epistemic-novelty
  interpretation.

These were the important observations. The fact that most disagreement came
from the leaf value was not, by itself, a failure.

**Why leaf-value dominance was expected.** MountainCar gives reward `-1` on
almost every nonterminal transition. Every member therefore receives nearly
the same immediate reward labels. For a depth-$d$ rollout,

$$
Q_k
  = \underbrace{\sum_{j=0}^{d-1}\gamma^j r_{j+1}^{(k)}}_{
      \text{almost identical across members}}
    + \gamma^d v_d^{(k)},
$$

so it is natural that

$$
\sigma_Q \approx \gamma^d\operatorname{std}_k\!\left(v_d^{(k)}\right).
$$

The measurements reflected this. Across the main run, predicted-reward
standard deviation was only about `0.06-0.15`, while leaf-value standard
deviation was about `2.77-8.96`. The problem was not *where* Qstd came from; it
was that the leaf disagreement did not behave like reducible uncertainty about
novel consequences.

**Original hypothesis.** Every MuZero component was independently initialized
and trained for every member. Each member was intended to represent one
coherent learned-world hypothesis. Disagreement between their backed-up Q
values should then be low where data constrain all worlds and high where their
predictions require unsupported extrapolation.

**What the complete ensemble meant.** This was not an ensemble of only value
heads. Member $k$ had its own:

- representation function $h_k$, mapping observations to latents;
- recurrent dynamics and reward function $g_k$;
- policy and value prediction function $f_k$; and
- corresponding parameters and training path.

This fully private arrangement was the starting point, not the only sharing
boundary tested in the project. Later experiments used several selected
factorizations:

1. **Complete ensemble:** representation, dynamics/reward, and policy/value
   prediction were private to each member.
2. **Dynamics-isolation ensemble:** the representation network, policy/value
   prediction network, and self-supervised projection modules were shared,
   while each member retained its own dynamics/reward network. This made the
   members alternative transition models operating in a common latent and
   evaluation system.
3. **Shared operational intrinsic-reward head:** the core member models
   remained private, but the head predicting RND reward for MCTS was tied
   across members.
4. **Shared auxiliary RND reader:** the core member models and operational
   reward heads remained private, while one additional decoder was applied to
   every member's latent. We also tested a separate decoder per member.

These were targeted causal comparisons, not a complete factorial over every
possible shared/private component. In particular, a run named `shared_tree`
did not share neural-network parameters: it placed all member predictions in
one MCTS tree so their statistics could be aggregated at the same action
nodes.

For an action sequence $a_0,\ldots,a_{d-1}$, member $k$ produced its own
rollout:

$$
z_0^{(k)} = h_k(o_t),
$$

$$
\left(r_{j+1}^{(k)}, z_{j+1}^{(k)}\right)
    = g_k\left(z_j^{(k)}, a_j\right),
\qquad
\left(p_j^{(k)}, v_j^{(k)}\right) = f_k\left(z_j^{(k)}\right).
$$

The shared MCTS tree stored one node per action history, but each node contained
one latent and one Q statistic per member. Member $k$'s latent was advanced only
by $g_k$ and evaluated only by $f_k$. We never combined, for example, member
1's encoder with member 2's transition and member 3's value head.

The exploration signal was

$$
\bar Q = \frac{1}{K}\sum_{k=1}^{K} Q_k,
\qquad
\sigma_Q = \sqrt{\frac{1}{K}\sum_{k=1}^{K}(Q_k-\bar Q)^2},
$$

with schematic optimistic score

$$
Q_{\mathrm{opt}} = \bar Q + \beta\sigma_Q.
$$

If the members agreed on familiar experience but extrapolated differently
outside it, $\sigma_Q$ would resemble uncertainty over possible learned
worlds. This was Bayesian in motivation only: the members were not posterior
samples, and nothing in their losses guaranteed calibrated or contracting
ensemble variance.

**Issue 1: member-wide value offsets.** Members could learn similarly shaped
value functions at different absolute levels. Write

$$
V_k(s) = \widetilde V_k(s) + b_k,
$$

where $b_k$ is a nearly state-independent member bias. Then

$$
Q_{k,d}
  = R_{k,d} + \gamma^d\widetilde V_k(s_d) + \gamma^d b_k.
$$

Even if all $\widetilde V_k$ agree on a familiar state, Qstd contains the term
generated by the offsets $b_k$. This is not uncertainty about that state; it is
a calibration mismatch between members. At the fixed-horizon checkpoint, the
three members' mean values on the same replay sample were approximately
`-127.0`, `-96.9`, and `-95.0`. The corresponding centered offsets were
`-20.7`, `9.4`, and `11.3`, a large disagreement baseline before asking which
action led somewhere novel.

Concretely, suppose all three members have learned that action `right` is worth
ten more return units than action `left`, but one member predicts every state
about thirty units lower than the other two. Their *relative* control knowledge
can agree while their numerical Q values still have a large standard
deviation at every familiar state. More visits need not remove this variance:
the bootstrapped MuZero objective can preserve several differently calibrated
value functions that induce similar policies. An optimistic bonus then treats
this global choice of numerical origin as if it were local uncertainty about
an action's consequences.

The replay calibration estimated

$$
b_k
  = \mathbb E_{s\sim\mathcal D}[V_k(s)]
    - \frac{1}{K}\sum_{m=1}^{K}
      \mathbb E_{s\sim\mathcal D}[V_m(s)]
$$

and subtracted $\gamma^d b_k$ from the member's backed-up leaf contribution.
In other words, calibration aligned each member's average value level on a
common replay distribution before comparing them. This corrected a genuine
nuisance component and changed the global density correlation of imagined
Qstd from positive to negative. It did not force agreement state by state, did
not calibrate recurrently imagined latents outside replay support, and did not
make Qstd a reliable action-ranking signal: high calibrated-Qstd sequences
still did not consistently select unseen endpoints.

**Issue 2: search-depth effects.** Different root actions were backed up from
different mixtures of leaf depths. A member offset contributes
$\gamma^d b_k$, the amount of accumulated `-1` reward changes with $d$, and
recurrent model error compounds with each transition. Therefore Qstd measured
at depth 2 is not directly comparable with Qstd measured at depth 10. In a
later depth diagnostic, mean raw disagreement in one phase increased from
about `7.9` at depth 1 to `14.3` at depth 5 and `26.0` at depth 11. An action
could receive a larger uncertainty bonus because search reached a different
depth, not because its physical endpoint was less familiar.

Fixed-horizon sequences, depth-matched comparisons, and depth-wise
normalization were introduced to remove this confound. They made action scores
more comparable, but could not supply novelty semantics to the residual
disagreement.

**Issue 3: independent training did not imply convergence on familiar
states.** The members had no loss requiring their value functions to agree.
Independent initialization was intended to preserve epistemic diversity, but
the same freedom also allowed persistent global biases, different propagated
termination estimates, and different approximations of the bootstrapped value
target. Goal discovery could actually increase disagreement temporarily as
members learned and propagated the shorter successful return at different
rates. Frequent visitation reduced individual prediction error, but did not
guarantee that cross-member variance would vanish.

This gives a plausible explanation for the observed rise and oscillation in
Qstd, but we did not isolate one component as the sole cause of the behavioral
regressions. The supported conclusion is narrower: the ensemble was
mechanically coherent and sometimes behaviorally useful, yet its raw Q
variance was not a calibrated measure of state novelty and did not reliably
decay with familiarity.

## Idea 2: Make disagreement itself the intrinsic objective

**Hypothesis.** If extrinsic reward was obscuring the signal, directly rewarding
model disagreement should make the agent seek unsupported consequences.

**What we tried.** We tested pure intrinsic trees, disagreement between
member-predicted next latents, Qstd as an immediate learned intrinsic reward,
fixed action-sequence horizons, depth-matched comparisons, depth normalization,
value-offset calibration, and bounded optimistic member envelopes. We also
tested nearest-history episodic novelty and an NGU-style learned novelty model.

The two simplest candidate bonuses were return disagreement,

$$
r_t^{Q\text{-disc}} = \sigma_Q(s_t,a_t),
$$

and disagreement between member-predicted next latents,

$$
r_t^{z\text{-disc}}
  = \frac{1}{K}\sum_{k=1}^{K}
      \left\|z_{t+1}^{(k)}-\bar z_{t+1}\right\|_2^2,
\qquad
\bar z_{t+1}=\frac{1}{K}\sum_{k=1}^{K}z_{t+1}^{(k)}.
$$

**Experiment: isolate disagreement to the dynamics model.** For the complete
ensemble, the second expression is conceptually fragile:
independently trained encoders need not assign the same meaning to the same
latent coordinate. A large Euclidean variance can reflect coordinate choice
rather than different beliefs about the physical successor. We therefore also
tested a dynamics-isolation architecture with

$$
z_t = h(o_t),
\qquad
(r_{t+1}^{(k)},z_{t+1}^{(k)}) = g_k(z_t,a_t),
\qquad
(p_{t+1}^{(k)},v_{t+1}^{(k)}) = f(z_{t+1}^{(k)}),
$$

where $h$ and $f$ were shared and only $g_k$ was member-specific. The
self-supervised projection path was shared as well. Here, "shared $f$" meant
one set of prediction-network parameters applied separately to every member
latent:

$$
(p_{t+1}^{(1)},v_{t+1}^{(1)})=f(z_{t+1}^{(1)}),\quad
(p_{t+1}^{(2)},v_{t+1}^{(2)})=f(z_{t+1}^{(2)}),\quad
(p_{t+1}^{(3)},v_{t+1}^{(3)})=f(z_{t+1}^{(3)}).
$$

We did not choose one member latent and did not average the latents before
evaluating them. Each tree node retained all three $z^{(k)}$ and all three
$v^{(k)}$. In a normal or optimistic shared tree, each value would be backed up
through its corresponding member rewards, producing

$$
Q_k = r_1^{(k)} + \gamma r_2^{(k)} + \cdots + \gamma^d v_d^{(k)},
$$

after which search could use their mean and, for optimistic search, their
standard deviation. The member policy distributions were averaged to obtain
one action prior; the member latents, rewards, values, and Q backups were not
averaged before this point.

The dynamics-disagreement experiment itself used a separate
`intrinsic_tree_only` collector, however. Its action score did **not** use the
three ordinary $v^{(k)}$ predictions. The immediate intrinsic edge reward was
the average pairwise distance between the member successor latents, and the
leaf contribution came from a separately trained intrinsic-value model whose
member predictions were averaged into one scalar. The shared ordinary
prediction network still supplied the averaged policy prior and was trained on
latents from every $g_k$, encouraging all private dynamics models to emit
latents that the same downstream function could interpret.

Thus, this was not another complete-world ensemble. It deliberately ensembled
only the transition/reward model while holding the state coordinate system and
downstream policy/value interpretation common. The experiment asked a cleaner
question: do alternative dynamics models disagree specifically on unfamiliar
consequences?

**Result.** They did not do so reliably enough to guide exploration. The
representative frozen-prior dynamics-disagreement run found no goal across 34
phases. The no-prior and projected-SSL variants likewise produced no collection
success. Some runs moved the exploration frontier or improved evaluation, but
the disagreement itself did not consistently point toward novel physical
states. Sharing the coordinate system made the statistic better defined; it
did not give the statistic novelty semantics.

**What happened.** Some Q-disagreement variants found the MountainCar goal and
the learned-Qstd bridge experiment was behaviorally strong. Direct dynamics
disagreement was much less reliable. It could be large without producing a
useful direction, or small because different members organized unsupported
futures into similarly shaped latent regions. Early episodic/NGU variants
occasionally moved the frontier farther, but their imagined novelty predictions
were not reliable enough to guide planning.

Calibration and fixed horizons improved action comparability. They addressed
real measurement problems: two Q values backed up from different depths or
members with different global offsets should not be compared naively. They did
not address the semantic problem. A better normalized disagreement can still be
disagreement about value extrapolation rather than state novelty.

**Why we moved on.** These experiments showed that disagreement can be a useful
control signal without being the quantity the thesis intended to estimate.
Designing increasingly elaborate transformations of Qstd could improve
MountainCar while making the method harder to interpret.

## Idea 3: Preserve successful exploration with calibration and replay

**Hypothesis.** If useful exploratory trajectories were too rare to influence
learning, calibrated optimism, mixed search-value targets, and elite replay
could retain and propagate them.

**What happened.** These changes improved practical MountainCar control. Rare
goal-reaching trajectories were learned rather than immediately forgotten, and
some configurations solved the task much earlier.

**Why this was not the intended solution.** Elite replay preserves a trajectory
because it has high external value. A pure novelty-seeking agent should stop
preferring that trajectory after it becomes familiar. A mixed intrinsic and
extrinsic collection policy also means the intrinsic reward/value model is
trained on data generated by a different policy from the one it is meant to
evaluate. This was a useful engineering baseline, but not a clean experiment in
intrinsic exploration.

**Conclusion from Ideas 1-3.** Full-model ensemble diversity can help
exploration, but Qstd does not acquire novelty semantics merely because all
MuZero components are ensembled. Calibration improves the measurement of
disagreement; it does not define what the disagreement means.

## Idea 4: Create novelty with representation and value priors

**Hypothesis.** A frozen member-specific prior should make an unsupported state
look different in each learned world. As the trainable model observes that
state, it can learn to compensate for the prior. Ensemble disagreement would
then be high where data are absent and low where data are plentiful.

**Mechanism.** This program used two related but distinct frozen priors. They
should not be conflated.

First, a **representation prior** added a member-specific state-dependent
offset $P_k$ to member $k$'s learned representation $E_k$:

$$
z_k^{\mathrm{direct}}(s)
  = C\left(E_k(s) + \alpha P_k(s)\right),
$$

where $C$ denotes the chosen composition or normalization rule. The recurrent
model produced an imagined state using the same member throughout:

$$
\hat z_{t+d}^{(k)}
  = g_k^{(d)}\left(h_k(s_t),a_{t:t+d-1}\right).
$$

If self-supervision made $\hat z_{t+d}^{(k)}$ match the direct encoding of the
actual future state, then a novel endpoint should carry the member-specific
representation prior into planning.

Second, a **value-function prior** was added specifically to turn unsupported
states into member disagreement at the output:

$$
V_k^{\mathrm{total}}(z)
  = V_k^{\mathrm{learned}}(z) + \lambda P_k^V(z).
$$

The intended learning process was asymmetric. On familiar states, supervised
value targets should train $V_k^{\mathrm{learned}}$ to cancel that member's
frozen prior, causing the total member predictions to agree. On unsupported
states, there is no local target enforcing cancellation, so the different
$P_k^V$ terms should remain visible as value and Q disagreement. The
representation prior was meant to encode member-specific state novelty;
self-supervised consistency was meant to transport that geometry into imagined
latents; and the value prior was meant to expose unsupported imagined states in
the quantity used by planning.

We tested normalization changes, VICReg-style regularization, random value
initialization, projected and raw consistency losses, frozen state priors,
frozen value priors, and local hash/grid priors.

**What happened.** The representation prior worked at the encoder but not
through recurrent dynamics. A dedicated probe found a strong correlation
between physical novelty and directly encoded latent novelty, about `0.84`,
but almost no correlation for the recurrently imagined latent, about `0.05`.
Within a root,
imagined latents ranked novel and familiar action sequences no better than
chance.

Projected self-supervision explained one escape route. Its objective was of
the form

$$
\mathcal L_{\mathrm{proj}}
  = \left\|q\!\left(\pi(\hat z)\right)
       - \operatorname{sg}\!\left(\pi(z^{\mathrm{direct}})\right)\right\|_2^2,
$$

where $\pi$ and $q$ are projection and prediction networks. They could map
different raw latent distributions to similar normalized outputs. Batch
statistics made training-mode alignment look strong, while evaluation-mode
statistics exposed the raw mismatch. We therefore tested

$$
\mathcal L_{\mathrm{raw}}
  = \left\|\hat z
       - \operatorname{sg}\!\left(z^{\mathrm{direct}}\right)\right\|_2^2.
$$

Raw matching made recurrent and direct latents substantially more comparable,
including at depth five. It still did not make unsupported states separate in
a useful way. Consistency says only that two constructions of the same state
should agree. This is why we had added the representation and value priors: the
priors, not consistency itself, were supposed to define which states remained
uncertain.

The value-prior implementation initially had a categorical-support bug: adding
a scalar directly to categorical logits did not produce a fixed scalar shift
in decoded value. After correction, a regression probe showed total predictions
converging near the same target while the trainable components learned equal
and opposite residuals for priors of `-2`, `0`, and `+2`. That was evidence that
the intended cancellation mechanism worked on supervised states, not evidence
against the idea.

The unresolved failure was **locality of cancellation**. The experiments did
not show that cancellation was confined to familiar support while frozen prior
diversity survived on unsupported imagined states. The learned value functions
could generalize their compensation beyond directly trained states, and the
value prior was evaluated on a learned latent whose geometry continued to
change. We therefore did not obtain the desired pattern of low disagreement in
visited regions and reliably high disagreement only in novel regions.

The representation prior had a separate scaling escape route. The learned
encoder output grew to a norm around forty while its prior contribution had
norm one, making the representation prior negligible. Fixed-norm composition
removed that loophole and local hash/grid priors reduced broad generalization,
but neither made the full encoder-dynamics-value chain produce sustained,
novelty-calibrated disagreement.

We also considered indexing a prior by root state and action sequence. That
avoids applying a frozen function to a drifting latent, but it estimates
history novelty. Two action histories that reach the same physical state can
remain different forever, which was not the intended quantity.

**Decision.** We paused the random-prior program. It established that direct
encoder novelty is insufficient: novelty must survive recurrent dynamics,
retain physical locality, resist scaling and cancellation that generalizes
beyond familiar support, and identify states rather than arbitrary histories.

## Idea 5: Define novelty explicitly with density and RND

**Hypothesis.** Instead of hoping that ensemble disagreement acquires novelty
semantics, define novelty independently and test whether MuZero can learn and
plan with that target.

**Mechanism.** We first used exact state-density rewards as an interpretable
MountainCar diagnostic. We then used Random Network Distillation. For a
normalized state representation $\phi(s)$, a fixed random target network $T$
and a trained predictor $P_\tau$ define

$$
N_\tau(s)
  = \left\|T(\phi(s)) - P_\tau(\phi(s))\right\|_2^2.
$$

The snapshot index $\tau$ matters: $N_\tau$ is the novelty landscape produced
by experience available at that point in training. The target network remains
fixed; the predictor changes as observations become familiar.

The density experiment was useful because it separated a correct reward
definition from the model's ability to propagate it. Even with exact novelty,
early leaf values were inaccurate. The agent had not trained on many of the
states needed to predict the return of exploratory action sequences, and the
initial novelty-return scale was very large. This explained why apparently
wrong tree preferences did not necessarily imply a wrong novelty target.

**What happened.** The first RND implementation contained a real bug. Its
normalization collapsed many continuous MountainCar states into effectively
repeated inputs. After normalizing position and velocity using fixed physical
bounds, the random target became smoothly correlated with physical distance,
and predictor error tracked both visitation density and distance from previous
support.

RND did not produce a binary boundary between seen and unseen states. That is
not a defect. In a continuous space, a geometrically unvisited state next to a
dense familiar region should receive only moderately more error. Across the
diagnostics, genuinely unsupported regions had higher error, but often by tens
of percent rather than orders of magnitude. The important result was that RND
had the intended local geometry and changed as experience accumulated.

Target timing also mattered. The RND predictor is updated using newly collected
observations. If replay targets are captured before that update, the learned
reward describes the novelty landscape that has just been consumed, not the
landscape that will guide the next collection. We therefore refreshed the
target snapshot after the fresh RND update and trained MuZero against that
post-update landscape.

**Decision.** Correctly normalized RND became the reference novelty signal. It
had interpretable local geometry and changed as support accumulated, unlike
Qstd. The remaining question was whether an imagined future could be assigned
this novelty before it had been visited.

## Idea 6: Train a fully intrinsic MuZero agent

**Hypothesis.** To claim pure exploration, the policy that generates replay
must itself optimize intrinsic reward. Otherwise an intrinsic model can appear
successful while external reward determines which states it sees.

**Mechanism.** For a transition into $s_{t+1}$, the agent received

$$
r_t^{\mathrm{int}} = N_\tau(s_{t+1}).
$$

Its intrinsic value target was eventually a bootstrapped $n$-step return,

$$
G_t^{(n)}
  = \sum_{i=0}^{n-1}\gamma^i r_{t+i}^{\mathrm{int}}
    + \gamma^n V_{\bar\theta}^{\mathrm{int}}(s_{t+n}),
$$

where $V_{\bar\theta}^{\mathrm{int}}$ is the target intrinsic value model. The
MountainCar proof of concept first used full Monte Carlo returns, equivalent to
running $n$ to the episode end and omitting the bootstrap. The valid long
Branching Rooms comparisons later used $n=20$ with a target-value bootstrap.

We initially mixed intrinsic and extrinsic returns in one tree. This allowed
novelty to propagate toward the start state, but incorrect external values also
controlled which data were collected. The intrinsic heads were therefore
learning under a policy that was not actually their exploration policy. To
remove that confound, we built a fully intrinsic agent: collection came from an
intrinsic tree; replay came only from that policy; and encoder, dynamics,
intrinsic reward, intrinsic value, and policy were trained from intrinsic
targets. External reward was retained only as a diagnostic. This agent reached
the distant MountainCar region without external reward guiding either
collection or training.

Full Monte Carlo intrinsic targets were useful for this proof of concept
because they avoided bootstrap feedback. They were not adopted as the preferred
general solution. They are high variance and tied strongly to the current
exploration policy. Later Branching Rooms comparisons therefore used an
explicit bootstrapped twenty-step target.

**What happened.** The fully intrinsic MountainCar agent reached the distant
goal region without external reward guiding collection or training. This
established that RND could drive the complete MuZero learning loop, rather than
merely annotate trajectories found by another policy.

**Decision.** We retained the fully intrinsic design and moved from Monte Carlo
to valid bootstrapped targets for longer environments. The unresolved problem
was no longer how to define real-state novelty. It was how to predict
$N_\tau(s_{t+d})$ at an imagined state before direct experience supplied that
target.

## Idea 7: Actively query the current novelty landscape

**Problem.** An ordinary reward head sees a latent and learns from rewards
attached to replay transitions. For an imagined novel future, that creates a
chicken-and-egg problem: the head is expected to predict high RND before replay
has supplied an RND target from that region. Moreover, RND is nonstationary.
The same physical state becomes less novel as the predictor trains, so novelty
cannot be inferred from state identity alone without some representation of
the current RND landscape.

**Hypothesis.** A root-conditioned query policy could take a small number of
measurements of the current RND field. A reader could then use those
measurements to infer novelty at recurrently imagined states, including states
not yet represented by direct reward targets in replay.

**Mechanism.** Given root state $s_t$, the query network generated $m$ physical
states,

$$
x_i = Q_\psi(s_t)_i,
\qquad
e_i = N_\tau(x_i),
\qquad i=1,\ldots,m.
$$

A reader combined their embeddings and measured errors with the imagined
latent:

$$
\widehat N_{t+d}
  = R_\omega\!\left(
      \hat z_{t+d},
      \left\{\left(h_q(x_i),e_i\right)\right\}_{i=1}^{m}
    \right).
$$

In principle, $Q_\psi$ could learn which measurements of the novelty field
were useful for planning from this particular root.

**Why this introduced a second reward predictor.** The ordinary MuZero model
already had one private intrinsic reward head per ensemble member. Those heads
took only an imagined latent as input. The proposed reader additionally needed
root-specific queried states and their current RND errors, which were produced
outside the recurrent MuZero model and changed with the RND snapshot. It could
therefore not be inserted into the existing head without changing that head's
interface, recurrent inference, and role in MCTS. We conservatively added it as
a side branch and retained the ordinary heads as the baseline operational
reward model and as a fallback before the query system became active.

When query rewards were enabled in MCTS, the two predictions were not summed.
The query reader replaced the ordinary intrinsic edge reward for that tree
expansion. Both predictors were nevertheless trained. Keeping the ordinary
path made it possible to compare reader training versus reader deployment,
detach only the reader gradient, and later remove private reward-head training
as a separate intervention. Once query context was shown to be unnecessary,
the side branch ceased to have a compelling interpretation as a query reader;
it remained as an auxiliary RND decoder. The resulting duplication was an
experimental scaffold, not an a priori claim that two reward predictors should
be part of the final architecture.

**Implementation correction.** The first implementation was too detached from
actual MuZero learning. It sampled roots separately and used known environment
dynamics to create counterfactual supervision. That contradicted the
model-free purpose of MuZero and did not train on the predictions that mattered
during collection. We removed environment-dynamics supervision and integrated
the objective into ordinary replay. Every MuZero minibatch now supplied a real
root, its actual action sequence, recurrent latents at each unrolled depth, and
current RND targets recomputed at the corresponding real future observations.

The query design itself also evolved. A diversity loss was removed because
useful queries may legitimately cluster near a frontier. Pairwise or
within-root ranking was considered, but a replay root only gives factual
targets along the trajectory that was actually executed. The final learner
therefore used all valid depths of the ordinary recurrent unroll rather than
inventing counterfactual labels.

**What happened.** This system produced the strongest MountainCar exploration
seen in the intrinsic experiments. More importantly, it appeared to prefer
trajectories toward a distant novel region before replay contained direct
rewards from that region. Initially this looked like evidence that the queries
had allowed the reader to reconstruct the current novelty landscape.

The initial three-seed matrix made the behavioral change clear:

| Condition | First collection goal phase, seeds 0/1/2 | Successful segments, mean +/- sample SD |
|---|---:|---:|
| Baseline | `8 / 8 / 13` | `13.3 +/- 10.3` |
| Query reader trained with world model detached | `none / 8 / 8` | `14.3 +/- 12.4` |
| Query reader jointly trains world model and supplies tree rewards | `4 / 4 / 6` | `49.7 +/- 17.5` |
| Joint world-model training without query context | `4 / 5 / 8` | `60.7 +/- 29.7` |

The last row was the first warning that the behavior might come from the
training objective rather than the queried measurements.

**Intervention test.** That interpretation did not survive intervention.

- Reversing RND errors among query locations left `97.8%` of preferred actions
  unchanged.
- Replacing all query errors by their mean left `98.7%` unchanged.
- Supplying another root's query context left `97.2%` unchanged.
- Zeroing query latents and errors still left `86%` unchanged, and the
  momentum-aligned action preference remained on `83%` of states.

A no-context model also performed at least as well as the query model.
MountainCar behavior was therefore not being driven by the specific
measurements chosen by the query policy.

The more plausible explanation was simpler. The auxiliary loss added RND
prediction gradients through the encoder and dynamics across the recurrent
unroll. This encouraged imagined latents to expose whatever features made
future novelty predictable. In MountainCar, momentum makes that extrapolation
easy: continuing along the current direction often moves outside recent
support. The model did not need a detailed map of the current RND field to
learn that regularity.

**Decision.** We rejected the claim that active queries caused the MountainCar
gain. The result still contained a useful mechanism, but identifying it
required isolating the query context, auxiliary prediction, and gradient path.

## Idea 8: Train imagined latents to expose current novelty

**Question.** Was the gain caused by the information in the query context, by
using the auxiliary prediction in MCTS, or by the auxiliary loss changing the
world model?

**Mechanism.** For a replay root and its factual action sequence, MuZero
generated recurrent latents

$$
\hat z_{t+d}^{(k)}
  = g_k^{(d)}\left(h_k(s_t),a_{t:t+d-1}\right).
$$

An auxiliary decoder $D_{\omega,k}$ was trained against current RND at every
valid unrolled depth:

$$
\mathcal L_{\mathrm{aux}}
  = \frac{1}{K}\sum_{k=1}^{K}\sum_{d=1}^{D}
    \ell\!\left(
      D_{\omega,k}(\hat z_{t+d}^{(k)}),
      \operatorname{sg}\!\left(N_\tau(s_{t+d})\right)
    \right).
$$

In the shared-reader condition, $D_{\omega,k}\equiv D_\omega$ for every
member. The same decoder weights are applied separately to each member's
latent; the member latents are not concatenated or averaged. This encourages
each member to expose RND-relevant information in a form readable by one
common function. In a member-specific condition, each $D_{\omega,k}$ has its
own parameters.

In the detached condition,

$$
\nabla_{\theta_h,\theta_g}\mathcal L_{\mathrm{aux}} = 0,
$$

so only the decoder learned. In the joint condition this gradient entered the
encoder and recurrent dynamics. In the train-only condition the auxiliary
prediction was discarded during MCTS; the private operational reward heads
still supplied all tree-edge rewards.

This distinction must not be overstated. The ordinary MuZero reward loss is
also applied along the recurrent unroll:

$$
\mathcal L_{\mathrm{reward}}
  = \frac{1}{K}\sum_{k=1}^{K}\sum_{d=1}^{D}
    \ell\!\left(
      H_k^r(\hat z_{t+d}^{(k)}),
      N_\tau(s_{t+d})
    \right),
$$

and it can also train the world model. Unlike the train-only auxiliary output,
$H_k^r$ supplies operational MCTS edge rewards. The auxiliary branch did not
reveal labels unavailable to the reward head. It added a separate decoder,
target transform, weighting, and gradient path while leaving the operational
reward channel unchanged. The question was therefore about optimization and
representation pressure, not extra target information.

This point is essential: the baseline runs used
`intrinsic_head_only=False`. Consequently, the ordinary private raw-RND reward
loss already backpropagated at every recurrent depth through the latent state,
the recurrent dynamics, and the root encoder. The auxiliary branch did **not**
introduce RND representation learning for the first time. It added a second,
independently parameterized and usually log-scaled RND objective on top of RND
representation learning that was already present.

### MountainCar breakthrough scorecard

The cleanest comparison is the fixed-initialization confirmatory baseline
against the selected shared-log joint train-only auxiliary condition. The
auxiliary prediction was discarded by MCTS in the latter condition; private
raw-RND heads remained the operational tree rewards. A successful segment is a
collected 200-step trajectory segment that reached the MountainCar goal. The
training and collection policy was intrinsic-only: external reward did not
select actions and its reward/value losses were disabled. Evaluation also used
the deterministic intrinsic tree, but reported the environment's extrinsic
return averaged over 16 episodes. Thus `-200` means that this intrinsic policy
did not reach the goal within the time limit, and less-negative values are
better. It is a diagnostic of where exploration led, not the optimized
objective.

| Metric | Baseline | Shared-log auxiliary, joint train-only | Change |
|---|---:|---:|---:|
| Unique position-velocity bins, seeds 0/1/2 | `3796 / 3848 / 2893` | `4230 / 4359 / 3894` | More coverage in every seed |
| Unique state bins, mean +/- sample SD | `3512 +/- 537` | `4161 +/- 240` | `+649`, or `+18.5%` |
| Fraction of the configured 6300-bin diagnostic grid | `55.8%` | `66.0%` | `+10.3` percentage points |
| First collection goal phase, seeds 0/1/2 | `8 / 8 / 13` | `5 / 6 / 7` | Earlier in every seed |
| Successful segments, seeds 0/1/2 | `22 / 16 / 2` | `92 / 33 / 38` | `+70 / +17 / +36` |
| Successful segments, mean +/- sample SD | `13.3 +/- 10.3` | `54.3 +/- 32.7` | `4.1x`, or `+41.0` segments |
| Best 16-episode evaluator return, seeds 0/1/2 | `-177.0 / -157.4 / -197.5` | `-146.6 / -145.3 / -170.1` | Better in every seed |
| Mean of per-seed best evaluator returns | `-177.3` | `-154.0` | `+23.3` return points |
| Last recorded evaluator return, mean across seeds | `-187.6` | `-181.4` | `+6.2` return points |

These rows come from confirmatory baseline jobs `30218-30220` and joint
train-only jobs `30221`, `30222`, and `30224`; collection counts and evaluator
returns were read from their saved diagnostics and evaluator logs. State-bin
coverage was reconstructed from all recorded collection-root positions and
velocities using the experiment's configured `0.02 x 0.002` discretization.
The percentage is coverage of that rectangular diagnostic grid, not a claim
that every bin is dynamically reachable. Position-only coverage saturated at
`85` bins in all six runs, so the joint position-velocity coverage is the more
discriminating measure.

Coverage and collection behavior are the central exploration results. The
auxiliary increased cumulative state-bin coverage by `18.5%`, moved goal
discovery from phases `8/8/13` to `5/6/7`, and increased successful
goal-reaching segments by about fourfold. Evaluator performance also improved,
but it was secondary and not monotonic because the policy was optimized for
intrinsic novelty rather than for repeatedly solving MountainCar after
discovering the goal.

For context, deploying the same auxiliary reader as the MCTS edge reward
reached the goal in phases `4/5/8`, produced `51/94/37` successful segments
(`60.7 +/- 29.7`), and had a mean per-seed best evaluator return of `-152.4`.
This was only a modest increment over train-only, which is why the planning-time
reader was not considered necessary. The ordinary-sized matched auxiliary
decoder had the highest successful-segment mean, `76.0 +/- 44.9`, but was much
more variable (`35/124/69` segments and first goals in phases `12/5/4`). It was
therefore evidence that the specialized reader was unnecessary, not a cleanly
superior final configuration.

**Causal result 1: the gradient into the world model mattered.** A detached
decoder learned the task but behaved like baseline. Allowing the same objective
to update the encoder and dynamics substantially improved exploration. In the
confirmatory three-seed family, baseline and detached training both averaged
roughly thirteen successful distant-region segments, while joint train-only
training averaged roughly fifty-four. The counts were variable, but this
contrast was large and replicated.

**Causal result 2: using the decoder in MCTS was not necessary.** Most of the
gain remained when the auxiliary prediction was discarded at planning time.
The useful intervention happened during representation learning, while the
ordinary intrinsic reward heads continued to determine tree rewards.

**Causal result 3: no special reader architecture was necessary.** We varied
shared versus member-specific readers, raw versus log-transformed targets,
normalized targets, architecture size, learning rate, and loss weight. No
single distinction explained all results. Shared log targets were stable, but
member-specific and ordinary-sized decoders could also work.

**Reward-head controls.** Simply doubling the operational reward loss did not
reproduce the result. A log target sometimes helped the ordinary head, while a
single shared operational log head was worse than the auxiliary configuration.
However, the final pre-registered family showed that an ordinary-sized
*additional* decoder with matched auxiliary training could also improve
exploration. Thus the evidence does not establish that the specialized reader,
sharing, log targets, or large architecture is individually necessary.

**Decision.** The supported mechanism is narrower than the original query
story:

> An additional, jointly trained current-RND prediction objective can reshape
> MuZero's encoder and recurrent dynamics in a way that improves exploration,
> even when its prediction is never used by MCTS.

This remains architecturally redundant. Two modules can predict nearly the
same target, and the experiments do not yet isolate whether the benefit comes
from decoder separation, gradient geometry, loss scaling, or reduced conflict
with the operational reward model. They do establish that merely fitting a
detached reader does nothing, active query context is unnecessary in
MountainCar, and the auxiliary prediction need not control planning directly.

## Idea 9: Test transfer in Branching Rooms

**Why this test was necessary.** MountainCar was valuable because its state was
known, its dynamics were easy to inspect, and distant-region discovery was
unambiguous. Those same properties made it a weak final test. Exploration
mostly consists of building momentum, and the regularity learned by the
auxiliary objective can be summarized as "continue moving into less familiar
territory." A model can extrapolate this without understanding a complicated
local novelty field.

**Test environment.** We therefore designed Branching Rooms. It is still
controlled and exactly measurable, but it uses partial RGB observations,
orientation-dependent actions, shared corridors, branching choices, and dead
ends. Total cell coverage measures breadth, while rooms and designated dead
ends measure topological reach. This distinction became important because an
agent can cover many nearby cells while failing to enter distant branches.

**Validity corrections.** The initial jobs exposed two mechanical bugs, one in
MLP construction and one at a terminal replay boundary. They were fixed and
excluded from scientific comparison. A later target experiment was also
classified correctly as invalid for its intended question: what was described
as twenty-step TD had twenty rewards followed by zero continuation value. That
is a finite-horizon return, not a bootstrapped TD target. We retained it as
development history but did not use it to decide whether bootstrapping helped.

**What happened.** The corrected long comparison used a twenty-step intrinsic
return plus a target value bootstrap. The auxiliary model did not transform
raw coverage: position and pose totals remained close to baseline. The more
interesting difference was where exploration occurred. Baseline seeds
consistently reached thirteen of sixteen rooms and two dead ends. Two auxiliary
seeds reached branches that no baseline seed reached, raising the average
dead-end count from two to three, although one auxiliary seed underperformed.
This is evidence of better topological reach, but not yet a statistically
decisive general improvement.

The phase-by-phase heatmaps clarified the behavior. Neither agent continuously
moved into new cells. Both repeatedly traversed familiar rooms and corridors,
occasionally expanding the frontier. Some revisitation is structurally
necessary because corridors connect branches. Some immediate backtracking is
not: it exposed a weakness in phase-frozen RND that became the next research
question.

**Decision.** The auxiliary novelty-prediction objective shows tentative
transfer beyond MountainCar, especially in distant branch discovery. The
effect is smaller and more variable than in MountainCar, so it should be
presented as promising evidence rather than a solved exploration method.

## Idea 10: Query through a compact observation space

**Problem.** The original MountainCar query network could emit real
two-dimensional states directly. That does not scale to images. We considered
learning an inverse of the MuZero encoder that would return the set or
distribution of observations compatible with a latent. With stop-gradient at
the MuZero representation, such an inverse would not force the encoder to
retain every visual detail. It was an interesting interpretability idea, but
it solved the reverse problem: decoding a MuZero latent, not giving a query
policy a compact space in which to propose states.

**Mechanism.** For Branching Rooms we instead trained a separate image
autoencoder $(E_\eta,D_\eta)$. Real observations supplied compact codes
$c=E_\eta(o)$ and reconstructions $\tilde o=D_\eta(c)$. The query policy emitted
codes directly:

$$
c_i = Q_\psi(s_t)_i,
\qquad
\tilde o_i = D_\eta(c_i),
\qquad
e_i = N_\tau(\tilde o_i).
$$

The reader used $(c_i,e_i)$ as context for predicting novelty at imagined
MuZero latents. This did not use known environment dynamics and, in principle,
gave the policy a compact space in which to ask image-level questions.

**What happened.** Unlike MountainCar, withholding query context modestly
reduced topological coverage. The context condition reached slightly more rooms
and dead ends and often predicted replay RND better. That was the first
suggestion that explicit context might matter when the environment has genuine
branching.

**Failure diagnostic.** The geometry diagnostic prevented an overly strong
interpretation. Query codes were much farther from sampled real-state codes
than real-state codes were from one another. The same query slot changed very
little across roots, while different slots formed a mostly fixed global set.
The network had found off-manifold regions of the autoencoder code space whose
decoded images were useful to the loss. It had not learned root-local probes
corresponding to reachable physical states.

This is why a low prediction loss is not sufficient evidence for meaningful
querying. A query is only scientifically interpretable if it lies on a valid
state manifold, changes appropriately with the root, and measures information
that affects the reader's prediction or the planner's action. The implemented
compact query system did not satisfy those conditions.

**Decision.** Query context may provide some useful signal in a branched pixel
environment, but the current generator is not a valid local state-query
mechanism. Its apparent benefit remains confounded with auxiliary world-model
training and synthetic off-manifold probes.

## Idea 11: Suppress within-episode revisits with latent memory

**Problem.** RND is deliberately updated between collection phases rather than
after every environment action. This keeps the novelty target stationary while
MCTS and replay targets are generated. In Branching Rooms it creates a concrete
failure mode. If state `A` is novel at the start of the phase, the transition
`A -> B -> A` can reward the final return to `A`, because the frozen RND model
has not yet learned that `A` was visited earlier in the same episode.

**Hypothesis.** Keep phase-level RND fixed, but correct its tree reward with a
fast episode-local memory. This should suppress exact revisits without making
the RND model itself nonstationary after every action.

**Mechanism.** Let $\kappa_k(z)$ be member $k$'s projected or raw comparison
key. For an imagined candidate, compute its nearest squared distance to either
the direct encodings of real states already visited in the episode,
$\mathcal M_t$, or proper-prefix imagined states on the current search path,
$\mathcal P$:

$$
d_{\min}(\hat z)
  = \min_{m\in\mathcal M_t\cup\mathcal P}
      \frac{1}{K}\sum_{k=1}^{K}
      \left\|\kappa_k(\hat z^{(k)})-m^{(k)}\right\|_2^2.
$$

The memory estimates a typical one-step latent distance $\delta_t$ from actual
consecutive states. It is used only to normalize scale:

$$
u_{\mathrm{epi}}(\hat z)
  = 1 - \exp\!\left(
      -\frac{d_{\min}(\hat z)}{b\max(\delta_t,\epsilon)}
    \right),
$$

$$
f_{\mathrm{epi}}
  = \rho + (1-\rho)u_{\mathrm{epi}},
\qquad
\hat r_{\mathrm{tree}}
  = \hat r_{\mathrm{RND}}f_{\mathrm{epi}}.
$$

Here $b$ is the kernel bandwidth and $\rho=0.05$ is the minimum retained reward
factor in the scheduled experiment. In `A -> B -> A`, the final imagined `A`
is compared with the first stored `A`; the possibly large transition distance
from `B` to `A` is irrelevant. The gate changes immediate imagined edge reward
only, leaving the intrinsic leaf value unchanged.

**Controls.** Two comparison spaces were prepared. The projected-memory
condition uses the normal self-supervised comparison space. The raw-memory
condition uses an explicit raw-latent consistency loss, with a
raw-SSL/no-memory condition to separate the effect of changing representation
training from the effect of memory itself.

**What went wrong.** Complete three-seed reruns showed that neither comparison
space delivered the intended selective revisit correction. In projected SSL
space, the memory was mostly inactive: the mean one-step reward factor was
`0.997`, and selected and unselected root actions received effectively the same
factor. Even at depth five the mean factor was `0.883`. Final position coverage
changed only from `0.761` without memory to `0.777` with memory, with paired
seed changes of `+0.061`, `0.000`, and `-0.010`. This is not evidence that the
projected key reliably recognized `A -> B -> A`; imagined recurrent states were
usually not close enough to stored direct-state keys for the gate to act.

Raw-latent memory had the opposite failure. It acted strongly, but too broadly.
The mean reward factor fell from `0.912` at depth one to `0.357` at depth five,
where about 80% of edges had a factor below `0.5`. The selected and unselected
one-step factors were again nearly identical (`0.912` versus `0.911`), so the
gate did little useful root-action discrimination while increasingly damping
longer imagined paths. Relative to the matched raw-SSL/no-memory control, final
position coverage fell by `0.127`, pose coverage by `0.116`, room coverage by
`0.125`, and dead-end coverage by `0.200` on average. Position coverage was
lower in all three paired seeds.

The raw-SSL/no-memory control itself reached the highest mean position and pose
coverage in this four-condition comparison. That means changing the
representation objective was not the cause of the raw-memory failure; adding
the memory gate was. The diagnostics do not label the physical identity of
every gated imagined edge, so they cannot directly count exact `A -> B -> A`
corrections. However, the depth-wide suppression and consistent behavioral
loss show that the raw gate was not selective enough.

**Decision.** Reject this episodic-memory mechanism in its current form.
Projected keys do not make direct and imagined revisits sufficiently
comparable, while raw keys make too many deep imagined states appear familiar.
Any future revisit correction should first be validated against physical-state
identity labels for exact revisits, nearby new poses, and genuinely new states
before it is allowed to alter MCTS rewards.

## What the project has ruled out

Several conclusions are negative but valuable:

1. **Raw Q disagreement is not a novelty estimator by default.** It can improve
   control while mostly measuring value-head variation.
2. **Dynamics disagreement is not sufficient merely because it occurs in a
   learned model.** Latent disagreement has no guaranteed physical meaning.
3. **Direct encoder novelty does not automatically transfer to imagined
   novelty.** The recurrent model can destroy the geometry that the encoder
   prior created.
4. **Projected SSL does not imply raw latent comparability.** A projection head
   can absorb substantial distribution mismatch.
5. **Raw latent comparability does not itself create epistemic separation.** It
   closes one loophole but does not define novelty.
6. **Random priors require more than random initialization.** Their locality,
   magnitude, predictability, cancellation, and behavior under latent drift all
   matter.
7. **MountainCar query context was not the cause of the exploration gain.**
   Direct interventions and no-context controls established this.
8. **Low query-reader loss does not validate the queries.** The compact query
   policy exploited latent holes and emitted nearly root-independent probes.
9. **More goal-reaching is not by itself proof of better pure exploration.** A
   novelty agent should move on after a successful trajectory becomes familiar.
10. **A finite-horizon target without continuation value is not bootstrapped
    TD.** Runs using that accidental target cannot answer the intended target
    comparison.
11. **Latent proximity is not yet a safe episodic-revisit gate.** Projected
    proximity barely changed rewards, while raw proximity increasingly
    suppressed deep imagined paths and reduced exploration in every seed.

These failures progressively narrowed the research question. They prevented us
from attributing useful behavior to mechanisms that the evidence did not
support.

## What remains supported

Three findings are sufficiently well supported to form the core of the thesis
story.

First, RND can provide a physically meaningful, smoothly generalizing novelty
target when its input geometry is handled correctly. This gives us an external
reference against which learned uncertainty can be judged.

Second, MuZero can be trained as a fully intrinsic agent. It can collect its own
exploration data and train its policy, reward, value, encoder, and dynamics
without an external reward policy determining where it goes.

Third, and most importantly, an additional jointly trained current-RND
prediction objective can improve pure exploration through the gradient it
sends into the encoder and recurrent dynamics. The detached-versus-joint
contrast identifies representation shaping as the causal mechanism. The
effect does not require the auxiliary prediction to replace the reward used by
MCTS, and it does not require the specialized query architecture used in the
first implementation.

The Branching Rooms result is best described as tentative transfer evidence.
The auxiliary objective changed topological reach more than raw coverage, which
is consistent with the intended mechanism, but the sample is small and
variable. The query and episodic-memory components remain open research rather
than established contributions.

## Thesis-level interpretation

The original ambition was to create a planner whose ensemble disagreement
directly represented epistemic uncertainty over imagined physical states. The
experiments did not validate that mechanism. They instead suggest a different
lesson: uncertainty useful for exploration may need to be taught to the world
model through an explicit auxiliary prediction task rather than expected to
emerge from independent initialization.

This reframes the contribution. The most defensible claim is not that we solved
active novelty-field querying, nor that ensemble variance is calibrated. It is
that a MuZero representation optimized only for policy, value, reward, and
standard consistency can omit information needed for novelty-aware planning.
Adding explicit current-novelty supervision along recurrent unrolls changes the
representation and can materially improve intrinsic exploration.

The remaining scientific question is whether this principle survives stronger
environments and can be expressed without redundant heads or invalid synthetic
queries. Branching Rooms is the first step in that direction. Episodic memory
tests whether phase-current novelty can be corrected using episode-current
experience. A future valid-state query mechanism would test whether a planner
can actively measure a changing novelty landscape in high-dimensional domains.

For a supervisor meeting, the project can therefore be summarized in one
sentence:

> We began by trying to extract novelty from ensemble disagreement, found that
> the disagreement had the wrong semantics, replaced it with explicit RND
> novelty, and discovered through ablation that the most useful intervention was
> teaching MuZero's recurrent world model to represent current novelty, not the
> query mechanism originally designed to supply it.
