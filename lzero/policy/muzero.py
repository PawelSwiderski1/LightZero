import copy
from typing import List, Dict, Any, Tuple, Union, Optional

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
from ding.model import model_wrap
from ding.policy.base_policy import Policy
from ding.torch_utils import to_tensor
from ding.utils import POLICY_REGISTRY
from torch.nn import L1Loss

from lzero.mcts import MuZeroMCTSCtree as MCTSCtree
from lzero.mcts import MuZeroMCTSPtree as MCTSPtree
from lzero.model import ImageTransforms
from lzero.model.utils import calculate_dormant_ratio
from lzero.policy import scalar_transform, InverseScalarTransform, cross_entropy_loss, phi_transform, \
    DiscreteSupport, to_torch_float_tensor, mz_network_output_unpack, select_action, negative_cosine_similarity, \
    prepare_obs, configure_optimizers


def _representation_prediction_parameter_groups(model: torch.nn.Module):
    """Split MCTS representation/prediction parameters from all other parameters."""
    representation_prediction_params = []
    other_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        path = f'.{name}.'
        if '.representation_network.' in path or '.prediction_network.' in path:
            representation_prediction_params.append(parameter)
        else:
            other_params.append(parameter)
    return representation_prediction_params, other_params


def _categorical_value_prior_training_terms(
        base_logits: torch.Tensor,
        scalar_prior: torch.Tensor,
        scalar_target: torch.Tensor,
        value_support: DiscreteSupport,
        inverse_scalar_transform,
        scalar_loss_weight: float,
):
    """Build a stable categorical cancellation loss for an additive scalar value prior."""
    scalar_target = scalar_target.reshape(-1, 1)
    scalar_prior = scalar_prior.reshape(-1, 1)
    base_scalar = inverse_scalar_transform(base_logits)
    final_scalar = base_scalar + scalar_prior

    # The categorical head learns the residual target. Detaching the prior on
    # this branch prevents gradients through a target distribution, while the
    # scalar term below provides the intended gradient through the prior input.
    adjusted_target = scalar_target - scalar_prior.detach()
    adjusted_target_categorical = phi_transform(
        value_support,
        scalar_transform(adjusted_target.squeeze(-1)),
    )
    categorical_loss = cross_entropy_loss(base_logits, adjusted_target_categorical)
    scalar_consistency_loss = F.smooth_l1_loss(
        scalar_transform(final_scalar),
        scalar_transform(scalar_target),
        reduction='none',
    ).reshape(base_logits.shape[0], -1).mean(dim=1)
    combined_loss = categorical_loss + float(scalar_loss_weight) * scalar_consistency_loss
    diagnostics = {
        'prior': scalar_prior,
        'adjusted_target': adjusted_target,
        'base_scalar': base_scalar,
        'final_scalar': final_scalar,
        'categorical_loss': categorical_loss,
        'scalar_consistency_loss': scalar_consistency_loss,
        'final_abs_error': (final_scalar - scalar_target).abs(),
    }
    return final_scalar, combined_loss, diagnostics


def _latent_variance_covariance_loss(
        latent_state: torch.Tensor,
        group_ids=None,
        valid_mask: Optional[torch.Tensor] = None,
        variance_target: float = 1.0,
        eps: float = 1e-4,
):
    """VICReg-style anti-collapse losses, calculated independently per ensemble member."""
    flat_latent = latent_state.reshape(latent_state.shape[0], -1).float()
    batch_size, feature_dim = flat_latent.shape
    if valid_mask is None:
        valid_mask = torch.ones(batch_size, dtype=torch.bool, device=flat_latent.device)
    else:
        valid_mask = torch.as_tensor(valid_mask, device=flat_latent.device).reshape(-1).bool()
        if valid_mask.numel() != batch_size:
            raise ValueError(f'Expected {batch_size} latent validity entries, got {valid_mask.numel()}.')

    if group_ids is None:
        group_ids = [0] * batch_size
    elif torch.is_tensor(group_ids):
        group_ids = group_ids.detach().cpu().reshape(-1).tolist()
    else:
        group_ids = list(group_ids)
    if len(group_ids) != batch_size:
        raise ValueError(f'Expected {batch_size} latent group ids, got {len(group_ids)}.')

    grouped_indices = {}
    for row_index, group_id in enumerate(group_ids):
        grouped_indices.setdefault(int(group_id), []).append(row_index)

    variance_losses = []
    covariance_losses = []
    std_means = []
    std_mins = []
    std_maxes = []
    covariance_abs_means = []
    valid_sample_count = 0
    for indices in grouped_indices.values():
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=flat_latent.device)
        index_tensor = index_tensor[valid_mask.index_select(0, index_tensor)]
        if index_tensor.numel() < 2:
            continue
        group_latent = flat_latent.index_select(0, index_tensor)
        centered = group_latent - group_latent.mean(dim=0, keepdim=True)
        feature_std = torch.sqrt(group_latent.var(dim=0, unbiased=True) + float(eps))
        covariance = centered.T @ centered / float(group_latent.shape[0] - 1)
        off_diagonal_mask = ~torch.eye(feature_dim, dtype=torch.bool, device=covariance.device)
        off_diagonal = covariance[off_diagonal_mask]

        variance_losses.append(torch.relu(float(variance_target) - feature_std).mean())
        covariance_losses.append(
            off_diagonal.pow(2).sum() / float(feature_dim) if off_diagonal.numel() > 0 else covariance.sum() * 0.0
        )
        std_means.append(feature_std.detach().mean())
        std_mins.append(feature_std.detach().min())
        std_maxes.append(feature_std.detach().max())
        covariance_abs_means.append(
            off_diagonal.detach().abs().mean()
            if off_diagonal.numel() > 0
            else covariance.detach().sum() * 0.0
        )
        valid_sample_count += int(group_latent.shape[0])

    zero = flat_latent.sum() * 0.0
    if not variance_losses:
        return zero, zero, {
            'group_count': 0,
            'sample_count': 0,
            'feature_std_mean': 0.0,
            'feature_std_min': 0.0,
            'feature_std_max': 0.0,
            'covariance_abs_mean': 0.0,
        }

    return torch.stack(variance_losses).mean(), torch.stack(covariance_losses).mean(), {
        'group_count': len(variance_losses),
        'sample_count': valid_sample_count,
        'feature_std_mean': float(torch.stack(std_means).mean().item()),
        'feature_std_min': float(torch.stack(std_mins).mean().item()),
        'feature_std_max': float(torch.stack(std_maxes).mean().item()),
        'covariance_abs_mean': float(torch.stack(covariance_abs_means).mean().item()),
    }


def _raw_latent_consistency_terms(
        predicted_latent: torch.Tensor,
        target_latent: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Per-sample raw-latent alignment terms with a stop-gradient target branch."""
    predicted_flat = predicted_latent.reshape(predicted_latent.shape[0], -1).float()
    target_flat = target_latent.detach().reshape(target_latent.shape[0], -1).float()
    if predicted_flat.shape != target_flat.shape:
        raise ValueError(
            f'Raw latent SSL requires equal latent shapes, got '
            f'{tuple(predicted_flat.shape)} and {tuple(target_flat.shape)}.'
        )
    predicted_l2 = predicted_flat.norm(p=2, dim=1)
    target_l2 = target_flat.norm(p=2, dim=1)
    cosine = (predicted_flat * target_flat).sum(dim=1) / (
        predicted_l2 * target_l2
    ).clamp_min(1e-8)
    return {
        'loss': (predicted_flat - target_flat).pow(2).mean(dim=1),
        'cosine': cosine,
        'pred_l2': predicted_l2,
        'target_l2': target_l2,
    }


@POLICY_REGISTRY.register('muzero')
class MuZeroPolicy(Policy):
    """
    Overview:
        if self._cfg.model.model_type in ["conv", "mlp"]:
            The policy class for MuZero.
        if self._cfg.model.model_type == ["conv_context", "mlp_context"]:
            The policy class for MuZero w/ Context, a variant of MuZero.
            This variant retains the same training settings as MuZero but diverges during inference
            by employing a k-step recursively predicted latent representation at the root node,
            proposed in the UniZero paper https://arxiv.org/abs/2406.10667.
    """

    # The default_config for MuZero policy.
    config = dict(
        model=dict(
            # (str) The model type. For 1-dimensional vector obs, we use mlp model. For the image obs, we use conv model.
            model_type='conv',  # options={'mlp', 'conv'}
            # (bool) If True, the action space of the environment is continuous, otherwise discrete.
            continuous_action_space=False,
            # (tuple) The stacked obs shape.
            # observation_shape=(1, 96, 96),  # if frame_stack_num=1
            observation_shape=(4, 96, 96),  # if frame_stack_num=4
            # (bool) Whether to use the self-supervised learning loss.
            self_supervised_learning_loss=False,
            # (bool) Whether to use discrete support to represent categorical distribution for value/reward/value_prefix.
            # reference: http://proceedings.mlr.press/v80/imani18a/imani18a.pdf, https://arxiv.org/abs/2403.03950
            categorical_distribution=True,
            # (int) The image channel in image observation.
            image_channel=1,
            # (int) The number of frames to stack together.
            frame_stack_num=1,
            # (int) The number of res blocks in MuZero model.
            num_res_blocks=1,
            # (int) The number of channels of hidden states in MuZero model.
            num_channels=64,
            # (tuple) The range of supports used in categorical distribution.
            # These variables are only effective when ``model.categorical_distribution=True``.
            reward_support_range=(-300., 301., 1.),
            value_support_range=(-300., 301., 1.),
            # (bool) whether to learn bias in the last linear layer in value and policy head.
            bias=True,
            # (str) The type of action encoding. Options are ['one_hot', 'not_one_hot']. Default to 'one_hot'.
            discrete_action_encoding_type='one_hot',
            # (bool) whether to use res connection in dynamics.
            res_connection_in_dynamics=True,
            # (str) The type of normalization in MuZero model. Options are ['BN', 'LN']. Default to 'LN'.
            norm_type='BN',
            # (bool) Whether to analyze simulation normalization.
            analysis_sim_norm=False,
            # (bool) Whether to analyze dormant ratio. More details can be found in https://proceedings.mlr.press/v202/sokar23a/sokar23a.pdf.
            analysis_dormant_ratio=False,
            # (bool) Whether to use HarmonyDream to balance weights between different losses. Default to False.
            # More details can be found in https://arxiv.org/abs/2310.00344.
            harmony_balance=False
        ),
        # ****** common ******
        # (bool) Whether to use wandb to log the training process.
        use_wandb=False,
        # (bool) whether to use rnd model.
        use_rnd_model=False,
        # (bool) Whether to use multi-gpu training.
        multi_gpu=False,
        # (bool) Whether to enable the sampled-based algorithm (e.g. Sampled EfficientZero)
        # this variable is used in ``collector``.
        sampled_algo=False,
        # (bool) Whether to enable the gumbel-based algorithm (e.g. Gumbel Muzero)
        gumbel_algo=False,
        # (bool) Whether to use C++ MCTS in policy. If False, use Python implementation.
        mcts_ctree=True,
        # (bool) Whether to use cuda for network.
        cuda=True,
        # (int) The number of environments used in collecting data.
        collector_env_num=8,
        # (int) The number of environments used in evaluating policy.
        evaluator_env_num=3,
        # (str) The type of environment. Options are ['not_board_games', 'board_games'].
        env_type='not_board_games',
        # (str) The type of action space. Options are ['fixed_action_space', 'varied_action_space'].
        action_type='fixed_action_space',
        # (str) The type of battle mode. Options are ['play_with_bot_mode', 'self_play_mode'].
        battle_mode='play_with_bot_mode',
        # (bool) Whether to monitor extra statistics in tensorboard.
        monitor_extra_statistics=True,
        # (int) The transition number of one ``GameSegment``.
        game_segment_length=200,
        # (bool): Indicates whether to perform an offline evaluation of the checkpoint (ckpt).
        # If set to True, the checkpoint will be evaluated after the training process is complete.
        # IMPORTANT: Setting eval_offline to True requires configuring the saving of checkpoints to align with the evaluation frequency.
        # This is done by setting the parameter learn.learner.hook.save_ckpt_after_iter to the same value as eval_freq in the train_muzero.py automatically.
        eval_offline=False,
        # (bool) Whether to calculate the dormant ratio.
        calculate_dormant_ratio=False,
        # (bool) Whether to analyze simulation normalization.
        analysis_sim_norm=False,
        # (bool) Whether to analyze dormant ratio.
        analysis_dormant_ratio=False,

        # ****** observation ******
        # (bool) Whether to transform image to string to save memory.
        transform2string=False,
        # (bool) Whether to use gray scale image.
        gray_scale=False,
        # (bool) Whether to use data augmentation.
        use_augmentation=False,
        # (list) The style of augmentation.
        augmentation=['shift', 'intensity'],

        # ******* learn ******
        # (bool) Whether to ignore the done flag in the training data. Typically, this value is set to False.
        # However, for some environments with a fixed episode length, to ensure the accuracy of Q-value calculations,
        # we should set it to True to avoid the influence of the done flag.
        ignore_done=False,
        # (int) How many updates(iterations) to train after collector's one collection.
        # Bigger "update_per_collect" means bigger off-policy.
        # collect data -> update policy-> collect data -> ...
        # For different env, we have different episode_length,
        # we usually set update_per_collect = collector_env_num * episode_length / batch_size * reuse_factor.
        # If we set update_per_collect=None, we will set update_per_collect = collected_transitions_num * cfg.policy.replay_ratio automatically.
        update_per_collect=None,
        # (float) The ratio of the collected data used for training. Only effective when ``update_per_collect`` is not None.
        replay_ratio=0.25,
        # (int) Minibatch size for one gradient descent.
        batch_size=256,
        # (str) Optimizer for training policy network. ['SGD', 'Adam']
        optim_type='SGD',
        # (float) Learning rate for training policy network. Initial lr for manually decay schedule.
        learning_rate=0.2,
        # (int) Frequency of target network update.
        target_update_freq=100,
        # (int) Frequency of target network update.
        target_update_freq_for_intrinsic_reward=1000,
        # (float) Weight decay for training policy network.
        weight_decay=1e-4,
        # Optional weight decay override for the MCTS-facing representation and prediction networks.
        # ``None`` preserves the global weight decay; set to 0 to exempt these pathways.
        representation_prediction_weight_decay=None,
        # VICReg-style anti-collapse regularization on the planning latent. Both weights default to zero.
        latent_variance_loss_weight=0.0,
        latent_covariance_loss_weight=0.0,
        latent_variance_target=1.0,
        latent_regularization_eps=1e-4,
        # (float) One-order Momentum in optimizer, which stabilizes the training process (gradient direction).
        momentum=0.9,
        # (float) The maximum constraint value of gradient norm clipping.
        grad_clip_value=10,
        # (int) The number of episodes in each collecting stage when use muzero_collector.
        n_episode=8,
        # (int) The number of num_segments in each collecting stage when use muzero_segment_collector.
        num_segments=8,
        # (int) the number of simulations in MCTS.
        num_simulations=50,
        # (float) Discount factor (gamma) for returns.
        discount_factor=0.997,
        # (int) The number of steps for calculating target q_value.
        td_steps=5,
        # (int) The number of unroll steps in dynamics network.
        num_unroll_steps=5,
        # (float) The weight of reward loss.
        reward_loss_weight=1,
        # (float) The weight of value loss.
        value_loss_weight=0.25,
        # (float) The weight of policy loss.
        policy_loss_weight=1,
        # (float) The weight of policy entropy loss.
        policy_entropy_weight=0,
        # (float) The weight of ssl (self-supervised learning) loss.
        ssl_loss_weight=0,
        # (float) Weight of direct MSE between recurrent latents and detached encoded future latents.
        # Unlike the projected SSL loss, this constrains the exact planning latent used by MCTS.
        raw_latent_ssl_loss_weight=0.0,
        # (bool) Whether to use piecewise constant learning rate decay.
        # i.e. lr: 0.2 -> 0.02 -> 0.002
        piecewise_decay_lr_scheduler=True,
        # (int) The number of final training iterations to control lr decay, which is only used for manually decay.
        threshold_training_steps_for_final_lr=int(5e4),
        # (bool) Whether to use manually decayed temperature.
        manual_temperature_decay=False,
        # (int) The number of final training iterations to control temperature, which is only used for manually decay.
        threshold_training_steps_for_final_temperature=int(1e5),
        # (float) The fixed temperature value for MCTS action selection, which is used to control the exploration.
        # The larger the value, the more exploration. This value is only used when manual_temperature_decay=False.
        fixed_temperature_value=0.25,
        # (bool) Whether to use the true chance in MCTS in some environments with stochastic dynamics, such as 2048.
        use_ture_chance_label_in_chance_encoder=False,
        # (bool) Whether to add noise to roots during reanalyze process.
        reanalyze_noise=True,
        # (bool) Whether to reuse the root value between batch searches.
        reuse_search=False,
        # (bool) whether to use the pure policy to collect data. If False, use the MCTS guided with policy.
        collect_with_pure_policy=False,

        # ****** Priority ******
        # (bool) Whether to use priority when sampling training data from the buffer.
        use_priority=False,
        # (float) The degree of prioritization to use. A value of 0 means no prioritization,
        # while a value of 1 means full prioritization.
        priority_prob_alpha=0.6,
        # (float) The degree of correction to use. A value of 0 means no correction,
        # while a value of 1 means full correction.
        priority_prob_beta=0.4,
        # (str) Initial PER priority rule for newly collected trajectories.
        # 'bellman_max' mimics EfficientZero V2 insertion priority: compute collection-time
        # n-step value Bellman errors from real rewards, then insert the whole trajectory with
        # max(existing replay priority, max trajectory error). Use 'pred_vs_search' for the
        # older LightZero behavior based on |V_collection - MCTS_search_value|.
        priority_initial_mode='bellman_max',
        priority_new_data_use_max=True,

        # ****** UCB ******
        # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of search tree.
        root_dirichlet_alpha=0.3,
        # (float) The noise weight at the root node of the search tree.
        root_noise_weight=0.25,

        # ****** Explore by random collect ******
        # (int) The number of episodes to collect data randomly before training.
        random_collect_episode_num=0,

        # ****** Explore by eps greedy ******
        eps=dict(
            # (bool) Whether to use eps greedy exploration in collecting data.
            eps_greedy_exploration_in_collect=False,
            # (str) The type of decaying epsilon. Options are 'linear', 'exp'.
            type='linear',
            # (float) The start value of eps.
            start=1.,
            # (float) The end value of eps.
            end=0.05,
            # (int) The decay steps from start to end eps.
            decay=int(1e5),
        ),
    )

    def default_model(self) -> Tuple[str, List[str]]:
        """
        Overview:
            Return this algorithm default model setting for demonstration.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): model name and model import_names.
                - model_type (:obj:`str`): The model type used in this algorithm, which is registered in ModelRegistry.
                - import_names (:obj:`List[str]`): The model class path list used in this algorithm.
        .. note::
            The user can define and use customized network model but must obey the same interface definition indicated \
            by import_names path. For MuZero, ``lzero.model.muzero_model.MuZeroModel``
        """
        if self._cfg.model.model_type == "conv":
            return 'MuZeroModel', ['lzero.model.muzero_model']
        elif self._cfg.model.model_type == "mlp":
            return 'MuZeroModelMLP', ['lzero.model.muzero_model_mlp']
        elif self._cfg.model.model_type == "conv_context":
            return 'MuZeroContextModel', ['lzero.model.muzero_context_model']
        else:
            raise ValueError("model type {} is not supported".format(self._cfg.model.model_type))

    def set_train_iter_env_step(self, train_iter, env_step) -> None:
        """
        Overview:
            Set the train_iter and env_step for the policy.
        Arguments:
            - train_iter (:obj:`int`): The train_iter for the policy.
            - env_step (:obj:`int`): The env_step for the policy.
        """
        self.train_iter = train_iter
        self.env_step = env_step

    def _init_learn(self) -> None:
        """
        Overview:
            Learn mode init method. Called by ``self.__init__``. Initialize the learn model, optimizer and MCTS utils.
        """
        assert self._cfg.optim_type in ['SGD', 'Adam', 'AdamW'], self._cfg.optim_type
        representation_prediction_weight_decay = self._cfg.get(
            'representation_prediction_weight_decay', None
        )
        if representation_prediction_weight_decay is None:
            optimizer_parameters = self._model.parameters()
        else:
            representation_prediction_params, other_params = _representation_prediction_parameter_groups(self._model)
            if not representation_prediction_params:
                raise ValueError(
                    'representation_prediction_weight_decay was set, but no representation_network or '
                    'prediction_network parameters were found.'
                )
            optimizer_parameters = []
            if other_params:
                optimizer_parameters.append(
                    {'params': other_params, 'weight_decay': float(self._cfg.weight_decay)}
                )
            optimizer_parameters.append(
                {
                    'params': representation_prediction_params,
                    'weight_decay': float(representation_prediction_weight_decay),
                }
            )
        # NOTE: in board_games, for fixed lr 0.003, 'Adam' is better than 'SGD'.
        if self._cfg.optim_type == 'SGD':
            self._optimizer = optim.SGD(
                optimizer_parameters,
                lr=self._cfg.learning_rate,
                momentum=self._cfg.momentum,
                weight_decay=0.0 if representation_prediction_weight_decay is not None else self._cfg.weight_decay,
            )
        elif self._cfg.optim_type == 'Adam':
            self._optimizer = optim.Adam(
                optimizer_parameters,
                lr=self._cfg.learning_rate,
                weight_decay=0.0 if representation_prediction_weight_decay is not None else self._cfg.weight_decay,
            )
        elif self._cfg.optim_type == 'AdamW':
            if representation_prediction_weight_decay is None:
                self._optimizer = configure_optimizers(
                    model=self._model,
                    weight_decay=self._cfg.weight_decay,
                    learning_rate=self._cfg.learning_rate,
                    device_type=self._cfg.device,
                )
            else:
                self._optimizer = optim.AdamW(optimizer_parameters, lr=self._cfg.learning_rate)

        if self._cfg.piecewise_decay_lr_scheduler:
            from torch.optim.lr_scheduler import LambdaLR
            max_step = self._cfg.threshold_training_steps_for_final_lr
            # NOTE: the 1, 0.1, 0.01 is the decay rate, not the lr.
            lr_lambda = lambda step: 1 if step < max_step * 0.5 else (0.1 if step < max_step else 0.01)  # noqa
            self.lr_scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        # use model_wrapper for specialized demands of different modes
        self._target_model = copy.deepcopy(self._model)
        self._target_model = model_wrap(
            self._target_model,
            wrapper_name='target',
            update_type='assign',
            update_kwargs={'freq': self._cfg.target_update_freq}
        )
        self._learn_model = self._model

        if self._cfg.use_augmentation:
            self.image_transforms = ImageTransforms(
                self._cfg.augmentation,
                image_shape=(self._cfg.model.observation_shape[1], self._cfg.model.observation_shape[2])
            )
        self.value_support = DiscreteSupport(*self._cfg.model.value_support_range, self._cfg.device)
        self.reward_support = DiscreteSupport(*self._cfg.model.reward_support_range, self._cfg.device)
        assert self.value_support.size == self._learn_model.value_support_size          # if these assertions fails, somebody introduced...
        assert self.reward_support.size == self._learn_model.reward_support_size        # ...incoherence between policy and model
        self.value_inverse_scalar_transform_handle = InverseScalarTransform(self.value_support, self._cfg.model.categorical_distribution)
        self.reward_inverse_scalar_transform_handle = InverseScalarTransform(self.reward_support, self._cfg.model.categorical_distribution)
        
        # ==============================================================
        # harmonydream (learnable weights for different losses)
        # ==============================================================
        if self._cfg.model.harmony_balance:
            # List of parameter names
            harmony_names = ["harmony_dynamics", "harmony_policy", "harmony_value", "harmony_reward", "harmony_entropy"]
            # Initialize and name each parameter
            for name in harmony_names:
                param = torch.nn.Parameter(-torch.log(torch.tensor(1.0)))
                setattr(self, name, param)
            
        if self._cfg.use_rnd_model:
            if self._cfg.target_model_for_intrinsic_reward_update_type == 'assign':
                self._target_model_for_intrinsic_reward = model_wrap(
                    self._target_model,
                    wrapper_name='target',
                    update_type='assign',
                    update_kwargs={'freq': self._cfg.target_update_freq_for_intrinsic_reward}
                )
            elif self._cfg.target_model_for_intrinsic_reward_update_type == 'momentum':
                self._target_model_for_intrinsic_reward = model_wrap(
                    self._target_model,
                    wrapper_name='target',
                    update_type='momentum',
                    update_kwargs={'theta': self._cfg.target_update_theta_for_intrinsic_reward}
                )

        # ========= logging for analysis =========
        self.l2_norm_before = 0.
        self.l2_norm_after = 0.
        self.grad_norm_before = 0.
        self.grad_norm_after = 0.
        self.dormant_ratio_encoder = 0.
        self.dormant_ratio_dynamics = 0.

        if self._cfg.use_wandb:
            # TODO: add the model to wandb
            wandb.watch(self._learn_model.representation_network, log="all")

    def _latent_regularization_group_ids(self, batch_size: int):
        active_member_ids = getattr(self._learn_model, 'active_member_id', None)
        if torch.is_tensor(active_member_ids):
            active_member_ids = active_member_ids.detach().cpu().reshape(-1).tolist()
        if isinstance(active_member_ids, (list, tuple)):
            if len(active_member_ids) != batch_size:
                raise ValueError(
                    f'Expected {batch_size} active member ids for latent regularization, '
                    f'got {len(active_member_ids)}.'
                )
            return list(active_member_ids)
        return None

    def _value_training_prediction_and_loss(
            self,
            value_logits: torch.Tensor,
            latent_state: torch.Tensor,
            scalar_target: torch.Tensor,
            categorical_target: torch.Tensor,
    ):
        base_scalar = self.value_inverse_scalar_transform_handle(value_logits)
        prior_scale = float(self._cfg.model.get('random_prior_scale', 0.0))
        if prior_scale <= 0.0:
            return base_scalar, cross_entropy_loss(value_logits, categorical_target), None
        if not bool(self._cfg.model.get('categorical_distribution', True)):
            raise NotImplementedError(
                'The additive value-prior training loss currently requires categorical value predictions.'
            )
        if not hasattr(self._learn_model, 'value_prior_from_latent'):
            raise RuntimeError(
                'random_prior_scale is enabled, but the model does not expose value_prior_from_latent().'
            )
        scalar_prior = self._learn_model.value_prior_from_latent(latent_state)
        if scalar_prior is None:
            raise RuntimeError('random_prior_scale is enabled, but the model has no value-prior networks.')
        scalar_loss_weight = float(
            self._cfg.model.get('value_prior_scalar_loss_weight', 1.0)
        )
        return _categorical_value_prior_training_terms(
            value_logits,
            scalar_prior,
            scalar_target,
            self.value_support,
            self.value_inverse_scalar_transform_handle,
            scalar_loss_weight,
        )

    def _replay_auxiliary_loss(
            self,
            data,
            obs_batch_ori,
            mask_batch: torch.Tensor,
            weights: torch.Tensor,
            latent_states: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return an optional loss evaluated on the primary MuZero unroll."""
        return latent_states[0].sum() * 0.0, {}

    def _after_replay_auxiliary_backward(self) -> Dict[str, float]:
        """Synchronize or clip parameters owned by a replay auxiliary loss."""
        return {}

    def _forward_learn(self, data: Tuple[torch.Tensor]) -> Dict[str, Union[float, int]]:
        """
        Overview:
            The forward function for learning policy in learn mode, which is the core of the learning process.
            The data is sampled from replay buffer.
            The loss is calculated by the loss function and the loss is backpropagated to update the model.
        Arguments:
            - data (:obj:`Tuple[torch.Tensor]`): The data sampled from replay buffer, which is a tuple of tensors.
                The first tensor is the current_batch, the second tensor is the target_batch.
        Returns:
            - info_dict (:obj:`Dict[str, Union[float, int]]`): The information dict to be logged, which contains \
                current learning loss and learning statistics.
        """
        self._learn_model.train()
        self._target_model.train()
        if self._cfg.use_rnd_model:
            self._target_model_for_intrinsic_reward.train()

        current_batch, target_batch = data
        obs_batch_ori, action_batch, mask_batch, indices, weights, make_time = current_batch
        target_reward, target_value, target_policy = target_batch

        projected_ssl_enabled = (
            bool(self._cfg.model.self_supervised_learning_loss)
            and float(self._cfg.ssl_loss_weight) > 0.0
        )
        raw_latent_ssl_weight = float(self._cfg.get('raw_latent_ssl_loss_weight', 0.0))
        raw_latent_ssl_enabled = raw_latent_ssl_weight > 0.0
        obs_batch, obs_target_batch = prepare_obs(obs_batch_ori, self._cfg)
        if (projected_ssl_enabled or raw_latent_ssl_enabled) and obs_target_batch is None:
            raise ValueError('Future observations are required when a latent SSL loss is enabled.')

        # do augmentations
        if self._cfg.use_augmentation:
            obs_batch = self.image_transforms.transform(obs_batch)
            if projected_ssl_enabled or raw_latent_ssl_enabled:
                obs_target_batch = self.image_transforms.transform(obs_target_batch)

        # shape: (batch_size, num_unroll_steps, action_dim)
        # NOTE: .long() is only  for discrete action space.
        action_batch = torch.from_numpy(action_batch).to(self._cfg.device).unsqueeze(-1).long()
        data_list = [mask_batch, target_reward,
            target_value, target_policy, weights
        ]
        [mask_batch, target_reward, target_value, target_policy,
         weights] = to_torch_float_tensor(data_list, self._cfg.device)

        target_reward = target_reward.view(self._cfg.batch_size, -1)
        target_value = target_value.view(self._cfg.batch_size, -1)

        assert obs_batch.size(0) == self._cfg.batch_size == target_reward.size(0)

        # ``scalar_transform`` to transform the original value to the scaled value,
        # i.e. h(.) function in paper https://arxiv.org/pdf/1805.11593.pdf.
        transformed_target_reward = scalar_transform(target_reward)
        transformed_target_value = scalar_transform(target_value)

        # transform a scalar to its categorical_distribution. After this transformation, each scalar is
        # represented as the linear combination of its two adjacent supports.
        target_reward_categorical = phi_transform(self.reward_support, transformed_target_reward)
        target_value_categorical = phi_transform(self.value_support, transformed_target_value)
        value_prior_training_records = []

        # ==============================================================
        # the core initial_inference in MuZero policy.
        # ==============================================================
        network_output = self._learn_model.initial_inference(obs_batch)

        # value_prefix shape: (batch_size, 10), the ``value_prefix`` at the first step is zero padding.
        latent_state, reward, value, policy_logits = mz_network_output_unpack(network_output)
        replay_latent_states = [latent_state]

        latent_regularization_enabled = (
            float(self._cfg.get('latent_variance_loss_weight', 0.0)) > 0.0
            or float(self._cfg.get('latent_covariance_loss_weight', 0.0)) > 0.0
        )
        latent_regularization_records = {'root': [], 'dynamics': [], 'target': []}
        latent_group_ids = self._latent_regularization_group_ids(int(latent_state.shape[0]))

        def record_latent_regularization(source: str, state: torch.Tensor, valid_mask=None) -> None:
            if not latent_regularization_enabled:
                return
            variance_loss, covariance_loss, stats = _latent_variance_covariance_loss(
                state,
                group_ids=latent_group_ids,
                valid_mask=valid_mask,
                variance_target=float(self._cfg.get('latent_variance_target', 1.0)),
                eps=float(self._cfg.get('latent_regularization_eps', 1e-4)),
            )
            if stats['group_count'] > 0:
                latent_regularization_records[source].append((variance_loss, covariance_loss, stats))

        record_latent_regularization('root', latent_state)

        # ========= logging for analysis =========
        # calculate dormant ratio of encoder
        if self._cfg.calculate_dormant_ratio:
            self.dormant_ratio_encoder = calculate_dormant_ratio(self._learn_model.representation_network, obs_batch.detach(),
                                                           percentage=self._cfg.dormant_threshold)
        # calculate L2 norm of latent state
        latent_state_l2_norms = torch.norm(latent_state.view(latent_state.shape[0], -1), p=2, dim=1).mean()
        # ========= logging for analysis ===============

        original_value, value_loss, value_prior_record = self._value_training_prediction_and_loss(
            value,
            latent_state,
            target_value[:, 0],
            target_value_categorical[:, 0],
        )
        if value_prior_record is not None:
            value_prior_training_records.append(value_prior_record)
        root_predicted_value = original_value.squeeze(-1).detach().cpu().numpy()

        # Note: The following lines are just for debugging.
        predicted_rewards = []
        if self._cfg.monitor_extra_statistics:
            predicted_values, predicted_policies = original_value.detach().cpu(), torch.softmax(
                policy_logits, dim=1
            ).detach().cpu()

        # calculate the new priorities for each transition.
        value_priority = L1Loss(reduction='none')(original_value.squeeze(-1), target_value[:, 0])
        value_priority = value_priority.data.cpu().numpy() + 1e-6

        # ==============================================================
        # calculate policy and value loss for the first step.
        # ==============================================================
        policy_loss = cross_entropy_loss(policy_logits, target_policy[:, 0])

        prob = torch.softmax(policy_logits, dim=-1)
        entropy = -(prob * prob.log()).sum(-1)
        policy_entropy_loss = -entropy

        reward_loss = torch.zeros(self._cfg.batch_size, device=self._cfg.device)
        consistency_loss = torch.zeros(self._cfg.batch_size, device=self._cfg.device)
        raw_latent_consistency_loss = torch.zeros(self._cfg.batch_size, device=self._cfg.device)
        raw_latent_consistency_records = []
        target_policy_entropy = 0

        # ==============================================================
        # the core recurrent_inference in MuZero policy.
        # ==============================================================
        for step_k in range(self._cfg.num_unroll_steps):
            # unroll with the dynamics function: predict the next ``latent_state``, ``reward``,
            # given current ``latent_state`` and ``action``.
            # And then predict policy_logits and value with the prediction function.
            network_output = self._learn_model.recurrent_inference(latent_state, action_batch[:, step_k])
            latent_state, reward, value, policy_logits = mz_network_output_unpack(network_output)
            replay_latent_states.append(latent_state)
            record_latent_regularization('dynamics', latent_state, mask_batch[:, step_k])

            # ========= logging for analysis ===============
            if step_k == self._cfg.num_unroll_steps - 1 and self._cfg.calculate_dormant_ratio:
                # calculate dormant ratio of encoder
                action_tmp = action_batch[:, step_k]
                if len(action_tmp.shape) == 1:
                    action = action.unsqueeze(-1)
                # transform action to one-hot encoding.
                # action_one_hot shape: (batch_size, action_space_size), e.g., (8, 4)
                action_one_hot = torch.zeros(action_tmp.shape[0], policy_logits.shape[-1], device=action_tmp.device)
                # transform action to torch.int64
                action_tmp = action_tmp.long()
                action_one_hot.scatter_(1, action_tmp, 1)
                action_encoding_tmp = action_one_hot.unsqueeze(-1).unsqueeze(-1)
                action_encoding = action_encoding_tmp.expand(
                    latent_state.shape[0], policy_logits.shape[-1], latent_state.shape[2], latent_state.shape[3]
                )
                state_action_encoding = torch.cat((latent_state, action_encoding), dim=1)
                self.dormant_ratio_dynamics = calculate_dormant_ratio(self._learn_model.dynamics_network,
                                                                state_action_encoding.detach(),
                                                                percentage=self._cfg.dormant_threshold)
            # ========= logging for analysis ===============

            original_value, step_value_loss, value_prior_record = self._value_training_prediction_and_loss(
                value,
                latent_state,
                target_value[:, step_k + 1],
                target_value_categorical[:, step_k + 1],
            )
            if value_prior_record is not None:
                value_prior_training_records.append(value_prior_record)

            if projected_ssl_enabled or raw_latent_ssl_enabled:
                # ==============================================================
                # calculate consistency loss for the next ``num_unroll_steps`` unroll steps.
                # ==============================================================
                # Obtain the oracle latent state once for both projected and raw SSL.
                beg_index, end_index = self._get_target_obs_index_in_step_k(step_k)
                target_network_output = self._learn_model.initial_inference(
                    obs_target_batch[:, beg_index:end_index]
                )
                latent_state = to_tensor(latent_state)
                representation_state = to_tensor(target_network_output.latent_state)
                record_latent_regularization('target', representation_state, mask_batch[:, step_k])

                if projected_ssl_enabled:
                    # NOTE: no grad for the representation_state branch
                    dynamic_proj = self._learn_model.project(latent_state, with_grad=True)
                    observation_proj = self._learn_model.project(representation_state, with_grad=False)
                    temp_loss = negative_cosine_similarity(dynamic_proj, observation_proj) * mask_batch[:, step_k]
                    consistency_loss += temp_loss

                if raw_latent_ssl_enabled:
                    raw_terms = _raw_latent_consistency_terms(latent_state, representation_state)
                    step_mask = mask_batch[:, step_k]
                    raw_latent_consistency_loss += raw_terms['loss'] * step_mask
                    raw_record = {
                        key: value.detach() for key, value in raw_terms.items()
                    }
                    raw_record['mask'] = step_mask.detach()
                    raw_latent_consistency_records.append(raw_record)

            # NOTE: the target policy is calculated in game buffer now.
            # ==============================================================
            # calculate policy loss for the next ``num_unroll_steps`` unroll steps.
            # NOTE: the +=.
            # ==============================================================
            policy_loss += cross_entropy_loss(policy_logits, target_policy[:, step_k + 1])

            # Here we take the hypothetical step k = step_k + 1
            prob = torch.softmax(policy_logits, dim=-1)
            entropy = -(prob * prob.log()).sum(-1)
            policy_entropy_loss += -entropy

            target_normalized_visit_count = target_policy[:, step_k + 1]

            # ******* NOTE: target_policy_entropy is only for debug.  ******
            non_masked_indices = torch.nonzero(mask_batch[:, step_k + 1]).squeeze(-1)
            # Check if there are any unmasked rows
            if len(non_masked_indices) > 0:
                target_normalized_visit_count_masked = torch.index_select(
                    target_normalized_visit_count, 0, non_masked_indices
                )
                target_policy_entropy += -((target_normalized_visit_count_masked + 1e-6) * (
                        target_normalized_visit_count_masked + 1e-6).log()).sum(-1).mean()
            else:
                # Set target_policy_entropy to log(|A|) if all rows are masked
                target_policy_entropy += torch.log(torch.tensor(target_normalized_visit_count.shape[-1]))

            value_loss += step_value_loss
            reward_loss += cross_entropy_loss(reward, target_reward_categorical[:, step_k])

            if self._cfg.monitor_extra_statistics:
                original_rewards = self.reward_inverse_scalar_transform_handle(reward)
                original_rewards_cpu = original_rewards.detach().cpu()

                predicted_values = torch.cat(
                    (predicted_values, original_value.detach().cpu())
                )
                predicted_rewards.append(original_rewards_cpu)
                predicted_policies = torch.cat((predicted_policies, torch.softmax(policy_logits, dim=1).detach().cpu()))

        latent_regularization_log = {}
        all_latent_regularization_records = [
            record
            for source_records in latent_regularization_records.values()
            for record in source_records
        ]
        if all_latent_regularization_records:
            latent_variance_loss = torch.stack(
                [record[0] for record in all_latent_regularization_records]
            ).mean()
            latent_covariance_loss = torch.stack(
                [record[1] for record in all_latent_regularization_records]
            ).mean()
            for source, source_records in latent_regularization_records.items():
                if not source_records:
                    continue
                latent_regularization_log.update({
                    f'latent_regularization/{source}_variance_loss': float(
                        torch.stack([record[0].detach() for record in source_records]).mean().item()
                    ),
                    f'latent_regularization/{source}_covariance_loss': float(
                        torch.stack([record[1].detach() for record in source_records]).mean().item()
                    ),
                    f'latent_regularization/{source}_feature_std_mean': float(
                        np.mean([record[2]['feature_std_mean'] for record in source_records])
                    ),
                    f'latent_regularization/{source}_feature_std_min': float(
                        np.mean([record[2]['feature_std_min'] for record in source_records])
                    ),
                    f'latent_regularization/{source}_feature_std_max': float(
                        np.mean([record[2]['feature_std_max'] for record in source_records])
                    ),
                    f'latent_regularization/{source}_covariance_abs_mean': float(
                        np.mean([record[2]['covariance_abs_mean'] for record in source_records])
                    ),
                })
        else:
            latent_variance_loss = latent_state.sum() * 0.0
            latent_covariance_loss = latent_state.sum() * 0.0

        weighted_latent_variance_loss = (
            float(self._cfg.get('latent_variance_loss_weight', 0.0)) * latent_variance_loss
        )
        weighted_latent_covariance_loss = (
            float(self._cfg.get('latent_covariance_loss_weight', 0.0)) * latent_covariance_loss
        )

        raw_latent_consistency_log = {}
        if raw_latent_consistency_records:
            valid_count = torch.stack([
                record['mask'].sum() for record in raw_latent_consistency_records
            ]).sum()
            valid_denominator = valid_count.clamp_min(1.0)

            def raw_valid_mean(name: str) -> torch.Tensor:
                return torch.stack([
                    (record[name] * record['mask']).sum()
                    for record in raw_latent_consistency_records
                ]).sum() / valid_denominator

            raw_loss_mean = raw_valid_mean('loss')
            raw_latent_consistency_log.update({
                'raw_latent_ssl/loss': float(raw_loss_mean.item()),
                'raw_latent_ssl/weighted_loss': float(
                    (raw_latent_ssl_weight * raw_loss_mean).item()
                ),
                'raw_latent_ssl/cosine': float(raw_valid_mean('cosine').item()),
                'raw_latent_ssl/pred_l2_mean': float(raw_valid_mean('pred_l2').item()),
                'raw_latent_ssl/target_l2_mean': float(raw_valid_mean('target_l2').item()),
                'raw_latent_ssl/valid_step_fraction': float(
                    valid_count.item()
                    / max(1, self._cfg.batch_size * self._cfg.num_unroll_steps)
                ),
            })
            for depth, record in enumerate(raw_latent_consistency_records, start=1):
                depth_count = record['mask'].sum().clamp_min(1.0)
                raw_latent_consistency_log[f'raw_latent_ssl/depth_{depth}_loss'] = float(
                    ((record['loss'] * record['mask']).sum() / depth_count).item()
                )
                raw_latent_consistency_log[f'raw_latent_ssl/depth_{depth}_cosine'] = float(
                    ((record['cosine'] * record['mask']).sum() / depth_count).item()
                )

        # ==============================================================
        # the core learn model update step.
        # ==============================================================
        # weighted loss with masks (some invalid states which are out of trajectory.)
        # Nan appear when consistency loss or policy entropy loss uses harmony parameter as coefficient.
        
        # Please refer to https://github.com/thuml/HarmonyDream/blob/main/wmlib-torch/wmlib/agents/dreamerv2.py#L161
        # ["harmony_dynamics", "harmony_policy", "harmony_value", "harmony_reward", "harmony_entropy"]
        if self._cfg.model.harmony_balance:
            loss = (
                  (consistency_loss.mean() * self._cfg.ssl_loss_weight)
                + (raw_latent_consistency_loss.mean() * raw_latent_ssl_weight)
                + (policy_loss.mean() / torch.exp(self.harmony_policy))
                + (value_loss.mean() / torch.exp(self.harmony_value)) 
                + (reward_loss.mean() / torch.exp(self.harmony_reward))
            ) 
            weighted_total_loss = loss.mean()
            weighted_total_loss += (
                torch.log(torch.exp(self.harmony_policy) + 1) +
                torch.log(torch.exp(self.harmony_value) + 1) + 
                torch.log(torch.exp(self.harmony_reward) + 1) 
            )
        else:  
            loss = (
                    self._cfg.ssl_loss_weight * consistency_loss + self._cfg.policy_loss_weight * policy_loss +
                    self._cfg.value_loss_weight * value_loss + self._cfg.reward_loss_weight * reward_loss +
                    self._cfg.policy_entropy_weight * policy_entropy_loss +
                    raw_latent_ssl_weight * raw_latent_consistency_loss
            )
            weighted_total_loss = (weights * loss).mean()
        weighted_muzero_loss = weighted_total_loss
        weighted_total_loss = (
            weighted_muzero_loss + weighted_latent_variance_loss + weighted_latent_covariance_loss
        )
        auxiliary_replay_loss, auxiliary_replay_log = self._replay_auxiliary_loss(
            data,
            obs_batch_ori,
            mask_batch,
            weights,
            replay_latent_states,
        )
        if auxiliary_replay_loss.dim() != 0:
            raise ValueError(
                "Replay auxiliary loss must be scalar, got shape "
                f"{tuple(auxiliary_replay_loss.shape)}."
            )
        weighted_total_loss = weighted_total_loss + auxiliary_replay_loss

        gradient_scale = 1 / self._cfg.num_unroll_steps
        prediction_gradient_log = {}
        prediction_weight_parameters = []
        if bool(self._cfg.get('prediction_gradient_diagnostics', False)):
            prediction_weight_parameters = [
                parameter
                for name, parameter in self._learn_model.named_parameters()
                if name.endswith('prediction_network.fc_prediction_common.0.weight')
            ]
            if prediction_weight_parameters:
                if self._cfg.model.harmony_balance:
                    value_objective = value_loss.mean() / torch.exp(self.harmony_value)
                    policy_objective = policy_loss.mean() / torch.exp(self.harmony_policy)
                else:
                    value_objective = (
                        weights * self._cfg.value_loss_weight * value_loss
                    ).mean()
                    policy_objective = (
                        weights * self._cfg.policy_loss_weight * policy_loss
                    ).mean()
                value_gradients = torch.autograd.grad(
                    value_objective,
                    prediction_weight_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                policy_gradients = torch.autograd.grad(
                    policy_objective,
                    prediction_weight_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                value_gradient_norms = [
                    0.0 if gradient is None else float((gradient * gradient_scale).norm().item())
                    for gradient in value_gradients
                ]
                policy_gradient_norms = [
                    0.0 if gradient is None else float((gradient * gradient_scale).norm().item())
                    for gradient in policy_gradients
                ]
                prediction_gradient_log['prediction_trunk/value_grad_norm_mean'] = float(
                    np.mean(value_gradient_norms)
                )
                prediction_gradient_log['prediction_trunk/policy_grad_norm_mean'] = float(
                    np.mean(policy_gradient_norms)
                )
                for member_id, gradient_norm in enumerate(value_gradient_norms):
                    prediction_gradient_log[
                        f'prediction_trunk/member_{member_id}_value_grad_norm'
                    ] = float(gradient_norm)
                for member_id, gradient_norm in enumerate(policy_gradient_norms):
                    prediction_gradient_log[
                        f'prediction_trunk/member_{member_id}_policy_grad_norm'
                    ] = float(gradient_norm)

        # Preserve LightZero's historical unroll scaling for MuZero losses, while
        # keeping the explicit latent-regularization coefficients literal.
        weighted_muzero_loss.register_hook(lambda grad: grad * gradient_scale)
        self._optimizer.zero_grad()
        weighted_total_loss.backward()

        # ============= for analysis =============
        if self._cfg.analysis_sim_norm:
            del self.l2_norm_before
            del self.l2_norm_after
            del self.grad_norm_before
            del self.grad_norm_after
            self.l2_norm_before, self.l2_norm_after, self.grad_norm_before, self.grad_norm_after = self._learn_model.encoder_hook.analyze()
            self._target_model.encoder_hook.clear_data()
        # ============= for analysis =============

        if self._cfg.multi_gpu:
            self.sync_gradients(self._learn_model)
        total_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(self._learn_model.parameters(),
                                                                     self._cfg.grad_clip_value)
        auxiliary_gradient_log = self._after_replay_auxiliary_backward()
        self._optimizer.step()
        if prediction_weight_parameters:
            weight_norms = [float(parameter.detach().norm().item()) for parameter in prediction_weight_parameters]
            prediction_gradient_log['prediction_trunk/first_weight_norm_mean'] = float(np.mean(weight_norms))
            for member_id, weight_norm in enumerate(weight_norms):
                prediction_gradient_log[
                    f'prediction_trunk/member_{member_id}_first_weight_norm'
                ] = float(weight_norm)
        if self._cfg.piecewise_decay_lr_scheduler:
            self.lr_scheduler.step()

        # ==============================================================
        # the core target model update step.
        # ==============================================================
        self._target_model.update(self._learn_model.state_dict())
        if self._cfg.use_rnd_model:
            self._target_model_for_intrinsic_reward.update(self._learn_model.state_dict())

        if self._cfg.monitor_extra_statistics:
            predicted_rewards = torch.stack(predicted_rewards).transpose(1, 0).squeeze(-1)
            predicted_rewards = predicted_rewards.reshape(-1).unsqueeze(-1)

        value_prior_training_log = {}
        if value_prior_training_records:
            def record_mean(name: str) -> float:
                values = torch.cat(
                    [record[name].detach().reshape(-1) for record in value_prior_training_records]
                )
                return float(values.mean().item())

            priors = torch.cat(
                [record['prior'].detach().reshape(-1) for record in value_prior_training_records]
            )
            value_prior_training_log = {
                'value_prior/prior_mean': float(priors.mean().item()),
                'value_prior/prior_std': float(priors.std(unbiased=False).item()),
                'value_prior/base_prediction_mean': record_mean('base_scalar'),
                'value_prior/final_prediction_mean': record_mean('final_scalar'),
                'value_prior/adjusted_target_mean': record_mean('adjusted_target'),
                'value_prior/categorical_loss': record_mean('categorical_loss'),
                'value_prior/scalar_consistency_loss': record_mean('scalar_consistency_loss'),
                'value_prior/final_prediction_mae': record_mean('final_abs_error'),
            }

        return_log_dict = {
            'collect_mcts_temperature': self._collect_mcts_temperature,
            'collect_epsilon': self.collect_epsilon,
            'cur_lr': self._optimizer.param_groups[0]['lr'],
            'weighted_total_loss': weighted_total_loss.item(),
            'total_loss': (
                loss.mean() + weighted_latent_variance_loss + weighted_latent_covariance_loss
                + auxiliary_replay_loss
            ).item(),
            'policy_loss': policy_loss.mean().item(),
            'policy_entropy': - policy_entropy_loss.mean().item() / (self._cfg.num_unroll_steps + 1),
            'target_policy_entropy': target_policy_entropy.item() / (self._cfg.num_unroll_steps + 1),
            'reward_loss': reward_loss.mean().item(),
            'value_loss': value_loss.mean().item(),
            'consistency_loss': consistency_loss.mean().item() / self._cfg.num_unroll_steps,
            'target_reward': target_reward.mean().item(),
            'target_value': target_value.mean().item(),
            'transformed_target_reward': transformed_target_reward.mean().item(),
            'transformed_target_value': transformed_target_value.mean().item(),
            'predicted_rewards': predicted_rewards.mean().item(),
            'predicted_values': predicted_values.mean().item(),
            'root_predicted_value': float(np.mean(root_predicted_value)),
            'root_predicted_value_orig': root_predicted_value,
            'total_grad_norm_before_clip': total_grad_norm_before_clip.item(),
            'latent_regularization/variance_loss': float(latent_variance_loss.detach().item()),
            'latent_regularization/covariance_loss': float(latent_covariance_loss.detach().item()),
            'latent_regularization/weighted_variance_loss': float(
                weighted_latent_variance_loss.detach().item()
            ),
            'latent_regularization/weighted_covariance_loss': float(
                weighted_latent_covariance_loss.detach().item()
            ),
            'optimizer/representation_prediction_weight_decay': float(
                self._cfg.weight_decay
                if self._cfg.get('representation_prediction_weight_decay', None) is None
                else self._cfg.representation_prediction_weight_decay
            ),
            # ==============================================================
            # priority related
            # ==============================================================
            'value_priority': value_priority.mean().item(),
            'value_priority_orig': value_priority,  # torch.tensor compatible with ddp settings

            'analysis/dormant_ratio_encoder': self.dormant_ratio_encoder,
            'analysis/dormant_ratio_dynamics': self.dormant_ratio_dynamics,
            'analysis/latent_state_l2_norms': latent_state_l2_norms.item(),
            'analysis/l2_norm_before': self.l2_norm_before,
            'analysis/l2_norm_after': self.l2_norm_after,
            'analysis/grad_norm_before': self.grad_norm_before,
            'analysis/grad_norm_after': self.grad_norm_after,
        }
        return_log_dict.update(latent_regularization_log)
        return_log_dict.update(raw_latent_consistency_log)
        return_log_dict.update(prediction_gradient_log)
        return_log_dict.update(value_prior_training_log)
        return_log_dict.update(auxiliary_replay_log)
        return_log_dict.update(auxiliary_gradient_log)
        
        # ["harmony_dynamics", "harmony_policy", "harmony_value", "harmony_reward", "harmony_entropy"]
        if self._cfg.model.harmony_balance:
            harmony_dict = {
                "harmony_dynamics": self.harmony_dynamics.item(), 
                "harmony_dynamics_exp_recip": (1 / torch.exp(self.harmony_dynamics)).item(),
                "harmony_policy": self.harmony_policy.item(),
                "harmony_policy_exp_recip": (1 / torch.exp(self.harmony_policy)).item(),
                "harmony_value": self.harmony_value.item(),
                "harmony_value_exp_recip": (1 / torch.exp(self.harmony_value)).item(),
                "harmony_reward": self.harmony_reward.item(),
                "harmony_reward_exp_recip": (1 / torch.exp(self.harmony_reward)).item(),
                "harmony_entropy": self.harmony_entropy.item(),
                "harmony_entropy_exp_recip": (1 / torch.exp(self.harmony_entropy)).item(),
            }
            return_log_dict.update(harmony_dict)

        if self._cfg.use_wandb:
            wandb.log({'learner_step/' + k: v for k, v in return_log_dict.items()}, step=self.env_step)
            wandb.log({"learner_iter_vs_env_step": self.train_iter}, step=self.env_step)

        return return_log_dict
    
    def _init_collect(self) -> None:
        """
        Overview:
            Collect mode init method. Called by ``self.__init__``. Initialize the collect model and MCTS utils.
        """
        self._collect_model = self._model
        if self._cfg.mcts_ctree:
            self._mcts_collect = MCTSCtree(self._cfg)
        else:
            self._mcts_collect = MCTSPtree(self._cfg)
        self._collect_mcts_temperature = 1.
        self.collect_epsilon = 0.0
        if self._cfg.model.model_type == 'conv_context':
            self.last_batch_obs = torch.zeros([8, self._cfg.model.observation_shape[0], 64, 64]).to(self._cfg.device)
            self.last_batch_action = [-1 for i in range(8)]

    def _forward_collect(
            self,
            data: torch.Tensor,
            action_mask: list = None,
            temperature: float = 1,
            to_play: List = [-1],
            epsilon: float = 0.25,
            ready_env_id: np.array = None,
            **kwargs,
    ) -> Dict:
        """
        Overview:
            The forward function for collecting data in collect mode. Use model to execute MCTS search.
            Choosing the action through sampling during the collect mode.
        Arguments:
            - data (:obj:`torch.Tensor`): The input data, i.e. the observation.
            - action_mask (:obj:`list`): The action mask, i.e. the action that cannot be selected.
            - temperature (:obj:`float`): The temperature of the policy.
            - to_play (:obj:`int`): The player to play.
            - epsilon (:obj:`float`): The epsilon of the eps greedy exploration.
            - ready_env_id (:obj:`list`): The id of the env that is ready to collect.
        Shape:
            - data (:obj:`torch.Tensor`):
                - For Atari, :math:`(N, C*S, H, W)`, where N is the number of collect_env, C is the number of channels, \
                    S is the number of stacked frames, H is the height of the image, W is the width of the image.
                - For lunarlander, :math:`(N, O)`, where N is the number of collect_env, O is the observation space size.
            - action_mask: :math:`(N, action_space_size)`, where N is the number of collect_env.
            - temperature: :math:`(1, )`.
            - to_play: :math:`(N, 1)`, where N is the number of collect_env.
            - epsilon: :math:`(1, )`.
            - ready_env_id: None
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, the keys including ``action``, ``distributions``, \
                ``visit_count_distribution_entropy``, ``value``, ``pred_value``, ``policy_logits``.
        """
        self._collect_model.eval()
        self._collect_mcts_temperature = temperature
        self.collect_epsilon = epsilon
        active_collect_env_num = data.shape[0]
        if ready_env_id is None:
            ready_env_id = np.arange(active_collect_env_num)
        output = {i: None for i in ready_env_id}
        with torch.no_grad():
            if self._cfg.model.model_type in ["conv", "mlp"]:
                network_output = self._collect_model.initial_inference(data)
            elif self._cfg.model.model_type == "conv_context":
                network_output = self._collect_model.initial_inference(self.last_batch_obs, self.last_batch_action,
                                                                       data)

            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            pred_values = self.value_inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()
            latent_state_roots = latent_state_roots.detach().cpu().numpy()
            policy_logits = policy_logits.detach().cpu().numpy().tolist()

            legal_actions = [np.nonzero(action_mask[j])[0].tolist() for j in range(active_collect_env_num)]
            if not self._cfg.collect_with_pure_policy:
                # the only difference between collect and eval is the dirichlet noise
                noises = [
                    np.random.dirichlet([self._cfg.root_dirichlet_alpha] * int(sum(action_mask[j]))
                                        ).astype(np.float32).tolist() for j in range(active_collect_env_num)
                ]
                if self._cfg.mcts_ctree:
                    # cpp mcts_tree
                    roots = MCTSCtree.roots(active_collect_env_num, legal_actions)
                else:
                    # python mcts_tree
                    roots = MCTSPtree.roots(active_collect_env_num, legal_actions)

                roots.prepare(self._cfg.root_noise_weight, noises, reward_roots, policy_logits, to_play)
                self._mcts_collect.search(roots, self._collect_model, latent_state_roots, to_play)

                # list of list, shape: ``{list: batch_size} -> {list: action_space_size}``
                roots_visit_count_distributions = roots.get_distributions()
                roots_values = roots.get_values()  # shape: {list: batch_size}

                batch_action = []
                for i, env_id in enumerate(ready_env_id):
                    distributions, value = roots_visit_count_distributions[i], roots_values[i]
                    if self._cfg.eps.eps_greedy_exploration_in_collect:
                        # eps greedy collect
                        action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                            distributions, temperature=self._collect_mcts_temperature, deterministic=True
                        )
                        action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]
                        if np.random.rand() < self.collect_epsilon:
                            action = np.random.choice(legal_actions[i])
                    else:
                        # normal collect
                        # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                        # the index within the legal action set, rather than the index in the entire action set.
                        action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                            distributions, temperature=self._collect_mcts_temperature, deterministic=False
                        )
                        # NOTE: Convert the ``action_index_in_legal_action_set`` to the corresponding ``action`` in the entire action set.
                        action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]
                    output[env_id] = {
                        'action': action,
                        'visit_count_distributions': distributions,
                        'visit_count_distribution_entropy': visit_count_distribution_entropy,
                        'searched_value': value,
                        'predicted_value': pred_values[i],
                        'predicted_policy_logits': policy_logits[i],
                    }
                    if self._cfg.model.model_type in ["conv_context"]:
                        batch_action.append(action)

                if self._cfg.model.model_type in ["conv_context"]:
                    self.last_batch_obs = data
                    self.last_batch_action = batch_action
            else:
                for i, env_id in enumerate(ready_env_id):
                    policy_values = torch.softmax(torch.tensor([policy_logits[i][a] for a in legal_actions[i]]),
                                                  dim=0).tolist()
                    policy_values = policy_values / np.sum(policy_values)
                    action_index_in_legal_action_set = np.random.choice(len(legal_actions[i]), p=policy_values)
                    action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]
                    output[env_id] = {
                        'action': action,
                        'searched_value': pred_values[i],
                        'predicted_value': pred_values[i],
                        'predicted_policy_logits': policy_logits[i],
                    }

        return output

    def _get_target_obs_index_in_step_k(self, step):
        """
        Overview:
            Get the begin index and end index of the target obs in step k.
        Arguments:
            - step (:obj:`int`): The current step k.
        Returns:
            - beg_index (:obj:`int`): The begin index of the target obs in step k.
            - end_index (:obj:`int`): The end index of the target obs in step k.
        Examples:
            >>> self._cfg.model.model_type = 'conv'
            >>> self._cfg.model.image_channel = 3
            >>> self._cfg.model.frame_stack_num = 4
            >>> self._get_target_obs_index_in_step_k(0)
            >>> (0, 12)
        """
        if self._cfg.model.model_type in ['conv', 'conv_context']:
            beg_index = self._cfg.model.image_channel * step
            end_index = self._cfg.model.image_channel * (step + self._cfg.model.frame_stack_num)
        elif self._cfg.model.model_type in ['mlp', 'mlp_context']:
            beg_index = self._cfg.model.observation_shape * step
            end_index = self._cfg.model.observation_shape * (step + self._cfg.model.frame_stack_num)
        return beg_index, end_index

    def _init_eval(self) -> None:
        """
        Overview:
            Evaluate mode init method. Called by ``self.__init__``. Initialize the eval model and MCTS utils.
        """
        self._eval_model = self._model
        if self._cfg.mcts_ctree:
            self._mcts_eval = MCTSCtree(self._cfg)
        else:
            self._mcts_eval = MCTSPtree(self._cfg)
        if self._cfg.model.model_type == 'conv_context':
            self.last_batch_obs = torch.zeros([3, self._cfg.model.observation_shape[0], 64, 64]).to(self._cfg.device)
            self.last_batch_action = [-1 for _ in range(3)]
        # elif self._cfg.model.model_type == 'mlp_context':
        #     self.last_batch_obs = torch.zeros([3, self._cfg.model.observation_shape]).to(self._cfg.device)
        #     self.last_batch_action = [-1 for _ in range(3)]

    def _forward_eval(self, data: torch.Tensor, action_mask: list, to_play: List = [-1],
                      ready_env_id: np.array = None, **kwargs) -> Dict:
        """
        Overview:
            The forward function for evaluating the current policy in eval mode. Use model to execute MCTS search.
            Choosing the action with the highest value (argmax) rather than sampling during the eval mode.
        Arguments:
            - data (:obj:`torch.Tensor`): The input data, i.e. the observation.
            - action_mask (:obj:`list`): The action mask, i.e. the action that cannot be selected.
            - to_play (:obj:`int`): The player to play.
            - ready_env_id (:obj:`list`): The id of the env that is ready to collect.
        Shape:
            - data (:obj:`torch.Tensor`):
                - For Atari, :math:`(N, C*S, H, W)`, where N is the number of collect_env, C is the number of channels, \
                    S is the number of stacked frames, H is the height of the image, W is the width of the image.
                - For lunarlander, :math:`(N, O)`, where N is the number of collect_env, O is the observation space size.
            - action_mask: :math:`(N, action_space_size)`, where N is the number of collect_env.
            - to_play: :math:`(N, 1)`, where N is the number of collect_env.
            - ready_env_id: None
        Returns:
            - output (:obj:`Dict[int, Any]`): Dict type data, the keys including ``action``, ``distributions``, \
                ``visit_count_distribution_entropy``, ``value``, ``pred_value``, ``policy_logits``.
        """
        self._eval_model.eval()
        active_eval_env_num = data.shape[0]
        if ready_env_id is None:
            ready_env_id = np.arange(active_eval_env_num)
        output = {i: None for i in ready_env_id}
        with torch.no_grad():
            if self._cfg.model.model_type in ["conv", "mlp"]:
                network_output = self._eval_model.initial_inference(data)
            elif self._cfg.model.model_type == "conv_context":
                network_output = self._eval_model.initial_inference(self.last_batch_obs, self.last_batch_action, data)
            latent_state_roots, reward_roots, pred_values, policy_logits = mz_network_output_unpack(network_output)

            if not self._eval_model.training:
                # if not in training, obtain the scalars of the value/reward
                pred_values = self.value_inverse_scalar_transform_handle(pred_values).detach().cpu().numpy()  # shape（B, 1）
                latent_state_roots = latent_state_roots.detach().cpu().numpy()
                policy_logits = policy_logits.detach().cpu().numpy().tolist()  # list shape（B, A）

            legal_actions = [np.nonzero(action_mask[j])[0].tolist() for j in range(active_eval_env_num)]
            if self._cfg.mcts_ctree:
                # cpp mcts_tree
                roots = MCTSCtree.roots(active_eval_env_num, legal_actions)
            else:
                # python mcts_tree
                roots = MCTSPtree.roots(active_eval_env_num, legal_actions)
            roots.prepare_no_noise(reward_roots, policy_logits, to_play)
            self._mcts_eval.search(roots, self._eval_model, latent_state_roots, to_play)

            # list of list, shape: ``{list: batch_size} -> {list: action_space_size}``
            roots_visit_count_distributions = roots.get_distributions()
            roots_values = roots.get_values()  # shape: {list: batch_size}

            batch_action = []
            for i, env_id in enumerate(ready_env_id):
                distributions, value = roots_visit_count_distributions[i], roots_values[i]
                # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
                # the index within the legal action set, rather than the index in the entire action set.
                #  Setting deterministic=True implies choosing the action with the highest value (argmax) rather than
                # sampling during the evaluation phase.
                action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
                    distributions, temperature=1, deterministic=True
                )
                # NOTE: Convert the ``action_index_in_legal_action_set`` to the corresponding ``action`` in the
                # entire action set.
                action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]

                output[env_id] = {
                    'action': action,
                    'visit_count_distributions': distributions,
                    'visit_count_distribution_entropy': visit_count_distribution_entropy,
                    'searched_value': value,
                    'predicted_value': pred_values[i],
                    'predicted_policy_logits': policy_logits[i],
                }
                if self._cfg.model.model_type in ["conv_context"]:
                    batch_action.append(action)

            if self._cfg.model.model_type in ["conv_context"]:
                self.last_batch_obs = data
                self.last_batch_action = batch_action

        return output

    def _reset_collect(self, data_id: Optional[List[int]] = None, task_id: int = None) -> None:
        """
        Overview:
            Reset the observation and action for the collector environment.
        Arguments:
            - data_id (`Optional[List[int]]`): List of data ids to reset (not used in this implementation).
            - task_id (:obj:`int`): The task id. Default is None, which means MuZero is in the single-task mode.
        """
        if self._cfg.model.model_type in ["conv_context"]:
            self.last_batch_obs = initialize_zeros_batch(
                self._cfg.model.observation_shape,
                self._cfg.collector_env_num,
                self._cfg.device
            )
            self.last_batch_action = [-1 for _ in range(self._cfg.collector_env_num)]

    def _reset_eval(self, data_id: Optional[List[int]] = None, task_id: int = None) -> None:
        """
        Overview:
            Reset the observation and action for the evaluator environment.
        Arguments:
            - data_id (:obj:`Optional[List[int]]`): List of data ids to reset (not used in this implementation).
            - task_id (:obj:`int`): The task id. Default is None, which means MuZero is in the single-task mode.
        """
        if self._cfg.model.model_type in ["conv_context"]:
            self.last_batch_obs = initialize_zeros_batch(
                self._cfg.model.observation_shape,
                self._cfg.evaluator_env_num,
                self._cfg.device
            )
            self.last_batch_action = [-1 for _ in range(self._cfg.evaluator_env_num)]
    
    def _monitor_vars_learn(self) -> List[str]:
        """
        Overview:
            Register the variables to be monitored in learn mode. The registered variables will be logged in
            tensorboard according to the return value ``_forward_learn``.
        """
        return_list = [
            'analysis/dormant_ratio_encoder',
            'analysis/dormant_ratio_dynamics',
            'analysis/latent_state_l2_norms',
            'analysis/l2_norm_before',
            'analysis/l2_norm_after',
            'analysis/grad_norm_before',
            'analysis/grad_norm_after',

            'collect_mcts_temperature',
            'cur_lr',
            'weighted_total_loss',
            'total_loss',
            'policy_loss',
            'policy_entropy',
            'target_policy_entropy',
            'reward_loss',
            'value_loss',
            'consistency_loss',
            'value_priority',
            'target_reward',
            'target_value',
            'predicted_rewards',
            'predicted_values',
            'transformed_target_reward',
            'transformed_target_value',
            'total_grad_norm_before_clip',
        ]
        if (
                float(self._cfg.get('latent_variance_loss_weight', 0.0)) > 0.0
                or float(self._cfg.get('latent_covariance_loss_weight', 0.0)) > 0.0
        ):
            return_list.extend([
                'latent_regularization/variance_loss',
                'latent_regularization/covariance_loss',
                'latent_regularization/weighted_variance_loss',
                'latent_regularization/weighted_covariance_loss',
                'latent_regularization/root_variance_loss',
                'latent_regularization/root_covariance_loss',
                'latent_regularization/root_feature_std_mean',
                'latent_regularization/root_feature_std_min',
                'latent_regularization/root_feature_std_max',
                'latent_regularization/root_covariance_abs_mean',
                'latent_regularization/dynamics_variance_loss',
                'latent_regularization/dynamics_covariance_loss',
                'latent_regularization/dynamics_feature_std_mean',
                'latent_regularization/dynamics_feature_std_min',
                'latent_regularization/dynamics_feature_std_max',
                'latent_regularization/dynamics_covariance_abs_mean',
                'latent_regularization/target_variance_loss',
                'latent_regularization/target_covariance_loss',
                'latent_regularization/target_feature_std_mean',
                'latent_regularization/target_feature_std_min',
                'latent_regularization/target_feature_std_max',
                'latent_regularization/target_covariance_abs_mean',
                'optimizer/representation_prediction_weight_decay',
            ])
        if float(self._cfg.get('raw_latent_ssl_loss_weight', 0.0)) > 0.0:
            return_list.extend([
                'raw_latent_ssl/loss',
                'raw_latent_ssl/weighted_loss',
                'raw_latent_ssl/cosine',
                'raw_latent_ssl/pred_l2_mean',
                'raw_latent_ssl/target_l2_mean',
                'raw_latent_ssl/valid_step_fraction',
            ])
            for depth in range(1, int(self._cfg.num_unroll_steps) + 1):
                return_list.extend([
                    f'raw_latent_ssl/depth_{depth}_loss',
                    f'raw_latent_ssl/depth_{depth}_cosine',
                ])
        # ["harmony_dynamics", "harmony_policy", "harmony_value", "harmony_reward", "harmony_entropy"]
        if self._cfg.model.harmony_balance:
            harmony_list = [
                'harmony_dynamics', 'harmony_dynamics_exp_recip',
                'harmony_policy', 'harmony_policy_exp_recip',
                'harmony_value', 'harmony_value_exp_recip',
                'harmony_reward', 'harmony_reward_exp_recip',
                'harmony_entropy', 'harmony_entropy_exp_recip',
            ]
            return_list.extend(harmony_list)
        if bool(self._cfg.get('prediction_gradient_diagnostics', False)):
            return_list.extend([
                'prediction_trunk/first_weight_norm_mean',
                'prediction_trunk/value_grad_norm_mean',
                'prediction_trunk/policy_grad_norm_mean',
            ])
            ensemble_size = int(self._cfg.model.get('ensemble_size', 1))
            for member_id in range(ensemble_size):
                return_list.extend([
                    f'prediction_trunk/member_{member_id}_first_weight_norm',
                    f'prediction_trunk/member_{member_id}_value_grad_norm',
                    f'prediction_trunk/member_{member_id}_policy_grad_norm',
                ])
        if float(self._cfg.model.get('random_prior_scale', 0.0)) > 0.0:
            return_list.extend([
                'value_prior/prior_mean',
                'value_prior/prior_std',
                'value_prior/base_prediction_mean',
                'value_prior/final_prediction_mean',
                'value_prior/adjusted_target_mean',
                'value_prior/categorical_loss',
                'value_prior/scalar_consistency_loss',
                'value_prior/final_prediction_mae',
            ])
        return return_list

    def _state_dict_learn(self) -> Dict[str, Any]:
        """
        Overview:
            Return the state_dict of learn mode, usually including model, target_model and optimizer.
        Returns:
            - state_dict (:obj:`Dict[str, Any]`): The dict of current policy learn state, for saving and restoring.
        """
        return {
            'model': self._learn_model.state_dict(),
            'target_model': self._target_model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

    def _load_state_dict_learn(self, state_dict: Dict[str, Any]) -> None:
        """
        Overview:
            Load the state_dict variable into policy learn mode.
        Arguments:
            - state_dict (:obj:`Dict[str, Any]`): The dict of policy learn state saved before.
        """
        self._learn_model.load_state_dict(state_dict['model'])
        self._target_model.load_state_dict(state_dict['target_model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])

    def __del__(self):
        cfg = getattr(self, '_cfg', None)
        model_cfg = getattr(cfg, 'model', {})
        if not bool(model_cfg.get('analysis_sim_norm', False)):
            return
        # Remove hooks after training.
        for model_name in ('_collect_model', '_target_model'):
            model = getattr(self, model_name, None)
            encoder_hook = getattr(model, 'encoder_hook', None)
            if encoder_hook is not None:
                encoder_hook.remove_hooks()

    def _process_transition(self, obs, policy_output, timestep):
        # be compatible with DI-engine Policy class
        pass

    def _get_train_sample(self, data):
        # be compatible with DI-engine Policy class
        pass
