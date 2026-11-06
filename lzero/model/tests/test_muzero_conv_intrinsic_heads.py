import torch

from lzero.model.muzero_model import MuZeroModel


def test_convolutional_muzero_intrinsic_heads_follow_mlp_contract():
    model = MuZeroModel(
        observation_shape=(3, 84, 84),
        action_space_size=3,
        num_channels=16,
        num_res_blocks=1,
        downsample=True,
        categorical_distribution=False,
        intrinsic_reward_head_hidden_channels=(32,),
        intrinsic_value_head_hidden_channels=(32,),
        intrinsic_reward_target_transform='log',
        intrinsic_reward_log_reference=0.1,
        last_linear_layer_init_zero=False,
    )
    model.eval()
    observations = torch.rand(2, 3, 84, 84)

    with torch.no_grad():
        latent = model.initial_inference(observations).latent_state
        reward_prediction = model.intrinsic_reward_prediction_from_latent(latent)
        reward = model.intrinsic_reward_from_latent(latent)
        value = model.intrinsic_value_from_latent(latent)

    assert latent.shape == (2, 16, 6, 6)
    assert reward_prediction.shape == (2, 1)
    assert reward.shape == (2, 1)
    assert value.shape == (2, 1)
    assert torch.all(reward >= 0)

    target = torch.tensor([0.01, 0.5])
    transformed = model.transform_intrinsic_reward_target(target)
    reconstructed = model.inverse_transform_intrinsic_reward_prediction(transformed)
    assert torch.allclose(reconstructed, target, atol=1e-6, rtol=1e-6)
