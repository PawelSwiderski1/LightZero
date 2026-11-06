import numpy as np
import torch
from easydict import EasyDict

from lzero.policy.ensemble_muzero import EnsembleMuZeroPolicy, _ImageRND
from lzero.policy.rnd_auxiliary_decoder import SharedRNDAuxiliaryDecoder


def test_extract_convolutional_observation_sequence_uses_sliding_stacks():
    policy = object.__new__(EnsembleMuZeroPolicy)
    policy._cfg = EasyDict(
        device='cpu',
        model=dict(
            model_type='conv',
            observation_shape=(6, 4, 5),
            image_channel=3,
            frame_stack_num=2,
        ),
    )
    frames = np.stack(
        [np.full((3, 4, 5), value, dtype=np.float32) for value in range(7)],
        axis=0,
    )
    replay_observations = frames.reshape(1, 7 * 3, 4, 5)

    sequence = policy._extract_observation_sequence(replay_observations)

    assert sequence.shape == (1, 6, 6, 4, 5)
    assert torch.all(sequence[0, 0, :3] == 0)
    assert torch.all(sequence[0, 0, 3:] == 1)
    assert torch.all(sequence[0, 5, :3] == 5)
    assert torch.all(sequence[0, 5, 3:] == 6)
    rnd_frames = policy._select_rnd_observation_features(sequence)
    assert rnd_frames.shape == (1, 6, 3, 4, 5)
    assert torch.all(rnd_frames[0, 0] == 1)
    assert torch.all(rnd_frames[0, 5] == 6)


def test_image_rnd_preserves_image_shape_and_only_trains_predictor():
    torch.manual_seed(0)
    model = _ImageRND(input_channels=3, hidden_dim=32, feature_dim=16)
    inputs = torch.rand(12, 3, 84, 84)
    target_before = {
        key: value.detach().clone() for key, value in model.target.state_dict().items()
    }
    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=1e-3)
    initial_error = model.error(inputs).mean().item()

    for _ in range(8):
        prediction, target = model(inputs)
        loss = (prediction - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert model.error(inputs).mean().item() < initial_error
    assert all(not parameter.requires_grad for parameter in model.target.parameters())
    for key, value in model.target.state_dict().items():
        assert torch.equal(value, target_before[key])


def test_shared_auxiliary_decoder_trains_all_member_latents():
    torch.manual_seed(1)
    decoder = SharedRNDAuxiliaryDecoder(
        latent_shape=(8, 3, 3),
        hidden_dim=16,
        target_transform='log',
        target_log_reference=0.1,
    )
    latents = torch.randn(4, 5, 3, 8, 3, 3, requires_grad=True)
    target_errors = torch.rand(4, 5) + 0.01
    target_mask = torch.ones(4, 5)
    sample_weights = torch.ones(4)

    loss, terms = decoder.loss(
        latents, target_errors, target_mask, sample_weights
    )
    loss.backward()

    assert loss.ndim == 0
    assert latents.grad is not None
    assert torch.all(latents.grad.reshape(4, 5, 3, -1).norm(dim=-1) > 0)
    assert 'depth_5_prediction_mae_raw' in terms
    assert terms['valid_target_fraction'].item() == 1.0
