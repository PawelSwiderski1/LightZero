import pytest
import torch

from lzero.policy.rnd_compact_query import CompactImageRNDLandscapeQuery
from lzero.policy.rnd_landscape_query import MountainCarRNDLandscapeQuery


def _offset_targets(roots: torch.Tensor, target_count: int) -> torch.Tensor:
    offsets = torch.stack(
        [
            torch.linspace(-0.12, 0.12, target_count),
            torch.linspace(-0.025, 0.025, target_count),
        ],
        dim=-1,
    )
    targets = roots.unsqueeze(1) + offsets.unsqueeze(0)
    targets[..., 0].clamp_(-1.2, 0.6)
    targets[..., 1].clamp_(-0.07, 0.07)
    return targets


def _synthetic_latents(states: torch.Tensor, ensemble_size: int = 2) -> torch.Tensor:
    position = states[..., 0]
    velocity = states[..., 1]
    features = torch.stack(
        [
            position,
            velocity,
            position.square(),
            velocity.square(),
            position * velocity,
            torch.sin(position),
            torch.cos(position),
            torch.ones_like(position),
        ],
        dim=-1,
    )
    member_offset = torch.linspace(
        0.0, 0.03, ensemble_size, device=states.device
    ).reshape(*((1,) * (states.dim() - 1)), ensemble_size, 1)
    return features.unsqueeze(-2) + member_offset


def _synthetic_image_latents(
        observations: torch.Tensor,
        ensemble_size: int = 2,
        latent_dim: int = 8,
) -> torch.Tensor:
    flat = observations.flatten(start_dim=-3)
    pooled = torch.stack(
        [
            flat.mean(dim=-1),
            flat.std(dim=-1, unbiased=False),
            flat.amin(dim=-1),
            flat.amax(dim=-1),
        ],
        dim=-1,
    )
    repeats = (latent_dim + pooled.shape[-1] - 1) // pooled.shape[-1]
    features = pooled.repeat_interleave(repeats, dim=-1)[..., :latent_dim]
    offsets = torch.linspace(
        0.0, 0.03, ensemble_size, device=observations.device
    ).reshape(*((1,) * (observations.dim() - 3)), ensemble_size, 1)
    return features.unsqueeze(-2) + offsets


@pytest.mark.unittest
def test_compact_image_query_decodes_images_and_trains_all_context_components():
    torch.manual_seed(23)
    model = CompactImageRNDLandscapeQuery(
        observation_shape=(3, 84, 84),
        latent_dim=8,
        query_code_dim=12,
        reconstruction_loss_weight=0.5,
        query_count=3,
        hidden_dim=16,
        reader_use_context=True,
        ensemble_size=2,
        target_transform='log',
    )
    roots = torch.rand(2, 3, 84, 84)
    imagined_observations = torch.rand(2, 2, 3, 84, 84)
    imagined_latents = _synthetic_image_latents(imagined_observations)
    target_errors = 0.01 + imagined_observations.mean(dim=(2, 3, 4))

    def error_fn(observations):
        return 0.01 + observations.square().mean(dim=(1, 2, 3))

    loss, terms = model.loss_on_observations(
        roots,
        imagined_latents,
        target_errors,
        error_fn=error_fn,
        query_latent_fn=_synthetic_image_latents,
        diversity_weight=0.0,
    )
    loss.backward()

    with torch.no_grad():
        context = model.query(roots, error_fn)
    assert context.query_states.shape == (2, 3, 3, 84, 84)
    assert context.root_query_codes.shape == (2, 12)
    assert context.query_codes.shape == (2, 3, 12)
    assert torch.all(context.query_states >= 0.0)
    assert torch.all(context.query_states <= 1.0)
    assert terms['reconstruction_loss'].item() >= 0.0
    assert model.query_policy[-2].weight.grad is not None
    assert model.observation_encoder_conv[0].weight.grad is not None
    assert model.observation_decoder_conv[-2].weight.grad is not None
    assert model.latent_reader[-1].weight.grad is not None


@pytest.mark.unittest
def test_query_policy_returns_valid_queries_and_positive_latent_predictions():
    torch.manual_seed(3)
    model = MountainCarRNDLandscapeQuery(latent_dim=8, query_count=5, hidden_dim=32)
    roots = torch.tensor([[-0.5, 0.0], [-0.3, 0.02]], dtype=torch.float32)

    def error_fn(states):
        return 0.01 + (states[:, 0] + 0.2).pow(2) + 10.0 * states[:, 1].pow(2)

    context = model.query(roots, error_fn)
    context.query_latents = _synthetic_latents(context.query_states)
    target_states = _offset_targets(roots, target_count=4)
    imagined_latents = _synthetic_latents(target_states)
    predictions = model.predict_latent_error(context, imagined_latents)

    assert context.query_states.shape == (2, 5, 2)
    assert context.query_log_errors.shape == (2, 5, 1)
    assert context.query_latents.shape == (2, 5, 2, 8)
    assert torch.all(context.query_states[..., 0] >= -1.2)
    assert torch.all(context.query_states[..., 0] <= 0.6)
    assert torch.all(context.query_states[..., 1] >= -0.07)
    assert torch.all(context.query_states[..., 1] <= 0.07)
    assert predictions.shape == (2, 4)
    assert torch.all(torch.isfinite(predictions))
    assert torch.all(predictions > 0.0)


@pytest.mark.unittest
def test_latent_only_reader_is_invariant_to_query_context():
    torch.manual_seed(5)
    model = MountainCarRNDLandscapeQuery(
        latent_dim=8,
        query_count=4,
        hidden_dim=16,
        reader_use_context=False,
    )
    roots = torch.tensor([[-0.5, 0.0], [-0.3, 0.02]], dtype=torch.float32)

    def error_fn(states):
        return 0.01 + states[:, 0].square() + states[:, 1].square()

    context = model.query(roots, error_fn)
    context.query_latents = _synthetic_latents(context.query_states)
    target_states = _offset_targets(roots, target_count=3)
    imagined_latents = _synthetic_latents(target_states)
    before = model.predict_latent_member_log_error(context, imagined_latents)

    context.query_log_errors = context.query_log_errors + 7.0
    context.query_latents = torch.randn_like(context.query_latents) * 100.0
    after = model.predict_latent_member_log_error(context, imagined_latents)

    assert torch.equal(before, after)


@pytest.mark.unittest
def test_raw_target_transform_does_not_log_rnd_errors():
    errors = torch.tensor([1e-4, 1e-2, 0.5], dtype=torch.float32)
    raw_model = MountainCarRNDLandscapeQuery(
        latent_dim=8,
        query_count=2,
        hidden_dim=16,
        reader_use_context=False,
        target_transform='raw',
    )
    log_model = MountainCarRNDLandscapeQuery(
        latent_dim=8,
        query_count=2,
        hidden_dim=16,
        reader_use_context=False,
        target_transform='log',
    )

    assert torch.equal(raw_model._transform_errors(errors), errors)
    assert torch.allclose(log_model._transform_errors(errors), torch.log(errors))


@pytest.mark.unittest
def test_ordinary_reader_uses_centered_log_bundle_and_zero_output_init():
    model = MountainCarRNDLandscapeQuery(
        latent_dim=8,
        query_count=2,
        hidden_dim=16,
        reader_use_context=False,
        reader_architecture='ordinary',
        reader_hidden_channels=(32,),
        reader_norm_type='BN',
        reader_last_linear_layer_init_zero=True,
        target_transform='log',
        target_log_reference=1e-2,
        error_epsilon=1e-8,
        max_raw_error=5.0,
    )
    roots = torch.tensor([[-0.5, 0.0], [-0.4, 0.01]], dtype=torch.float32)

    def error_fn(states):
        return 0.01 + states[:, 0].square() + states[:, 1].square()

    context = model.query(roots, error_fn)
    context.query_latents = _synthetic_latents(context.query_states)
    imagined_latents = _synthetic_latents(_offset_targets(roots, target_count=3))
    model.eval()
    transformed_prediction = model.predict_latent_member_target(
        context, imagined_latents
    )
    raw_prediction = model.predict_latent_error(context, imagined_latents)

    assert transformed_prediction.shape == (2, 3, 2)
    assert torch.equal(
        transformed_prediction, torch.zeros_like(transformed_prediction)
    )
    assert torch.allclose(raw_prediction, torch.full_like(raw_prediction, 1e-2))
    assert torch.allclose(
        model._transform_errors(torch.tensor([1e-3, 1e-2, 1e-1])),
        torch.log(torch.tensor([0.1, 1.0, 10.0])),
    )


@pytest.mark.unittest
def test_ordinary_reader_rejects_query_context():
    with pytest.raises(ValueError, match='latent-only'):
        MountainCarRNDLandscapeQuery(
            latent_dim=8,
            reader_architecture='ordinary',
            reader_use_context=True,
        )


@pytest.mark.unittest
def test_normalized_raw_target_uses_fixed_scale_and_clips():
    model = MountainCarRNDLandscapeQuery(
        latent_dim=8,
        query_count=2,
        hidden_dim=16,
        reader_use_context=False,
        target_transform='normalized_raw',
        normalized_raw_clip=10.0,
    )
    model.set_target_scale(0.002)
    errors = torch.tensor([0.001, 0.004, 0.03], dtype=torch.float32)

    transformed = model._transform_errors(errors)
    recovered = model._inverse_transform_errors(transformed)

    assert torch.allclose(transformed, torch.tensor([0.5, 2.0, 10.0]))
    assert torch.allclose(recovered, torch.tensor([0.001, 0.004, 0.02]))
    assert model.state_dict()['target_scale'].item() == pytest.approx(0.002)

    with pytest.raises(ValueError):
        model.set_target_scale(0.0)


@pytest.mark.unittest
def test_member_specific_reader_parameters_only_change_their_member_prediction():
    torch.manual_seed(9)
    model = MountainCarRNDLandscapeQuery(
        latent_dim=8,
        query_count=3,
        hidden_dim=16,
        reader_use_context=False,
        ensemble_size=2,
        member_specific_reader=True,
    )
    roots = torch.tensor([[-0.5, 0.0], [-0.3, 0.02]], dtype=torch.float32)

    def error_fn(states):
        return 0.01 + states[:, 0].square() + states[:, 1].square()

    context = model.query(roots, error_fn)
    context.query_latents = _synthetic_latents(context.query_states)
    target_states = _offset_targets(roots, target_count=3)
    imagined_latents = _synthetic_latents(target_states)
    before = model.predict_latent_member_target(context, imagined_latents)

    with torch.no_grad():
        model.latent_readers[0][-1].bias.add_(1.0)
    after = model.predict_latent_member_target(context, imagined_latents)

    assert before.shape == (2, 3, 2)
    assert not torch.equal(before[..., 0], after[..., 0])
    assert torch.equal(before[..., 1], after[..., 1])


@pytest.mark.unittest
def test_query_reader_learns_rnd_targets_from_imagined_latents():
    torch.manual_seed(11)
    model = MountainCarRNDLandscapeQuery(latent_dim=8, query_count=6, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    positions = torch.linspace(-1.2, 0.6, 25)
    velocities = torch.linspace(-0.07, 0.07, 19)
    p, v = torch.meshgrid(positions, velocities, indexing='ij')
    first = 0.002 + 0.04 * (p + 0.1).pow(2) + 2.0 * (v - 0.02).pow(2)
    second = 0.001 + 0.03 * (p + 0.7).pow(2) + 3.0 * (v + 0.03).pow(2)
    grids = torch.stack([first, second, first, second], dim=0)
    roots = torch.tensor(
        [[-0.55, 0.0], [-0.4, 0.01], [-0.7, -0.01], [-0.2, 0.03]],
        dtype=torch.float32,
    )
    target_states = _offset_targets(roots, target_count=8)
    imagined_latents = _synthetic_latents(target_states)
    first_targets = (
        0.002
        + 0.04 * (target_states[..., 0] + 0.1).pow(2)
        + 2.0 * (target_states[..., 1] - 0.02).pow(2)
    )
    second_targets = (
        0.001
        + 0.03 * (target_states[..., 0] + 0.7).pow(2)
        + 3.0 * (target_states[..., 1] + 0.03).pow(2)
    )
    target_errors = torch.stack(
        [first_targets[0], second_targets[1], first_targets[2], second_targets[3]],
        dim=0,
    )

    with torch.no_grad():
        before, _ = model.loss_on_latents(
            grids, roots, imagined_latents, target_errors, _synthetic_latents,
            diversity_weight=0.0,
        )
    for _ in range(250):
        loss, _ = model.loss_on_latents(
            grids, roots, imagined_latents, target_errors, _synthetic_latents,
            diversity_weight=0.0,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        after, terms = model.loss_on_latents(
            grids, roots, imagined_latents, target_errors, _synthetic_latents,
            diversity_weight=0.0,
        )

    assert after.item() < before.item() * 0.4
    assert terms['prediction_mae_log'].item() < 0.2


@pytest.mark.unittest
def test_centered_prediction_loss_is_separate_from_diversity_loss():
    torch.manual_seed(17)
    model = MountainCarRNDLandscapeQuery(latent_dim=8, query_count=4, hidden_dim=16)
    positions = torch.linspace(-1.2, 0.6, 11)
    velocities = torch.linspace(-0.07, 0.07, 9)
    p, v = torch.meshgrid(positions, velocities, indexing='ij')
    grids = (0.001 + 0.03 * (p + 0.4).pow(2) + 2.0 * v.pow(2)).unsqueeze(0)
    roots = torch.tensor([[-0.5, 0.0]], dtype=torch.float32)
    target_states = _offset_targets(roots, target_count=8)
    imagined_latents = _synthetic_latents(target_states)
    target_errors = (
        0.001
        + 0.03 * (target_states[..., 0] + 0.4).pow(2)
        + 2.0 * target_states[..., 1].pow(2)
    )

    loss, terms = model.loss_on_latents(
        grids,
        roots,
        imagined_latents,
        target_errors,
        _synthetic_latents,
        diversity_weight=0.0,
        centered_prediction_weight=2.0,
    )

    expected = terms['prediction_loss'] + 2.0 * terms['centered_prediction_loss']
    assert loss.item() == pytest.approx(expected.item())
    assert torch.isfinite(terms['centered_prediction_correlation'])
    assert terms['query_pair_distance_mean'].item() >= terms['query_pair_distance_min'].item()
    assert terms['query_root_distance_mean'].item() >= 0.0
