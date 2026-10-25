import numpy as np
import pytest
import torch

from dgm_mc.market import equicorrelated
from dgm_mc.neural import PriceNetwork, gradient_callback, residual, train
from dgm_mc.provenance import seed_for


torch.set_num_threads(1)


@pytest.mark.parametrize("d", [1, 2, 6, 10])
def test_exact_residual_manufactured_mixed_polynomial(d):
    m = equicorrelated(d, sigma=np.linspace(.1, .3, d))
    t = torch.full((4, 1), .2, dtype=torch.float64, requires_grad=True)
    x = torch.full((4, d), .1, dtype=torch.float64, requires_grad=True)
    model = lambda t, x: t ** 2 + x.sum(dim=1, keepdim=True) ** 2
    actual = residual(model, t, x, m)
    expected = 2 * .2 + 2 * d * .1 * m.drift.sum() + m.covariance.sum() - m.rate * (.2 ** 2 + (d * .1) ** 2)
    assert torch.allclose(actual, torch.full_like(actual, expected), atol=1e-12)


@pytest.mark.parametrize("mode", ["constant", "linear"])
def test_residual_handles_zero_second_derivatives(mode):
    m = equicorrelated(2)
    t = torch.zeros((3, 1), dtype=torch.float64, requires_grad=True)
    x = torch.zeros((3, 2), dtype=torch.float64, requires_grad=True)
    model = (lambda t, x: torch.ones_like(t)) if mode == "constant" else (lambda t, x: x.sum(1, keepdim=True))
    expected = -m.rate if mode == "constant" else m.drift.sum()
    assert torch.allclose(residual(model, t, x, m), torch.full_like(t, expected))


@pytest.mark.parametrize("architecture", ["dgm", "mlp"])
def test_mode_identity_and_deterministic_loss_gradient(architecture):
    torch.manual_seed(7)
    model = PriceNetwork(6, 8, 1, architecture)
    m = equicorrelated(6)
    t = torch.full((3, 1), .4, dtype=torch.float64, requires_grad=True)
    x = torch.randn((3, 6), dtype=torch.float64, requires_grad=True)
    model.train()
    before = model(t, x).detach()
    model.eval()
    assert torch.equal(before, model(t, x).detach())
    losses, grads = [], []
    for _ in range(2):
        model.zero_grad()
        loss = residual(model, t, x, m).square().mean()
        loss.backward()
        losses.append(loss.detach())
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
    assert torch.equal(losses[0], losses[1])
    assert torch.equal(grads[0], grads[1])


def test_fresh_training_reproducible_before_initialization():
    config = dict(steps=3, width=8, depth=1, batch=8, learning_rate=.001, terminal_weight=10, collocation_std=.3)
    first, _ = train(equicorrelated(1), config, 11)
    torch.manual_seed(999)
    second, _ = train(equicorrelated(1), config, 11)
    assert all(torch.equal(first.state_dict()[k], v) for k, v in second.state_dict().items())


def test_callback_gradient_matches_finite_difference():
    torch.manual_seed(9)
    model = PriceNetwork(2, 8, 1)
    x = np.array([[.1, -.1], [.01, -.02]])
    actual = gradient_callback(model)(.3, x)
    eps = 1e-5
    for i in range(2):
        direction = np.zeros_like(x)
        direction[:, i] = eps
        t = torch.full((2, 1), .3, dtype=torch.float64)
        finite = (model(t, torch.tensor(x + direction)) - model(t, torch.tensor(x - direction))).detach().numpy().ravel() / (2 * eps)
        assert np.allclose(actual[:, i], finite, atol=1e-8)


def test_rng_namespaces_do_not_overlap():
    seeds = [seed_for(role, d, s) for role in ["train", "pilot", "eval", "reference"] for d in [1, 5] for s in range(10)]
    assert len(set(seeds)) == len(seeds)
