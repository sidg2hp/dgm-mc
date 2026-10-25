"""Smooth DGM/MLP with an exact, deterministic Black--Scholes residual."""
from time import perf_counter

import numpy as np
import torch
from torch import nn


class Gate(nn.Module):
    def __init__(self, inputs, width):
        super().__init__()
        self.input = nn.ModuleList([nn.Linear(inputs, width) for _ in range(4)])
        self.state = nn.ModuleList([nn.Linear(width, width, bias=False) for _ in range(4)])

    def forward(self, z, h):
        carry, update, reset = [torch.tanh(self.input[i](z) + self.state[i](h)) for i in range(3)]
        candidate = torch.tanh(self.input[3](z) + self.state[3](h * reset))
        return (1 - update) * candidate + carry * h


class PriceNetwork(nn.Module):
    def __init__(self, d, width=32, depth=2, architecture="dgm"):
        super().__init__()
        if architecture not in {"dgm", "mlp"} or min(d, width, depth) < 1:
            raise ValueError("Invalid architecture")
        self.architecture = architecture
        self.embedding = nn.Linear(d + 1, width)
        self.layers = nn.ModuleList([
            Gate(d + 1, width) if architecture == "dgm" else nn.Linear(width, width)
            for _ in range(depth)
        ])
        self.output = nn.Linear(width, 1)
        self.double()
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, t, x):
        z = torch.cat([t, x], dim=1)
        h = torch.tanh(self.embedding(z))
        for layer in self.layers:
            h = layer(z, h) if self.architecture == "dgm" else torch.tanh(layer(h))
        return self.output(h)


def derivative(y, x):
    """Handle constant/linear manufactured solutions without fake derivatives."""
    if not y.requires_grad:
        return x * 0
    value = torch.autograd.grad(y.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
    return x * 0 if value is None else value + x * 0


def residual(model, t, x, market):
    u = model(t, x)
    ut, ux = derivative(u, t), derivative(u, x)
    cov = torch.as_tensor(market.covariance, dtype=x.dtype, device=x.device)
    drift = torch.as_tensor(market.drift, dtype=x.dtype, device=x.device)
    trace = torch.zeros_like(u)
    for i in range(market.d):
        row = derivative(ux[:, i:i + 1], x)
        trace = trace + (row * cov[i]).sum(dim=1, keepdim=True)
    return ut + (ux * drift).sum(dim=1, keepdim=True) + 0.5 * trace - market.rate * u


def train(market, config, seed, architecture="dgm"):
    """Fixed-budget fresh training, seeded BEFORE initialization; no test access."""
    torch.manual_seed(seed)
    model = PriceNetwork(market.d, config["width"], config["depth"], architecture)
    rng = torch.Generator().manual_seed(seed + 1_000_000)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    batch = config["batch"]
    if batch < 2 or config["steps"] < 1:
        raise ValueError("Positive training budget required")
    history = []
    start = perf_counter()
    for step in range(config["steps"]):
        # Gaussian collocation has full support; no false truncation boundaries.
        # This is a declared collocation law, not claimed exact importance sampling.
        t = (torch.rand((batch, 1), generator=rng, dtype=torch.float64) * market.maturity).requires_grad_()
        x = (torch.randn((batch, market.d), generator=rng, dtype=torch.float64) * config["collocation_std"]).requires_grad_()
        terminal_x = torch.randn((batch, market.d), generator=rng, dtype=torch.float64) * config["collocation_std"]
        terminal_t = torch.full((batch, 1), market.maturity, dtype=torch.float64)
        target = market.strike * torch.clamp(torch.exp(terminal_x).mean(dim=1, keepdim=True) - 1, min=0)
        pde_loss = residual(model, t, x, market).square().mean()
        terminal_loss = (model(terminal_t, terminal_x) - target).square().mean()
        loss = pde_loss + config["terminal_weight"] * terminal_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % max(1, config["steps"] // 20) == 0:
            history.append(dict(step=step + 1, loss=loss.item(), pde=pde_loss.item(), terminal=terminal_loss.item()))
    elapsed = perf_counter() - start
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, dict(seconds=elapsed, history=history, seed=seed,
                       parameters=sum(p.numel() for p in model.parameters()),
                       architecture=architecture, config=config, market=market.to_dict())


def gradient_callback(model, batch_size=4096):
    """Frozen-model log gradient, with bounded memory and no parameter gradients."""
    def callback(t, x):
        parts = []
        with torch.enable_grad():
            for start in range(0, len(x), batch_size):
                xx = torch.tensor(x[start:start + batch_size], dtype=torch.float64, requires_grad=True)
                tt = torch.full((len(xx), 1), t, dtype=torch.float64)
                value = model(tt, xx)
                parts.append(torch.autograd.grad(value.sum(), xx)[0].detach().numpy())
        return np.concatenate(parts)
    return callback
