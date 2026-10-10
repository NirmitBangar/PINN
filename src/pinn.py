"""Core models, physics and training utilities for PINN dynamics experiments."""

from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn


@dataclass
class OscillatorConfig:
    m: float = 1.0
    mu: float = 4.0
    k: float = 400.0
    t_end: float = 1.0

    @property
    def delta(self):
        return self.mu / (2 * self.m)

    @property
    def omega0(self):
        return math.sqrt(self.k / self.m)

    @property
    def omega_d(self):
        return math.sqrt(max(self.omega0**2 - self.delta**2, 0.0))


def exact_solution(t, cfg: OscillatorConfig):
    """Closed-form under-damped solution for u(0)=1, u'(0)=0."""
    d, wd = cfg.delta, cfg.omega_d
    return torch.exp(-d*t) * (
        torch.cos(wd*t) + (d/wd) * torch.sin(wd*t)
    )


def oscillator_residual(t, u, m, mu, k):
    """Compute m*u'' + mu*u' + k*u using torch autograd."""
    du = torch.autograd.grad(
        u, t, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    d2u = torch.autograd.grad(
        du, t, torch.ones_like(du), create_graph=True, retain_graph=True
    )[0]
    return m*d2u + mu*du + k*u, du, d2u


class MLP(nn.Module):
    def __init__(self, hidden=64, layers=4):
        super().__init__()
        blocks = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            blocks += [nn.Linear(hidden, hidden), nn.Tanh()]
        blocks += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*blocks)

    def forward(self, t):
        return self.net(t)


class FourierFeatures(nn.Module):
    """Deterministic Fourier features for a 1D time input."""
    def __init__(self, frequencies):
        super().__init__()
        self.register_buffer(
            "frequencies", torch.tensor(frequencies, dtype=torch.float32).view(1, -1)
        )

    def forward(self, t):
        x = 2 * math.pi * t @ self.frequencies
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


class FourierPINN(nn.Module):
    def __init__(self, frequencies=(1, 2, 4, 8, 16, 32), hidden=64, layers=4):
        super().__init__()
        self.features = FourierFeatures(frequencies)
        d = 2 * len(frequencies)
        blocks = [nn.Linear(d, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            blocks += [nn.Linear(hidden, hidden), nn.Tanh()]
        blocks += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*blocks)

    def forward(self, t):
        return self.net(self.features(t))


class PositiveParameter(nn.Module):
    """Train a positive physical parameter through softplus."""
    def __init__(self, initial_value):
        super().__init__()
        inv = math.log(math.exp(initial_value) - 1.0) if initial_value > 1e-4 else -5.0
        self.raw = nn.Parameter(torch.tensor([inv], dtype=torch.float32))

    def forward(self):
        return torch.nn.functional.softplus(self.raw) + 1e-6


def losses_forward(model, t_phys, cfg, lambda_bc=10.0):
    t0 = torch.zeros((1, 1), device=t_phys.device, requires_grad=True)
    u0 = model(t0)
    r, du0, _ = oscillator_residual(t0, u0, cfg.m, cfg.mu, cfg.k)
    # residual evaluated at t_phys
    up = model(t_phys)
    rp, _, _ = oscillator_residual(t_phys, up, cfg.m, cfg.mu, cfg.k)
    bc = (u0 - 1).pow(2).mean() + du0.pow(2).mean()
    phys = rp.pow(2).mean()
    return phys + lambda_bc*bc, {"physics": phys.detach(), "boundary": bc.detach()}


def losses_inverse(model, t_phys, t_obs, u_obs, m, k, mu_parameter,
                   lambda_phys=1.0, lambda_data=100.0, lambda_bc=10.0):
    t0 = torch.zeros((1, 1), requires_grad=True)
    u0 = model(t0)
    r0, du0, _ = oscillator_residual(t0, u0, m, mu_parameter(), k)
    up = model(t_phys)
    r, _, _ = oscillator_residual(t_phys, up, m, mu_parameter(), k)
    pred = model(t_obs)
    bc = (u0 - 1).pow(2).mean() + du0.pow(2).mean()
    phys = r.pow(2).mean()
    data = (pred - u_obs).pow(2).mean()
    total = lambda_phys*phys + lambda_data*data + lambda_bc*bc
    return total, {
        "physics": phys.detach(),
        "data": data.detach(),
        "boundary": bc.detach(),
        "mu": mu_parameter().detach()
    }


def make_observations(cfg, n=30, noise_std=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand((n, 1), generator=g) * cfg.t_end
    u = exact_solution(t, cfg)
    noisy = u + noise_std * torch.randn(u.shape, generator=g)
    return t, noisy


def relative_l2(pred, truth):
    return torch.linalg.vector_norm(pred-truth) / torch.linalg.vector_norm(truth)
