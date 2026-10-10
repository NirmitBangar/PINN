# PINN Dynamics: Learning Hidden Physics from Sparse Observations

A research-style implementation of Physics-Informed Neural Networks (PINNs) for
forward simulation and inverse system identification of a damped harmonic oscillator.

## Research question

Can a neural network recover the dynamics of a physical system when observations are
sparse and noisy, while the governing differential equation is available as prior knowledge?

We study four questions:

1. Can a vanilla PINN recover the solution of the damped oscillator ODE?
2. Can the PINN identify an unknown damping coefficient from sparse noisy observations?
3. How does parameter recovery degrade as measurement noise increases?
4. Why do standard PINNs become harder to train for high-frequency dynamics, and can
   Fourier features improve convergence?

## Physics

The damped oscillator satisfies

    m u''(t) + mu u'(t) + k u(t) = 0

with

    u(0) = 1,   u'(0) = 0.

For the under-damped regime,

    delta = mu/(2m)
    omega_0 = sqrt(k/m)
    omega_d = sqrt(omega_0^2 - delta^2).

The analytical solution is used only for evaluation, not as training supervision.

## Methods

### Vanilla PINN

A neural network approximates u_theta(t). Automatic differentiation provides u' and u''.
The objective combines:

    L = lambda_phys L_phys + lambda_data L_data + lambda_bc L_bc

where the physics residual is

    r(t) = m u'' + mu u' + k u.

### Inverse PINN

The damping coefficient mu is treated as a trainable physical parameter:

    mu_hat = softplus(rho)

so positivity is guaranteed.

### Fourier-feature PINN

For high-frequency dynamics, the input is transformed using sinusoidal features:

    phi(t) = [sin(2*pi*f_i*t), cos(2*pi*f_i*t)]

before the MLP. This tests whether a richer input representation reduces the
spectral bias of a standard MLP.

## Experiments

- `01_forward.ipynb`: forward solution and analytical benchmark
- `02_inverse.ipynb`: recover unknown damping from sparse observations
- `03_noise_robustness.ipynb`: parameter recovery versus noise level
- `04_high_frequency.ipynb`: vanilla PINN versus Fourier-feature PINN

The core implementation is in `src/pinn.py` and can also be imported into notebooks.

## Reproducibility

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m pytest
```

Then run the notebooks or scripts in `experiments/`.

## What makes this different from a tutorial

The project does not stop at fitting one oscillator. It treats PINNs as a scientific
inference method and evaluates:

- solution error against a closed-form reference,
- physical-parameter recovery,
- robustness to measurement noise,
- failure under increasing frequency,
- an architectural intervention using Fourier features.

## Resume title

**Physics-Informed Neural Networks for Dynamical System Identification**

## Resume-ready bullets after running the experiments

- Built a PyTorch PINN using automatic differentiation to solve damped-oscillator dynamics
  from governing-equation residuals rather than dense solution labels.
- Formulated an inverse PINN that jointly learns the state trajectory and unknown damping
  coefficient from sparse noisy observations, enforcing physical parameter constraints.
- Benchmarked parameter recovery across measurement-noise and oscillation-frequency regimes,
  and tested Fourier-feature inputs to improve PINN convergence on high-frequency dynamics.
