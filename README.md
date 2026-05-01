# WGAN-GP vs Score-Based Generative Models (SDE)

Empirical comparison of two generative modeling approaches — Wasserstein GAN with Gradient Penalty (WGAN-GP) and Score-Based Generative Models via Stochastic Differential Equations (SGM/SDE) — on 2D synthetic distributions.

## Context
Final project for the Deep Learning course — M2 ISIFAR (Applied Mathematics for Finance), Université Paris-Cité, 2026.

Joint work with Berat DERIN.

## Methods

### WGAN-GP
- Critic / generator architecture: 3-4 layer MLPs with ReLU
- Wasserstein-1 distance estimation via Kantorovich-Rubinstein duality
- Lipschitz constraint enforced through gradient penalty (λ = 10)
- Adam optimizer with β = (0, 0.9), n_critic = 5

### Score-Based Generative Model
- VP-SDE: dx = -½β(t)x dt + √β(t) dW with linear β schedule
- Denoising score matching with ε-prediction parametrization
- Time embedding via separate MLP with additive modulation
- DDPM-style sampling over 2000 Euler-Maruyama steps

## Datasets
Three 2D distributions (10,000 points each), each designed to expose a known weakness:
- **8 Gaussians** in a circle — tests mode collapse
- **Spiral** (Archimedean, noisy) — tests fine continuous structure
- **Unit circle** — tests sub-manifold behavior (score divergence)

## Key findings

| Dataset | WGAN-GP | SGM | Best |
|---|---|---|---|
| 8 Gaussians (MMD) | 0.0130 | 0.0275 | WGAN-GP |
| Spiral (MMD) | 0.1686 | 0.0668 | SGM |
| Circle (MMD) | 0.0037 | 0.0161 | WGAN-GP |

- **SGM captures fine geometry better** (spiral, sub-manifolds) thanks to incremental denoising
- **WGAN-GP distributes mass more uniformly** across modes but suffers severe mode collapse on the spiral
- **MMD as a single metric is misleading**: contradicts visual inspection on 2/3 datasets
- **SGM is far more stable to train** (monotonic convergence vs noisy adversarial dynamics)
- **WGAN-GP is much faster at generation** (1 forward pass vs 2000 SDE steps)

## Stack
Python, PyTorch, NumPy, Matplotlib

## Files
- `Code.ipynb` — full implementation
- `Raport_Latex.pdf` — detailed methodology and analysis
- `figures/` — generated samples and training curves

## References
- Arjovsky et al. (2017) — Wasserstein GAN
- Gulrajani et al. (2017) — Improved Training of Wasserstein GANs
- Song et al. (2020) — Score-Based Generative Modeling through SDEs
