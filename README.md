# PETRIC2-PATRIC: Learning Kernels to Modify Updates

This repository contains algorithms developed by the PATRIC team submitted to
the [2025/2026 PETRIC2 reconstruction challenge](https://github.com/SyneRBI/PETRIC2/wiki),
building on the MaGeZ algorithm that won the first [PETRIC challenge](https://www.ccpsynerbi.ac.uk/petric/).

## Authors

- Patrick Fahy, University of Bath, United Kingdom
- Matthias Ehrhardt, University of Bath, United Kingdom
- Mohammad Golbabaee, University of Bristol, United Kingdom
- Zeljko Kereta, University College London, United Kingdom

## Method overview

We start from the MaGeZ preconditioned SVRG algorithm and replace the scalar step size at each iteration with a **learned 3D convolution kernel** applied to the preconditioned gradient. The key idea is that this generalises the scalar step size to a richer spatial operator while keeping the parameter count small (5×5×5 kernels).

### Base algorithm: MaGeZ (preconditioned SVRG)

The base update rule is

$$x_{t+1} = \big[ x_t - \alpha_t P_t \tilde{g}_t \big]_+$$

where $\tilde{g}_t$ is the SVRG gradient estimate, $P_t$ is a diagonal preconditioner based on the harmonic mean of $x / (A^\top \mathbf{1})$ and the inverse diagonal Hessian of the Relative Difference Prior, and $[\cdot]_+$ enforces non-negativity.

For PETRIC2, we additionally apply a Gaussian pre-filter (FWHM = 6mm) to the OSEM warm-start image before beginning the iteration.

### Our contribution: learned convolution kernels

We replace the scalar step size $\alpha_t$ with a learned 3D convolution kernel $K_t$:

$$\Delta x_t = K_t * (P_t \tilde{g}_t)$$

Since convolution is linear in the kernel, the training objective is a **linear least-squares** problem that can be solved efficiently using **Conjugate Gradients (CG)** — no backpropagation, unrolling, or automatic differentiation is required. Each kernel is learned in under 100 CG iterations (seconds, not minutes).

This approach is based on: Fahy, Golbabaee, Ehrhardt. [*Greedy learning to optimize with convergence guarantees*](https://arxiv.org/abs/2406.00260). arXiv:2406.00260, 2024.

### Training

Kernels were trained on a small subset of the available datasets (5 out of 13). Due to the heterogeneity of PET scanner geometries across datasets, the kernels are trained to be **iteration-dependent but not data-adaptive** — the same kernel $K_t$ is applied to all datasets at iteration $t$, relying on the preconditioned gradient $P_t \tilde{g}_t$ to absorb scanner-specific differences.

Learned kernels are used for the **first epoch only** (when the full gradient is computed). Beyond the first epoch, stochastic subset gradients made the learning signal too noisy — learned kernels converged to approximately zero. After the first epoch, the algorithm switches back to standard MaGeZ with a hand-tuned decreasing step size schedule.

## Submitted algorithms

### Submission 1: Whole-object kernel loss

The kernel regression loss is computed over the **whole-object mask**:

$$K_t^* = \arg\min_K \sum_{n=1}^{N} \| x_{t,n} - K * (P_{t,n} \tilde{g}_{t,n}) - x_n^\star \|_{M_{\mathrm{obj},n}}^2$$

### Submission 2: VOI-aware kernel loss

The kernel regression loss uses a **weighted combination** that directly targets the PETRIC2 evaluation metrics. For whole-object and background masks, per-voxel MSE is used (proxy for NRMSE). For each VOI region, the squared error of means is used (proxy for AEM):

$$K_t^* = \arg\min_K \sum_{n=1}^{N} \frac{1}{C_n} \left[ \sum_{j \in \{\mathrm{obj}, \mathrm{bg}\}} \frac{1}{|M_{j,n}|} \sum_{i \in M_{j,n}} (\hat{x}_n^{(i)} - x_n^{\star(i)})^2 + \sum_{k=1}^{K_n} \big(\mathrm{mean}(\hat{x}_n; R_{k,n}) - \mathrm{mean}(x_n^\star; R_{k,n})\big)^2 \right]$$

where $\hat{x}_n = x_{t,n} - K * (P_{t,n} \tilde{g}_{t,n})$.

## Full algorithm

1. **Epoch 1** ($t = 0, \ldots, S-1$): use learned kernels

$$x_{t+1} = \big[ x_t - K_t^* * (P_t \tilde{g}_t) \big]_+$$

2. **Epochs ≥ 2**: standard MaGeZ with $\alpha_t = 1.5$ for $t \le 60$, and $\alpha_t = 1$ otherwise

$$x_{t+1} = \big[ x_t - \alpha_t P_t \tilde{g}_t \big]_+$$

## Acknowledgements

We thank the PETRIC2 organisers for a very interesting challenge. This work builds on the [MaGeZ algorithm](https://arxiv.org/abs/2506.04976) by Ehrhardt, Schramm, and Kereta.
