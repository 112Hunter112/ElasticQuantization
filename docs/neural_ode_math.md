# Neural ODE Temporal Consistency Engine: Mathematical Background

**Authors:** Parth Aditya  
**Last Updated:** January 2026  
**Version:** 1.0

---

## Table of Contents

1. [Introduction](#introduction)
2. [Neural Ordinary Differential Equations](#neural-ordinary-differential-equations)
3. [Latent ODE Framework](#latent-ode-framework)
4. [Variational Inference](#variational-inference)
5. [Temporal Attention Mechanism](#temporal-attention-mechanism)
6. [Continuous Normalizing Flows](#continuous-normalizing-flows)
7. [Uncertainty Quantification](#uncertainty-quantification)
8. [ODE Solver Theory](#ode-solver-theory)
9. [Training Objectives](#training-objectives)
10. [Stiffness Detection](#stiffness-detection)
11. [Computational Complexity](#computational-complexity)
12. [References](#references)

---

## 1. Introduction

This document provides a comprehensive mathematical foundation for the Neural ODE Temporal Consistency Engine implemented in the Aether Auditor system. The engine leverages continuous-time dynamics modeling to predict temporal evolution of embedding vectors and detect anomalies in CDC event streams.

### 1.1 Problem Formulation

Given an irregular time series of observations:

$$\mathcal{D} = \{(\mathbf{x}_i, t_i)\}_{i=1}^N$$

where $\mathbf{x}_i \in \mathbb{R}^d$ is an observation vector and $t_i \in \mathbb{R}^+$ is its timestamp, we seek to:

1. **Model temporal dynamics:** Learn a continuous-time representation of the system's evolution
2. **Predict future states:** Forecast $\mathbf{x}(t)$ for $t > t_N$
3. **Quantify uncertainty:** Estimate both epistemic and aleatoric uncertainty
4. **Detect anomalies:** Identify observations that deviate from expected dynamics

---

## 2. Neural Ordinary Differential Equations

### 2.1 Foundation

A Neural ODE parameterizes the derivative of a hidden state as a neural network:

$$\frac{d\mathbf{z}(t)}{dt} = f_\theta(\mathbf{z}(t), t)$$

where:
- $\mathbf{z}(t) \in \mathbb{R}^{d_z}$ is the latent state at time $t$
- $f_\theta: \mathbb{R}^{d_z} \times \mathbb{R} \to \mathbb{R}^{d_z}$ is a neural network with parameters $\theta$

### 2.2 Initial Value Problem

Given an initial condition $\mathbf{z}(t_0) = \mathbf{z}_0$, the solution at time $t_1$ is:

$$\mathbf{z}(t_1) = \mathbf{z}_0 + \int_{t_0}^{t_1} f_\theta(\mathbf{z}(t), t) \, dt$$

This integral is computed numerically using adaptive ODE solvers (e.g., Dormand-Prince, Runge-Kutta methods).

### 2.3 Adjoint Sensitivity Method

For efficient gradient computation, we use the adjoint sensitivity method. The gradient of a scalar loss $L$ with respect to parameters $\theta$ is:

$$\frac{dL}{d\theta} = -\int_{t_1}^{t_0} \mathbf{a}(t)^T \frac{\partial f_\theta(\mathbf{z}(t), t)}{\partial \theta} \, dt$$

where the adjoint state $\mathbf{a}(t)$ satisfies:

$$\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t)^T \frac{\partial f_\theta(\mathbf{z}(t), t)}{\partial \mathbf{z}(t)}$$

with boundary condition $\mathbf{a}(t_1) = \frac{\partial L}{\partial \mathbf{z}(t_1)}$.

**Computational Advantage:** Memory cost is $O(1)$ instead of $O(T)$ for standard backpropagation through time, where $T$ is the number of solver steps.

---

## 3. Latent ODE Framework

### 3.1 Architecture Overview

Our system consists of three components:

1. **Encoder:** $q_\phi(\mathbf{z}_0 | \mathcal{D})$ - Maps irregular observations to initial latent state
2. **ODE Dynamics:** $\mathbf{z}(t) = \text{ODESolve}(\mathbf{z}_0, f_\theta, t_0, t)$
3. **Decoder:** $p_\psi(\mathbf{x} | \mathbf{z})$ - Maps latent states to observations

### 3.2 Encoder: Irregular Time Series to Latent State

The encoder handles irregular timestamps using a GRU-based approach:

$$\mathbf{h}_i = \text{GRU}([\mathbf{x}_i; \Delta t_i], \mathbf{h}_{i-1})$$

where $\Delta t_i = t_i - t_{i-1}$ is the time difference.

After processing the sequence, we apply temporal attention:

$$\mathbf{c} = \sum_{i=1}^N \alpha_i \mathbf{h}_i$$

where attention weights are:

$$\alpha_i = \frac{\exp(\text{score}(\mathbf{h}_i))}{\sum_{j=1}^N \exp(\text{score}(\mathbf{h}_j))}$$

The latent initial condition is then parameterized as a Gaussian:

$$q_\phi(\mathbf{z}_0 | \mathcal{D}) = \mathcal{N}(\boldsymbol{\mu}_0, \text{diag}(\boldsymbol{\sigma}_0^2))$$

where $\boldsymbol{\mu}_0 = f_\mu(\mathbf{c})$ and $\log \boldsymbol{\sigma}_0^2 = f_\sigma(\mathbf{c})$.

### 3.3 ODE Dynamics Network

The dynamics function $f_\theta$ is implemented as a time-conditioned MLP with residual connections:

$$f_\theta(\mathbf{z}, t) = \text{MLP}([\mathbf{z}; \phi(t)])$$

where $\phi(t)$ is a sinusoidal time embedding:

$$\phi(t) = [\sin(\omega_1 t), \cos(\omega_1 t), \ldots, \sin(\omega_k t), \cos(\omega_k t)]$$

with $\omega_i = 10000^{-2i/d_{emb}}$ following the Transformer positional encoding.

### 3.4 Decoder

The decoder reconstructs observations from latent states:

$$p_\psi(\mathbf{x} | \mathbf{z}) = \mathcal{N}(\mathbf{x}; g_\psi(\mathbf{z}), \sigma_x^2 \mathbf{I})$$

where $g_\psi$ is a multi-layer perceptron.

---

## 4. Variational Inference

### 4.1 Evidence Lower Bound (ELBO)

We maximize the ELBO:

$$\mathcal{L}(\theta, \phi, \psi) = \mathbb{E}_{q_\phi(\mathbf{z}_0|\mathcal{D})} \left[ \sum_{i=1}^N \log p_\psi(\mathbf{x}_i | \mathbf{z}(t_i)) \right] - \text{KL}(q_\phi(\mathbf{z}_0|\mathcal{D}) \| p(\mathbf{z}_0))$$

where:
- First term: Expected log-likelihood (reconstruction)
- Second term: KL divergence to prior (regularization)

### 4.2 Reconstruction Loss

The reconstruction term is:

$$\mathcal{L}_{\text{recon}} = -\frac{1}{2\sigma_x^2} \sum_{i=1}^N \|\mathbf{x}_i - g_\psi(\mathbf{z}(t_i))\|_2^2 - \frac{Nd}{2}\log(2\pi\sigma_x^2)$$

### 4.3 KL Divergence

With a standard Gaussian prior $p(\mathbf{z}_0) = \mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$\text{KL}(q_\phi \| p) = \frac{1}{2} \sum_{j=1}^{d_z} \left( \mu_{0,j}^2 + \sigma_{0,j}^2 - \log \sigma_{0,j}^2 - 1 \right)$$

### 4.4 Reparameterization Trick

To enable gradient-based optimization:

$$\mathbf{z}_0 = \boldsymbol{\mu}_0 + \boldsymbol{\sigma}_0 \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

This allows gradients to flow through the sampling operation.

---

## 5. Temporal Attention Mechanism

### 5.1 Multi-Head Self-Attention

For a sequence of hidden states $\mathbf{H} = [\mathbf{h}_1, \ldots, \mathbf{h}_N] \in \mathbb{R}^{N \times d_h}$:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

where:
- $\mathbf{Q} = \mathbf{H}\mathbf{W}_Q$ (queries)
- $\mathbf{K} = \mathbf{H}\mathbf{W}_K$ (keys)  
- $\mathbf{V} = \mathbf{H}\mathbf{W}_V$ (values)
- $d_k$ is the dimension per head

### 5.2 Multi-Head Extension

With $h$ attention heads:

$$\text{MultiHead}(\mathbf{H}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}_O$$

where:

$$\text{head}_i = \text{Attention}(\mathbf{H}\mathbf{W}_Q^i, \mathbf{H}\mathbf{W}_K^i, \mathbf{H}\mathbf{W}_V^i)$$

### 5.3 Temporal Bias (Optional Extension)

To incorporate temporal information explicitly:

$$\text{score}(i, j) = \frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}} + b(t_i - t_j)$$

where $b(\Delta t)$ is a learned bias function of time differences.

---

## 6. Continuous Normalizing Flows

### 6.1 Change of Variables

For a bijective transformation $\mathbf{z}_1 = T(\mathbf{z}_0)$:

$$p(\mathbf{z}_1) = p(\mathbf{z}_0) \left| \det \frac{\partial T}{\partial \mathbf{z}_0} \right|^{-1}$$

### 6.2 Instantaneous Change of Variables

With Neural ODEs, the log-determinant evolves according to:

$$\frac{d}{dt} \log p(\mathbf{z}(t)) = -\text{Tr}\left( \frac{\partial f_\theta(\mathbf{z}(t), t)}{\partial \mathbf{z}(t)} \right)$$

### 6.3 Hutchinson's Trace Estimator

Computing the trace exactly is $O(d_z^2)$. We use an unbiased stochastic estimator:

$$\text{Tr}(\mathbf{J}) = \mathbb{E}_{\boldsymbol{\epsilon}}\left[ \boldsymbol{\epsilon}^T \mathbf{J} \boldsymbol{\epsilon} \right]$$

where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and $\mathbf{J} = \frac{\partial f_\theta}{\partial \mathbf{z}}$.

In practice:

$$\text{Tr}(\mathbf{J}) \approx \boldsymbol{\epsilon}^T \left( \frac{\partial f_\theta(\mathbf{z}, t)}{\partial \mathbf{z}} \boldsymbol{\epsilon} \right)$$

which is computed via vector-Jacobian products (VJP) in $O(d_z)$ time.

### 6.4 Augmented Dynamics

The full system with log-determinant tracking:

$$\frac{d}{dt} \begin{bmatrix} \mathbf{z}(t) \\ \log p(\mathbf{z}(t)) \end{bmatrix} = \begin{bmatrix} f_\theta(\mathbf{z}(t), t) \\ -\text{Tr}\left( \frac{\partial f_\theta}{\partial \mathbf{z}} \right) \end{bmatrix}$$

---

## 7. Uncertainty Quantification

### 7.1 Sources of Uncertainty

1. **Epistemic Uncertainty:** Model uncertainty due to limited data
2. **Aleatoric Uncertainty:** Inherent stochasticity in the system

### 7.2 Monte Carlo Estimation

For a prediction at time $t^*$, we sample:

$$\mathbf{z}_0^{(i)} \sim q_\phi(\mathbf{z}_0 | \mathcal{D}), \quad i = 1, \ldots, M$$

Solve the ODE for each sample:

$$\mathbf{z}^{(i)}(t^*) = \text{ODESolve}(\mathbf{z}_0^{(i)}, f_\theta, t_0, t^*)$$

Decode predictions:

$$\mathbf{x}^{(i)}(t^*) = g_\psi(\mathbf{z}^{(i)}(t^*))$$

### 7.3 Predictive Distribution

The predictive mean and variance are:

$$\bar{\mathbf{x}}(t^*) = \frac{1}{M} \sum_{i=1}^M \mathbf{x}^{(i)}(t^*)$$

$$\mathbf{\Sigma}(t^*) = \frac{1}{M-1} \sum_{i=1}^M (\mathbf{x}^{(i)}(t^*) - \bar{\mathbf{x}}(t^*))(\mathbf{x}^{(i)}(t^*) - \bar{\mathbf{x}}(t^*))^T$$

### 7.4 Anomaly Score

We define the anomaly score as:

$$A(t^*) = \|\mathbf{x}_{\text{obs}}(t^*) - \bar{\mathbf{x}}(t^*)\|_{\mathbf{\Sigma}(t^*)}$$

where $\|\mathbf{v}\|_\mathbf{\Sigma} = \sqrt{\mathbf{v}^T \mathbf{\Sigma}^{-1} \mathbf{v}}$ is the Mahalanobis distance.

An observation is flagged as anomalous if $A(t^*) > \tau$ for threshold $\tau$.

### 7.5 Uncertainty Decomposition

Total variance can be decomposed:

$$\text{Var}[\mathbf{x}(t^*)] = \underbrace{\mathbb{E}[\text{Var}[\mathbf{x}|\mathbf{z}]]}_{\text{Aleatoric}} + \underbrace{\text{Var}[\mathbb{E}[\mathbf{x}|\mathbf{z}]]}_{\text{Epistemic}}$$

In our model:
- Aleatoric: $\sigma_x^2 \mathbf{I}$ (decoder noise)
- Epistemic: Variance across sampled trajectories

---

## 8. ODE Solver Theory

### 8.1 Runge-Kutta Methods

A general $s$-stage Runge-Kutta method:

$$\mathbf{k}_i = f\left(\mathbf{z}_n + h\sum_{j=1}^s a_{ij}\mathbf{k}_j, t_n + c_i h\right), \quad i = 1, \ldots, s$$

$$\mathbf{z}_{n+1} = \mathbf{z}_n + h \sum_{i=1}^s b_i \mathbf{k}_i$$

where $h$ is the step size.

### 8.2 Dormand-Prince (Dopri5)

A 5th-order method with embedded 4th-order error estimate:

$$\mathbf{z}_{n+1}^{(5)} = \mathbf{z}_n + h \sum_{i=1}^7 b_i \mathbf{k}_i$$

$$\mathbf{z}_{n+1}^{(4)} = \mathbf{z}_n + h \sum_{i=1}^7 b_i^* \mathbf{k}_i$$

Local error estimate:

$$e_{n+1} = \mathbf{z}_{n+1}^{(5)} - \mathbf{z}_{n+1}^{(4)} = h \sum_{i=1}^7 (b_i - b_i^*) \mathbf{k}_i$$

### 8.3 Adaptive Step Size Control

The step size is adjusted to maintain error tolerance:

$$h_{\text{new}} = h_{\text{old}} \left( \frac{\epsilon}{\|e\|} \right)^{1/(q+1)} \cdot \text{safety factor}$$

where:
- $\epsilon$ is the error tolerance
- $q$ is the order of the method
- Safety factor is typically 0.9

### 8.4 Stiff Systems: Implicit Methods

For stiff ODEs (characterized by widely separated timescales), implicit methods are required. We use Kvaerno5, a 5th-order SDIRK (Singly Diagonally Implicit Runge-Kutta) method.

The implicit stage equation:

$$\mathbf{k}_i = f\left(\mathbf{z}_n + h\sum_{j=1}^i a_{ij}\mathbf{k}_j, t_n + c_i h\right)$$

This requires solving nonlinear systems at each stage, typically via Newton iteration.

---

## 9. Training Objectives

### 9.1 Complete Loss Function

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}} + \gamma \cdot \mathcal{L}_{\text{smooth}}$$

where:

$$\mathcal{L}_{\text{recon}} = \frac{1}{N} \sum_{i=1}^N \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|_2^2$$

$$\mathcal{L}_{\text{KL}} = \text{KL}(q_\phi(\mathbf{z}_0|\mathcal{D}) \| p(\mathbf{z}_0))$$

$$\mathcal{L}_{\text{smooth}} = \frac{1}{N-2} \sum_{i=2}^{N-1} \|\mathbf{x}_{i+1} - 2\mathbf{x}_i + \mathbf{x}_{i-1}\|_2^2$$

### 9.2 Hyperparameter Annealing

We use $\beta$-annealing to avoid posterior collapse:

$$\beta(t) = \min\left(1, \frac{t}{T_{\text{anneal}}}\right)$$

where $T_{\text{anneal}}$ is the annealing period (typically 10-20% of total training).

### 9.3 Optimization

We use AdamW optimizer with learning rate schedule:

$$\eta(t) = \eta_{\text{max}} \cdot \min\left(\frac{t}{T_{\text{warmup}}}, \frac{1 + \cos(\pi \cdot (t - T_{\text{warmup}})/(T_{\text{total}} - T_{\text{warmup}}))}{2}\right)$$

This combines:
- Linear warmup for $t < T_{\text{warmup}}$
- Cosine annealing for $t \geq T_{\text{warmup}}$

### 9.4 Gradient Clipping

To prevent gradient explosion in ODE backpropagation:

$$\mathbf{g}_{\text{clipped}} = \min\left(1, \frac{C}{\|\mathbf{g}\|_2}\right) \mathbf{g}$$

where $C = 1.0$ is the clipping threshold.

---

## 10. Stiffness Detection

### 10.1 Stiffness Ratio

A system is considered stiff if the Jacobian has widely separated eigenvalues:

$$\text{Stiffness Ratio} = \frac{\max_i |\lambda_i|}{\min_i |\lambda_i|}$$

where $\lambda_i$ are eigenvalues of $\mathbf{J} = \frac{\partial f_\theta}{\partial \mathbf{z}}$.

### 10.2 Computational Stiffness Detection

Computing all eigenvalues is expensive. We use:

1. **Power iteration** for largest eigenvalue $\lambda_{\text{max}}$
2. **Inverse power iteration** for smallest eigenvalue $\lambda_{\text{min}}$

Power iteration:

$$\mathbf{v}^{(k+1)} = \frac{\mathbf{J}\mathbf{v}^{(k)}}{\|\mathbf{J}\mathbf{v}^{(k)}\|}$$

Converges to eigenvector of $\lambda_{\text{max}}$, with:

$$\lambda_{\text{max}} \approx \frac{\mathbf{v}^{(k)T}\mathbf{J}\mathbf{v}^{(k)}}{\mathbf{v}^{(k)T}\mathbf{v}^{(k)}}$$

### 10.3 Adaptive Solver Selection

$$\text{Solver} = \begin{cases}
\text{Dopri5/Tsit5} & \text{if } \rho < 100 \\
\text{Kvaerno5} & \text{if } \rho \geq 100
\end{cases}$$

where $\rho$ is the stiffness ratio.

---

## 11. Computational Complexity

### 11.1 Forward Pass

| Component | Complexity |
|-----------|-----------|
| Encoder (GRU) | $O(N \cdot d_h^2)$ |
| Temporal Attention | $O(N^2 \cdot d_h)$ |
| ODE Solve | $O(T \cdot d_z \cdot C_f)$ |
| Decoder | $O(d_z \cdot d_x)$ |

where:
- $N$: Sequence length
- $d_h$: Hidden dimension
- $d_z$: Latent dimension
- $d_x$: Observation dimension
- $T$: Number of ODE solver steps
- $C_f$: Cost of evaluating $f_\theta$

### 11.2 Backward Pass (Adjoint Method)

Memory: $O(d_z + d_\theta)$ (constant in $T$)

Time: $O(T \cdot d_z \cdot C_f)$ (same as forward)

### 11.3 Uncertainty Quantification

For $M$ Monte Carlo samples:

Total cost = $M \times \text{(Forward Pass Cost)}$

Typical $M = 50$ for predictions, $M = 10$ for real-time inference.

### 11.4 Practical Considerations

**Input Dimension Scaling:**

| Input Dim | Latent Dim | Attention Heads | Parameters | Inference Time (CPU) |
|-----------|------------|-----------------|------------|---------------------|
| 2 | 16 | 1 | ~18K | 3ms |
| 128 | 64 | 4 | ~245K | 8ms |
| 768 | 128 | 8 | ~1.2M | 15ms |

**GPU Acceleration:**

- 3-5x speedup for large batches ($B > 16$)
- Minimal benefit for single predictions due to overhead

---

## 12. References

### Foundational Papers

1. **Neural Ordinary Differential Equations**  
   Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018).  
   *Advances in Neural Information Processing Systems*, 31.

2. **Latent ODEs for Irregularly-Sampled Time Series**  
   Rubanova, Y., Chen, R. T., & Duvenaud, D. K. (2019).  
   *Advances in Neural Information Processing Systems*, 32.

3. **Augmented Neural ODEs**  
   Dupont, E., Doucet, A., & Teh, Y. W. (2019).  
   *Advances in Neural Information Processing Systems*, 32.

### ODE Solvers

4. **A Family of Embedded Runge-Kutta Formulae**  
   Dormand, J. R., & Prince, P. J. (1980).  
   *Journal of Computational and Applied Mathematics*, 6(1), 19-26.

5. **Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems**  
   Hairer, E., & Wanner, G. (1996).  
   Springer Series in Computational Mathematics.

### Variational Inference

6. **Auto-Encoding Variational Bayes**  
   Kingma, D. P., & Welling, M. (2013).  
   *arXiv preprint arXiv:1312.6114*.

7. **Understanding disentangling in β-VAE**  
   Burgess, C. P., et al. (2018).  
   *arXiv preprint arXiv:1804.03599*.

### Continuous Normalizing Flows

8. **FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models**  
   Grathwohl, W., Chen, R. T., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2018).  
   *arXiv preprint arXiv:1810.01367*.

### Attention Mechanisms

9. **Attention is All You Need**  
   Vaswani, A., et al. (2017).  
   *Advances in Neural Information Processing Systems*, 30.

### Uncertainty Quantification

10. **What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?**  
    Kendall, A., & Gal, Y. (2017).  
    *Advances in Neural Information Processing Systems*, 30.

11. **Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles**  
    Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017).  
    *Advances in Neural Information Processing Systems*, 30.

### Implementation References

12. **JAX: Composable transformations of Python+NumPy programs**  
    Bradbury, J., et al. (2018).  
    http://github.com/google/jax

13. **Equinox: A JAX library for parameterised functions**  
    Kidger, P. (2021).  
    https://github.com/patrick-kidger/equinox

14. **Diffrax: Numerical differential equation solvers in JAX**  
    Kidger, P. (2021).  
    https://github.com/patrick-kidger/diffrax

---

## Appendix A: Notation Table

| Symbol | Description |
|--------|-------------|
| $\mathbf{x}_i$ | Observation vector at time $i$ |
| $t_i$ | Timestamp of observation $i$ |
| $\mathbf{z}(t)$ | Latent state at time $t$ |
| $f_\theta$ | Neural ODE dynamics function |
| $q_\phi$ | Encoder (approximate posterior) |
| $p_\psi$ | Decoder (likelihood) |
| $d_x$ | Observation dimension |
| $d_z$ | Latent dimension |
| $d_h$ | Hidden dimension |
| $\theta, \phi, \psi$ | Neural network parameters |
| $N$ | Number of observations |
| $M$ | Number of Monte Carlo samples |
| $h$ | ODE solver step size |

---

## Appendix B: Derivations

### B.1 KL Divergence for Diagonal Gaussians

Given $q(\mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ and $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$\text{KL}(q \| p) = \int q(\mathbf{z}) \log \frac{q(\mathbf{z})}{p(\mathbf{z})} d\mathbf{z}$$

For diagonal covariance:

$$= \frac{1}{2} \left[ \sum_{j=1}^{d_z} \left( \mu_j^2 + \sigma_j^2 \right) - \log \prod_{j=1}^{d_z} \sigma_j^2 - d_z \right]$$

$$= \frac{1}{2} \sum_{j=1}^{d_z} \left( \mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1 \right)$$

### B.2 Gradient of ELBO

$$\nabla_\theta \mathcal{L} = \nabla_\theta \mathbb{E}_{q_\phi(\mathbf{z}_0)} \left[ \log p_\psi(\mathbf{x}|\mathbf{z}) \right]$$

Using the reparameterization trick $\mathbf{z}_0 = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$:

$$= \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0,\mathbf{I})} \left[ \nabla_\theta \log p_\psi(\mathbf{x}|\mathbf{z}(\boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon})) \right]$$

This gradient can be estimated via Monte Carlo sampling.

### B.3 Vector-Jacobian Product for Trace Estimation

For Hutchinson's estimator:

$$\text{Tr}(\mathbf{J}) = \mathbb{E}_{\boldsymbol{\epsilon}}[\boldsymbol{\epsilon}^T \mathbf{J} \boldsymbol{\epsilon}]$$

In automatic differentiation, we compute:

$$\text{VJP}(f, \mathbf{z}, \boldsymbol{\epsilon}) = \boldsymbol{\epsilon}^T \frac{\partial f(\mathbf{z})}{\partial \mathbf{z}}$$

Then:

$$\text{Tr}(\mathbf{J}) \approx \boldsymbol{\epsilon}^T \cdot \text{VJP}(f, \mathbf{z}, \boldsymbol{\epsilon})$$

---

**Document Version Control:**

- v1.0 (January 2026): Initial release
- Future updates will include empirical validation results and extended case studies

**Feedback:** For corrections or questions, please contact the Aether team or open an issue in the repository.