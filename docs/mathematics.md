# Mathematical Foundations of GAMLSS

This document provides detailed mathematical derivations for the GAMLSS implementation in `gamlss_rs`. It covers distribution-specific derivatives, special functions, numerical algorithms, and convergence theory.

## Table of Contents

1. [Distribution Derivatives](#1-distribution-derivatives)
   - 1.1 [Gaussian](#11-gaussian-distribution)
   - 1.2 [Student-t](#12-student-t-distribution)
   - 1.3 [Poisson](#13-poisson-distribution)
   - 1.4 [Gamma](#14-gamma-distribution)
   - 1.5 [Negative Binomial](#15-negative-binomial-distribution)
   - 1.6 [Beta](#16-beta-distribution)
   - 1.7 [Binomial](#17-binomial-distribution)
   - 1.8 [Example: Gaussian Derivatives](#18-example-computing-derivatives-for-gaussian-data)
2. [Special Functions](#2-special-functions)
   - 2.1 [Digamma Function](#21-digamma-function)
   - 2.2 [Trigamma Function](#22-trigamma-function)
3. [Numerical Stability](#3-numerical-stability-and-implementation)
4. [GAMLSS Iteration (P-IRLS)](#4-gamlss-iteration-p-irls)
5. [Smoothing and Penalty Matrices](#5-smoothing-and-penalty-matrices)
   - 5.1 [Block-Sparse Penalties](#51-block-sparse-penalty-matrices)
6. [Effective Degrees of Freedom](#6-effective-degrees-of-freedom)
7. [Link Functions](#7-link-functions)
8. [GCV Optimization](#8-gcv-gradient-for-smoothing-parameter-optimization)
   - 8.1 [Cholesky Decomposition](#81-efficient-pwls-solution-via-cholesky-decomposition)
9. [Mean-Variance Relationships](#9-mean-variance-relationships)
10. [Convergence and Initialization](#10-convergence-and-initialization)
11. [Model Selection and Diagnostics](#11-model-selection-and-diagnostics)
12. [References](#references)

---

## 1. Distribution Derivatives

This section provides detailed derivations of score functions (first derivatives of log-likelihood) and Fisher information (expected second derivatives) for all implemented distributions. These quantities drive the penalized iteratively reweighted least squares (P-IRLS) algorithm.

**Key quantities**:
- **Score** $u = \frac{\partial \ell}{\partial \eta}$: Direction of steepest ascent for the log-likelihood
- **Fisher information** $w = I_\eta$: Expected curvature (used as weight in IRLS)
- **Working response** $z = \eta + u/w$: Adjusted target for the weighted least squares problem

For each distribution, we derive these quantities accounting for the link function via the chain rule.

### 1.1 Gaussian Distribution

**Parameterization**: $Y \sim N(\mu, \sigma^2)$

**Parameters**:
- $\mu$ (mean): identity link
- $\sigma$ (standard deviation): log link

**Variance**: $\text{Var}(Y) = \sigma^2$

#### Log-Likelihood

$$
\ell(\mu, \sigma | y) = -\frac{1}{2}\log(2\pi) - \log(\sigma) - \frac{(y - \mu)^2}{2\sigma^2}
$$

#### Derivatives for $\mu$ (identity link)

Score:
$$
\frac{\partial \ell}{\partial \mu} = \frac{y - \mu}{\sigma^2}
$$

Fisher information:
$$
I_\mu = \mathbb{E}\left[-\frac{\partial^2 \ell}{\partial \mu^2}\right] = \frac{1}{\sigma^2}
$$

Since we use identity link ($\eta = \mu$), no chain rule needed:
$$
u_\mu = \frac{y - \mu}{\sigma^2}, \quad w_\mu = \frac{1}{\sigma^2}
$$

#### Derivatives for $\sigma$ (log link)

Let $\eta_\sigma = \log(\sigma)$, so $\sigma = e^{\eta_\sigma}$.

Score with respect to $\sigma$:
$$
\frac{\partial \ell}{\partial \sigma} = -\frac{1}{\sigma} + \frac{(y - \mu)^2}{\sigma^3}
$$

Chain rule for log link:
$$
\frac{\partial \ell}{\partial \eta_\sigma} = \sigma \cdot \frac{\partial \ell}{\partial \sigma} = -1 + \frac{(y - \mu)^2}{\sigma^2} = \frac{(y - \mu)^2 - \sigma^2}{\sigma^2}
$$

Fisher information on $\eta_\sigma$ scale:
$$
I_{\eta_\sigma} = \mathbb{E}\left[-\frac{\partial^2 \ell}{\partial \eta_\sigma^2}\right] = 2
$$

This comes from $\mathbb{E}[(Y-\mu)^2/\sigma^2] = 1$ and $\text{Var}[(Y-\mu)^2/\sigma^2] = 2$ for Gaussian.

**Implementation**:
$$
u_\sigma = \frac{(y - \mu)^2 - \sigma^2}{\sigma^2}, \quad w_\sigma = 2
$$

---

### 1.2 Student-t Distribution

### Log-Likelihood

For observation $y$ with parameters $\mu$ (location), $\sigma$ (scale), and $\nu$ (degrees of freedom):

$$
\ell(\mu, \sigma, \nu | y) = \log \Gamma\left(\frac{\nu+1}{2}\right) - \log \Gamma\left(\frac{\nu}{2}\right) - \frac{1}{2}\log(\nu\pi\sigma^2) - \frac{\nu+1}{2}\log\left(1 + \frac{z^2}{\nu}\right)
$$

where $z = \frac{y - \mu}{\sigma}$.

### First Derivatives (Score Functions)

Define the robustified weight:
$$
w = \frac{\nu + 1}{\nu + z^2}
$$

#### Location parameter $\mu$:
$$
\frac{\partial \ell}{\partial \mu} = \frac{w \cdot z}{\sigma}
$$

#### Scale parameter $\sigma$ (log link):
$$
\frac{\partial \ell}{\partial \log(\sigma)} = w \cdot z^2 - 1
$$

#### Degrees of freedom $\nu$ (log link):
$$
\frac{\partial \ell}{\partial \log(\nu)} = \nu \left[\frac{1}{2}\left(\psi\left(\frac{\nu+1}{2}\right) - \psi\left(\frac{\nu}{2}\right) - \log\left(1 + \frac{z^2}{\nu}\right) + \frac{w \cdot z^2 - 1}{\nu}\right)\right]
$$

where $\psi(x) = \frac{d}{dx}\log\Gamma(x)$ is the digamma function.

### Expected Fisher Information

#### For $\mu$:
$$
I_\mu = \mathbb{E}\left[\frac{w}{\sigma^2}\right] = \frac{\nu + 1}{\sigma^2(\nu + 3)}
$$

In practice, we use the observed information (plug in $w$ directly).

#### For $\sigma$ (log link):
$$
I_{\log(\sigma)} = \frac{2\nu}{\nu + 3}
$$

#### For $\nu$ (log link):
$$
I_{\log(\nu)} = \frac{\nu^2}{4}\left[\psi'\left(\frac{\nu}{2}\right) - \psi'\left(\frac{\nu+1}{2}\right) + \frac{2(\nu+3)}{\nu(\nu+1)}\right]
$$

where $\psi'(x)$ is the trigamma function.

### Numerical Considerations

1. **Minimum $\nu$**: Require $\nu > 2$ to ensure finite variance
2. **Minimum $\sigma$**: Enforce $\sigma \geq 10^{-6}$ to prevent division by zero
3. **Weight clamping**: The denominator $\nu + z^2$ should be $\geq 10^{-10}$
4. **Information positivity**: Ensure $I_{\log(\nu)} \geq 10^{-6}$

---

### 1.3 Poisson Distribution

**Parameterization**: $Y \sim \text{Poisson}(\mu)$

**Parameters**:
- $\mu$ (rate/mean): log link

**Variance**: $\text{Var}(Y) = \mu$ (equidispersion)

#### Log-Likelihood

$$
\ell(\mu | y) = y \log(\mu) - \mu - \log(y!)
$$

#### Derivatives for $\mu$ (log link)

Let $\eta = \log(\mu)$, so $\mu = e^\eta$.

Score with respect to $\mu$:
$$
\frac{\partial \ell}{\partial \mu} = \frac{y}{\mu} - 1 = \frac{y - \mu}{\mu}
$$

Chain rule for log link:
$$
\frac{\partial \ell}{\partial \eta} = \mu \cdot \frac{\partial \ell}{\partial \mu} = y - \mu
$$

Fisher information on $\eta$ scale:
$$
I_\eta = \mathbb{E}\left[-\frac{\partial^2 \ell}{\partial \eta^2}\right] = \mu
$$

**Implementation**:
$$
u_\mu = y - \mu, \quad w_\mu = \mu
$$

---

### 1.4 Gamma Distribution

**Parameterization**: $Y \sim \text{Gamma}(\alpha, \theta)$ where $\alpha = 1/\sigma^2$ (shape), $\theta = \mu\sigma^2$ (scale)

**Parameters**:
- $\mu$ (mean): log link
- $\sigma$ (coefficient of variation): log link

**Variance**: $\text{Var}(Y) = \mu^2 \sigma^2$

**Mean-CV relationship**: $\text{CV}(Y) = \sigma$

#### Log-Likelihood

$$
\ell(\mu, \sigma | y) = -\alpha\log(\theta) - \log\Gamma(\alpha) + (\alpha-1)\log(y) - \frac{y}{\theta}
$$

where $\alpha = 1/\sigma^2$ and $\theta = \mu\sigma^2$.

Simplifying:
$$
\ell = -\frac{1}{\sigma^2}\log(\mu\sigma^2) - \log\Gamma\left(\frac{1}{\sigma^2}\right) + \left(\frac{1}{\sigma^2}-1\right)\log(y) - \frac{y}{\mu\sigma^2}
$$

#### Derivatives for $\mu$ (log link)

Let $\eta_\mu = \log(\mu)$.

Score with respect to $\mu$:
$$
\frac{\partial \ell}{\partial \mu} = -\frac{\alpha}{\mu} + \frac{y}{\mu^2\theta} = -\frac{1}{\mu\sigma^2} + \frac{y}{\mu^2\sigma^2} = \frac{y - \mu}{\mu^2\sigma^2}
$$

Chain rule for log link:
$$
\frac{\partial \ell}{\partial \eta_\mu} = \mu \cdot \frac{\partial \ell}{\partial \mu} = \frac{y - \mu}{\mu\sigma^2}
$$

Fisher information on $\eta_\mu$ scale:
$$
I_{\eta_\mu} = \frac{1}{\sigma^2}
$$

**Implementation**:
$$
u_\mu = \frac{y - \mu}{\mu\sigma^2}, \quad w_\mu = \frac{1}{\sigma^2}
$$

#### Derivatives for $\sigma$ (log link)

Let $\eta_\sigma = \log(\sigma)$.

Score with respect to $\sigma$:
$$
\frac{\partial \ell}{\partial \sigma} = -\frac{2\alpha}{\sigma} + \frac{2\alpha}{\sigma}\psi(\alpha) + \frac{2\alpha}{\sigma}\log(y) - \frac{2\alpha}{\sigma}\log(\theta) + \frac{2y}{\mu\sigma^3}
$$

where $\psi(x) = \frac{d}{dx}\log\Gamma(x)$ is the digamma function.

Chain rule for log link ($\frac{d\sigma}{d\eta_\sigma} = \sigma$):
$$
\frac{\partial \ell}{\partial \eta_\sigma} = \frac{2}{\sigma^2}\left[\psi\left(\frac{1}{\sigma^2}\right) + 2\log(\sigma) - \log\left(\frac{y}{\mu}\right) + \frac{y}{\mu} - 1\right]
$$

Fisher information involves trigamma $\psi'(x) = \frac{d^2}{dx^2}\log\Gamma(x)$:
$$
I_{\eta_\sigma} = \frac{4}{\sigma^4}\psi'\left(\frac{1}{\sigma^2}\right) - \frac{2}{\sigma^2}
$$

**Implementation**:
$$
u_\sigma = \frac{2}{\sigma^2}\left[\psi(\alpha) + 2\log(\sigma) - \log(y/\mu) + y/\mu - 1\right], \quad w_\sigma = \frac{4}{\sigma^4}\psi'(\alpha) - \frac{2}{\sigma^2}
$$

---

### 1.5 Negative Binomial Distribution

**Parameterization**: $Y \sim \text{NB}(r, p)$ with mean $\mu$ and dispersion $\sigma$ (NB2 parameterization)

**Parameters**:
- $\mu$ (mean): log link
- $\sigma$ (overdispersion): log link

**Variance**: $\text{Var}(Y) = \mu + \sigma\mu^2$ (overdispersion relative to Poisson)

**Relationship to standard NB**: $r = 1/\sigma$, $p = 1/(1 + \sigma\mu)$

#### Log-Likelihood

$$
\ell(\mu, \sigma | y) = \log\Gamma(y + r) - \log\Gamma(r) - \log(y!) + r\log(p) + y\log(1-p)
$$

where $r = 1/\sigma$ and $p = 1/(1 + \sigma\mu)$.

Simplifying in terms of $(\mu, \sigma)$:
$$
\ell = \log\Gamma(y + 1/\sigma) - \log\Gamma(1/\sigma) - \log(y!) + \frac{1}{\sigma}\log\left(\frac{1}{1+\sigma\mu}\right) + y\log\left(\frac{\sigma\mu}{1+\sigma\mu}\right)
$$

#### Derivatives for $\mu$ (log link)

Let $\eta_\mu = \log(\mu)$.

Score with respect to $\mu$:
$$
\frac{\partial \ell}{\partial \mu} = \frac{y - \mu}{1 + \sigma\mu}
$$

Chain rule for log link ($\frac{d\mu}{d\eta_\mu} = \mu$):
$$
\frac{\partial \ell}{\partial \eta_\mu} = \mu \cdot \frac{\partial \ell}{\partial \mu} = \frac{\mu(y - \mu)}{1 + \sigma\mu}
$$

However, in IRLS we use the observed Fisher information $w_\mu = \frac{\mu}{1 + \sigma\mu}$, giving working response:
$$
u_\mu = \frac{y - \mu}{1 + \sigma\mu}, \quad w_\mu = \frac{\mu}{1 + \sigma\mu}
$$

This follows from the score being proportional to observed weight, as noted in the full derivation in Rigby & Stasinopoulos (2005).

**Implementation** (for IRLS):
$$
u_\mu = \frac{y - \mu}{1 + \sigma\mu}, \quad w_\mu = \frac{\mu}{1 + \sigma\mu}
$$

#### Derivatives for $\sigma$ (log link)

Let $\eta_\sigma = \log(\sigma)$ and $r = 1/\sigma$.

Score with respect to $\sigma$:
$$
\frac{\partial \ell}{\partial \sigma} = -\frac{1}{\sigma^2}\left[\psi(y + r) - \psi(r) - \log(1+\sigma\mu) + \frac{y - \mu}{1+\sigma\mu}\right]
$$

Chain rule for log link:
$$
\frac{\partial \ell}{\partial \eta_\sigma} = \sigma \cdot \frac{\partial \ell}{\partial \sigma} = -\frac{1}{\sigma}\left[\psi(y + r) - \psi(r) - \log(1+\sigma\mu) + \frac{y - \mu}{1+\sigma\mu}\right]
$$

Fisher information (using approximation):
$$
I_{\eta_\sigma} = \frac{1}{\sigma^2}\psi'(r)
$$

**Implementation**:
$$
u_\sigma = -\frac{1}{\sigma}\left[\psi(y + 1/\sigma) - \psi(1/\sigma) - \log(1+\sigma\mu) + \frac{y - \mu}{1+\sigma\mu}\right], \quad w_\sigma = \frac{\psi'(1/\sigma)}{\sigma^2}
$$

---

### 1.6 Beta Distribution

**Parameterization**: $Y \sim \text{Beta}(\alpha, \beta)$ where $\alpha = \mu\phi$ and $\beta = (1-\mu)\phi$

**Parameters**:
- $\mu$ (mean): logit link
- $\phi$ (precision): log link

**Variance**: $\text{Var}(Y) = \frac{\mu(1-\mu)}{1 + \phi}$

#### Log-Likelihood

$$
\ell(\mu, \phi | y) = \log\Gamma(\phi) - \log\Gamma(\alpha) - \log\Gamma(\beta) + (\alpha-1)\log(y) + (\beta-1)\log(1-y)
$$

where $\alpha = \mu\phi$ and $\beta = (1-\mu)\phi$.

#### Derivatives for $\mu$ (logit link)

Let $\eta_\mu = \text{logit}(\mu) = \log(\mu/(1-\mu))$.

Score with respect to $\mu$:
$$
\frac{\partial \ell}{\partial \mu} = \phi\left[\log(y) - \log(1-y) - \psi(\alpha) + \psi(\beta)\right]
$$

Chain rule for logit link ($\frac{d\mu}{d\eta_\mu} = \mu(1-\mu)$):
$$
\frac{\partial \ell}{\partial \eta_\mu} = \mu(1-\mu) \cdot \frac{\partial \ell}{\partial \mu} = \mu(1-\mu)\phi\left[\log(y) - \log(1-y) - \psi(\alpha) + \psi(\beta)\right]
$$

Fisher information:
$$
I_{\eta_\mu} = [\mu(1-\mu)]^2 \phi^2[\psi'(\alpha) + \psi'(\beta)]
$$

**Implementation**:
$$
u_\mu = \mu(1-\mu)\phi\left[\log(y) - \log(1-y) - \psi(\mu\phi) + \psi((1-\mu)\phi)\right]
$$
$$
w_\mu = [\mu(1-\mu)]^2 \phi^2[\psi'(\mu\phi) + \psi'((1-\mu)\phi)]
$$

#### Derivatives for $\phi$ (log link)

Let $\eta_\phi = \log(\phi)$.

Score with respect to $\phi$:
$$
\frac{\partial \ell}{\partial \phi} = \psi(\phi) - \mu\psi(\alpha) - (1-\mu)\psi(\beta) + \mu\log(y) + (1-\mu)\log(1-y)
$$

Chain rule for log link:
$$
\frac{\partial \ell}{\partial \eta_\phi} = \phi \cdot \frac{\partial \ell}{\partial \phi}
$$

Fisher information:
$$
I_{\eta_\phi} = \phi^2\left[\psi'(\phi) - \mu^2\psi'(\alpha) - (1-\mu)^2\psi'(\beta)\right]
$$

**Implementation**:
$$
u_\phi = \phi\left[\psi(\phi) - \mu\psi(\mu\phi) - (1-\mu)\psi((1-\mu)\phi) + \mu\log(y) + (1-\mu)\log(1-y)\right]
$$
$$
w_\phi = \phi^2\left[\psi'(\phi) - \mu^2\psi'(\mu\phi) - (1-\mu)^2\psi'((1-\mu)\phi)\right]
$$

---

### 1.7 Binomial Distribution

**Parameterization**: $Y \sim \text{Binomial}(n, \mu)$ where $Y$ is the number of successes in $n$ trials

**Parameters**:
- $\mu$ (probability): logit link

**Variance**: $\text{Var}(Y) = n\mu(1-\mu)$

#### Log-Likelihood

$$
\ell(\mu | y) = y\log(\mu) + (n-y)\log(1-\mu) + \log\binom{n}{y}
$$

#### Derivatives for $\mu$ (logit link)

Let $\eta = \text{logit}(\mu)$.

Score with respect to $\mu$:
$$
\frac{\partial \ell}{\partial \mu} = \frac{y}{\mu} - \frac{n-y}{1-\mu} = \frac{y - n\mu}{\mu(1-\mu)}
$$

Chain rule for logit link ($\frac{d\mu}{d\eta} = \mu(1-\mu)$):
$$
\frac{\partial \ell}{\partial \eta} = \mu(1-\mu) \cdot \frac{\partial \ell}{\partial \mu} = y - n\mu
$$

Fisher information on $\eta$ scale:
$$
I_\eta = n\mu(1-\mu)
$$

**Implementation**:
$$
u_\mu = y - n\mu, \quad w_\mu = n\mu(1-\mu)
$$

---

### 1.8 Example: Computing Derivatives for Gaussian Data

Suppose we have data $y = [2.1, 1.9, 3.2]$ and current parameter estimates $\mu = [2.0, 2.0, 3.0]$, $\sigma = [0.5, 0.5, 0.5]$.

**For $\mu$ (identity link)**:

Residuals: $r = y - \mu = [0.1, -0.1, 0.2]$

Weights: $w_\mu = 1/\sigma^2 = [4.0, 4.0, 4.0]$

Score: $u_\mu = r \cdot w_\mu = [0.4, -0.4, 0.8]$

Working response: $z_\mu = \mu + u_\mu/w_\mu = \mu + r = y$

(For identity link with constant weights, the working response is just $y$)

**For $\sigma$ (log link)**:

Squared residuals: $r^2 = [0.01, 0.01, 0.04]$

Score on $\eta_\sigma = \log(\sigma)$ scale:
$$
u_\sigma = \frac{r^2 - \sigma^2}{\sigma^2} = \frac{[0.01, 0.01, 0.04] - 0.25}{0.25} = [-0.96, -0.96, -0.84]
$$

Weights: $w_\sigma = [2.0, 2.0, 2.0]$

Working response on log scale:
$$
z_\sigma = \log(\sigma) + u_\sigma/w_\sigma = [-0.693, -0.693, -0.693] + [-0.48, -0.48, -0.42] = [-1.17, -1.17, -1.11]
$$

The PWLS solver then regresses $z_\sigma$ on the design matrix for $\sigma$ with weights $w_\sigma$ to update $\sigma$.

---

## 2. Special Functions

### 2.1 Digamma Function

The digamma function $\psi(x) = \frac{d}{dx}\log\Gamma(x) = \frac{\Gamma'(x)}{\Gamma(x)}$ appears in derivatives for Gamma, Negative Binomial, and Beta distributions.

#### Recurrence Relation

For $x < 5$, use:
$$
\psi(x) = \psi(x+1) - \frac{1}{x}
$$

Repeatedly apply until $x \geq 5$.

#### Asymptotic Expansion

For $x \geq 5$, use the series (Abramowitz & Stegun 6.3.18):
$$
\psi(x) \sim \log(x) - \frac{1}{2x} - \frac{1}{12x^2} + \frac{1}{120x^4} - \frac{1}{252x^6} + O(x^{-8})
$$

#### Known Values (for testing)

- $\psi(1) = -\gamma \approx -0.5772156649$ (Euler-Mascheroni constant)
- $\psi(2) = 1 - \gamma \approx 0.4227843351$
- $\psi(1/2) = -\gamma - 2\log(2) \approx -1.9635100260$

### 2.2 Trigamma Function

The trigamma function $\psi'(x) = \frac{d^2}{dx^2}\log\Gamma(x)$ is needed for Student-t Fisher information.

### Recurrence Relation

For $x < 5$, use:
$$
\psi'(x) = \psi'(x+1) + \frac{1}{x^2}
$$

### Asymptotic Expansion

For $x \geq 5$, use the series (Abramowitz & Stegun 6.4.11):
$$
\psi'(x) \sim \frac{1}{x} + \frac{1}{2x^2} + \frac{1}{6x^3} - \frac{1}{30x^5} + \frac{1}{42x^7} - \frac{1}{30x^9} + O(x^{-11})
$$

### Known Values (for testing)

- $\psi'(1) = \frac{\pi^2}{6} \approx 1.644934067$
- $\psi'(2) = \frac{\pi^2}{6} - 1 \approx 0.644934067$
- $\psi'(1/2) = \frac{\pi^2}{2} \approx 4.934802201$

---

## 3. Numerical Stability and Implementation

### 3.1 Parameter Bounds

To prevent numerical issues, parameters are clamped to safe ranges:

**Minimum positive value**: $10^{-10}$
- Used for: $\mu$ (when must be positive), $\sigma$, $\phi$, probabilities

**Minimum weight**: $10^{-6}$
- Fisher information weights are floored at this value to ensure positive definiteness of the weight matrix

**Link function bounds**:
- Log/logit links: $\eta \in [-30, 30]$
- Prevents overflow in $\exp(\eta)$ (since $e^{30} \approx 10^{13}$ and $e^{-30} \approx 10^{-14}$)

### 3.2 Distribution-Specific Safeguards

**Student-t**:
- Require $\nu > 2$ to ensure finite variance
- Ensure $\nu + z^2 \geq 10^{-10}$ to prevent division issues in robustifying weight

**Gamma**:
- Clamp $\sigma$ to reasonable range to prevent extreme shape parameters

**Beta**:
- Clamp $y$ to $(10^{-10}, 1-10^{-10})$ before computing $\log(y)$ and $\log(1-y)$
- Clamp $\mu$ similarly before computing logit

**Negative Binomial**:
- Ensure $1 + \sigma\mu$ stays positive and bounded away from zero

### 3.3 Batched Special Function Computation

Digamma and trigamma functions are computed in batched vectorized form for performance:

- For $n < 10{,}000$ observations: sequential computation
- For $n \geq 10{,}000$: parallel computation via Rayon (when `parallel` feature enabled)

The switchpoint is empirically tuned to balance overhead vs. speedup.

---

## 4. GAMLSS Iteration (P-IRLS)

### Algorithm Overview

For each parameter $\theta_k$ (e.g., $\mu$, $\sigma$, $\nu$):

1. **Working response**:
   $$
   z_k = \eta_k + \frac{u_k}{w_k}
   $$
   where $u_k = \frac{\partial \ell}{\partial \eta_k}$ and $w_k$ is the expected Fisher information

2. **Penalized weighted least squares**:
   $$
   \hat{\beta}_k = \arg\min_\beta \|W_k^{1/2}(z_k - X_k\beta)\|^2 + \sum_j \lambda_j \beta^T S_j \beta
   $$

3. **Update linear predictor**:
   $$
   \eta_k^{(new)} = X_k \hat{\beta}_k
   $$

4. **Smoothing parameter selection** via GCV:
   $$
   \text{GCV}(\lambda) = \frac{n \cdot \text{RSS}}{(n - \text{EDF})^2}
   $$
   where $\text{EDF} = \text{tr}(V \cdot X^TWX)$ and $V = (X^TWX + \sum_j \lambda_j S_j)^{-1}$

### Convergence Criterion

Stop when:
$$
\max_k \|\beta_k^{(t+1)} - \beta_k^{(t)}\|_1 < \epsilon
$$

Typical: $\epsilon = 10^{-3}$, max iterations = 200.

---

## 5. Smoothing and Penalty Matrices

### P-Spline Penalty (Eilers & Marx 1996)

For 2nd-order differences:
$$
S = D^T D
$$

where $D$ is the $(m-2) \times m$ difference matrix:
$$
D = \begin{bmatrix}
1 & -2 & 1 & 0 & \cdots & 0 \\
0 & 1 & -2 & 1 & \cdots & 0 \\
\vdots & & \ddots & & & \vdots \\
0 & \cdots & 0 & 1 & -2 & 1
\end{bmatrix}
$$

This penalizes curvature: $\lambda \beta^T S \beta \approx \lambda \int (\beta''(x))^2 dx$

### Tensor Product Penalty

For bivariate smooth $f(x_1, x_2)$ using bases $B_1$ and $B_2$:

$$
S_1 = S_1^{(row)} \otimes I_{k_2}, \quad S_2 = I_{k_1} \otimes S_2^{(col)}
$$

Total penalty:
$$
\lambda_1 \beta^T S_1 \beta + \lambda_2 \beta^T S_2 \beta
$$

This allows different smoothness in each dimension.

### Random Effect Penalty

For group effects $\alpha_1, \ldots, \alpha_G$:
$$
S = I_G
$$

Penalty $\lambda \sum_{g=1}^G \alpha_g^2$ shrinks effects toward zero (ridge penalty).

---

### 5.1 Block-Sparse Penalty Matrices

In practice, penalty matrices $S_j$ are often **block-sparse**: they only penalize a subset of the coefficient vector. For example, if the model has an intercept, linear terms, and a smooth term:

$$
\beta = [\beta_{\text{intercept}}, \beta_{\text{linear}}, \beta_{\text{smooth}}]^T
$$

The penalty for the smooth term is:
$$
S_{\text{smooth}} = \begin{bmatrix}
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & S_{\text{block}}
\end{bmatrix}
$$

where $S_{\text{block}}$ is the $(n_{\text{splines}} \times n_{\text{splines}})$ penalty for the spline coefficients.

**Efficient Storage**: Instead of storing the full $p \times p$ matrix (mostly zeros), we store:
- `block`: The non-zero sub-matrix $S_{\text{block}}$
- `offset`: Starting index in the full coefficient vector
- `full_dim`: Total dimension $p$

**Operations**:

*Scaled addition*: $A \leftarrow A + \lambda S$
$$
A[i:i+k, j:j+k] \leftarrow A[i:i+k, j:j+k] + \lambda \cdot S_{\text{block}}
$$
where $i = \text{offset}$ and $k = \dim(S_{\text{block}})$.

*Matrix-vector product*: $v = S\beta$
$$
v[i:i+k] = S_{\text{block}} \cdot \beta[i:i+k], \quad v[\text{elsewhere}] = 0
$$

This exploits sparsity for computational efficiency, especially important when many terms have independent penalties.

---

## 6. Effective Degrees of Freedom

The "wiggliness" of the fit is measured by:
$$
\text{EDF} = \text{tr}(H)
$$

where $H = X(X^TWX + \lambda S)^{-1}X^TWX$ is the hat matrix.

**Interpretation**:
- EDF ≈ 2 for linear fit (intercept + slope)
- EDF ≈ number of basis functions for unpenalized spline
- EDF between these extremes reflects smoothing

---

## 7. Link Functions

### Identity Link
- **Function**: $g(\mu) = \mu$
- **Inverse**: $g^{-1}(\eta) = \eta$
- **Domain**: $\mu \in \mathbb{R}$
- **Use**: Gaussian mean, Student-t location
- **Derivative**: $\frac{d\mu}{d\eta} = 1$

### Log Link
- **Function**: $g(\mu) = \log(\mu)$
- **Inverse**: $g^{-1}(\eta) = \exp(\eta)$
- **Domain**: $\mu > 0$
- **Use**: Poisson rate, Gamma mean, scale parameters, degrees of freedom
- **Derivative**: $\frac{d\mu}{d\eta} = \mu$
- **Numerical safeguard**: Clamp $\eta \in [-30, 30]$ to prevent overflow

### Logit Link
- **Function**: $g(\mu) = \log\left(\frac{\mu}{1-\mu}\right)$
- **Inverse**: $g^{-1}(\eta) = \frac{1}{1 + e^{-\eta}} = \frac{e^\eta}{1 + e^\eta}$
- **Domain**: $\mu \in (0, 1)$
- **Use**: Binomial probability, Beta mean
- **Derivative**: $\frac{d\mu}{d\eta} = \mu(1-\mu)$
- **Numerical safeguards**:
  - Clamp $\mu \in [10^{-10}, 1-10^{-10}]$ before applying link
  - Clamp $\eta \in [-30, 30]$ to prevent overflow in inverse

### Chain Rule for Link Functions

When modeling parameter $\theta$ with link function $g$, we have $\eta = g(\theta)$. The score and Fisher information transform as:

$$
\frac{\partial \ell}{\partial \eta} = \frac{d\theta}{d\eta} \cdot \frac{\partial \ell}{\partial \theta}
$$

$$
I_\eta = \left(\frac{d\theta}{d\eta}\right)^2 I_\theta
$$

**For log link** ($\eta = \log(\theta)$):
$$
\frac{d\theta}{d\eta} = \theta, \quad \frac{\partial \ell}{\partial \eta} = \theta \cdot \frac{\partial \ell}{\partial \theta}, \quad I_\eta = \theta^2 I_\theta
$$

**For logit link** ($\eta = \text{logit}(\theta)$):
$$
\frac{d\theta}{d\eta} = \theta(1-\theta), \quad \frac{\partial \ell}{\partial \eta} = \theta(1-\theta) \cdot \frac{\partial \ell}{\partial \theta}, \quad I_\eta = [\theta(1-\theta)]^2 I_\theta
$$

**For identity link** ($\eta = \theta$):
$$
\frac{d\theta}{d\eta} = 1, \quad \frac{\partial \ell}{\partial \eta} = \frac{\partial \ell}{\partial \theta}, \quad I_\eta = I_\theta
$$

### Working Response and IRLS

The working response in the IRLS algorithm is:
$$
z = \eta + \frac{u}{w}
$$

where $u = \frac{\partial \ell}{\partial \eta}$ and $w = I_\eta$.

For a Poisson model with log link:
- Current: $\eta = \log(\hat{\mu})$
- Score: $u = y - \hat{\mu}$
- Weight: $w = \hat{\mu}$
- Working response: $z = \log(\hat{\mu}) + \frac{y - \hat{\mu}}{\hat{\mu}}$

For a Binomial model with logit link:
- Current: $\eta = \text{logit}(\hat{\mu})$
- Score: $u = y - n\hat{\mu}$
- Weight: $w = n\hat{\mu}(1-\hat{\mu})$
- Working response: $z = \text{logit}(\hat{\mu}) + \frac{y - n\hat{\mu}}{n\hat{\mu}(1-\hat{\mu})}$

The PWLS problem then becomes:
$$
\hat{\beta}^{(t+1)} = \arg\min_\beta \sum_i w_i (z_i - x_i^T\beta)^2 + \sum_j \lambda_j \beta^T S_j \beta
$$

---

## 8. GCV Gradient for Smoothing Parameter Optimization

The GCV score is minimized using L-BFGS optimization. The gradient with respect to $\log(\lambda_j)$ requires careful derivation.

### Setup

Let $V = (X^TWX + \sum_j \lambda_j S_j)^{-1}$ and $\hat{\beta}(\lambda) = V X^T W z$.

### Key Derivatives

**Derivative of $\beta$ with respect to $\lambda_j$:**
$$
\frac{\partial \hat{\beta}}{\partial \lambda_j} = -V S_j \hat{\beta}
$$

**Derivative of RSS:**
$$
\frac{\partial \text{RSS}}{\partial \lambda_j} = 2 (X^T W r)^T V S_j \hat{\beta}
$$

where $r = z - X\hat{\beta}$ are the residuals.

**Derivative of EDF:**
$$
\frac{\partial \text{EDF}}{\partial \lambda_j} = -\text{tr}(V S_j V X^T W X)
$$

### Full GCV Gradient

Using the quotient rule on $\text{GCV} = \frac{n \cdot \text{RSS}}{(n - \text{EDF})^2}$:

$$
\frac{\partial \text{GCV}}{\partial \lambda_j} = \frac{n}{(n - \text{EDF})^3} \left[ \frac{\partial \text{RSS}}{\partial \lambda_j}(n - \text{EDF}) + 2 \cdot \text{RSS} \cdot \frac{\partial \text{EDF}}{\partial \lambda_j} \right]
$$

### Chain Rule for Log-Space

Since we optimize in log-space:
$$
\frac{\partial \text{GCV}}{\partial \log(\lambda_j)} = \lambda_j \cdot \frac{\partial \text{GCV}}{\partial \lambda_j}
$$

This ensures $\lambda_j > 0$ without explicit constraints.

---

### 8.1 Efficient PWLS Solution via Cholesky Decomposition

The penalized weighted least squares problem:
$$
\hat{\beta} = \arg\min_\beta \|W^{1/2}(z - X\beta)\|^2 + \sum_j \lambda_j \beta^T S_j \beta
$$

has the normal equations:
$$
(X^TWX + S_\lambda)\hat{\beta} = X^TWz
$$

where $S_\lambda = \sum_j \lambda_j S_j$.

**Key observation**: $X^TWX + S_\lambda$ is symmetric positive definite (SPD) when $S_j$ are positive semi-definite and $W$ has positive diagonal entries.

#### Cholesky Approach

1. Form the weighted design matrix: $X_w = W^{1/2} X$
2. Form the coefficient matrix: $A = X_w^T X_w + S_\lambda$
3. Compute Cholesky decomposition: $A = L L^T$ where $L$ is lower triangular
4. Solve $L y = X^T W z$ for $y$ (forward substitution)
5. Solve $L^T \hat{\beta} = y$ for $\hat{\beta}$ (backward substitution)

**Advantages over LU decomposition**:
- Exploits SPD structure: $O(p^3/3)$ instead of $O(2p^3/3)$
- Numerically stable for well-conditioned systems
- Automatic failure detection: Cholesky fails if matrix is not positive definite

**Fallback**: If Cholesky fails (due to numerical issues or near-singularity), fall back to LU decomposition with partial pivoting.

#### Covariance Matrix

The covariance matrix for $\hat{\beta}$ is:
$$
\text{Cov}(\hat{\beta}) = (X^TWX + S_\lambda)^{-1} = A^{-1}
$$

With Cholesky $A = LL^T$:
$$
A^{-1} = (L^T)^{-1} L^{-1}
$$

We compute $L^{-1}$ via triangular inversion, then form $(L^{-1})^T L^{-1}$.

#### Gradient Computation for GCV

For GCV optimization, we need $\frac{\partial \text{EDF}}{\partial \lambda_j}$:
$$
\frac{\partial \text{EDF}}{\partial \lambda_j} = -\text{tr}(V S_j V X^T W X)
$$

where $V = A^{-1}$.

**Exploiting block structure**: When $S_j$ is block-sparse, only compute:
$$
\text{tr}(V[i:i+k, :] S_{\text{block}} V[:, i:i+k] X^T W X)
$$

This avoids forming the full $p \times p$ products when $S_{\text{block}} \ll p$.

---

## 9. Mean-Variance Relationships

A key feature distinguishing different distributions is how variance relates to the mean. This determines which distribution is appropriate for different data structures.

| Distribution | Mean | Variance | Relationship |
|--------------|------|----------|--------------|
| **Gaussian** | $\mu$ | $\sigma^2$ | Constant (homoskedastic) |
| **Poisson** | $\mu$ | $\mu$ | Equidispersion: $\text{Var} = \text{Mean}$ |
| **Gamma** | $\mu$ | $\mu^2\sigma^2$ | Proportional to $\mu^2$ (constant CV) |
| **Negative Binomial** | $\mu$ | $\mu + \sigma\mu^2$ | Overdispersion relative to Poisson |
| **Binomial** | $n\mu$ | $n\mu(1-\mu)$ | Quadratic in mean, max at $\mu=0.5$ |
| **Beta** | $\mu$ | $\frac{\mu(1-\mu)}{1+\phi}$ | Max at $\mu=0.5$, decreases with $\phi$ |
| **Student-t** | $\mu$ | $\frac{\nu\sigma^2}{\nu-2}$ | Heavier tails than Gaussian |

### Choosing a Distribution

**Count data**:
- Equidispersed: Poisson
- Overdispersed (variance > mean): Negative Binomial
- Known number of trials: Binomial

**Continuous positive**:
- Constant coefficient of variation: Gamma
- Heavy tails: Student-t with $\nu$ small
- Heteroskedastic with mean: Gamma or lognormal (not implemented)

**Proportions/rates in $(0,1)$**:
- Beta (continuous)
- Binomial (discrete counts / $n$ trials)

**Continuous unbounded**:
- Light tails, constant variance: Gaussian
- Heavy tails: Student-t

### Overdispersion

**Overdispersion** occurs when the observed variance exceeds what the distribution predicts from the mean alone.

For count data:
- If $\text{Var}(Y) > \mu$: Use Negative Binomial instead of Poisson
- The parameter $\sigma$ in NB quantifies the overdispersion: as $\sigma \to 0$, NB $\to$ Poisson

For continuous data:
- Gamma allows variance to scale with $\mu^2$
- Student-t allows heavier tails (more extreme values) than Gaussian

---

## 10. Convergence and Initialization

### Initialization Strategy

The RS algorithm requires starting values for all parameters. Default initialization:

1. **$\mu$**:
   - Gaussian/Student-t: $\bar{y}$ (sample mean)
   - Poisson/Gamma/NB: $\bar{y}$ (then apply log link)
   - Binomial: $\bar{y}/n$ (sample proportion)
   - Beta: $\bar{y}$ (sample mean, clamped to $(0.1, 0.9)$)

2. **$\sigma$**:
   - Gaussian/Student-t: $s_y$ (sample std dev)
   - Gamma/NB/Beta: 1.0 (or estimated from sample CV)

3. **$\nu$** (Student-t): 5.0

4. **$\phi$** (Beta): 1.0

### Convergence Criterion

The algorithm stops when:
$$
\max_k \|\beta_k^{(t+1)} - \beta_k^{(t)}\|_1 < \epsilon
$$

where $k$ indexes parameters ($\mu$, $\sigma$, etc.) and $\epsilon = 10^{-3}$ by default.

**Maximum iterations**: 200 (default)

### Convergence Issues

**Non-convergence** can occur due to:
1. **Poor initialization**: Try better starting values, especially for $\nu$ in Student-t
2. **Model misspecification**: Wrong distribution family for the data
3. **Multicollinearity**: Highly correlated predictors (check condition number of design matrix)
4. **Insufficient smoothing**: Extremely small $\lambda$ can cause instability
5. **Numerical issues**: Extreme data values causing overflow/underflow

**Diagnostics**:
- Monitor deviance: should decrease monotonically
- Check for NaN/Inf in coefficients or fitted values
- Inspect condition number of $(X^TWX + \lambda S)$

---

## 11. Model Selection and Diagnostics

### 11.1 Information Criteria

**Akaike Information Criterion (AIC)**:
$$
\text{AIC} = -2\ell(\hat{\theta}) + 2k
$$

where $\ell(\hat{\theta})$ is the maximized log-likelihood and $k$ is the number of parameters.

For GAMLSS with smoothing:
$$
k = \sum_{j=1}^P \text{EDF}_j
$$

where $P$ is the number of distribution parameters and $\text{EDF}_j$ accounts for the effective degrees of freedom of smooth terms.

**Bayesian Information Criterion (BIC)**:
$$
\text{BIC} = -2\ell(\hat{\theta}) + k\log(n)
$$

BIC penalizes complexity more heavily than AIC for $n > 7$.

### 11.2 Deviance

The **deviance** measures lack of fit:
$$
D = 2[\ell_{\text{saturated}} - \ell_{\text{fitted}}]
$$

where $\ell_{\text{saturated}}$ is the log-likelihood of a saturated model (perfect fit).

For nested models, the deviance difference follows approximately $\chi^2$ distribution:
$$
D_1 - D_2 \sim \chi^2_{\text{EDF}_2 - \text{EDF}_1}
$$

### 11.3 Residuals

**Quantile (randomized quantile) residuals**:
$$
r_i = \Phi^{-1}(F(y_i | \hat{\theta}_i))
$$

where $F$ is the fitted CDF and $\Phi^{-1}$ is the inverse standard normal CDF.

If the model is correct, $r_i \sim N(0,1)$.

**Deviance residuals**:
$$
r_i^{(D)} = \text{sign}(y_i - \hat{\mu}_i) \sqrt{d_i}
$$

where $d_i$ is the contribution of observation $i$ to the total deviance.

**Pearson residuals**:
$$
r_i^{(P)} = \frac{y_i - \hat{\mu}_i}{\sqrt{\widehat{\text{Var}}(Y_i)}}
$$

### 11.4 Worm Plots

For each parameter $\theta_k$, plot quantile residuals against fitted quantiles:
- X-axis: Normal quantiles $\Phi^{-1}((i-0.5)/n)$
- Y-axis: Sorted quantile residuals

If the model is correct, points should lie on a horizontal line at zero. Systematic deviations indicate:
- U-shape: Underdispersion
- Inverse U-shape: Overdispersion
- S-shape: Skewness issues
- Linear trend: Location shift

### 11.5 Q-Q Plots

Plot theoretical quantiles against sample quantiles of residuals. Should be approximately linear if residuals are normal.

### 11.6 Goodness of Link Test

To test if link $g$ is appropriate, fit augmented model:
$$
g(\theta) = X\beta + \gamma [g(\theta)]^2
$$

Test $H_0: \gamma = 0$ using likelihood ratio test.

---

## References

1. Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized additive models for location, scale and shape. *Journal of the Royal Statistical Society: Series C*, 54(3), 507-554.

2. Eilers, P. H., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89-121.

3. Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapman and Hall/CRC.

4. Abramowitz, M., & Stegun, I. A. (1972). *Handbook of Mathematical Functions*. Dover Publications.