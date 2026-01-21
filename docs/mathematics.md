# Mathematical Foundations of GAMLSS

## 1. Student-t Distribution Derivatives

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
I_{\log(\nu)} = \frac{\nu^2}{4}\left[\psi'\left(\frac{\nu}{2}\right) - \psi'\left(\frac{\nu+1}{2}\right) - \frac{2(\nu+3)}{\nu(\nu+1)}\right]
$$

where $\psi'(x)$ is the trigamma function.

### Numerical Considerations

1. **Minimum $\nu$**: Require $\nu > 2$ to ensure finite variance
2. **Minimum $\sigma$**: Enforce $\sigma \geq 10^{-6}$ to prevent division by zero
3. **Weight clamping**: The denominator $\nu + z^2$ should be $\geq 10^{-10}$
4. **Information positivity**: Ensure $I_{\log(\nu)} \geq 10^{-6}$

---

## 2. Trigamma Function Implementation

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

## 3. GAMLSS Iteration (P-IRLS)

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

Typical: $\epsilon = 10^{-6}$, max iterations = 20.

---

## 4. Smoothing and Penalty Matrices

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

## 5. Effective Degrees of Freedom

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

## 6. Link Functions

### Identity Link
- **Function**: $g(\mu) = \mu$
- **Inverse**: $g^{-1}(\eta) = \eta$
- **Domain**: $\mu \in \mathbb{R}$
- **Use**: Gaussian mean

### Log Link
- **Function**: $g(\mu) = \log(\mu)$
- **Inverse**: $g^{-1}(\eta) = \exp(\eta)$
- **Domain**: $\mu > 0$
- **Use**: Poisson rate, Gaussian/Student-t scale, Student-t df
- **Numerical safeguard**: Clamp $\eta \in [-30, 30]$ to prevent overflow

---

## References

1. Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized additive models for location, scale and shape. *Journal of the Royal Statistical Society: Series C*, 54(3), 507-554.

2. Eilers, P. H., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89-121.

3. Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapman and Hall/CRC.

4. Abramowitz, M., & Stegun, I. A. (1972). *Handbook of Mathematical Functions*. Dover Publications.