# Mathematical Foundations of GAMLSS

This document provides detailed mathematical derivations for the GAMLSS implementation in `glissando`. It covers distribution-specific derivatives, special functions, numerical algorithms, and convergence theory.

> Generated PDF: run `bash docs/math/build-math-pdf.sh` to produce `docs/math/mathematics.pdf` (a numbered, hyperlinked, table-of-contents'd PDF). The PDF build uses pandoc + xelatex; see the script header for details.

---

## 1. Distribution Derivatives and Distributional Functions

This section provides detailed derivations of score functions (first derivatives of log-likelihood) and Fisher information (expected second derivatives) for all implemented distributions. These quantities drive the penalized iteratively reweighted least squares (P-IRLS) algorithm. [CDF-TRIO] then derives the cumulative distribution, density, and quantile (the CDF/PDF/quantile trio) that these same families expose for residuals, centiles, and predictive intervals.

> **Cross-references** use bracketed tags: a named subsection is cited as `[TAG]` (e.g. `[WEIBULL]`, `[CDF-TRIO]`) — search the tag to jump to its heading. Whole-chapter references keep the `§N` form.

**Key quantities**:
- **Score** $u = \frac{\partial \ell}{\partial \eta}$: Direction of steepest ascent for the log-likelihood
- **Fisher information** $w = I_\eta$: Expected curvature (used as weight in IRLS)
- **Working response** $z = \eta + u/w$: Adjusted target for the weighted least squares problem

For each distribution, we derive these quantities accounting for the link function via the chain rule.

### [GAUSSIAN] Gaussian Distribution

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

### [STUDENT-T] Student-t Distribution

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

The working weight uses this expected information (the same convention as
gamlss `TF()`), not the observed per-row weight $w/\sigma^2$: the two agree
only at $z^2 = 3$, and a data-dependent weight changes the PWLS subproblem,
hence $\lambda$ selection, EDF, and standard errors, relative to the
reference implementation.

#### For $\sigma$ (log link):
$$
I_{\log(\sigma)} = \frac{2\nu}{\nu + 3}
$$

#### For $\nu$ (log link)

Lange, Little & Taylor (1989), identical to gamlss `TF()`'s `d2ldv2`:

$$
I_{\log(\nu)} = \frac{\nu^2}{4}\left[\psi'\left(\frac{\nu}{2}\right) - \psi'\left(\frac{\nu+1}{2}\right) - \frac{2(\nu+5)}{\nu(\nu+1)(\nu+3)}\right]
$$

where $\psi'(x)$ is the trigamma function. The rational term is a small
correction: $I_{\log(\nu)}$ decays like $O(1/\nu)$, so a sign or degree error
here inflates the weight by orders of magnitude and effectively freezes
$\nu$ at its starting value (an earlier implementation used
$+2(\nu+3)/(\nu(\nu+1))$, ~50× too large at $\nu = 5$). The formula is
validated against a Monte-Carlo estimate of $\mathbb{E}[(\nu\,\partial\ell/\partial\nu)^2]$.

### Numerical Considerations

1. **Minimum $\nu$**: $\nu > 2$ is enforced to ensure finite variance, via a *floored log link* on $\nu$ ($\eta = \log\nu$, $\nu = \max(e^\eta, 2)$). The optimizer can explore the heavy-tail region without the variance $\sigma^2\nu/(\nu-2)$ becoming undefined; the floor never binds when the true $\nu$ is well above 2. See `FlooredLogLink` in `src/distributions/links.rs`.

   Where the floor binds, $d\nu/d\eta = 0$ and the score must not be forwarded
   blindly. The implementation applies a KKT-style aggregate projection: if the
   score summed over the pinned rows is $\leq 0$, the constrained optimum is at
   the boundary and those rows are frozen ($u = 0$, so the block reports a zero
   step and the outer loop converges); if it is $> 0$ the full chain rule is
   forwarded so $\nu$ can re-enter the interior. Forwarding a negative
   aggregate walks $\eta_\nu$ downward indefinitely; a per-row one-sided
   projection instead biases the aggregate upward and produces a lift-off /
   fall-back limit cycle at the boundary. The single summed score is exact for
   an intercept-only $\eta_\nu$ (the standard usage).
2. **Minimum $\sigma$**: Enforce $\sigma \geq 10^{-6}$ to prevent division by zero
3. **Weight clamping**: The denominator $\nu + z^2$ should be $\geq 10^{-10}$
4. **Information positivity**: Ensure $I_{\log(\nu)} \geq 10^{-6}$

---

### [POISSON] Poisson Distribution

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

### [GAMMA] Gamma Distribution

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

Fisher information involves trigamma $\psi'(x) = \frac{d^2}{dx^2}\log\Gamma(x)$.
On the natural scale $I_\sigma = \frac{4}{\sigma^6}\psi'(\alpha) - \frac{4}{\sigma^4}$,
so on the $\eta_\sigma$ scale ($\times\,\sigma^2$):
$$
I_{\eta_\sigma} = \frac{4}{\sigma^4}\psi'\left(\frac{1}{\sigma^2}\right) - \frac{4}{\sigma^2}
$$

This matches gamlss `GA`'s `d2ldd2` and the Monte-Carlo check
$\mathbb{E}[u_{\eta_\sigma}^2]$. Since $\psi'(1/\sigma^2) > \sigma^2$ for all
$\sigma > 0$, the expression is strictly positive, with limit
$I_{\eta_\sigma} \to 2$ as $\sigma \to 0$ (the Gaussian-like regime). An
earlier implementation used $-2/\sigma^2$ for the second term, inflating the
weight roughly five-fold at $\sigma = 0.5$ and over-damping every $\sigma$
update.

**Implementation**:
$$
u_\sigma = \frac{2}{\sigma^2}\left[\psi(\alpha) + 2\log(\sigma) - \log(y/\mu) + y/\mu - 1\right], \quad w_\sigma = \frac{4}{\sigma^4}\psi'(\alpha) - \frac{4}{\sigma^2}
$$

---

### [WEIBULL] Weibull Distribution

**Parameterization**: $Y \sim \text{Weibull}$ with scale $\mu$ and shape $\sigma$ (gamlss `WEI`), density $f(y) = \frac{\sigma}{\mu}\left(\frac{y}{\mu}\right)^{\sigma-1}\exp\left[-\left(\frac{y}{\mu}\right)^\sigma\right]$ on $y > 0$.

**Parameters**:
- $\mu$ (scale): log link
- $\sigma$ (shape): log link

**Moments**: $\mathbb{E}[Y] = \mu\,\Gamma(1 + 1/\sigma)$, $\text{Var}(Y) = \mu^2\left[\Gamma(1 + 2/\sigma) - \Gamma(1 + 1/\sigma)^2\right]$ — neither equals $\mu$, so both moment methods are overridden.

Throughout, write $z = (y/\mu)^\sigma$, which is $\text{Exp}(1)$-distributed at the true parameters.

#### Log-Likelihood

$$
\ell(\mu, \sigma \mid y) = \log\sigma - \sigma\log\mu + (\sigma - 1)\log y - \left(\frac{y}{\mu}\right)^\sigma
$$

#### Derivatives for $\mu$ (log link)

Let $\eta_\mu = \log\mu$. Since $\partial z/\partial\mu = -\sigma z/\mu$,
$$
\frac{\partial \ell}{\partial \mu} = -\frac{\sigma}{\mu} + \frac{\sigma}{\mu}\left(\frac{y}{\mu}\right)^\sigma = \frac{\sigma}{\mu}(z - 1).
$$

Chain rule for log link ($d\mu/d\eta_\mu = \mu$):
$$
\frac{\partial \ell}{\partial \eta_\mu} = \sigma(z - 1).
$$

Using $\mathbb{E}[z] = 1$, the $\eta$-scale Fisher weight is constant:
$$
u_\mu = \sigma(z - 1), \quad w_\mu = \sigma^2.
$$

#### Derivatives for $\sigma$ (log link)

Let $\eta_\sigma = \log\sigma$. With $\partial z/\partial\sigma = z\log(y/\mu)$,
$$
\frac{\partial \ell}{\partial \sigma} = \frac{1}{\sigma} + \log\!\frac{y}{\mu} - \left(\frac{y}{\mu}\right)^\sigma \log\!\frac{y}{\mu} = \frac{1}{\sigma} + \log\!\frac{y}{\mu}\,(1 - z).
$$

Chain rule for log link ($d\sigma/d\eta_\sigma = \sigma$):
$$
\frac{\partial \ell}{\partial \eta_\sigma} = 1 + \sigma\log\!\frac{y}{\mu}\,(1 - z).
$$

The expected information on the $\eta_\sigma$ scale is the Weibull shape constant
$$
I_{\eta_\sigma} = \frac{\pi^2}{6} + (1 - \gamma)^2 \approx 1.8237,
$$
where $\gamma \approx 0.5772$ is the Euler–Mascheroni constant. Hence
$$
u_\sigma = 1 + \sigma\log(y/\mu)\,(1 - z), \quad w_\sigma = \frac{\pi^2}{6} + (1 - \gamma)^2.
$$

#### CDF, Density, and Quantile

$$
F(y \mid \mu, \sigma) = 1 - \exp\!\left[-\left(\frac{y}{\mu}\right)^\sigma\right], \qquad
Q(p \mid \mu, \sigma) = \mu\,\bigl[-\log(1 - p)\bigr]^{1/\sigma},
$$
with density $f$ as above. All three are closed-form; see [CDF-TRIO] for how the trio feeds residuals and centiles.

---

### [NEG-BINOMIAL] Negative Binomial Distribution

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

**Why observed information here:** for $\mu$ in NB the expected information $I_{\eta_\mu} = \mathbb{E}[\mu (Y - \mu)/(1 + \sigma\mu)^2]$ collapses to $\mu/(1 + \sigma\mu)$ after expectation, which coincides numerically with the observed-information form because of the variance identity $\mathrm{Var}(Y) = \mu(1 + \sigma\mu)$. Rigby & Stasinopoulos (2005, eq. (15)) use this simplification; `src/distributions/negative_binomial.rs` follows it. As $\sigma \to 0$ the weight reduces to $\mu$ — exactly the Poisson Fisher weight — confirming the limiting Poisson behavior.

**Implementation** (for IRLS):
$$
u_\mu = \frac{y - \mu}{1 + \sigma\mu}, \quad w_\mu = \frac{\mu}{1 + \sigma\mu}
$$

#### Derivatives for $\sigma$ (log link)

Let $\eta_\sigma = \log(\sigma)$ and $r = 1/\sigma$.

Differentiating with respect to $r$ first,
$$
\frac{\partial \ell}{\partial r} = \psi(y + r) - \psi(r) - \log(1+\sigma\mu) + \frac{\mu - y}{r+\mu},
$$
then $\frac{\partial \ell}{\partial \eta_\sigma} = \sigma\,\frac{\partial \ell}{\partial \sigma} = -\frac{1}{\sigma}\frac{\partial \ell}{\partial r}$ (since $r = 1/\sigma$):
$$
\frac{\partial \ell}{\partial \eta_\sigma} = -\frac{1}{\sigma}\left[\psi(y + r) - \psi(r) - \log(1+\sigma\mu) + \frac{\mu - y}{r+\mu}\right]
$$

**Working weight**: the exact expected information has no closed form (it
involves $\mathbb{E}[\psi'(Y + r)]$), so the squared-score convention of
gamlss `NBI` (`d2ldd2 = -dldd^2`) is used:
$$
w_{\eta_\sigma} = u_{\eta_\sigma}^2 \quad (\text{floored at } 10^{-6})
$$

An earlier implementation used the partial expected-information approximation
$\psi'(r)/\sigma^2$, which drops same-order terms, behaves like $1/\sigma$
near the Poisson boundary (over-damping $\sigma$ updates), and does not track
the gamlss oracle's $\lambda$/EDF selection for $\sigma$ smooths.

**Implementation**:
$$
u_\sigma = -\frac{1}{\sigma}\left[\psi(y + 1/\sigma) - \psi(1/\sigma) - \log(1+\sigma\mu) + \frac{\mu - y}{1/\sigma+\mu}\right], \quad w_\sigma = \max(u_\sigma^2,\ 10^{-6})
$$

---

### [BETA] Beta Distribution

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
I_{\eta_\phi} = \phi^2\left[\mu^2\psi'(\alpha) + (1-\mu)^2\psi'(\beta) - \psi'(\phi)\right]
$$

The bracket is $-\partial^2\ell/\partial\phi^2$; the sign convention matters
because $\psi'$ is decreasing and convex, which makes
$\mu^2\psi'(\mu\phi) + (1-\mu)^2\psi'((1-\mu)\phi) > \psi'(\phi)$ for all
$\mu \in (0,1)$, $\phi > 0$, so $I_{\eta_\phi}$ is strictly positive as an
information must be. (An earlier version wrote the bracket negated and relied
on an absolute value.)

**Implementation**:
$$
u_\phi = \phi\left[\psi(\phi) - \mu\psi(\mu\phi) - (1-\mu)\psi((1-\mu)\phi) + \mu\log(y) + (1-\mu)\log(1-y)\right]
$$
$$
w_\phi = \phi^2\left[\mu^2\psi'(\mu\phi) + (1-\mu)^2\psi'((1-\mu)\phi) - \psi'(\phi)\right]
$$

---

### [BINOMIAL] Binomial Distribution

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

### [OCAT] Ordered Categorical Distribution (`Ocat`)

**Parameterization**: Proportional-odds / cumulative-logit for ordinal response $y \in \{1, \ldots, R\}$, $R \in \{2, 3, 4, 5\}$.

**Parameters**:
- $\mu$ (latent linear predictor): identity link
- $\delta_1$ (first threshold $\theta_1$): identity link
- $\delta_k$ (positive increment $\theta_k - \theta_{k-1}$), $k = 2, \ldots, R-1$: log link

**Threshold reconstruction**: $\theta_1 = \delta_1$ (unconstrained); $\theta_k = \theta_{k-1} + \exp(\delta_k)$ for $k \ge 2$, ensuring strict monotonicity $\theta_1 < \cdots < \theta_{R-1}$.

#### Cumulative-Logit Probabilities

Define the logistic CDF (argument clamped to $[-30, 30]$) and density:
$$
F_k = \text{logistic}(\theta_k - \mu) = \frac{1}{1 + e^{-(\theta_k - \mu)}}, \qquad f_k = F_k(1 - F_k)
$$
with boundary conventions $F_0 = 0$, $F_R = 1$. Category probabilities:
$$
\pi_r = F_r - F_{r-1}, \quad r = 1, \ldots, R.
$$
Each $\pi_r$ is floored at $\mathrm{MIN\_PROB} = 10^{-10}$ and the vector is renormalized to sum to 1.

#### Log-Likelihood

$$
\ell_i = \log \pi_{y_i} = \log(F_{y_i} - F_{y_i - 1})
$$

#### Derivatives for $\mu$ (identity link)

Using boundary conventions $f_0 = f_R = 0$, for observation $y_i = r$:

Score:
$$
u_\mu = \frac{f_{r-1} - f_r}{\pi_r}
$$

Expected Fisher information (sum over all categories):
$$
w_\mu = \sum_{r=1}^{R} \frac{(f_{r-1} - f_r)^2}{\pi_r}
$$

**Implementation**:
$$
u_\mu = \frac{f_{r-1} - f_r}{\pi_r}, \quad w_\mu = \sum_{r=1}^{R} \frac{(f_{r-1} - f_r)^2}{\pi_r}
$$

Both floored at $\mathrm{MIN\_WEIGHT}$ where needed.

#### Derivatives for $\delta_k$ (identity link for $k=1$, log link for $k \ge 2$)

The Jacobian of the $\eta_k \to \theta_k$ map is:
$$
\mathrm{jac}_k = \begin{cases} 1 & k = 1 \text{ (identity link)} \\ \exp(\eta_k) = \delta_k & k \ge 2 \text{ (log link; stored as response-scale value)} \end{cases}
$$

For observation $y_i = r$, threshold $\delta_k$ only affects $\pi_r$ when $r \ge k$:

Score:
$$
u_k = \begin{cases}
0 & r < k \\[4pt]
\dfrac{\mathrm{jac}_k\,\bigl(f_r\,\mathbf{1}\{r \le R-1\} - f_{r-1}\,\mathbf{1}\{r > k\}\bigr)}{\pi_r} & r \ge k
\end{cases}
$$

Expected Fisher information (sum over categories $r = k, \ldots, R$):
$$
w_k = \sum_{r=k}^{R} \frac{\Bigl(\mathrm{jac}_k\,\bigl(f_r\,\mathbf{1}\{r \le R-1\} - f_{r-1}\,\mathbf{1}\{r > k\}\bigr)\Bigr)^2}{\pi_r}
$$
floored at $\mathrm{MIN\_WEIGHT}$.

#### Moments

$$
\mathbb{E}[Y] = \sum_{r=1}^{R} r\,\pi_r, \qquad \mathrm{Var}(Y) = \sum_{r=1}^{R} r^2\,\pi_r - \bigl(\mathbb{E}[Y]\bigr)^2
$$

(variance clamped to $\ge 0$).

> **Implementation note**: `Ocat` carries `n_categories` state and cannot be recovered from a name string alone. It is excluded from `from_name` and from the WASM / JSON string-dispatch routes. Construct it explicitly in Rust or via the Python `Ocat(n_categories=R)` class. The fitting loop treats each threshold $\delta_k$ as an intercept-only formula — no new solver machinery is required beyond what drives any other scalar distribution parameter.

Source: `src/distributions/ocat.rs`.

---

### [BCCG] Box-Cox–Cole-Green (BCCG) Distribution

The BCCG family (Cole & Green 1992) models a **skew positive** response $y > 0$ by a Box-Cox power transform to a standard normal. It is parameterized by the median $\mu$ (log link), the approximate coefficient of variation $\sigma$ (log link), and the skewness / Box-Cox power $\nu$ (identity link). It is the engine behind LMS centile curves (growth charts). BCCG is the worked template for the Box-Cox family; BCT and BCPE extend the same spine, replacing the standard normal that the transformed residual follows with a Student-$t$ (extra df $\tau$) or power-exponential (extra kurtosis $\tau$).

#### The Box-Cox z-score

The shared spine transforms $y > 0$ to a standardized residual $z$:
$$
z = \begin{cases}
\dfrac{(y/\mu)^\nu - 1}{\nu\sigma} & \nu \ne 0 \\[8pt]
\dfrac{\log(y/\mu)}{\sigma} & \nu = 0 \quad (\text{limit})
\end{cases}
$$
Numerically the $\nu \ne 0$ branch is evaluated as $\operatorname{expm1}(\nu L)/(\nu\sigma)$ with $L = \log(y/\mu)$, which is accurate (no cancellation) as $\nu \to 0$, so the density is smooth through $\nu = 0$.

#### Log-Likelihood

For BCCG the transformed residual is standard normal, $z \sim N(0,1)$, so adding the Jacobian $\mathrm{d}z/\mathrm{d}y$ gives the pointwise log-density
$$
\ell = -\tfrac{1}{2} z^2 - \tfrac{1}{2}\log(2\pi) + (\nu - 1)\log(y) - \nu\log(\mu) - \log(\sigma).
$$
(The exact gamlss density truncates the lower tail at $y = 0$ and renormalizes by $\Phi(1/(\sigma|\nu|))$; in the usual regime $1/(\sigma|\nu|)$ is large so $\Phi \approx 1$ and the correction is negligible — the implementation uses the un-truncated form.)

By the definition of $z$ we have the identity $T \equiv (y/\mu)^\nu = 1 + \nu\sigma z$, which collapses the score functions to clean forms.

#### First Derivatives (Score Functions)

Partial derivatives of $z$:
$$
\frac{\partial z}{\partial \mu} = -\frac{T}{\mu\sigma}, \qquad
\frac{\partial z}{\partial \sigma} = -\frac{z}{\sigma}, \qquad
\frac{\partial z}{\partial \nu} = \frac{\nu T L - (T - 1)}{\nu^2 \sigma},
$$
with the limit $\partial z/\partial\nu \to L^2/(2\sigma)$ as $\nu \to 0$. On the parameter scale,
$$
\frac{\partial \ell}{\partial \mu} = \frac{zT}{\mu\sigma} - \frac{\nu}{\mu}, \qquad
\frac{\partial \ell}{\partial \sigma} = \frac{z^2 - 1}{\sigma}, \qquad
\frac{\partial \ell}{\partial \nu} = -z\,\frac{\partial z}{\partial \nu} + \log(y/\mu).
$$
Applying the chain rule for the links ($\mu, \sigma$ log $\Rightarrow u_\eta = \theta\,\partial\ell/\partial\theta$; $\nu$ identity $\Rightarrow u_\eta = \partial\ell/\partial\nu$) and substituting $T = 1 + \nu\sigma z$ gives the $\eta$-scale scores the implementation returns:
$$
u_\mu = \frac{z}{\sigma} + \nu(z^2 - 1), \qquad
u_\sigma = z^2 - 1, \qquad
u_\nu = -z\,\frac{\partial z}{\partial \nu} + \log(y/\mu).
$$

#### Expected Fisher Information

Using $z \sim N(0,1)$ (so $E[z^2] = 1$, $E[z^4] = 3$) and a small-$\sigma$ expansion for $\nu$, the expected information on the parameter scale is
$$
I_{\mu\mu} = \frac{1}{\mu^2\sigma^2} + \frac{2\nu^2}{\mu^2}, \qquad
I_{\sigma\sigma} = \frac{2}{\sigma^2}, \qquad
I_{\nu\nu} = \frac{7\sigma^2}{4},
$$
matching the gamlss BCCG expected second derivatives. The $\eta$-scale weights ($W_\eta = (\mathrm{d}\theta/\mathrm{d}\eta)^2 I_{\theta\theta}$, floored to keep $W$ positive definite) are
$$
w_\mu = \mu^2 I_{\mu\mu} = \frac{1}{\sigma^2} + 2\nu^2, \qquad
w_\sigma = \sigma^2 I_{\sigma\sigma} = 2, \qquad
w_\nu = I_{\nu\nu} = \frac{7\sigma^2}{4}.
$$

#### CDF and Quantile

Both reduce to the standard normal $\Phi$ of the Box-Cox z-score:
$$
F(y) = \Phi(z), \qquad
Q(p) = \mu\bigl(1 + \nu\sigma\,\Phi^{-1}(p)\bigr)^{1/\nu} \quad (\nu \ne 0; \ \ \mu\,e^{\sigma\Phi^{-1}(p)} \text{ at } \nu = 0).
$$
Since $\mathrm{d}z/\mathrm{d}y = (y/\mu)^{\nu-1}/(\sigma\mu) > 0$ for $y > 0$, $F$ is a valid increasing CDF. Two special cases give independent oracles: at $\nu = 1$, BCCG is $N(\mu, \mu\sigma)$; at $\nu = 0$, it is $\mathrm{LogNormal}(\log\mu, \sigma)$. The mean is not $\mu$ (which is the median); the implementation uses the second-order approximation $E[Y] \approx \mu(1 + \tfrac{1}{2}\sigma^2(1-\nu))$, exact at $\nu \in \{0, 1\}$.

#### BCT and BCPE: the same spine, a heavier-tailed $z$

BCT and BCPE add a fourth parameter $\tau > 0$ (log link) by replacing the standard normal that $z$ follows. The Box-Cox spine ($z$, $\partial z/\partial\nu$, the Jacobian) is **unchanged**; only $\log h(z)$, the $\mu/\sigma/\nu$ scores' dependence on $\mathrm{d}\ell/\mathrm{d}z$, and the new $\tau$ column differ. Writing $D = -\mathrm{d}\ell/\mathrm{d}z$ (so $D = z$ for the normal), the $\eta$-scale scores keep one shape:
$$
u_\mu = \frac{D\,T}{\sigma} - \nu, \qquad
u_\sigma = zD - 1, \qquad
u_\nu = -D\,\frac{\partial z}{\partial \nu} + \log(y/\mu).
$$

**BCT** (Box-Cox-$t$): $z \sim t_\tau$ (Student-$t$, $\tau$ degrees of freedom), so $\log h(z) = \log\Gamma(\tfrac{\tau+1}{2}) - \log\Gamma(\tfrac{\tau}{2}) - \tfrac12\log(\pi\tau) - \tfrac{\tau+1}{2}\log(1 + z^2/\tau)$. The robustifying weight $w_t = (\tau+1)/(\tau+z^2)$ gives $D = w_t z$, so $u_\mu = w_t zT/\sigma - \nu$, $u_\sigma = w_t z^2 - 1$. The $\tau$ score and information mirror `StudentT` (see [STUDENT-T])'s df parameter; $F(y) = T_\tau(z)$. As $\tau \to \infty$, $t \to$ normal and BCT $\to$ BCCG.

**BCPE** (Box-Cox power-exponential): $z$ follows a variance-standardized power-exponential with shape $\tau$. With $c^2 = 2^{-2/\tau}\,\Gamma(1/\tau)/\Gamma(3/\tau)$,
$$
\log h(z) = N(\tau) - \tfrac12\bigl|z/c\bigr|^\tau, \qquad
N(\tau) = \log\tau - \log 2 - \tfrac32\log\Gamma(1/\tau) + \tfrac12\log\Gamma(3/\tau),
$$
so $D = \tfrac{\tau}{2c}|z/c|^{\tau-1}\operatorname{sign}(z)$ and $u_\sigma = \tfrac{\tau}{2}|z/c|^\tau - 1$. The CDF is the regularized incomplete gamma $F(y) = \tfrac12 + \tfrac12\operatorname{sign}(z)\,P\!\bigl(1/\tau,\ \tfrac12|z/c|^\tau\bigr)$, inverted for the quantile via a $\mathrm{Gamma}(1/\tau, 1)$ quantile. $\tau = 2$ is the normal, so BCPE $\to$ BCCG; $\tau < 2$ is leptokurtic, $\tau > 2$ platykurtic. The exact $\tau$ Fisher information uses the $\mathrm{Gamma}(1/\tau, 1)$ moments of $v = \tfrac12|z/c|^\tau$: writing $\partial\ell/\partial\tau = N'(\tau) + P v + Q\,v\log v$, $\;\mathrm{E}[(\partial\ell/\partial\tau)^2]$ closes in $\psi, \psi'$ at $a = 1/\tau$.

---

### [WORKED-EXAMPLE] Example: Computing Derivatives for Gaussian Data

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

### [CDF-TRIO] Cumulative Distributions, Densities, and Quantiles

Beyond the score/Fisher quantities that drive fitting, each family exposes the **distributional trio** — the cumulative distribution $F$, the density/mass $f$, and the quantile (inverse CDF) $Q$. These are what every statistically distinctive GAMLSS output needs (quantile residuals in [RESIDUALS], centile curves, predictive intervals).

#### Definitions

$$
F(y \mid \theta) = P(Y \le y \mid \theta), \qquad
f(y \mid \theta) = \begin{cases} \dfrac{\partial F}{\partial y} & \text{continuous} \\[4pt] F(y) - F(y^-) & \text{discrete} \end{cases}, \qquad
Q(p \mid \theta) = \inf\{\, y : F(y \mid \theta) \ge p \,\}.
$$

The density is obtained for free from the pointwise log-likelihood of [GAUSSIAN]–[OCAT]:
$$
f(y \mid \theta) = \exp\bigl(\ell_{\text{pointwise}}(y \mid \theta)\bigr),
$$
which is exact because every family's $\ell_{\text{pointwise}}$ returns a **normalized** log-density (continuous) or log-mass (discrete). Discrete families set an `is_discrete` flag; their $F$ is the right-continuous step function evaluated at $\lfloor y \rfloor$.

#### Why the trio matters: the probability integral transform

If $Y$ is continuous with CDF $F$, then
$$
U = F(Y \mid \theta) \sim \text{Uniform}(0,1) \quad\Longrightarrow\quad r = \Phi^{-1}(U) \sim N(0,1).
$$
So $r_i = \Phi^{-1}(F(y_i \mid \hat\theta_i))$ is standard normal whenever the fitted model is correct, **regardless of the family** — the foundation of the randomized quantile residuals of [RESIDUALS].

#### Per-family CDF and quantile

Each family maps its gamlss (mean/dispersion) parameters to the standard arguments of one special function. All special functions are in `statrs`: the error function $\mathrm{erf}$; the regularized lower / upper incomplete gamma $P(a,x) = \gamma(a,x)/\Gamma(a)$ and $Q(a,x) = 1 - P(a,x)$; and the regularized incomplete beta $I_x(a,b) = B(x;a,b)/B(a,b)$.

| Family | gamlss params | standard params | $F(y \mid \theta)$ | $Q(p \mid \theta)$ |
| --- | --- | --- | --- | --- |
| Gaussian | $\mu,\sigma$ | loc $\mu$, sd $\sigma$ | $\tfrac12\bigl[1 + \mathrm{erf}\bigl(\tfrac{y-\mu}{\sigma\sqrt2}\bigr)\bigr]$ | $\mu + \sigma\sqrt2\,\mathrm{erf}^{-1}(2p-1)$ |
| Gamma | $\mu,\sigma$ | shape $\alpha=1/\sigma^2$, scale $s=\mu\sigma^2$ | $P(\alpha,\,y/s)$ | $\text{Gamma}(\alpha,1/s)^{-1}(p)$ |
| Weibull | $\mu,\sigma$ | scale $\mu$, shape $\sigma$ | $1 - e^{-(y/\mu)^\sigma}$ | $\mu\,[-\ln(1-p)]^{1/\sigma}$ |
| Student-t | $\mu,\sigma,\nu$ | std $t$ on $z=(y-\mu)/\sigma$ | $T_\nu(z)$ | $\mu + \sigma\,T_\nu^{-1}(p)$ |
| Beta | $\mu,\phi$ | $a=\mu\phi,\; b=(1-\mu)\phi$ | $I_y(a,b)$ | $\text{Beta}(a,b)^{-1}(p)$ |
| Poisson *(disc.)* | $\mu$ | rate $\mu$ | $Q(\lfloor y\rfloor + 1,\, \mu)$ | search (below) |
| Neg. binomial *(disc.)* | $\mu,\sigma$ | size $r=1/\sigma$, prob $q=r/(r+\mu)$ | $I_q(r,\, \lfloor y\rfloor + 1)$ | search |
| Binomial *(disc.)* | $n$ (state), $\mu$ | $n$ trials, prob $\mu$ | $I_{1-\mu}(n-\lfloor y\rfloor,\, \lfloor y\rfloor + 1)$ | search |
| Ocat *(disc.)* | $\mu,\delta$ thresholds | cumulative-logit | $\mathrm{logistic}(\theta_{\lfloor y\rfloor} - \mu)$, $=1$ at $y \ge R$ | smallest level $r$ with $F(r) \ge p$ |

Note Beta uses the **precision** parameterization $(\mu,\phi)$ consistent with [BETA] and the glossary, so $a=\mu\phi$ and $b=(1-\mu)\phi$ (not a $(\mu,\sigma)$ form).

#### Discrete CDFs via incomplete special functions

The discrete CDFs avoid a summation loop through a special-function identity. For the Poisson (representative of the discrete cases),
$$
F(m \mid \mu) = \sum_{k=0}^{m} e^{-\mu}\frac{\mu^k}{k!} = Q(m+1, \mu) = \frac{\Gamma(m+1, \mu)}{\Gamma(m+1)}, \qquad m = \lfloor y \rfloor,
$$
i.e. the regularized **upper** incomplete gamma at an integer first argument equals the Poisson CDF. The negative-binomial and binomial CDFs are the regularized incomplete beta evaluations in the table; Ocat is the proportional-odds cumulative-logit $P(Y \le r) = \mathrm{logistic}(\theta_r - \mu)$ (the same $F_k$ used in [OCAT]).

#### Discrete quantile search

For the discrete families, $Q$ is the smallest integer $k$ with $F(k) \ge p$, found by a doubling-bracket then bisection (the shared `discrete_quantile` helper):
$$
\text{double } hi \text{ until } F(hi) \ge p, \quad \text{then bisect } [lo, hi] \text{ to the smallest } k \text{ with } F(k) \ge p,
$$
which is $O(\log k)$ even for large means. The right-continuous convention — evaluating the discrete CDF at $\lfloor y \rfloor$ — gives the identity
$$
F(k) - F(k-1) = f(k) \quad \text{(pmf at integer support)},
$$
which the randomized quantile residuals of [RESIDUALS] rely on to de-lump each atom.

### [STRUCTURAL] Structural Likelihoods: Censoring, Truncation, and Hurdles

These are **not new families** — they are transformations of a base family's
likelihood given extra per-observation information the analyst supplies. Each is
a *wrapper distribution* that holds a boxed base `Distribution` and fits the base
family's parameters through the ordinary RS loop; only `loglik_pointwise` and
`derivatives` change. Throughout, $F$ and $f$ are the base CDF and density, and
for a parameter $\theta$ with linear predictor $\eta_\theta$ write
$F' = \partial F / \partial \eta_\theta$ and $F'' = \partial^2 F / \partial \eta_\theta^2$.
The score is $u = \partial \ell / \partial \eta$ and the IRLS working weight is the
**observed information** $w = -\partial^2 \ell / \partial \eta^2$, floored at
$\texttt{MIN\_WEIGHT}$ to keep $X^T W X + S_\lambda$ positive definite (large
steps are then tamed by the step-halving line search of [BACKFIT]).

#### [STRUCT-1] Censoring

Each observation is exact, or known only to lie below / above / within an
interval. The pointwise log-likelihood swaps the density for a survival /
interval probability built from the base CDF:
$$
\ell =
\begin{cases}
\log f(y) & \text{event (exact } y) \\
\log S(y) = \log\!\big(1 - F(y)\big) & \text{right-censored } (Y > y) \\
\log F(y) & \text{left-censored } (Y \le y) \\
\log\!\big(F(\text{hi}) - F(y)\big) & \text{interval } [y, \text{hi}]
\end{cases}
$$
Differentiating with respect to $\eta$ gives the score and observed-information
weight per censored row:
$$
\text{right:} \quad u = \frac{-F'}{1 - F}, \quad w = \frac{F''}{1 - F} + \left(\frac{F'}{1 - F}\right)^2,
$$
$$
\text{left:} \quad u = \frac{F'}{F}, \quad w = -\frac{F''}{F} + \left(\frac{F'}{F}\right)^2,
$$
$$
\text{interval } (D = F(\text{hi}) - F(y)): \quad u = \frac{D'}{D}, \quad w = -\frac{D''}{D} + \left(\frac{D'}{D}\right)^2,
$$
with $D' = F'(\text{hi}) - F'(y)$, $D'' = F''(\text{hi}) - F''(y)$. Event rows keep
the base family's $(u, w)$, so an all-event `Censored` reduces exactly to the base
log-likelihood. Source: `src/distributions/censored.rs`.

#### [STRUCT-2] Truncation

The response is observed only within $(\text{lo}, \text{hi})$; out-of-range values
are *absent*, not censored, so the density renormalizes by the in-support mass
$D = F(\text{hi}) - F(\text{lo})$:
$$
\ell = \log f(y) - \log D, \qquad \text{lo} < y < \text{hi}.
$$
The score and weight add the normalizer's contribution to the base family's:
$$
u = u_{\text{base}} - \frac{D'}{D}, \qquad w = w_{\text{base}} + \frac{D''}{D} - \left(\frac{D'}{D}\right)^2 .
$$
A $(-\infty, \infty)$ truncation reduces to the base. The wrapper's CDF/quantile
are renormalized onto the truncated support, $F_T(y) = (F(y) - F(\text{lo}))/D$.
Source: `src/distributions/truncated.rs`.

#### [STRUCT-3] Hurdle / two-part

A point mass at zero plus a *zero-truncated* base for the positive part, with a
logit-linked atom $\xi = P(Y = 0)$:
$$
\ell =
\begin{cases}
\log \xi & y = 0 \\
\log(1 - \xi) + \log f(y) - \log\!\big(1 - F(0)\big) & y > 0
\end{cases}
$$
The $\xi$ atom is a Bernoulli on the zero indicator; with the logit link
$\eta_\xi = \operatorname{logit}\xi$,
$$
u_\xi = \mathbf{1}(y = 0) - \xi, \qquad w_\xi = \xi(1 - \xi).
$$
The base parameters receive the zero-truncation score (truncating at $0$) on the
positive rows and nothing on the zero rows. Contrast with zero-*inflation*, where
the base can still emit $0$. Source: `src/distributions/hurdle.rs`.

#### [STRUCT-CDF-ETA] Analytic CDF $\eta$-derivatives

The censoring/truncation score and weight need $F'$ and $F''$ for each parameter.
The `Distribution::cdf_eta_derivatives` hook supplies these **analytically for the
location/scale parameters**, while non-elementary **shape-parameter** derivatives
(which would require differentiating the regularized incomplete gamma/beta with
respect to its shape argument) fall back to a central difference of the base
`cdf` on that parameter's $\eta$. For $z = (y - \mu)/\sigma$ with standardized
density $g$ (and $g' = \mathrm{d}g/\mathrm{d}z$):
$$
\text{location } \mu \;(\text{identity}): \quad \frac{\partial F}{\partial \eta} = -\frac{g}{\sigma}, \quad \frac{\partial^2 F}{\partial \eta^2} = \frac{g'}{\sigma^2},
$$
$$
\text{scale } \sigma \;(\text{log}): \quad \frac{\partial F}{\partial \eta} = -z\,g, \quad \frac{\partial^2 F}{\partial \eta^2} = z\,g + z^2 g'.
$$
**Gaussian** uses $g = \varphi$, $g' = -z\varphi$, so $\partial^2 F/\partial \eta_\sigma^2 = z\varphi(1 - z^2)$.
**Student-t** uses the standardized $t$ density with $g' = -g\,(\nu+1)z/(\nu + z^2)$ for $\mu, \sigma$; its $\nu$ is numeric.
**Gamma** treats $\mu$ (log link) analytically: $\mu$ enters only the scale
$\theta = \mu\sigma^2$ at fixed shape $\alpha = 1/\sigma^2$, so with $x = y/\theta$,
$$
\frac{\partial F}{\partial \eta_\mu} = -\frac{x^{\alpha} e^{-x}}{\Gamma(\alpha)}, \qquad \frac{\partial^2 F}{\partial \eta_\mu^2} = (x - \alpha)\,\frac{\partial F}{\partial \eta_\mu},
$$
while Gamma's $\sigma$ (which enters the shape) is numeric; **Beta** (both shape
parameters) is fully numeric. Source: `gaussian.rs` / `student_t.rs` / `gamma.rs`
(`cdf_eta_derivatives`), `src/distributions/structural.rs` (numeric fallback +
the score/weight composition above).

---

## 2. Special Functions

### [DIGAMMA] Digamma Function

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

### [TRIGAMMA] Trigamma Function

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

### [PARAM-BOUNDS] Parameter Bounds

To prevent numerical issues, parameters are clamped to safe ranges:

**Minimum positive value**: $10^{-10}$
- Used for: $\mu$ (when must be positive), $\sigma$, $\phi$, probabilities

**Minimum weight**: $10^{-6}$
- Fisher information weights are floored at this value to ensure positive definiteness of the weight matrix

**Link function bounds**:
- Log/logit links: $\eta \in [-30, 30]$
- Prevents overflow in $\exp(\eta)$ (since $e^{30} \approx 10^{13}$ and $e^{-30} \approx 10^{-14}$)

### [DIST-SAFEGUARDS] Distribution-Specific Safeguards

**Student-t**:
- Enforce $\nu > 2$ (finite variance) via the floored log link on $\nu$ (§7)
- Ensure $\nu + z^2 \geq 10^{-10}$ to prevent division issues in robustifying weight

**Gamma**:
- Clamp $\sigma$ to reasonable range to prevent extreme shape parameters

**Beta**:
- Clamp $y$ to $(10^{-10}, 1-10^{-10})$ before computing $\log(y)$ and $\log(1-y)$
- Clamp $\mu$ similarly before computing logit

**Negative Binomial**:
- Ensure $1 + \sigma\mu$ stays positive and bounded away from zero

**Ordered Categorical**:
- Floor each category probability $\pi_r$ at $\mathrm{MIN\_PROB} = 10^{-10}$ before computing scores and Fisher info
- Renormalize: divide all $\pi_r$ by their sum after flooring, so that $\sum_r \pi_r = 1$ is maintained

### [BATCHED-SPECFUN] Batched Special Function Computation

Digamma and trigamma functions are computed in batched vectorized form for performance:

- For $n < 10{,}000$ observations: sequential computation
- For $n \geq 10{,}000$: parallel computation via Rayon (when `parallel` feature enabled)

The switchpoint is empirically tuned to balance overhead vs. speedup.

---

## 4. GAMLSS Iteration (P-IRLS)

### [BACKFIT] Backfitting Outer Loop

GAMLSS fits the joint model by cycling through distribution parameters one at a time, holding the others fixed (Rigby & Stasinopoulos 2005, §3). For a $P$-parameter family (e.g.\ Gaussian: $P=2$; Student-t: $P=3$):

```text
for cycle in 0..max_iterations:
    all_converged = true
    for each parameter theta_k in family.parameters():
        eta_k_old = eta_k
        eta_k     = p_irls_step(theta_k | other params held fixed)   # eta = X * beta
        delta     = max_abs(eta_k - eta_k_old)
        scale     = max(max_abs(eta_k), 1.0)   # floor at 1 for O(1) predictors
        if delta / scale >= tolerance:
            all_converged = false
    gd = global_deviance(all params)           # -2 * sum log f(y_i | theta_i)
    if all_converged and abs(gd_prev - gd) < gd_tolerance:
        break
    gd_prev = gd
```

Source: `src/fitting/mod.rs`. Convergence requires **two tests to pass together**:

1. **Per-parameter relative change of the linear predictor** $\eta_k = X_k\beta_k$:

   $$
   \frac{\|\eta_k^{(t+1)} - \eta_k^{(t)}\|_\infty}{\max(\|\eta_k^{(t+1)}\|_\infty,\, 1)} < \epsilon
   $$

   for every $k$. The test is in **fit space** rather than coefficient space:
   penalized designs can carry fit-irrelevant coefficient ridges (a flat REML
   valley where the per-cycle $\lambda$ re-optimization jitters between
   fit-equivalent $(\lambda, \beta)$ pairs) along which $\beta$ never becomes
   stationary even though the model ($\eta$, $\mu$, the deviance) already is.
   Using each parameter's own scale prevents a large parameter (e.g. $\mu$ on
   un-normalized data) from masking drift in a small one (e.g. a log-scale
   $\sigma$ near 0); the floor of 1 keeps the test equivalent to an absolute
   threshold for $O(1)$ predictors.

2. **Absolute global-deviance change** $|GD^{(t)} - GD^{(t+1)}| <
   \epsilon_{GD}$, the same convention as R gamlss's `c.crit` (an earlier
   *relative* form scaled its slack with $|GD|$ and stopped large-deviance
   fits far short of the optimum).

The active defaults are $\epsilon = 10^{-3}$, $\epsilon_{GD} = 10^{-3}$, max
iterations $= 200$ (`DEFAULT_TOLERANCE`, `DEFAULT_GD_TOLERANCE`,
`DEFAULT_MAX_ITER` in `src/fitting/mod.rs`).

### [PIRLS-INNER] Inner P-IRLS Step (Fisher Scoring)

For each parameter $\theta_k$ (e.g., $\mu$, $\sigma$, $\nu$):

1. **Working response**:
   $$
   z_k = \eta_k + \frac{u_k}{w_k}
   $$
   where $u_k = \frac{\partial \ell}{\partial \eta_k}$ and $w_k$ is the (expected, or in some cases observed) Fisher information.

2. **Penalized weighted least squares** (see [PWLS-CHOLESKY] for the Cholesky solve):
   $$
   \hat{\beta}_k = \arg\min_\beta \|W_k^{1/2}(z_k - X_k\beta)\|^2 + \sum_j \lambda_j \beta^T S_j \beta
   $$

3. **Update linear predictor**:
   $$
   \eta_k^{(\mathrm{new})} = X_k \hat{\beta}_k
   $$

4. **Smoothing parameter selection** — GCV, REML, or Fellner–Schall (§8, [REML-LAML]).

5. **Step-halving on the penalized deviance** (FIT-1): the accepted update is
   $\beta_k^{(t)} + \alpha\, d_k$ with $d_k = \hat\beta_k - \beta_k^{(t)}$ and
   $\alpha \in \{1, \tfrac12, \tfrac14, \dots\}$ backtracked until

   $$
   GD(\beta_k^{(t)} + \alpha d_k) + \sum_j \lambda_j\, (\beta_k^{(t)} + \alpha
   d_k)^T S_j (\beta_k^{(t)} + \alpha d_k)
   $$

   does not increase. The objective must be the **penalized** deviance: the
   Fisher direction $d_k$ is an ascent direction for the penalized
   log-likelihood only, and when the current $\beta$ is wigglier than the
   penalized optimum (e.g. $\lambda$ grew across cycles) a raw-deviance line
   search rejects every step and freezes the fit at a non-stationary point.
   The penalty change along the path is evaluated in the cancellation-free
   form $2\alpha\, d^T S_\lambda \beta + \alpha^2 d^T S_\lambda d$ so that a
   clamp-ceiling $\lambda$ (~$e^{30}$) cannot swamp the comparison with
   round-off. If the backtracking floor ($\alpha = 2^{-10}$) is reached with
   the objective still increasing, the block update is **rejected** ($\alpha =
   0$) rather than forced: accepting uphill micro-steps allows unbounded slow
   divergence.

Source: `src/fitting/scoring.rs`.

### [WORKING-RESPONSE] Working-Response Derivation

Why is $z = \eta + u/w$ the right pseudo-response? Expand the log-likelihood for one observation in $\eta$ around the current iterate $\eta_0$:

$$
\ell(\eta) \approx \ell(\eta_0) + u(\eta_0)\,(\eta - \eta_0) - \tfrac{1}{2}\,w(\eta_0)\,(\eta - \eta_0)^2
$$

where $u = \partial \ell/\partial \eta$ and $w = -\mathbb{E}[\partial^2 \ell/\partial \eta^2]$ (Fisher information, positive). Maximizing the quadratic Taylor surrogate over $\eta$ gives the one-Newton-step update $\eta - \eta_0 = u/w$. Setting $\eta = X\beta$ and stacking observations yields the Gauss–Newton normal equations:

$$
X^T W X \,\Delta\beta = X^T W \,(u/w)
$$

with $\Delta\beta = \beta - \beta_0$ and the diagonal weight $W = \mathrm{diag}(w_i)$. Equivalently, regress the **working response** $z = \eta_0 + u/w$ on $X$ with weights $W$:

$$
X^T W X \,\beta = X^T W z.
$$

Adding the penalty $\sum_j \lambda_j \beta^T S_j \beta$ to the surrogate produces the PWLS system of [PWLS-CHOLESKY]. This is exactly the IRLS construction of McCullagh & Nelder (1989) generalized to one distributional parameter at a time.

### [INNER-SAFEGUARDS] Numerical Safeguards in the Inner Loop

Cataloged here so the magic constants are auditable. All apply per observation, per inner iteration. Source: `src/fitting/scoring.rs`, `src/distributions/links.rs`.

| Symbol | Value | Where | Purpose |
| --- | --- | --- | --- |
| `MIN_WEIGHT` | $10^{-6}$ | `scoring.rs` | Floor on $w_i$ so the weight matrix stays positive definite; near-zero Fisher info would blow up $u/w$. Hits counted in `weight_floor_hits`. |
| `MAX_STEP` | $10^{6}$ | `scoring.rs` | Clamps $u_i/w_i$ in $\eta$-units: a pure anti-overflow guard for degenerate score/information combinations. It must be **large**: a tight clamp (an earlier value of 20) silently biased the working response, because when many rows clip, the update direction is decided by the *count* of positive vs negative rows rather than the score-weighted aggregate, which can point the Fisher step uphill. Overshoot robustness is provided by the deviance-guarded step-halving instead. Hits counted in `step_cap_hits`. |
| `MAX_ETA`, `MIN_ETA` | $\pm 30$ | `links.rs` | Clamps $\eta$ before applying the inverse log/logit link so $\exp(\eta)$ stays finite ($e^{30} \approx 10^{13}$). |
| `MIN_POSITIVE` | $10^{-10}$ | `links.rs`, `diagnostics.rs` | Floor on parameters that must be strictly positive ($\sigma$, $\phi$, probabilities), and on variances before the square-root in residual computation. |

Persistent non-zero `weight_floor_hits` or `step_cap_hits` at convergence signals model misspecification or ill-conditioned data; transient hits in early iterations are normal.

### [PRIOR-WEIGHTS] Prior / Observation Weights

Each call to the P-IRLS `step` function accepts an optional `prior_weights: Option<&Array1<f64>>` — a per-observation scale applied to each observation's likelihood contribution. When absent, a ones vector is substituted (uniform weights).

Let $\mathrm{safe\_w}_i = \max(w_i, \mathrm{MIN\_WEIGHT})$ where $w_i$ is the distribution's Fisher information for observation $i$.

**Working response — unchanged by prior weights**:
$$
z_i = \eta_i + \mathrm{clip}\!\left(\frac{u_i}{\mathrm{safe\_w}_i},\; \pm\mathrm{MAX\_STEP}\right)
$$
The step size $u_i / \mathrm{safe\_w}_i$ is a property of log-likelihood curvature and must not be scaled by the observation weight.

**Solve weight — carries the prior factor**:
$$
W_{ii} = \mathrm{prior}_i \cdot \mathrm{safe\_w}_i
$$

**Effect on the penalized normal equations**: because $z$ divides by $\mathrm{safe\_w}$ while $W$ multiplies by $\mathrm{prior} \cdot \mathrm{safe\_w}$, the prior weight scales each observation's contribution to both $X^T W X$ and $X^T W z$ exactly once — standard frequency / precision-weight semantics without distorting the IRLS step length:
$$
(X^T W X + S_\lambda)\hat{\beta} = X^T W z, \qquad W = \mathrm{diag}(\mathrm{prior}_i \cdot \mathrm{safe\_w}_i).
$$

When `prior_weights = None`, the formula reduces to $W = \mathrm{diag}(\mathrm{safe\_w}_i)$, identical to the unweighted case.

Source: `src/fitting/scoring.rs` (`step`).

### [STRUCT-4] Finite Mixtures via EM

A $K$-component finite mixture has density
$$
f(y) = \sum_{k=1}^{K} w_k\, g_k(y), \qquad w_k \ge 0, \quad \sum_k w_k = 1,
$$
where each component $g_k$ is a GAMLSS of the same family with its own fitted
parameters. The components' responsibilities couple the observations, so — unlike
the per-row structural wrappers of [STRUCTURAL] — a mixture cannot be a single
`Distribution`; it is fit by an **EM outer loop** over the existing
prior-weighted RS fit ([PRIOR-WEIGHTS]):

- **E-step** — posterior responsibility that observation $i$ came from component $k$:
$$
r_{ik} = \frac{w_k\, g_k(y_i)}{\sum_{j} w_j\, g_j(y_i)} .
$$
- **M-step** — refit each component by the prior-weighted RS fit with observation
  weights $r_{\cdot k}$, then update the mixing weights
  $w_k = \tfrac{1}{n}\sum_i r_{ik}$.

Iteration continues until the mixture log-likelihood
$$
\ell_{\text{mix}} = \sum_i \log \sum_k w_k\, g_k(y_i)
$$
stops improving (relative change below tolerance); EM makes $\ell_{\text{mix}}$
monotone non-decreasing. Initialization assigns each observation to the nearest of
$K$ randomly drawn seed observations (a 1-D $k$-means seeding that breaks the
symmetric "all components identical" fixed point), and a small responsibility
floor keeps every component non-empty. The fitted `MixtureModel` reports
$\ell_{\text{mix}}$, the component models, the weights, and AIC/BIC with effective
degrees of freedom $\sum_k \mathrm{EDF}_k + (K - 1)$ (the $K-1$ free mixing
weights). Source: `src/fitting/mixture.rs` (`fit_mixture`, `MixtureModel`).

---

## 5. Smoothing and Penalty Matrices

### [BSPLINE-BASIS] B-Spline Basis (Cox–de Boor)

A degree-$p$ B-spline basis with $k$ functions is defined on a knot vector
$$
\tau_0 < \tau_1 < \cdots < \tau_{k+p}.
$$

**Knot placement (P-spline, `bs="ps"`).** The implementation uses **equally-spaced** knots — the requirement of the Eilers–Marx P-spline, because the difference penalty $D^T D$ only approximates a roughness integral when the basis knots are uniform:
$$
\tau_j = \min(x) + (j - p)\,\Delta, \qquad \Delta = \frac{\max(x) - \min(x)}{k - p}, \quad j = 0, \ldots, k+p.
$$
This places $\tau_p = \min(x)$ and $\tau_k = \max(x)$, with $p$ extra knots extending beyond each end of the data range. The anchoring range $(\min x, \max x)$ is resolved **once from the training data** and stored on the term (`PSpline1D::range`, `TensorProduct::range_1/2`), so prediction on a grid, a subset, or out-of-range points is evaluated on the *training* knot grid: re-deriving the range from the prediction data (the previous behavior) silently applied the coefficients to a different basis. Source: `src/splines/pspline.rs` (`select_knots`, `create_basis_matrix_with_range`), `src/fitting/assembler.rs` (`resolve_terms`).

The **Cox–de Boor recursion** evaluates the basis stably (Piegl & Tiller 1997, Algorithm A2.2). Let $i$ be the **knot span** — the index satisfying $\tau_i \le x < \tau_{i+1}$, clamped to $[p, k-1]$. Initialize $N_0 = 1$. Then for $j = 1, \ldots, p$:

$$
\mathrm{left}[j] = x - \tau_{i+1-j}, \qquad \mathrm{right}[j] = \tau_{i+j} - x,
$$

and for $r = 0, \ldots, j-1$:

$$
\text{denom} = \mathrm{right}[r+1] + \mathrm{left}[j-r], \qquad \text{term} = N_r / \text{denom},
$$
$$
N_r \leftarrow N_r^{\mathrm{saved}} + \mathrm{right}[r+1]\cdot\text{term}, \qquad N_r^{\mathrm{saved}} \leftarrow \mathrm{left}[j-r]\cdot\text{term}.
$$

At the end of iteration $j$, $N_0, \ldots, N_j$ hold the $j+1$ non-zero degree-$j$ basis values at $x$. **Critical implementation note:** $\mathrm{left}[j]$ and $\mathrm{right}[j]$ must be computed *once, before the inner $r$-loop*, so that the full arrays $\mathrm{left}[1..j]$ and $\mathrm{right}[1..j]$ remain valid when the inner loop reads $\mathrm{left}[j-r]$ and $\mathrm{right}[r+1]$. Computing them inside the $r$-loop overwrites earlier slots and produces wrong interior basis values for degree $\ge 3$ (the endpoint functions are unaffected but the middle ones are wrong; both the correct and the broken form still satisfy partition-of-unity, so property-only tests do not catch the error).

After the full degree-$p$ pass, $N_0, \ldots, N_p$ are the $p+1$ non-zero values at $x$; they map to columns $[i-p, i]$ of the basis matrix. Source: `src/splines/pspline.rs` (`create_basis_matrix_with_range`, `evaluate_basis_functions_into`, `find_knot_span`).

Properties:

- **Partition of unity**: $\sum_{j=1}^{k} B_j(x_i) = 1$ for every $x_i$ in the interior of the knot range.
- **Non-negativity**: $B_j(x) \ge 0$.
- **Local support**: $B_{i,p}$ is non-zero only on $[\tau_i, \tau_{i+p+1}]$, so each row of the design matrix has at most $p+1$ non-zero entries.

### [SUM-TO-ZERO] Sum-to-Zero Reparameterization (Identifiability)

Partition-of-unity creates a rank deficiency when the model has both an intercept and a smooth term: the column vector $\mathbf{1}_n$ lies in the column space of $B$, so the design matrix $[\mathbf{1}_n \mid B]$ is column-rank-deficient. To restore identifiability, the smooth is reparameterized through an orthonormal basis of the subspace orthogonal to $\mathbf{1}_k$.

The implementation uses a **Householder reflector** (`src/splines/`). Choose
$$
v = \mathbf{1}_k + \sqrt{k}\, e_1
$$
which reflects $\mathbf{1}_k$ onto $-\sqrt{k}\,e_1$. Form
$$
H = I_k - \frac{2}{\|v\|^2}\, v v^T, \qquad Z = H_{:,2:k}
$$
i.e.\ drop the first column of $H$. Then $Z \in \mathbb{R}^{k \times (k-1)}$ satisfies

$$
Z^T Z = I_{k-1}, \qquad Z^T \mathbf{1}_k = 0.
$$

Reparameterize $\beta = Z\gamma$ to get the constrained basis and penalty:
$$
B_{\mathrm{new}} = B Z, \qquad S_{\mathrm{new}} = Z^T S\, Z.
$$

The new design matrix $[\mathbf{1}_n \mid B Z]$ has full column rank, and the constraint $\mathbf{1}_k^T \beta = 0$ — i.e.\ the smooth has mean zero across knots — is enforced automatically.

### [DESIGN-TERMS] Design-Matrix Terms: Factors, Interactions, and Offsets (DATA-1/2/3)

The assembler (`src/fitting/assembler.rs`) turns a parameter's term list into design
columns. Beyond the intercept, linear, and smooth blocks above, three parametric term
kinds expand here.

**Factors (contrast coding).** A categorical column with $L$ distinct level codes
expands into $L-1$ columns under a chosen contrast, so a factor sharing the parameter
with an intercept stays identifiable (the dropped degree of freedom is the baseline /
grand mean). With levels sorted $\ell_0 < \ell_1 < \dots < \ell_{L-1}$:

- **Treatment** (R `contr.treatment`, the default): column $j$ is the indicator
  $\mathbb{1}[g_i = \ell_{j+1}]$, with $\ell_0$ the baseline. Coefficient $j$ is the
  mean shift of level $\ell_{j+1}$ relative to $\ell_0$.
- **Sum-to-zero** (R `contr.sum`): column $j$ is $+1$ for level $\ell_j$, $-1$ for the
  last level $\ell_{L-1}$, and $0$ otherwise. Each column sums to zero over the levels,
  so the coefficients are contrasts against the grand mean and the last level is
  $-\sum_j \beta_j$.

Levels are resolved once from the training column and stored on the term, so prediction
replays the identical column mapping; an unseen level at predict time maps to an all-zero
row (the honest "no information" encoding).

**Interactions.** The term $a\!:\!b$ is the row-wise Kronecker product of the two
operands' design blocks: if $A \in \mathbb{R}^{n\times p}$ and $B \in \mathbb{R}^{n\times q}$
are the operand columns, the interaction block is $C \in \mathbb{R}^{n\times pq}$ with
$$
C_{i,\,(r-1)q + s} = A_{i r}\, B_{i s},
$$
the same `row_kronecker_into` primitive used for tensor smooths. Factor$\times$continuous
and factor$\times$factor both fall out of this product once factors are expanded; the
crossing $a\!*\!b$ desugars to $a + b + a\!:\!b$.

**Offsets.** An offset is a known per-row term entering the linear predictor with a fixed
coefficient of $1$:
$$
\eta = X\beta + o, \qquad o_i = \log(\text{exposure}_i)\ \text{(typical rate model)}.
$$
It contributes to $\eta$, not to $\beta$, so it carries no design column. In the
Rigby–Stasinopoulos working response ([WORKING-RESPONSE]) the solver fits $X\beta$ to
$z = \eta + u/w$; since $X\beta = \eta - o$, the offset is subtracted from the working
response before the penalized least-squares solve, $z' = z - o$, and $\eta$ is
reconstructed as $X\beta + o$ afterward. The solver therefore stays offset-unaware, and a
fit with offset $o$ is identical to folding $o$ into the response on the link scale (a
closed-form check for the identity link).

### [PSPLINE-PENALTY] P-Spline Penalty (Eilers & Marx 1996)

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

### [TENSOR-PENALTY] Tensor Product Smooth and Anisotropic Penalty

For a bivariate smooth $f(x_1, x_2)$ built from marginal bases $B_1 \in \mathbb{R}^{n \times k_1}$ and $B_2 \in \mathbb{R}^{n \times k_2}$, the joint basis is the **row-wise Kronecker product** of the marginals:
$$
B[i, :] = B_1[i, :] \otimes B_2[i, :] \in \mathbb{R}^{k_1 k_2}, \qquad B \in \mathbb{R}^{n \times (k_1 k_2)}.
$$
The coefficient vector $\beta \in \mathbb{R}^{k_1 k_2}$ is indexed lexicographically so $\beta_{(i-1) k_2 + j}$ multiplies $B_1[\cdot, i] \cdot B_2[\cdot, j]$.

Anisotropic penalties are obtained by Kronecker'ing each marginal difference penalty with an identity matrix:
$$
S^{(1)} = S_1 \otimes I_{k_2}, \qquad S^{(2)} = I_{k_1} \otimes S_2
$$
with two independent smoothing parameters:
$$
\lambda_1 \beta^T S^{(1)} \beta + \lambda_2 \beta^T S^{(2)} \beta.
$$
$S^{(1)}$ penalizes roughness in the $x_1$ direction at every $x_2$ slice; $S^{(2)}$ vice versa. Choosing two $\lambda$'s lets the GCV/REML routine pick different smoothness for each margin (Wood 2017, §5.6).

When an intercept is present, **one** sum-to-zero constraint ([SUM-TO-ZERO]) is applied to the **full** tensor basis, transforming both penalties with the same $Z \in \mathbb{R}^{k_1 k_2 \times (k_1 k_2 - 1)}$:
$$
B_{\mathrm{new}} = B Z, \qquad S^{(j)}_{\mathrm{new}} = Z^T S^{(j)} Z, \quad j = 1, 2.
$$
This is mgcv's `te()` treatment ($k_1 k_2 - 1$ coefficients) and removes exactly the constant function: the only direction collinear with the intercept, since the row-Kronecker of two partition-of-unity bases is itself partition-of-unity ($B\,\mathbf{1}_{k_1 k_2} = \mathbf{1}_n$).

Centering each **margin** before the Kronecker product ($B_1 Z_1 \otimes_{\text{row}} B_2 Z_2$) is *not* equivalent: the product of two constant-free marginal spaces excludes every function of the form $f(x_1)\cdot 1$ and $1\cdot g(x_2)$ (both main effects), leaving a pure-interaction (`ti()`-style) smooth that cannot represent additive structure. An earlier implementation made exactly this error. Consequence of the correct construction: the penalty null space after centering has dimension $d_1 d_2 - 1$ (e.g. $3$ for order-2 margins, matching mgcv), and a formula combining `te(x1, x2)` with a separate main effect of $x_1$ on the same parameter is rank-deficient: use a pure-interaction decomposition for that, as in mgcv.

Source: `src/fitting/assembler.rs` (`assemble_smooth`), `src/splines/reparam.rs`.

### [RANDEF-PENALTY] Random Effect Penalty

For an observation-to-group map $g : \{1, \ldots, n\} \to \{1, \ldots, G\}$, the design matrix is the binary indicator
$$
Z \in \{0, 1\}^{n \times G}, \qquad Z[i, g] = \mathbf{1}\{g(i) = g\}.
$$

Coupled with the **identity penalty** $S = I_G$, the model
$$
\eta = \mathbf{1} \beta_0 + Z \alpha, \qquad \lambda \alpha^T \alpha = \lambda \sum_{g=1}^G \alpha_g^2
$$
is equivalent to the Bayesian random-intercept prior $\alpha_g \sim N(0, 1/\lambda)$ — the ridge-penalized empirical Bayes interpretation. Since each row of $Z$ sums to 1, the sum-to-zero reparameterization of [SUM-TO-ZERO] is again applied when an intercept is present:
$$
Z_{\mathrm{new}} = Z\,Z_{\mathrm{c}} \in \mathbb{R}^{n \times (G-1)}, \qquad Z_{\mathrm{c}}^T I_G Z_{\mathrm{c}} = I_{G-1}.
$$

The group-to-column map is resolved **once at fit time** (levels sorted for
determinism) and stored on the term, so prediction maps each group to the
coefficient it was fitted with regardless of the row order or group subset of
the new data; an unseen level at predict time is an error (mgcv factor
semantics). Source: `src/fitting/assembler.rs` (`resolve_terms`,
`assemble_smooth`).

---

### [BLOCK-SPARSE] Block-Sparse Penalty Matrices

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
- $\mathrm{EDF} \approx 2$ for a linear fit (intercept + slope)
- $\mathrm{EDF} \approx$ number of basis functions for an unpenalized spline
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
- **Use**: Poisson rate, Gamma mean, scale parameters ($\sigma$)
- **Derivative**: $\frac{d\mu}{d\eta} = \mu$
- **Numerical safeguard**: Clamp $\eta \in [-30, 30]$ to prevent overflow

### Floored Log Link
- **Function**: $g(\mu) = \log(\max(\mu, c))$
- **Inverse**: $g^{-1}(\eta) = \max(\exp(\eta), c)$
- **Domain**: $\mu \geq c$
- **Use**: Student-t degrees of freedom $\nu$ with floor $c = 2$, keeping the variance $\sigma^2\nu/(\nu-2)$ finite as the optimizer explores the heavy-tail region
- **Derivative**: $\frac{d\mu}{d\eta} = \mu$ above the floor (the floor binds only transiently and never when the true $\nu$ is well above 2)
- **Numerical safeguard**: same $\eta \in [-30, 30]$ clamp as the log link, plus the lower floor $c$. See `FlooredLogLink` in `src/distributions/links.rs`.

### Logit Link
- **Function**: $g(\mu) = \log\left(\frac{\mu}{1-\mu}\right)$
- **Inverse**: $g^{-1}(\eta) = \frac{1}{1 + e^{-\eta}} = \frac{e^\eta}{1 + e^\eta}$
- **Domain**: $\mu \in (0, 1)$
- **Use**: Binomial probability, Beta mean
- **Derivative**: $\frac{d\mu}{d\eta} = \mu(1-\mu)$
- **Numerical safeguards**:
  - Clamp $\mu \in [10^{-10}, 1-10^{-10}]$ before applying link
  - Clamp $\eta \in [-30, 30]$ to prevent overflow in inverse

### Probit Link
- **Function**: $g(\mu) = \Phi^{-1}(\mu)$, the standard-normal quantile
- **Inverse**: $g^{-1}(\eta) = \Phi(\eta)$
- **Domain**: $\mu \in (0, 1)$
- **Use**: Binomial probability — the latent-normal alternative to logit (econometrics, bioassay)
- **Derivative**: $\frac{d\mu}{d\eta} = \phi(\eta) = \frac{1}{\sqrt{2\pi}}e^{-\eta^2/2}$, the standard-normal density
- **Reuse**: $\Phi$ is the same `erf`-based CDF used by `Gaussian::cdf`; $\Phi^{-1}$ is `math::std_normal_quantile`
- **Numerical safeguards**: clamp $\mu \in [10^{-10}, 1-10^{-10}]$, $\eta \in [-30, 30]$

### Complementary Log-Log Link
- **Function**: $g(\mu) = \log(-\log(1-\mu))$
- **Inverse**: $g^{-1}(\eta) = 1 - \exp(-e^{\eta})$
- **Domain**: $\mu \in (0, 1)$ (asymmetric: approaches the bounds at different rates)
- **Use**: discrete-time survival / complementary-risk, asymmetric rare-event data
- **Derivative**: $\frac{d\mu}{d\eta} = \exp(\eta - e^{\eta})$
- **Numerical safeguards**: clamp $\eta \in [-30, 30]$ before the inner $\exp$

### Inverse Link
- **Function**: $g(\mu) = 1/\mu$
- **Inverse**: $g^{-1}(\eta) = 1/\eta$
- **Domain**: $\mu \neq 0$ ($\mu$ and $\eta$ share a sign)
- **Use**: canonical link for the Gamma family
- **Derivative**: $\frac{d\mu}{d\eta} = -1/\eta^2$
- **Numerical safeguard**: keep $|\eta| \ge 10^{-10}$ (sign-preserving floor) so $1/\eta$ stays finite

### Inverse-Square Link
- **Function**: $g(\mu) = 1/\mu^2$
- **Inverse**: $g^{-1}(\eta) = \eta^{-1/2}$
- **Domain**: $\mu > 0$, $\eta > 0$
- **Use**: positive responses (e.g. an inverse-Gaussian mean)
- **Derivative**: $\frac{d\mu}{d\eta} = -\tfrac{1}{2}\eta^{-3/2}$
- **Numerical safeguard**: floor $\eta \ge 10^{-10}$

### Square-Root Link
- **Function**: $g(\mu) = \sqrt{\mu}$
- **Inverse**: $g^{-1}(\eta) = \eta^2$
- **Domain**: $\mu \ge 0$
- **Use**: Poisson variance-stabilizing link
- **Derivative**: $\frac{d\mu}{d\eta} = 2\eta$

### Cauchit Link
- **Function**: $g(\mu) = \tan(\pi(\mu - \tfrac{1}{2}))$
- **Inverse**: $g^{-1}(\eta) = \tfrac{1}{2} + \frac{\arctan(\eta)}{\pi}$
- **Domain**: $\mu \in (0, 1)$
- **Use**: bounded probability with heavier tails than logit/probit — robust to extreme points in the linear predictor
- **Derivative**: $\frac{d\mu}{d\eta} = \frac{1}{\pi(1 + \eta^2)}$
- **Numerical safeguards**: clamp $\mu \in [10^{-10}, 1-10^{-10}]$, $\eta \in [-30, 30]$

All six are selectable by name (`"probit"`, `"cloglog"`, `"inverse"`, `"inverse_square"`, `"sqrt"`, `"cauchit"`) via `FitConfig::links` and reconstructed through `distributions::link_from_name`; each supplies the analytic $\frac{d\mu}{d\eta}$ above so the IRLS weights below are exact rather than finite-differenced.

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

### [PWLS-CHOLESKY] Efficient PWLS Solution via Cholesky Decomposition

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

**Robust log-determinant** (`log_det_robust`, `src/linalg.rs`): For the REML cost term $\log|X^T W X + S_\lambda|$, the Cholesky log-det is tried first; on failure (evenly-spaced B-spline design matrices can develop tiny negative floating-point pivots) the symmetric eigensolver `dsyev` is used, clamping eigenvalues to $10^{-300}$ before summing the logs. This means the REML objective gracefully returns a very negative log-det rather than an error, steering the L-BFGS optimizer away from degenerate $\lambda$ regions.

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

### [REML-LAML] REML / Laplace-Approximate Marginal Likelihood

GCV is one option for picking the smoothing parameters; the default `criterion` in `FitConfig` is `Reml` (see `src/fitting/mod.rs`). REML in this setting is the Laplace-approximate marginal likelihood (LAML) of Wood (2011), evaluated at the working PWLS step. The selector is exposed as a three-variant enum:

```rust
pub enum SmoothingCriterion { Gcv, Reml /* default */, FellnerSchall }
```

`Gcv` minimizes the score of §8 via L-BFGS on $\log \lambda$. `Reml` minimizes $-V_r$ (below) via L-BFGS on $\log \lambda$. `FellnerSchall` targets the same $-V_r$ via a deterministic multiplicative fixed-point ([FELLNER-SCHALL]).

#### The LAML objective

Let $S_\lambda = \sum_j \lambda_j S_j$. The working-model LAML negative log-marginal likelihood is
$$
-V_r(\lambda) = \tfrac{1}{2}\,\log\bigl|X^T W X + S_\lambda\bigr|
\;-\;\tfrac{1}{2}\,\log\bigl|S_\lambda\bigr|_+
\;+\; \text{Gaussian PWLS terms (data, } \sigma\text{)}
$$
where $|\,\cdot\,|_+$ denotes the **pseudo-determinant** — the product of strictly positive eigenvalues — required because $S_\lambda$ is rank-deficient (its null space contains the unpenalized polynomial directions: constants for order-1 penalties, lines for order-2, etc., plus the unpenalized intercept and linear columns).

$\log|X^T W X + S_\lambda|$ is computed via `log_det_robust` (see [PWLS-CHOLESKY]): Cholesky on the fast path; symmetric eigensolver fallback for near-PD matrices with tiny negative floating-point pivots (common for evenly-spaced B-spline designs), clamping eigenvalues to $10^{-300}$ before the log so the REML optimizer naturally avoids degenerate $\lambda$ regions instead of crashing.

Source: `src/fitting/solver.rs` (`RemlCost`); `src/linalg.rs` (`log_det_robust`).

#### Pseudo-determinant via grouped block eigendecomposition

`penalty_eigen` takes the full list of penalty matrices and their $\lambda$ values and processes them in **coefficient-block groups**: `penalty_nonzero_block_range` identifies the non-zero row/column range of each penalty, and all penalties that share the exact same range are combined into one group before eigendecomposition.

**For each group $g$:**

1. Form the combined scaled block: $B_g = \sum_{j \in g} \lambda_j S_j\big|_{\mathrm{block}}$
2. Symmetric eigendecompose: $B_g = Q_g\,\mathrm{diag}(d_1^{(g)}, \ldots)\,Q_g^T$
3. Per-group relative threshold: $\tau_g = \max\!\bigl(\varepsilon \cdot d_{\max}^{(g)},\; 10^{-300}\bigr)$, where $\varepsilon = \mathtt{REML\_RANK\_TOL\_EPS} = 10^{-8}$

Then:
$$
\log |S_\lambda|_+ = \sum_g \sum_{d_i^{(g)} > \tau_g} \log d_i^{(g)},
\qquad
S_\lambda^+ = \mathrm{block\text{-}diag}\!\bigl\{ B_g^+ \bigr\},
\qquad
M_p = \sum_g \mathrm{null\_dim}(B_g).
$$

**Why grouped decomposition?** A single global threshold $\tau = \varepsilon \cdot \max(d_{\max}, 1)$ is dominated by the largest-$\lambda$ term when smoothing parameters span many orders of magnitude (e.g.\ $\lambda_1 \approx 10^{-8}$ and $\lambda_4 \approx 5 \times 10^{10}$). This misclassifies small-$\lambda$ non-null directions as null space, flipping the REML gradient sign. The per-group threshold is relative to each block's own spectral norm, eliminating this pollution. The $10^{-300}$ hard floor prevents collapse when $d_{\max}^{(g)} \approx 0$ (very small $\lambda$).

**Tensor-product smooths**: the two marginal penalties $\lambda_1(S_1 \otimes I_{k_2})$ and $\lambda_2(I_{k_1} \otimes S_2)$ share the same $k_1 k_2$ coefficient block and are therefore combined before eigendecomposition, so the identity $(\lambda_1 S_1 + \lambda_2 S_2)^+ \ne (\lambda_1 S_1)^+ + (\lambda_2 S_2)^+$ is never violated. For disjoint blocks (one smooth per coefficient block) the formula reduces to a per-penalty decomposition.

Source: `src/fitting/solver.rs` (`penalty_eigen`, `penalty_nonzero_block_range`).

#### REML gradient

From Wood (2011, §3) the gradient with respect to $\log \lambda_j$ is
$$
\frac{\partial (-V_r)}{\partial \log \lambda_j}
= \tfrac{1}{2}\,\lambda_j\,\hat\beta^T S_j\, \hat\beta
\;-\;\tfrac{1}{2}\,\lambda_j\,\mathrm{tr}\!\bigl(S_\lambda^+\, S_j\bigr)
\;+\;\tfrac{1}{2}\,\lambda_j\,\mathrm{tr}\!\bigl(V\, S_j\bigr)
$$
with $V = (X^T W X + S_\lambda)^{-1}$ (working scale $\varphi = 1$; the noise
scale enters through $W$). The implementation evaluates this analytically,
validates it against central finite differences in `reml_tests`, and feeds it
to L-BFGS; final $\log\lambda$ values are clamped to
$[-\mathtt{LOG\_LAMBDA\_CLAMP}, \mathtt{LOG\_LAMBDA\_CLAMP}] = [-30, 30]$.

**Fellner–Schall polish.** L-BFGS with a Moré–Thuente line search can stall at
a warm-start-dependent, non-stationary point when the LAML surface has flat
ridges (several smooths collapsing onto their null space with $\lambda$ at the
clamp ceiling). The resulting per-cycle $\lambda$ jitter prevents the outer RS
loop from ever seeing a stationary $\eta$. `run_optimization_reml` therefore
polishes the L-BFGS output with the deterministic Fellner–Schall fixed point
(§8.3) on the same target and keeps whichever $\lambda$ scores better, so the
polish can never worsen the fit; a linear-algebra failure inside the polish
falls back to the L-BFGS result.

Source: `src/fitting/solver.rs`.

### [FELLNER-SCHALL] Fellner–Schall Multiplicative Fixed Point

Wood & Fasiolo (2017) show that for the LAML target, the update
$$
\lambda_j^{(t+1)} \;=\; \lambda_j^{(t)} \cdot
\frac{\mathrm{tr}\!\bigl(S_\lambda^+\, S_j\bigr) - \mathrm{tr}\!\bigl(V\, S_j\bigr)}
{\hat\beta^T S_j\, \hat\beta}
$$
is **monotone non-increasing** in $-V_r$ under mild conditions, has no line search, and converges geometrically near a minimum. The implementation lives at `src/fitting/solver.rs` (`run_optimization_fellner_schall`) and applies three safeguards:

| Safeguard | Constant | Value | Role |
| --- | --- | --- | --- |
| Relative floor on the numerator $\mathrm{tr}(S_\lambda^+ S_j) - \mathrm{tr}(V S_j)$ | `FS_NUMERATOR_FLOOR_REL` | $10^{-12}$ | Prevents tiny / occasionally negative numerators from collapsing the update. |
| Floor on the denominator $\hat\beta^T S_j \hat\beta$ | `FS_DENOMINATOR_FLOOR` | $10^{-12}$ | Avoids division by zero when $\hat\beta$ lies in the null space of $S_j$. |
| Per-iteration clamp on $\log \lambda_j$ | `LOG_LAMBDA_CLAMP` | $\pm 30$ | Keeps $\lambda$ in $[e^{-30}, e^{30}]$, matching the REML L-BFGS clamp. |

Loop control: at most `FS_MAX_ITERS = 50` iterations; converged when
$$
\max_j \bigl|\log \lambda_j^{(t+1)} - \log \lambda_j^{(t)}\bigr| < \mathtt{FS\_TOL} = 10^{-4}.
$$
The tolerance is set at $10^{-4}$ (rather than the looser $10^{-3}$) because the F-S update is first-order convergent: stopping at $10^{-3}$ leaves $\lambda$ still moving by $\approx 0.1\%$ per step, which is measurable on flat LAML landscapes. The extra iterations cost little since each F-S step is a single PWLS solve. Source: `src/fitting/solver.rs` (`FS_TOL`, `FS_MAX_ITERS`).

#### When to pick which criterion

- **GCV**: classical choice, computationally cheap; tends to undersmooth at moderate $n$ and is more prone to multiple local minima on the GCV surface (Reiss & Ogden 2009). The GCV optimizer always runs L-BFGS from the warm-started $\lambda$ and does not apply an early-exit threshold — a previous skip triggered when the warm-start GCV score was below $10^{-6}$ could freeze $\lambda$ for parameters with small residual sums of squares (e.g.\ a log-scale $\sigma$ whose working RSS is tiny), preventing those parameters from adapting their smoothness across outer cycles.
- **REML** (default): preferred for stability; the Laplace-approximate marginal likelihood is asymptotically equivalent to true REML for the working PWLS model, and tends to undersmooth less than GCV at moderate sample sizes.
- **FellnerSchall**: same target as REML but with a deterministic multiplicative update — no L-BFGS, no line search. Fast and well-behaved for well-conditioned problems; can stall if the numerator drifts to its floor.

### [RESTART-GUARD] Basin Probes (Collapse and Corner Guards)

For a **single** P-spline, the $\lambda$-objective ($-V_r$ or the GCV score) is unimodal in $\log\lambda$ with one interior optimum, but it flattens into a near-horizontal **shelf** at large $\lambda$ where the smooth has been driven entirely onto its penalty null space (EDF $\to$ null dimension; for an order-2 penalty after centering, a straight line). On the shelf the gradient $\approx 0$ and, for Fellner–Schall, $\hat\beta^T S_j \hat\beta \to 0$ so the multiplicative ratio explodes upward; either optimizer can become stuck there.

For **multi-penalty** terms (the anisotropic tensor penalty $\lambda_1 S_1\!\otimes\!I + \lambda_2 I\!\otimes\!S_2$) the surface is genuinely **multimodal**: spurious stationary points appear as *corners* (one margin's $\lambda$ pinned at the clamp ceiling or the $\lambda$ floor while the true optimum has it interior), and they come in shapes no cheap detector reliably catches (observed on real data: a corner with $\mathrm{EDF} = 16.0$ scoring 5 LAML units worse than the interior optimum at $\mathrm{EDF} = 20.6$, which matches mgcv's selection). Gradient descent from any single seed can be captured by a basin boundary.

The selector in `src/fitting/scoring.rs::step` therefore probes alternative basins and keeps the $\lambda$ with the best objective value (`lambda_cost`):

1. **Trigger.** Every term, single- or multi-penalty, probes when a smooth's EDF has fallen to within `EDF_COLLAPSE_SLACK` $= 0.5$ of its null dimension, or when any $\lambda$ sits at the $\log\lambda$ clamp bounds; the trigger re-fires every cycle the state stays suspicious (an early probe against unconverged working weights finds nothing, so a skip-once rule would never re-probe at the converged state where the rescue is decidable). A prior revision *also* probed multi-penalty terms **unconditionally on every cycle**, to catch the one corner shape the cheap triggers miss: a margin's $\lambda$ merely very large but not pinned at a bound. That was reverted: each firing runs the full seed battery below (a $7^k$ grid plus several L-BFGS/Fellner–Schall solves, each an eigendecomposition on the term's coefficient block), so it cost $\sim\!30$ s (OpenBLAS) to $>2$ min (pure-rust) per default-$10\times10$ tensor fit in an unoptimized build, hanging the debug test suites, while the payoff (mgcv EDF parity on specific sweep seeds) is validated only by the `#[ignore]`d `benchmark/run_comparison.sh` comparison, not by any CI/pre-push test. The ceiling/floor corners are still caught by the bound trigger; only the merely-large-interior sub-case is dropped, within the sweep's 20–25% EDF tolerance.
2. **Seeds.** (a) A low-$\lambda$ restart seed $\exp(\log\lambda_{\text{cold}} - \mathtt{RESTART\_LOG\_OFFSET})$ with offset $8$, below both the interior optimum and the shelf; (b) a fresh cold start from the trace-ratio heuristic; (c) per-coordinate variants of the incumbent with each bound-pinned $\lambda_j$ individually dropped to the restart level, the targeted escape for tensor corners; (d) for terms with $\leq 2$ penalties, L-BFGS started from the best cell of a coarse $7^k$ grid of $\log\lambda$ offsets $\{-16,-12,\dots,+8\}$ around the cold start. The grid evaluation is derivative-free, so it cannot be captured by a basin boundary.
3. **Acceptance.** Every candidate (and the incumbent) is scored by the actual criterion value; the minimum wins. This makes the guard safe in both directions: a genuinely null-space-optimal fit (a strictly linear truth under an order-2 penalty) has the *better* marginal likelihood at the collapsed $\lambda$ and is preserved; only a spuriously collapsed or corner-trapped fit, where another basin scores better, is repaired.

Source: `src/fitting/scoring.rs` (`step`), `src/fitting/solver.rs` (`restart_seed`, `lambda_cost`, `initial_log_lambda`, `RESTART_LOG_OFFSET`).

**Determinism.** One historical *trigger* for spurious single-smooth collapse is the nondeterministic reduction order of multi-threaded OpenBLAS. This repo pins BLAS to a single thread for its own `cargo` runs (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` in `.cargo/config.toml`), making the dense linear algebra reproducible; under single-thread BLAS the recovery control cases collapse in $0/20$ repeats (`tests/lambda_bistability.rs`). The basin probes above are retained as defense-in-depth for multi-threaded execution paths, and, via the collapse/bound trigger, for the corner traps of multi-penalty terms.

---

## 9. Mean-Variance Relationships

A key feature distinguishing different distributions is how variance relates to the mean. This determines which distribution is appropriate for different data structures.

| Distribution | Mean | Variance | Relationship |
|--------------|------|----------|--------------|
| **Gaussian** | $\mu$ | $\sigma^2$ | Constant (homoskedastic) |
| **Poisson** | $\mu$ | $\mu$ | Equidispersion: $\text{Var} = \text{Mean}$ |
| **Gamma** | $\mu$ | $\mu^2\sigma^2$ | Proportional to $\mu^2$ (constant CV) |
| **Weibull** | $\mu\Gamma(1+1/\sigma)$ | $\mu^2[\Gamma(1+2/\sigma)-\Gamma(1+1/\sigma)^2]$ | Shape $\sigma$ controls skew and tail |
| **Negative Binomial** | $\mu$ | $\mu + \sigma\mu^2$ | Overdispersion relative to Poisson |
| **Binomial** | $n\mu$ | $n\mu(1-\mu)$ | Quadratic in mean, max at $\mu=0.5$ |
| **Beta** | $\mu$ | $\frac{\mu(1-\mu)}{1+\phi}$ | Max at $\mu=0.5$, decreases with $\phi$ |
| **Student-t** | $\mu$ | $\frac{\nu\sigma^2}{\nu-2}$ | Heavier tails than Gaussian |
| **Ordered Categorical** | $\sum_r r\,\pi_r$ | $\sum_r r^2\,\pi_r - (\mathbb{E}Y)^2$ | Discrete $\{1,\ldots,R\}$; depends on all thresholds |

### Choosing a Distribution

**Ordinal responses**:
- Fixed ordered categories $\{1, \ldots, R\}$: Ordered Categorical (`Ocat`)

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
   - Gaussian: $\bar{y}$ (sample mean)
   - Student-t: $\mathrm{median}(y)$ — a robust location seed. The sample mean is pulled by the heavy tails, biasing the first robustifying weights $w = (\nu+1)/(\nu+z^2)$.
   - Poisson/Gamma/NB: $\bar{y}$ (then apply log link)
   - Binomial: $\sum_i y_i / \sum_i n_i$ (pooled across observations so per-row trial counts do not bias the seed), clamped to $(0.1, 0.9)$
   - Beta: $\bar{y}$ (sample mean, clamped to $(0.1, 0.9)$)

2. **$\sigma$**:
   - Gaussian: $s_y$ (sample std dev)
   - Student-t: $1.4826 \cdot \mathrm{MAD}(y)$ — the MAD-to-$\sigma$ consistency factor for a normal core. A raw sample SD overestimates the scale under heavy tails. Floored at $10^{-4}$ (falling back to $1.0$).
   - Gamma: $\hat\sigma_0 = s_y / \bar{y}$ (sample CV), clamped to $[0.05, 10.0]$. $\sigma$ parameterizes the coefficient of variation, not the raw SD; a raw-SD start makes REML over-penalize the $\sigma$ smooth on the first RS iteration and warm-start into a full-collapse trap.
   - NB: the method-of-moments overdispersion estimate $(s_y^2 - \bar{y})/\bar{y}^2$, clamped to $[0.1, 10.0]$ (falling back to $0.5$ when undefined, e.g. $n = 1$). $\sigma$ is the overdispersion coefficient of $\mathrm{Var} = \mu + \sigma\mu^2$, so a raw-SD start is on the wrong scale entirely, often $10$–$30\times$ too large for count data.
   - Beta: 1.0

3. **$\nu$** (Student-t): $5.0$ — a fixed moderate seed, deliberately **not** a sample-kurtosis estimate. For a regression model the *marginal* kurtosis of $y$ reflects the spread of the mean structure rather than the noise tails, so inverting $\kappa = 6/(\nu-4)$ biases $\nu$; in the multi-smooth weighted case this biased seed tipped the optimizer into a degenerate over-smoothed basin. $5$ sits well clear of the $\nu > 2$ finite-variance boundary. Source: `StudentT::initial_value` in `src/distributions/student_t.rs`.

4. **$\phi$** (Beta): 1.0

5. **Smoothing parameters $\lambda$**: initialized from the trace-ratio cold-start heuristic
   $$
   \log\lambda_j^{(0)} = \log\!\left(\frac{\mathrm{tr}(X^T X)}{\mathrm{tr}(S_j)}\right)
   $$
   per penalty (unweighted, using the assembled $X$ for the current distribution parameter). The all-ones start $\lambda = 1$ can leave $X^T W X + S_\lambda$ near-singular for high-cardinality bases (e.g.\ $k = 20$) or prior-weighted models where $X^T W X$ is scaled by large prior weights. Source: `src/fitting/mod.rs`, `src/fitting/solver.rs` (`initial_log_lambda`).

### Convergence Criterion

The algorithm declares convergence when every distribution parameter $\theta_k$ independently satisfies the relative criterion:
$$
\frac{\|\beta_k^{(t+1)} - \beta_k^{(t)}\|_\infty}{\max\!\bigl(\|\beta_k^{(t+1)}\|_\infty,\; 1\bigr)} < \epsilon,
$$
where $k$ indexes parameters ($\mu$, $\sigma$, etc.) and $\epsilon = 10^{-3}$ by default. Each parameter is checked against its **own** coefficient scale rather than a global scale pooled across all parameters. The $\max(\cdot, 1)$ floor makes the test behave like an absolute threshold when all coefficients are $O(1)$ (typical for normalized data).

Using a shared global scale — dividing the maximum change across all parameters by the maximum $|\beta|$ across all parameters — was the previous behaviour. It caused the loop to declare convergence prematurely when one parameter (e.g.\ $\mu$) had large coefficients and dominated the denominator while another (e.g.\ log-scale $\sigma$) was still drifting in relative terms.

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

### [INFO-CRITERIA] Information Criteria

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

### [DEVIANCE] Deviance

The **deviance** measures lack of fit:
$$
D = 2[\ell_{\text{saturated}} - \ell_{\text{fitted}}]
$$

where $\ell_{\text{saturated}}$ is the log-likelihood of a saturated model (perfect fit).

For nested models, the deviance difference follows approximately $\chi^2$ distribution:
$$
D_1 - D_2 \sim \chi^2_{\text{EDF}_2 - \text{EDF}_1}
$$

### [RESIDUALS] Residuals

**Quantile (randomized quantile) residuals** (Dunn & Smyth 1996). For a **continuous** response:
$$
r_i = \Phi^{-1}(F(y_i | \hat{\theta}_i))
$$

where $F$ is the fitted CDF and $\Phi^{-1}$ is the inverse standard normal CDF. If the model is correct, $r_i \sim N(0,1)$ by the probability integral transform ([CDF-TRIO]).

For a **discrete** response, $F$ jumps, so $F(Y)$ is not uniform; the *randomized* PIT spreads each atom across its jump interval. With $v_i \sim \text{Uniform}(0,1)$:
$$
a_i = F(y_i - 1 \mid \hat\theta_i), \quad b_i = F(y_i \mid \hat\theta_i), \quad u_i = a_i + v_i\,(b_i - a_i), \quad r_i = \Phi^{-1}(u_i).
$$

The fitted CDF $F$, quantile $Q$, and the `is_discrete` flag that selects the branch are provided by the distributional trio of [CDF-TRIO]; the right-continuity identity $F(k) - F(k-1) = f(k)$ is what makes $u_i$ uniform on the jump interval.

**Deviance residuals**:
$$
r_i^{(D)} = \text{sign}(y_i - \hat{\mu}_i) \sqrt{d_i}
$$

where $d_i$ is the contribution of observation $i$ to the total deviance.

**Pearson residuals**:
$$
r_i^{(P)} = \frac{y_i - \hat{\mu}_i}{\sqrt{\widehat{\text{Var}}(Y_i)}}
$$

### [WORM-PLOTS] Worm Plots

For each parameter $\theta_k$, plot quantile residuals against fitted quantiles:
- X-axis: Normal quantiles $\Phi^{-1}((i-0.5)/n)$
- Y-axis: Sorted quantile residuals

If the model is correct, points should lie on a horizontal line at zero. Systematic deviations indicate:
- U-shape: Underdispersion
- Inverse U-shape: Overdispersion
- S-shape: Skewness issues
- Linear trend: Location shift

### [QQ-PLOTS] Q-Q Plots

Plot theoretical quantiles against sample quantiles of residuals. Should be approximately linear if residuals are normal.

### [LINK-TEST] Goodness of Link Test

To test if link $g$ is appropriate, fit augmented model:
$$
g(\theta) = X\beta + \gamma [g(\theta)]^2
$$

Test $H_0: \gamma = 0$ using likelihood ratio test.

---

## 12. Code Map

A reverse index from mathematical concept to the canonical implementation, for contributors jumping between formula and source.

| Concept (this doc) | Source location |
| --- | --- |
| Backfitting outer loop with per-parameter convergence ([BACKFIT]) | `src/fitting/mod.rs` (`fit_gamlss`) |
| P-IRLS inner step ([PIRLS-INNER]) | `src/fitting/scoring.rs` (`step`) |
| `MIN_WEIGHT`, `MAX_STEP` ([INNER-SAFEGUARDS]) | `src/fitting/scoring.rs:28,32` |
| Prior / observation weights ([PRIOR-WEIGHTS]) | `src/fitting/scoring.rs` (`step`) |
| `MAX_ETA`, `MIN_ETA`, `MIN_POSITIVE` ([INNER-SAFEGUARDS]) | `src/distributions/links.rs:8-12` |
| `IdentityLink`, `LogLink`, `LogitLink` (§7) | `src/distributions/links.rs:24-59` |
| Gaussian derivatives ([GAUSSIAN]) | `src/distributions/gaussian.rs` |
| Student-t derivatives ([STUDENT-T]) | `src/distributions/student_t.rs` |
| Poisson derivatives ([POISSON]) | `src/distributions/poisson.rs` |
| Gamma derivatives ([GAMMA]) | `src/distributions/gamma.rs` |
| Weibull derivatives ([WEIBULL]) | `src/distributions/weibull.rs` |
| Negative Binomial derivatives ([NEG-BINOMIAL]) | `src/distributions/negative_binomial.rs` |
| Beta derivatives ([BETA]) | `src/distributions/beta.rs` |
| Binomial derivatives ([BINOMIAL]) | `src/distributions/binomial.rs` |
| Ordered categorical derivatives ([OCAT]) | `src/distributions/ocat.rs` |
| Censoring wrapper ([STRUCT-1]) | `src/distributions/censored.rs` |
| Truncation wrapper ([STRUCT-2]) | `src/distributions/truncated.rs` |
| Hurdle wrapper ([STRUCT-3]) | `src/distributions/hurdle.rs` |
| Analytic CDF $\eta$-derivatives + numeric fallback ([STRUCT-CDF-ETA]) | `src/distributions/{gaussian,student_t,gamma}.rs` (`cdf_eta_derivatives`), `src/distributions/structural.rs` |
| Finite mixtures via EM ([STRUCT-4]) | `src/fitting/mixture.rs` (`fit_mixture`, `MixtureModel`) |
| `FamilyDescriptor` serialization (SER-1) | `src/distributions/descriptor.rs` |
| Digamma / trigamma batched (§2) | `src/math.rs` |
| Distribution trait (§1) | `src/distributions/mod.rs` |
| CDF / PDF / quantile / `is_discrete` trait methods ([CDF-TRIO]) | `src/distributions/mod.rs` + per-family files |
| `discrete_quantile` bracket+bisection ([CDF-TRIO]) | `src/distributions/mod.rs` (`discrete_quantile`) |
| B-spline basis (de Boor) & uniform knots ([BSPLINE-BASIS]) | `src/splines/pspline.rs` (`create_basis_matrix_with_range`, `evaluate_basis_functions_into`, `select_knots`) |
| Sum-to-zero Householder $Z$ ([SUM-TO-ZERO]) | `src/splines/reparam.rs` (`sum_to_zero_basis`) |
| Difference penalty matrix (§5) | `src/splines/penalty.rs` (`create_penalty_matrix`) |
| Tensor product assembly (§5) | `src/fitting/assembler.rs` |
| Random effect assembly (§5) | `src/fitting/assembler.rs` |
| Block-sparse penalty ([BLOCK-SPARSE]) | `src/fitting/assembler.rs`, `src/types/newtypes.rs` |
| PWLS / Cholesky solve ([PWLS-CHOLESKY]) | `src/fitting/solver.rs` (`fit_pwls`, `fit_pwls_with_grad_info`) |
| Robust log-det fallback ([PWLS-CHOLESKY], [REML-LAML]) | `src/linalg.rs` (`log_det_robust`) |
| Initial-$\lambda$ trace-ratio heuristic (§10) | `src/fitting/mod.rs`, `src/fitting/solver.rs` (`initial_log_lambda`) |
| EDF (§6) | `src/fitting/solver.rs` (`fit_pwls_with_grad_info`) |
| GCV cost and gradient (§8) | `src/fitting/solver.rs` (`GamlssCost`) |
| GCV gradient finite-difference test | `src/fitting/solver.rs` (`gcv_gradient_matches_finite_diff`) |
| GCV L-BFGS driver (§8) | `src/fitting/solver.rs` (`run_optimization`) |
| REML / LAML cost and gradient ([REML-LAML]) | `src/fitting/solver.rs` (`RemlCost`) |
| REML gradient finite-difference test | `src/fitting/solver.rs` (`reml_gradient_matches_finite_diff`) |
| Penalty grouped pseudo-determinant / pseudo-inverse ([REML-LAML]) | `src/fitting/solver.rs` (`penalty_eigen`, `penalty_nonzero_block_range`) |
| Fellner–Schall iteration ([FELLNER-SCHALL]) | `src/fitting/solver.rs` (`run_optimization_fellner_schall`) |
| `SmoothingCriterion` enum ([REML-LAML]) | `src/fitting/mod.rs` |
| Defaults (`max_iterations=200`, `tolerance=1e-3`) | `src/fitting/mod.rs:31-32` |
| Diagnostics (AIC/BIC/EDF/residuals, §11) | `src/fitting/diagnostics.rs` |

---

## 13. Notation Glossary

Symbols used throughout this document.

| Symbol | Meaning |
| --- | --- |
| $y, Y$ | Response value / random variable |
| $n$ | Number of observations |
| $p$ | Total number of coefficients across all terms; also (in [BSPLINE-BASIS]) B-spline degree |
| $P$ | Number of distribution parameters (e.g.\ $P = 2$ for Gaussian) |
| $\theta_k$ | $k$-th distribution parameter ($\mu, \sigma, \nu, \phi$) |
| $\mu, \sigma, \nu, \phi$ | Location, scale, degrees of freedom (Student-t), precision (Beta) |
| $R$ | Number of ordered categories in `Ocat` ($R \in \{2,3,4,5\}$) |
| $\theta_k$ | $k$-th threshold in `Ocat` ($k = 1,\ldots,R-1$); also used generically for the $k$-th distribution parameter — context distinguishes |
| $\delta_k$ | $k$-th threshold increment parameter fed to the RS loop: $\theta_1 = \delta_1$ (identity link), $\theta_k = \theta_{k-1} + e^{\delta_k}$ for $k \ge 2$ (log link) |
| $F_k, f_k$ | Cumulative-logit CDF and logistic density at threshold $\theta_k$: $F_k = \mathrm{logistic}(\theta_k - \mu)$, $f_k = F_k(1-F_k)$ |
| $\pi_r$ | Category probability in `Ocat`: $\pi_r = F_r - F_{r-1}$, $F_0=0$, $F_R=1$ |
| $F(y\mid\theta)$ | Cumulative distribution $P(Y\le y\mid\theta)$; right-continuous step at $\lfloor y\rfloor$ for discrete families ([CDF-TRIO]) |
| $f(y\mid\theta)$ | Density (continuous) or mass (discrete) $=\exp(\ell_{\text{pointwise}})$ ([CDF-TRIO]) |
| $Q(p\mid\theta)$ | Quantile / inverse CDF: $\inf\{y:F(y\mid\theta)\ge p\}$ ([CDF-TRIO]) |
| $P(a,x),\, Q(a,x)$ | Regularized lower / upper incomplete gamma $\gamma(a,x)/\Gamma(a)$, $1-P(a,x)$ |
| $I_x(a,b)$ | Regularized incomplete beta $B(x;a,b)/B(a,b)$ |
| $\Phi, \Phi^{-1}$ | Standard normal CDF and its inverse (quantile) |
| $\mathrm{prior}_i$ | Per-observation prior / observation weight ([PRIOR-WEIGHTS]); defaults to 1 |
| $\mathrm{safe\_w}_i$ | Floored Fisher weight: $\max(w_i, \mathrm{MIN\_WEIGHT})$ |
| $g, g^{-1}$ | Link function and its inverse |
| $\eta = g(\theta)$ | Linear predictor for a distribution parameter |
| $X$ | Design matrix ($n \times p$); collects intercept, linear, and smooth basis columns |
| $\beta$ | Coefficient vector ($p \times 1$) |
| $W$ | Diagonal weight matrix; $W_{ii} = w_i$ = Fisher info for observation $i$ |
| $u_i = \partial \ell / \partial \eta_i$ | Score (first derivative of log-likelihood on $\eta$-scale) |
| $w_i = -\mathbb{E}\bigl[\partial^2 \ell / \partial \eta_i^2\bigr]$ | Fisher information on $\eta$-scale |
| $z = \eta + u/w$ | Working response |
| $S_j$ | $j$-th penalty matrix (one per smooth term direction) |
| $\lambda_j$ | Smoothing parameter for $S_j$ |
| $S_\lambda = \sum_j \lambda_j S_j$ | Aggregate penalty matrix |
| $V = (X^T W X + S_\lambda)^{-1}$ | PWLS coefficient covariance |
| $H = X V X^T W$ | Hat matrix (projection onto fitted values) |
| $\mathrm{EDF} = \mathrm{tr}(H) = \mathrm{tr}(V X^T W X)$ | Effective degrees of freedom |
| $\mathrm{RSS} = (z - X\hat\beta)^T W (z - X\hat\beta)$ | Weighted residual sum of squares |
| $\psi(x), \psi'(x)$ | Digamma, trigamma |
| $|A|_+$ | Pseudo-determinant of PSD matrix $A$ (product of strictly positive eigenvalues) |
| $A^+$ | Moore–Penrose pseudo-inverse of $A$ |
| $\otimes$ | Kronecker product |
| $\odot$ (in §5) | Row-wise Kronecker product (one row of $B_1 \otimes B_2$ per row index) |
| $\mathbf{1}_k$ | Column vector of $k$ ones |
| $I_k$ | $k \times k$ identity matrix |
| `[TAG]` | Cross-reference to the named subsection whose heading carries that bracketed tag (e.g. `[CDF-TRIO]`); chapter-level references use §N |

---

## References

1. Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized additive models for location, scale and shape. *Journal of the Royal Statistical Society: Series C*, 54(3), 507-554. — Core RS backfitting algorithm and observed-info choice in NB.

2. Eilers, P. H., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89-121. — P-spline difference penalty.

3. de Boor, C. (1978). *A Practical Guide to Splines*. Springer. — Cox–de Boor B-spline recursion.

4. Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapman and Hall/CRC. — Tensor-product smooths, identifiability constraints, GAM reference.

5. Wood, S. N. (2011). Fast stable restricted maximum likelihood and marginal likelihood estimation of semiparametric generalized linear models. *Journal of the Royal Statistical Society: Series B*, 73(1), 3-36. — Laplace-approximate REML for smoothing selection.

6. Wood, S. N., & Fasiolo, M. (2017). A generalized Fellner–Schall method for smoothing parameter optimization with application to Tweedie location, scale and shape models. *Biometrics*, 73(4), 1071-1081. — Multiplicative fixed-point update.

7. Reiss, P. T., & Ogden, R. T. (2009). Smoothing parameter selection for a class of semiparametric linear models. *JRSS B*, 71(2), 505-523. — Comparison of GCV vs REML behavior.

8. Craven, P., & Wahba, G. (1979). Smoothing noisy data with spline functions. *Numerische Mathematik*, 31, 377-403. — Original GCV.

9. McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman and Hall. — IRLS / Fisher scoring derivation underlying [WORKING-RESPONSE].

10. Abramowitz, M., & Stegun, I. A. (1972). *Handbook of Mathematical Functions*. Dover Publications. — Digamma / trigamma asymptotic expansions (6.3.18, 6.4.11).

11. Piegl, L., & Tiller, W. (1997). *The NURBS Book* (2nd ed.). Springer. — Algorithm A2.2: de Boor–Cox recurrence for stable B-spline evaluation; the canonical reference for the `left`/`right` array structure used in `evaluate_basis_functions_into`.

12. McCullagh, P. (1980). Regression models for ordinal data. *Journal of the Royal Statistical Society: Series B*, 42(2), 109-142. — Proportional-odds / cumulative-logit model; foundational reference for the `Ocat` distribution ([OCAT]).

13. Dunn, P. K., & Smyth, G. K. (1996). Randomized quantile residuals. *Journal of Computational and Graphical Statistics*, 5(3), 236-244. — Randomized probability integral transform for discrete responses ([CDF-TRIO], [RESIDUALS]).