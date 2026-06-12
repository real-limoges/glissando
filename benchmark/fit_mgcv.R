#!/usr/bin/env Rscript
# Fits mgcv models matching glissando's compare_fit.rs scenarios.
# Output JSON shape mirrors `FitResult` in compare_fit.rs so orchestrate.py
# can splice the two side-by-side under `glissando` / `mgcv` keys.
#
# Coverage: scenarios that mgcv handles natively (Gaussian, Poisson, Gamma,
# Negative Binomial, Beta — in linear or P-spline form), plus a Gaussian
# location-scale smooth via `gaulss` (gaussian_sigma_smooth). All gam() fits use
# method="REML" to match glissando's default smoothing criterion. Student-t
# scenarios are skipped (they need `gamlss`); orchestrate.py marks them
# mgcv_capable=FALSE.

suppressPackageStartupMessages({
  library(arrow)
  library(mgcv)
  library(jsonlite)
  library(optparse)
})

opts <- parse_args(OptionParser(option_list = list(
  make_option(c("--data"), type = "character"),
  make_option(c("--scenario"), type = "character"),
  make_option(c("--output"), type = "character")
)))

df <- read_parquet(opts$data)

elapsed_ms <- function(start) as.numeric(Sys.time() - start, units = "secs") * 1000

emit <- function(path, m, coefficients, edf, fit_time_ms,
                 fitted_sigma = list(), error_msg = NA) {
  result <- list(
    converged    = isTRUE(m$converged),
    iterations   = if (!is.null(m$iter)) as.integer(m$iter) else 0L,
    fit_time_ms  = fit_time_ms,
    coefficients = coefficients,
    fitted_mu    = as.list(unname(fitted(m))),
    fitted_sigma = fitted_sigma,
    edf          = edf,
    log_likelihood = as.numeric(stats::logLik(m)),
    aic          = AIC(m),
    error        = error_msg
  )
  write_json(result, path, auto_unbox = TRUE, pretty = TRUE, na = "null")
}

# ----- Linear (parametric) fitters ------------------------------------------------

fit_gaussian_linear <- function(df, output) {
  start <- Sys.time()
  m <- gam(y ~ x, data = df, family = gaussian(), method = "REML")
  emit(
    output, m,
    coefficients = list(
      mu        = unname(coef(m)),
      log_sigma = list(0.5 * log(m$sig2))
    ),
    edf = list(mu = sum(m$edf), sigma = 1.0),
    fit_time_ms = elapsed_ms(start)
  )
}

fit_gaussian_multiple <- function(df, output) {
  start <- Sys.time()
  m <- gam(y ~ x1 + x2 + x3, data = df, family = gaussian(), method = "REML")
  emit(
    output, m,
    coefficients = list(
      mu        = unname(coef(m)),
      log_sigma = list(0.5 * log(m$sig2))
    ),
    edf = list(mu = sum(m$edf), sigma = 1.0),
    fit_time_ms = elapsed_ms(start)
  )
}

fit_gaussian_large <- fit_gaussian_linear  # same model, larger n

fit_poisson_linear <- function(df, output) {
  start <- Sys.time()
  m <- gam(y ~ x, data = df, family = poisson(link = "log"), method = "REML")
  emit(
    output, m,
    coefficients = list(log_mu = unname(coef(m))),
    edf = list(mu = sum(m$edf)),
    fit_time_ms = elapsed_ms(start)
  )
}

fit_gamma_linear <- function(df, output) {
  start <- Sys.time()
  m <- gam(y ~ x, data = df, family = Gamma(link = "log"), method = "REML")
  log_sigma <- 0.5 * log(summary(m)$dispersion)
  emit(
    output, m,
    coefficients = list(
      log_mu    = unname(coef(m)),
      log_sigma = list(log_sigma)
    ),
    edf = list(mu = sum(m$edf), sigma = 1.0),
    fit_time_ms = elapsed_ms(start)
  )
}

fit_negative_binomial_linear <- function(df, output) {
  start <- Sys.time()
  m <- gam(y ~ x, data = df, family = nb(), method = "REML")
  log_sigma <- log(1.0 / m$family$getTheta(TRUE))
  emit(
    output, m,
    coefficients = list(
      log_mu    = unname(coef(m)),
      log_sigma = list(log_sigma)
    ),
    edf = list(mu = sum(m$edf), sigma = 1.0),
    fit_time_ms = elapsed_ms(start)
  )
}

fit_beta_linear <- function(df, output) {
  start <- Sys.time()
  m <- gam(y ~ x, data = df, family = betar(link = "logit"), method = "REML")
  log_phi <- log(m$family$getTheta(TRUE))
  emit(
    output, m,
    coefficients = list(
      logit_mu = unname(coef(m)),
      log_phi  = list(log_phi)
    ),
    edf = list(mu = sum(m$edf), phi = 1.0),
    fit_time_ms = elapsed_ms(start)
  )
}

# ----- Smooth (P-spline) fitters --------------------------------------------------

# `s(x, bs="ps", k=K)` requests a penalised B-spline basis of size K with second-order
# difference penalty — matches glissando's `Smooth::PSpline1D { degree: 3, penalty_order: 2 }`.

fit_gaussian_smooth <- function(df, output) {
  start <- Sys.time()
  m <- gam(y ~ s(x, bs = "ps", k = 20), data = df, family = gaussian(), method = "REML")
  emit(
    output, m,
    coefficients = list(
      mu_smooth = unname(coef(m)),
      log_sigma = list(0.5 * log(m$sig2))
    ),
    edf = list(mu = sum(m$edf), sigma = 1.0),
    fit_time_ms = elapsed_ms(start)
  )
}

fit_gaussian_quadratic <- fit_gaussian_smooth  # Rust uses the same smooth body

fit_poisson_smooth <- function(df, output) {
  start <- Sys.time()
  m <- gam(y ~ s(x, bs = "ps", k = 20), data = df, family = poisson(link = "log"), method = "REML")
  emit(
    output, m,
    coefficients = list(log_mu = unname(coef(m))),
    edf = list(mu = sum(m$edf)),
    fit_time_ms = elapsed_ms(start)
  )
}

fit_gamma_smooth <- function(df, output) {
  start <- Sys.time()
  m <- gam(y ~ s(x, bs = "ps", k = 20), data = df, family = Gamma(link = "log"), method = "REML")
  log_sigma <- 0.5 * log(summary(m)$dispersion)
  emit(
    output, m,
    coefficients = list(
      log_mu    = unname(coef(m)),
      log_sigma = list(log_sigma)
    ),
    edf = list(mu = sum(m$edf), sigma = 1.0),
    fit_time_ms = elapsed_ms(start)
  )
}

fit_negative_binomial_smooth <- function(df, output) {
  start <- Sys.time()
  m <- gam(y ~ s(x, bs = "ps", k = 20), data = df, family = nb(), method = "REML")
  log_sigma <- log(1.0 / m$family$getTheta(TRUE))
  emit(
    output, m,
    coefficients = list(
      log_mu    = unname(coef(m)),
      log_sigma = list(log_sigma)
    ),
    edf = list(mu = sum(m$edf), sigma = 1.0),
    fit_time_ms = elapsed_ms(start)
  )
}

# ----- Scale-smooth fitter (gaulss) -----------------------------------------------

# Gaussian location-scale with a P-spline on the scale. mgcv's `gaulss` family
# takes a two-formula list (mean, then the scale predictor) and returns a
# two-column `fitted` matrix: column 1 is the mean μ, column 2 is `1/σ` (the
# precision-style parameter gaulss actually models). We invert that column to
# recover σ on the same scale glissando reports (`fitted_values` for `sigma`).
# NOTE: validate the gaulss parameterization on first run — `run_comparison.sh`
# is the check; this fitter is exercised only when R + mgcv are present.
fit_gaussian_sigma_smooth <- function(df, output) {
  start <- Sys.time()
  m <- gam(list(y ~ 1, ~ s(x, bs = "ps", k = 20)), data = df, family = gaulss())
  fv <- fitted(m)
  mu_hat <- fv[, 1]
  sigma_hat <- 1 / fv[, 2]
  # gaulss fits via outer optimization and leaves `m$converged` NULL; the
  # convergence flag lives in `m$outer.info$conv` instead.
  converged <- !is.null(m$outer.info) && grepl("full conv", m$outer.info$conv, ignore.case = TRUE)
  result <- list(
    converged    = isTRUE(converged),
    iterations   = if (!is.null(m$outer.info$iter)) as.integer(m$outer.info$iter) else 0L,
    fit_time_ms  = elapsed_ms(start),
    # Smooth scenario: coefficients aren't compared (bases differ), so emit the
    # mean intercept and the full coefficient vector for reference only.
    coefficients = list(
      mu               = list(unname(coef(m))[1]),
      log_sigma_smooth = unname(coef(m))
    ),
    fitted_mu    = as.list(unname(mu_hat)),
    fitted_sigma = as.list(unname(sigma_hat)),
    edf          = list(mu = 1.0, sigma = sum(m$edf)),
    log_likelihood = as.numeric(stats::logLik(m)),
    aic          = AIC(m),
    error        = NA
  )
  write_json(result, output, auto_unbox = TRUE, pretty = TRUE, na = "null")
}

# ----- B1: Gaussian + prior weights -----------------------------------------------
# Replicates mgcv::bam(..., weights=w) semantics for the B1 listing-weighted model.
# Five P-spline smooths + one binary linear term; weights=1/n_obs_per_listing.

fit_b1_weighted_gaussian <- function(df, output) {
  start <- Sys.time()
  m <- gam(
    y ~ s(x1, bs = "ps", k = 20) +
        s(x2, bs = "ps", k = 20) +
        s(x3, bs = "ps", k = 20) +
        s(x4, bs = "ps", k = 20) +
        s(x5, bs = "ps", k = 20) +
        d1,
    data    = df,
    weights = df$weights,
    family  = gaussian(),
    method  = "REML"
  )
  emit(
    output, m,
    coefficients = list(
      mu        = unname(coef(m)),
      log_sigma = list(0.5 * log(m$sig2))
    ),
    edf = list(mu = sum(m$edf), sigma = 1.0),
    fit_time_ms = elapsed_ms(start)
  )
}

# ----- Dispatch -------------------------------------------------------------------

dispatch <- list(
  gaussian_linear           = fit_gaussian_linear,
  gaussian_multiple         = fit_gaussian_multiple,
  gaussian_large            = fit_gaussian_large,
  gaussian_smooth           = fit_gaussian_smooth,
  gaussian_quadratic        = fit_gaussian_quadratic,
  gaussian_sigma_smooth     = fit_gaussian_sigma_smooth,
  poisson_linear            = fit_poisson_linear,
  poisson_smooth            = fit_poisson_smooth,
  gamma_linear              = fit_gamma_linear,
  gamma_smooth              = fit_gamma_smooth,
  negative_binomial_linear  = fit_negative_binomial_linear,
  negative_binomial_smooth  = fit_negative_binomial_smooth,
  beta_linear               = fit_beta_linear,
  b1_weighted_gaussian      = fit_b1_weighted_gaussian
)

if (is.null(dispatch[[opts$scenario]])) {
  cat(sprintf("scenario '%s' not supported by this script\n", opts$scenario))
  quit(status = 1)
}

dispatch[[opts$scenario]](df, opts$output)
