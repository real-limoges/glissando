#!/usr/bin/env Rscript
# Fits mgcv models matching glissando's compare_fit.rs scenarios.
# Output JSON shape mirrors `FitResult` in compare_fit.rs so orchestrate.py
# can splice the two side-by-side under `glissando` / `mgcv` keys.
#
# Coverage:
#   Gaussian (linear, multiple, large, smooth, quadratic, sigma-smooth / gaulss)
#   Poisson  (linear, smooth)
#   Binomial (linear, smooth) — binomial(logit)
#   Gamma    (linear, smooth, sigma-smooth / gammals)
#   Student-t (linear, smooth) — scat() scaled-t family
#   NegBin   (linear, smooth) — nb()
#   Beta     (linear, smooth) — betar()
#   Tensor   (Gaussian te) — te(x1, x2, bs="ps")
#   RandomEf (Gaussian re) — s(g, bs="re")
#   CR spline (Gaussian cr) — s(x, bs="cr")
#   B1: Gaussian + prior weights; B2: StudentT + prior weights (scat)
#
# All gam() fits default to method="REML" to match glissando's default
# smoothing criterion.

suppressPackageStartupMessages({
  library(mgcv)
  library(jsonlite)
  library(optparse)
})

# Shared helpers (parquet reader with arrow -> nanoparquet fallback).
local({
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else "."
  source(file.path(dir, "common.R"), local = FALSE)
})

opts <- parse_args(OptionParser(option_list = list(
  make_option(c("--data"),     type = "character"),
  make_option(c("--scenario"), type = "character"),
  make_option(c("--output"),   type = "character")
)))

df <- read_parquet(opts$data)

elapsed_ms <- function(start) as.numeric(Sys.time() - start, units = "secs") * 1000

# Helper: extract smoothing parameters as a named list (informational only).
sp_list <- function(m) {
  tryCatch(
    as.list(unname(m$sp)),
    error = function(e) list()
  )
}

# Helper: convergence flag that works for both standard gam() and extended
# families (gaulss, gammals, scat) that use outer optimization.
gam_converged <- function(m) {
  if (!is.null(m$converged)) return(isTRUE(m$converged))
  if (!is.null(m$outer.info))
    return(grepl("full conv", m$outer.info$conv, ignore.case = TRUE))
  TRUE  # assume converged when no flag is present
}

# Standard emit for single-predictor gam models.  Adds `sp` and `se_eta` to
# the JSON in addition to the existing fields already consumed by the Rust test.
emit <- function(path, m, coefficients, edf, fit_time_ms,
                 fitted_sigma = list(), error_msg = NA) {
  # Link-scale standard errors on the training data (mu only for
  # single-parameter families; extended families override below).
  se_pred <- tryCatch(
    predict(m, type = "link", se.fit = TRUE),
    error = function(e) NULL
  )
  se_eta_out <- if (!is.null(se_pred) && !is.matrix(se_pred$se.fit)) {
    list(mu = as.list(unname(se_pred$se.fit)))
  } else {
    list()
  }

  result <- list(
    converged      = gam_converged(m),
    iterations     = if (!is.null(m$iter)) as.integer(m$iter) else 0L,
    fit_time_ms    = fit_time_ms,
    coefficients   = coefficients,
    fitted_mu      = as.list(unname(fitted(m))),
    fitted_sigma   = fitted_sigma,
    edf            = edf,
    log_likelihood = as.numeric(stats::logLik(m)),
    aic            = AIC(m),
    sp             = sp_list(m),
    se_eta         = se_eta_out,
    error          = error_msg
  )
  write_json(result, path, auto_unbox = TRUE, pretty = TRUE, na = "null")
}

# ─── Linear (parametric) fitters ─────────────────────────────────────────────

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

fit_binomial_linear <- function(df, output) {
  # y must be 0/1 integer for binomial family.
  df$y <- as.integer(df$y)
  start <- Sys.time()
  m <- gam(y ~ x, data = df, family = binomial(link = "logit"), method = "REML")
  emit(
    output, m,
    coefficients = list(logit_mu = unname(coef(m))),
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

# ─── Smooth (P-spline) fitters ────────────────────────────────────────────────
# `s(x, bs="ps", k=K)` requests a penalised B-spline basis of size K with
# second-order difference penalty — matches glissando's
# `PSpline1D { degree: 3, penalty_order: 2 }`.

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

fit_binomial_smooth <- function(df, output) {
  df$y <- as.integer(df$y)
  start <- Sys.time()
  m <- gam(y ~ s(x, bs = "ps", k = 20), data = df, family = binomial(link = "logit"), method = "REML")
  emit(
    output, m,
    coefficients = list(logit_mu_smooth = unname(coef(m))),
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

fit_beta_smooth <- function(df, output) {
  start <- Sys.time()
  m <- gam(y ~ s(x, bs = "ps", k = 20), data = df, family = betar(link = "logit"), method = "REML")
  log_phi <- log(m$family$getTheta(TRUE))
  emit(
    output, m,
    coefficients = list(
      logit_mu_smooth = unname(coef(m)),
      log_phi         = list(log_phi)
    ),
    edf = list(mu = sum(m$edf), phi = 1.0),
    fit_time_ms = elapsed_ms(start)
  )
}

# ─── CR spline ────────────────────────────────────────────────────────────────
# Natural cubic regression spline; glissando CrSpline1D uses identical quantile
# knots so the comparison tests the knot-matching logic directly.

fit_gaussian_cr_smooth <- function(df, output) {
  start <- Sys.time()
  m <- gam(y ~ s(x, bs = "cr", k = 10), data = df, family = gaussian(), method = "REML")
  emit(
    output, m,
    coefficients = list(
      mu_cr_smooth = unname(coef(m)),
      log_sigma    = list(0.5 * log(m$sig2))
    ),
    edf = list(mu = sum(m$edf), sigma = 1.0),
    fit_time_ms = elapsed_ms(start)
  )
}

# ─── Tensor product ────────────────────────────────────────────────────────────
# te(x1, x2, bs="ps", k=c(8,8)) matches glissando TensorProduct with
# n_splines_{1,2}=8, penalty_order_{1,2}=2, degree=3.

fit_tensor_smooth <- function(df, output) {
  start <- Sys.time()
  m <- gam(y ~ te(x1, x2, bs = "ps", k = c(8, 8)), data = df,
           family = gaussian(), method = "REML")
  emit(
    output, m,
    coefficients = list(
      mu_tensor = unname(coef(m)),
      log_sigma = list(0.5 * log(m$sig2))
    ),
    edf = list(mu = sum(m$edf), sigma = 1.0),
    fit_time_ms = elapsed_ms(start)
  )
}

# ─── Random effects ────────────────────────────────────────────────────────────

fit_random_effect <- function(df, output) {
  # glissando stores group IDs as float (0.0, 1.0, …); convert to factor for mgcv.
  df$g <- factor(as.integer(df$g))
  start <- Sys.time()
  m <- gam(y ~ x + s(g, bs = "re"), data = df, family = gaussian(), method = "REML")
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

# Heteroskedastic Gaussian: mean AND log-scale both linear in x, via gaulss.
# gaulss linear predictors: η₁ = μ (identity); η₂ goes through the logb link
# (τ = 1/σ = b + exp(η₂), b = 0.01), so σ = 1/linkinv(η₂). Only μ coefficients
# are emitted — the σ model lives on a different link scale than glissando's
# log σ, so coefficient-level σ comparison is not meaningful, but fitted_mu /
# fitted_sigma / log-likelihood / SE[μ] all are.
fit_gaussian_heteroskedastic <- function(df, output) {
  start <- Sys.time()
  m <- gam(list(y ~ x, ~ x), data = df, family = gaulss())
  fv        <- fitted(m)
  mu_hat    <- fv[, 1]
  sigma_hat <- 1.0 / fv[, 2]

  se_pred <- tryCatch(predict(m, type = "link", se.fit = TRUE), error = function(e) NULL)
  se_eta_out <- if (!is.null(se_pred) && is.matrix(se_pred$se.fit)) {
    list(mu = as.list(unname(se_pred$se.fit[, 1])))
  } else {
    list()
  }

  result <- list(
    converged    = gam_converged(m),
    iterations   = if (!is.null(m$outer.info$iter)) as.integer(m$outer.info$iter) else 0L,
    fit_time_ms  = elapsed_ms(start),
    coefficients = list(mu = unname(coef(m))[1:2]),
    fitted_mu    = as.list(unname(mu_hat)),
    fitted_sigma = as.list(unname(sigma_hat)),
    # Coefficients 1:2 belong to the μ predictor (y ~ x), 3:4 to the σ
    # predictor (~ x); both are unpenalized so each EDF is exactly 2.
    edf          = list(mu = sum(m$edf[1:2]), sigma = sum(m$edf[3:4])),
    log_likelihood = as.numeric(stats::logLik(m)),
    aic          = AIC(m),
    sp           = sp_list(m),
    se_eta       = se_eta_out,
    error        = NA
  )
  write_json(result, output, auto_unbox = TRUE, pretty = TRUE, na = "null")
}

# ─── Scale-smooth fitters (LSS families) ─────────────────────────────────────

# Gaussian location-scale smooth (gaulss).
# gaulss fitted matrix: col 1 = μ, col 2 = 1/σ (precision). Invert col 2 → σ.
fit_gaussian_sigma_smooth <- function(df, output) {
  start <- Sys.time()
  m <- gam(list(y ~ 1, ~ s(x, bs = "ps", k = 20)), data = df, family = gaulss())
  fv        <- fitted(m)
  mu_hat    <- fv[, 1]
  sigma_hat <- 1.0 / fv[, 2]

  # Link-scale SEs from gaulss predict (returns 2-column se.fit matrix).
  se_pred <- tryCatch(predict(m, type = "link", se.fit = TRUE), error = function(e) NULL)
  se_eta_out <- if (!is.null(se_pred) && is.matrix(se_pred$se.fit)) {
    list(
      mu    = as.list(unname(se_pred$se.fit[, 1])),
      sigma = as.list(unname(se_pred$se.fit[, 2]))
    )
  } else {
    list()
  }

  result <- list(
    converged    = gam_converged(m),
    iterations   = if (!is.null(m$outer.info$iter)) as.integer(m$outer.info$iter) else 0L,
    fit_time_ms  = elapsed_ms(start),
    coefficients = list(
      mu               = list(unname(coef(m))[1]),
      log_sigma_smooth = unname(coef(m))
    ),
    fitted_mu    = as.list(unname(mu_hat)),
    fitted_sigma = as.list(unname(sigma_hat)),
    edf          = list(mu = 1.0, sigma = sum(m$edf)),
    log_likelihood = as.numeric(stats::logLik(m)),
    aic          = AIC(m),
    sp           = sp_list(m),
    se_eta       = se_eta_out,
    error        = NA
  )
  write_json(result, output, auto_unbox = TRUE, pretty = TRUE, na = "null")
}

# Gamma location-scale smooth (gammals).
# gammals: predictor 1 = log(μ), predictor 2 = log(CV²) = 2·log(CV).
# glissando σ = CV, so sigma_hat = exp(η₂/2) = sqrt(exp(η₂)).
fit_gamma_sigma_smooth <- function(df, output) {
  start <- Sys.time()
  m <- gam(list(y ~ 1, ~ s(x, bs = "ps", k = 20)), data = df, family = gammals())
  # gammals linear predictors (predict type="link"):
  #   η₁ = log(μ)  — "identity" link on the log-mean, so exp(η₁) = E[Y].
  #   η₂ — the SCALE predictor goes through gammals' `logb` link, NOT a plain
  #        log: θ = log(φ) = b + log(1 + exp(η₂)) with b = −7 by default.
  #        Treating η₂ as log(φ) directly (the previous code) produced σ̂ in the
  #        5–20 range for a true CV of 0.2–0.7. Use the family's own linkinv to
  #        recover θ = log(φ); glissando's σ = CV = sqrt(φ) = exp(θ/2).
  lp        <- predict(m, type = "link")
  mu_hat    <- exp(lp[, 1])                       # log-mean → E[Y]
  theta     <- m$family$linfo[[2]]$linkinv(lp[, 2])  # log(φ) via logb linkinv
  sigma_hat <- exp(theta / 2.0)                   # φ = CV² → σ = CV

  se_pred <- tryCatch(predict(m, type = "link", se.fit = TRUE), error = function(e) NULL)
  se_eta_out <- if (!is.null(se_pred) && is.matrix(se_pred$se.fit)) {
    list(
      mu    = as.list(unname(se_pred$se.fit[, 1])),
      sigma = as.list(unname(se_pred$se.fit[, 2]))
    )
  } else {
    list()
  }

  result <- list(
    converged    = gam_converged(m),
    iterations   = if (!is.null(m$outer.info$iter)) as.integer(m$outer.info$iter) else 0L,
    fit_time_ms  = elapsed_ms(start),
    coefficients = list(
      log_mu          = list(unname(coef(m))[1]),
      log_sigma_smooth = unname(coef(m))
    ),
    fitted_mu    = as.list(unname(mu_hat)),
    fitted_sigma = as.list(unname(sigma_hat)),
    edf          = list(mu = 1.0, sigma = sum(m$edf)),
    log_likelihood = as.numeric(stats::logLik(m)),
    aic          = AIC(m),
    sp           = sp_list(m),
    se_eta       = se_eta_out,
    error        = NA
  )
  write_json(result, output, auto_unbox = TRUE, pretty = TRUE, na = "null")
}

# Student-t via scat() scaled-t family.
# scat models μ linearly and treats σ, ν as global nuisance parameters.
# Gate only on fitted_mu; σ/ν parameterisations differ.
fit_studentt_linear <- function(df, output) {
  start <- Sys.time()
  m <- gam(y ~ x, data = df, family = scat(), method = "REML")
  emit(
    output, m,
    coefficients = list(mu = unname(coef(m))),
    edf = list(mu = sum(m$edf)),
    fit_time_ms = elapsed_ms(start)
  )
}

fit_studentt_smooth <- function(df, output) {
  start <- Sys.time()
  m <- gam(y ~ s(x, bs = "ps", k = 20), data = df, family = scat(), method = "REML")
  emit(
    output, m,
    coefficients = list(mu_smooth = unname(coef(m))),
    edf = list(mu = sum(m$edf)),
    fit_time_ms = elapsed_ms(start)
  )
}

# ─── B1: Gaussian + prior weights ─────────────────────────────────────────────
# Replicates mgcv::bam(..., weights=w) semantics for the B1 listing-weighted
# model: five P-spline smooths + one binary linear term.

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

# B2: StudentT + prior weights (4 smooths); use scat() for the mean.
fit_b2_weighted_studentt <- function(df, output) {
  start <- Sys.time()
  m <- gam(
    y ~ s(x1, bs = "ps", k = 20) +
        s(x2, bs = "ps", k = 20) +
        s(x3, bs = "ps", k = 20) +
        s(x4, bs = "ps", k = 20),
    data    = df,
    weights = df$weights,
    family  = scat(),
    method  = "REML"
  )
  emit(
    output, m,
    coefficients = list(mu_smooth = unname(coef(m))),
    edf = list(mu = sum(m$edf)),
    fit_time_ms = elapsed_ms(start)
  )
}

# ─── Dispatch ─────────────────────────────────────────────────────────────────

dispatch <- list(
  gaussian_linear           = fit_gaussian_linear,
  gaussian_heteroskedastic  = fit_gaussian_heteroskedastic,
  gaussian_multiple         = fit_gaussian_multiple,
  gaussian_large            = fit_gaussian_large,
  gaussian_smooth           = fit_gaussian_smooth,
  gaussian_quadratic        = fit_gaussian_quadratic,
  gaussian_sigma_smooth     = fit_gaussian_sigma_smooth,
  gaussian_cr_smooth        = fit_gaussian_cr_smooth,
  tensor_smooth             = fit_tensor_smooth,
  random_effect             = fit_random_effect,
  poisson_linear            = fit_poisson_linear,
  poisson_smooth            = fit_poisson_smooth,
  binomial_linear           = fit_binomial_linear,
  binomial_smooth           = fit_binomial_smooth,
  gamma_linear              = fit_gamma_linear,
  gamma_smooth              = fit_gamma_smooth,
  gamma_sigma_smooth        = fit_gamma_sigma_smooth,
  studentt_linear           = fit_studentt_linear,
  studentt_smooth           = fit_studentt_smooth,
  negative_binomial_linear  = fit_negative_binomial_linear,
  negative_binomial_smooth  = fit_negative_binomial_smooth,
  beta_linear               = fit_beta_linear,
  beta_smooth               = fit_beta_smooth,
  b1_weighted_gaussian      = fit_b1_weighted_gaussian,
  b2_weighted_studentt      = fit_b2_weighted_studentt
)

if (is.null(dispatch[[opts$scenario]])) {
  cat(sprintf("scenario '%s' not supported by this script\n", opts$scenario))
  quit(status = 1)
}

dispatch[[opts$scenario]](df, opts$output)
