#!/usr/bin/env Rscript
# Fits gamlss TF() models matching glissando's StudentT compare_fit.rs scenarios.
#
# Why a separate script from fit_mgcv.R: mgcv's scat() is a *different* algorithm
# (Wood's joint outer-BFGS) that folds σ and ν into internal nuisance scalars and
# exposes only `mu` as a modelled predictor. The gamlss TF() family is the SAME
# Rigby–Stasinopoulos algorithm and the SAME (μ, σ, ν) location-scale-df
# parameterization glissando implements, so it is the correct like-for-like oracle:
# it reports μ, σ, ν coefficients, their EDF, SEs, and the (unweighted) log-likelihood
# on the same footing as glissando. The JSON shape mirrors `FitResult` in
# compare_fit.rs so orchestrate.py can splice it under the `gamlss` key.
#
# gamlss TF density (mu.link=identity, sigma.link=log, nu.link=log):
#   f(y) = (1/σ)·[Γ((ν+1)/2)/(Γ(ν/2)·√(πν))]·(1 + (y−μ)²/(ν σ²))^(−(ν+1)/2)
# which is identical to glissando's StudentT::loglik_pointwise.

suppressPackageStartupMessages({
  library(arrow)
  library(gamlss)
  library(jsonlite)
  library(optparse)
})

opts <- parse_args(OptionParser(option_list = list(
  make_option(c("--data"),     type = "character"),
  make_option(c("--scenario"), type = "character"),
  make_option(c("--output"),   type = "character")
)))

df <- read_parquet(opts$data)

elapsed_ms <- function(start) as.numeric(Sys.time() - start, units = "secs") * 1000

# Link-scale standard errors for one distribution parameter, as a plain list.
# Wrapped in tryCatch because gamlss predict(se.fit=TRUE) can fail for some
# smoother configurations; an empty list then simply omits that parameter.
se_eta_for <- function(m, what) {
  tryCatch({
    p <- predict(m, what = what, type = "link", se.fit = TRUE)
    as.list(unname(p$se.fit))
  }, error = function(e) NULL)
}

# Emit the FitResult JSON shape for a fitted gamlss TF() model. `mu_label` is the
# coefficient key for μ ("mu" for linear, "mu_smooth" for a P-spline mean).
emit_tf <- function(path, m, mu_label, fit_time_ms) {
  # Coefficients on the predictor (link) scale: μ identity, σ/ν log. Wrap in
  # as.list() so length-1 intercept vectors serialize as JSON arrays (not scalars
  # unboxed by auto_unbox), matching the Vec<f64> the Rust FitResult expects.
  coefficients <- list()
  coefficients[[mu_label]] <- as.list(unname(m$mu.coefficients))
  coefficients[["log_sigma"]] <- as.list(unname(m$sigma.coefficients))
  coefficients[["log_nu"]] <- as.list(unname(m$nu.coefficients))

  se_eta_out <- list()
  se_mu <- se_eta_for(m, "mu")
  if (!is.null(se_mu)) se_eta_out[["mu"]] <- se_mu
  se_sigma <- se_eta_for(m, "sigma")
  if (!is.null(se_sigma)) se_eta_out[["sigma"]] <- se_sigma
  se_nu <- se_eta_for(m, "nu")
  if (!is.null(se_nu)) se_eta_out[["nu"]] <- se_nu

  result <- list(
    converged      = isTRUE(m$converged),
    iterations     = if (!is.null(m$iter)) as.integer(m$iter) else 0L,
    fit_time_ms    = fit_time_ms,
    coefficients   = coefficients,
    fitted_mu      = as.list(unname(fitted(m, "mu"))),
    fitted_sigma   = as.list(unname(fitted(m, "sigma"))),
    # Effective degrees of freedom per parameter (mu.df includes the intercept).
    edf            = list(mu = m$mu.df, sigma = m$sigma.df, nu = m$nu.df),
    # Unweighted log-likelihood = −½·global deviance, matching glissando's
    # ML log-likelihood (its diagnostics report the unweighted sum).
    log_likelihood = as.numeric(-m$G.deviance / 2),
    aic            = as.numeric(m$aic),
    sp             = list(),
    se_eta         = se_eta_out,
    error          = NA
  )
  write_json(result, path, auto_unbox = TRUE, pretty = TRUE, na = "null")
}

ctrl <- gamlss.control(trace = FALSE)

# Student-t, linear mean. σ and ν are intercept-only (global), matching the
# glissando formula (μ ~ 1 + x, σ ~ 1, ν ~ 1).
fit_studentt_linear <- function(df, output) {
  start <- Sys.time()
  m <- gamlss(y ~ x, sigma.formula = ~1, nu.formula = ~1,
              family = TF(), data = df, control = ctrl)
  emit_tf(output, m, "mu", elapsed_ms(start))
}

# Student-t, P-spline mean. pb() defaults (inter=20, degree=3, order=2) match the
# glissando PSpline1D { n_splines: 20, degree: 3, penalty_order: 2 }.
fit_studentt_smooth <- function(df, output) {
  start <- Sys.time()
  m <- gamlss(y ~ pb(x), sigma.formula = ~1, nu.formula = ~1,
              family = TF(), data = df, control = ctrl)
  emit_tf(output, m, "mu_smooth", elapsed_ms(start))
}

# B2: StudentT with four P-spline smooths and listing-level prior weights.
fit_b2_weighted_studentt <- function(df, output) {
  start <- Sys.time()
  m <- gamlss(y ~ pb(x1) + pb(x2) + pb(x3) + pb(x4),
              sigma.formula = ~1, nu.formula = ~1,
              family = TF(), data = df, weights = df$weights, control = ctrl)
  emit_tf(output, m, "mu_smooth", elapsed_ms(start))
}

dispatch <- list(
  studentt_linear      = fit_studentt_linear,
  studentt_smooth      = fit_studentt_smooth,
  b2_weighted_studentt = fit_b2_weighted_studentt
)

if (is.null(dispatch[[opts$scenario]])) {
  cat(sprintf("scenario '%s' not supported by fit_gamlss.R\n", opts$scenario))
  quit(status = 1)
}

dispatch[[opts$scenario]](df, opts$output)
