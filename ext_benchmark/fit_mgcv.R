#!/usr/bin/env Rscript
#
# GAMLSS Comparison Framework: R Fitting Script
#
# Reads parquet data, fits models using mgcv (gam) or gamlss package,
# and outputs standardized JSON results.
#
# For fair comparison, uses mgcv::gam for Gaussian/Poisson/Gamma
# and gamlss package for Student-t (which mgcv doesn't support).

suppressPackageStartupMessages({
  library(arrow)       # parquet I/O
  library(mgcv)        # primary GAM fitting
  library(jsonlite)    # output format
  library(optparse)
})

# Try to load gamlss for Student-t, but don't fail if unavailable
has_gamlss <- requireNamespace("gamlss", quietly = TRUE)
if (has_gamlss) {
  suppressPackageStartupMessages(library(gamlss))
}


fit_gaussian_linear <- function(df) {
  # mu ~ intercept + x, sigma constant
  start_time <- Sys.time()
  
  fit <- gam(y ~ x, data = df, family = gaussian())
  
  elapsed_ms <- as.numeric(difftime(Sys.time(), start_time, units = "secs")) * 1000
  
  list(
    converged = fit$converged,
    iterations = fit$iter,
    fit_time_ms = elapsed_ms,
    coefficients = list(
      mu = as.numeric(coef(fit))
    ),
    fitted_mu = as.numeric(fitted(fit)),
    fitted_sigma = rep(sqrt(fit$sig2), nrow(df)),
    edf = list(mu = sum(fit$edf)),
    log_likelihood = as.numeric(logLik(fit)),
    aic = AIC(fit),
    error = NULL
  )
}


fit_gaussian_heteroskedastic <- function(df) {
  # mu ~ x, log(sigma) ~ x
  # mgcv's gaulss family allows modeling both mean and variance
  start_time <- Sys.time()

  fit <- gam(
    list(
      y ~ x,          # mu formula
      ~ x             # log(sigma) formula
    ),
    data = df,
    family = gaulss()
  )

  elapsed_ms <- as.numeric(difftime(Sys.time(), start_time, units = "secs")) * 1000

  # gaulss returns coefficients for both mu and log(sigma)
  # Coefficient indices: first set for mu, second for log(sigma)
  n_mu_coef <- length(fit$coefficients) / 2  # assumes equal terms

  # Extract predictions for both parameters
  preds <- predict(fit, type = "response")

  # For gaulss family, converged may be a list or NULL; coerce to boolean
  is_converged <- if (is.logical(fit$converged)) fit$converged else TRUE

  list(
    converged = is_converged,
    iterations = fit$iter,
    fit_time_ms = elapsed_ms,
    coefficients = list(
      mu = as.numeric(coef(fit)[1:2]),
      log_sigma = as.numeric(coef(fit)[3:4])
    ),
    fitted_mu = as.numeric(preds[, 1]),
    fitted_sigma = as.numeric(preds[, 2]),
    edf = list(mu = fit$edf[1], sigma = fit$edf[2]),
    log_likelihood = as.numeric(logLik(fit)),
    aic = AIC(fit),
    error = NA  # NA serializes to null in JSON, not {}
  )
}


fit_gaussian_smooth <- function(df) {
  # mu ~ s(x), sigma constant
  # Using thin plate regression spline (mgcv default)
  start_time <- Sys.time()
  
  fit <- gam(y ~ s(x, bs = "tp", k = 20), data = df, family = gaussian())
  
  elapsed_ms <- as.numeric(difftime(Sys.time(), start_time, units = "secs")) * 1000
  
  list(
    converged = fit$converged,
    iterations = fit$iter,
    fit_time_ms = elapsed_ms,
    coefficients = list(
      mu_smooth = as.numeric(coef(fit))
    ),
    fitted_mu = as.numeric(fitted(fit)),
    fitted_sigma = rep(sqrt(fit$sig2), nrow(df)),
    edf = list(mu = sum(fit$edf)),
    log_likelihood = as.numeric(logLik(fit)),
    aic = AIC(fit),
    error = NULL
  )
}


fit_poisson_linear <- function(df) {
  # log(mu) ~ intercept + x
  start_time <- Sys.time()
  
  fit <- gam(y ~ x, data = df, family = poisson(link = "log"))
  
  elapsed_ms <- as.numeric(difftime(Sys.time(), start_time, units = "secs")) * 1000
  
  list(
    converged = fit$converged,
    iterations = fit$iter,
    fit_time_ms = elapsed_ms,
    coefficients = list(
      log_mu = as.numeric(coef(fit))
    ),
    fitted_mu = as.numeric(fitted(fit)),
    fitted_sigma = list(),  # Poisson has no separate sigma
    edf = list(mu = sum(fit$edf)),
    log_likelihood = as.numeric(logLik(fit)),
    aic = AIC(fit),
    error = NULL
  )
}


fit_gamma_linear <- function(df) {
  # log(mu) ~ intercept + x
  # Using Gamma with log link
  start_time <- Sys.time()
  
  fit <- gam(y ~ x, data = df, family = Gamma(link = "log"))
  
  elapsed_ms <- as.numeric(difftime(Sys.time(), start_time, units = "secs")) * 1000
  
  # Extract dispersion (phi = 1/shape, so CV = sqrt(phi))
  phi <- summary(fit)$dispersion
  cv <- sqrt(phi)
  
  list(
    converged = fit$converged,
    iterations = fit$iter,
    fit_time_ms = elapsed_ms,
    coefficients = list(
      log_mu = as.numeric(coef(fit)),
      sigma = cv  # coefficient of variation
    ),
    fitted_mu = as.numeric(fitted(fit)),
    fitted_sigma = rep(cv, nrow(df)),
    edf = list(mu = sum(fit$edf)),
    log_likelihood = as.numeric(logLik(fit)),
    aic = AIC(fit),
    error = NULL
  )
}


fit_studentt_linear <- function(df) {
  # Using gamlss package for Student-t distribution
  # mu ~ x, sigma constant, nu (df) constant
  
  if (!has_gamlss) {
    return(list(
      converged = FALSE,
      iterations = 0,
      fit_time_ms = 0,
      coefficients = list(),
      fitted_mu = list(),
      fitted_sigma = list(),
      edf = list(),
      log_likelihood = NULL,
      aic = NULL,
      error = "gamlss package not installed - required for Student-t"
    ))
  }
  
  start_time <- Sys.time()
  
  fit <- gamlss(
    y ~ x,
    sigma.formula = ~ 1,
    nu.formula = ~ 1,
    data = df,
    family = TF(),  # T-family (Student-t) in gamlss
    trace = FALSE
  )
  
  elapsed_ms <- as.numeric(difftime(Sys.time(), start_time, units = "secs")) * 1000
  
  list(
    converged = fit$converged,
    iterations = fit$iter,
    fit_time_ms = elapsed_ms,
    coefficients = list(
      mu = as.numeric(coef(fit, what = "mu")),
      log_sigma = as.numeric(coef(fit, what = "sigma")),
      log_nu = as.numeric(coef(fit, what = "nu"))
    ),
    fitted_mu = as.numeric(fitted(fit, what = "mu")),
    fitted_sigma = as.numeric(exp(predict(fit, what = "sigma"))),
    edf = list(
      mu = sum(fit$mu.df),
      sigma = fit$sigma.df,
      nu = fit$nu.df
    ),
    log_likelihood = as.numeric(logLik(fit)),
    aic = AIC(fit),
    error = NULL
  )
}


fit_negative_binomial_linear <- function(df) {
  # log(mu) ~ intercept + x, with overdispersion
  start_time <- Sys.time()

  fit <- gam(y ~ x, data = df, family = nb(link = "log"))

  elapsed_ms <- as.numeric(difftime(Sys.time(), start_time, units = "secs")) * 1000

  # nb() estimates theta = 1/sigma^2, so sigma = 1/sqrt(theta)
  theta <- fit$family$getTheta(TRUE)
  sigma <- 1 / sqrt(theta)

  list(
    converged = fit$converged,
    iterations = fit$iter,
    fit_time_ms = elapsed_ms,
    coefficients = list(
      log_mu = as.numeric(coef(fit)),
      sigma = sigma
    ),
    fitted_mu = as.numeric(fitted(fit)),
    fitted_sigma = rep(sigma, nrow(df)),
    edf = list(mu = sum(fit$edf)),
    log_likelihood = as.numeric(logLik(fit)),
    aic = AIC(fit),
    error = NULL
  )
}


fit_beta_linear <- function(df) {
  # logit(mu) ~ intercept + x, phi constant
  # Using mgcv's betar family
  start_time <- Sys.time()

  fit <- gam(y ~ x, data = df, family = betar(link = "logit"))

  elapsed_ms <- as.numeric(difftime(Sys.time(), start_time, units = "secs")) * 1000

  # betar estimates phi (precision)
  phi <- fit$family$getTheta(TRUE)

  list(
    converged = fit$converged,
    iterations = fit$iter,
    fit_time_ms = elapsed_ms,
    coefficients = list(
      logit_mu = as.numeric(coef(fit)),
      log_phi = log(phi)
    ),
    fitted_mu = as.numeric(fitted(fit)),
    fitted_sigma = rep(phi, nrow(df)),  # returning phi as "sigma" slot
    edf = list(mu = sum(fit$edf)),
    log_likelihood = as.numeric(logLik(fit)),
    aic = AIC(fit),
    error = NULL
  )
}


fit_poisson_smooth <- function(df) {
  # log(mu) ~ s(x)
  start_time <- Sys.time()

  fit <- gam(y ~ s(x, bs = "tp", k = 20), data = df, family = poisson(link = "log"))

  elapsed_ms <- as.numeric(difftime(Sys.time(), start_time, units = "secs")) * 1000

  list(
    converged = fit$converged,
    iterations = fit$iter,
    fit_time_ms = elapsed_ms,
    coefficients = list(
      log_mu_smooth = as.numeric(coef(fit))
    ),
    fitted_mu = as.numeric(fitted(fit)),
    fitted_sigma = list(),
    edf = list(mu = sum(fit$edf)),
    log_likelihood = as.numeric(logLik(fit)),
    aic = AIC(fit),
    error = NULL
  )
}


fit_gamma_smooth <- function(df) {
  # log(mu) ~ s(x)
  start_time <- Sys.time()

  fit <- gam(y ~ s(x, bs = "tp", k = 20), data = df, family = Gamma(link = "log"))

  elapsed_ms <- as.numeric(difftime(Sys.time(), start_time, units = "secs")) * 1000

  phi <- summary(fit)$dispersion
  cv <- sqrt(phi)

  list(
    converged = fit$converged,
    iterations = fit$iter,
    fit_time_ms = elapsed_ms,
    coefficients = list(
      log_mu_smooth = as.numeric(coef(fit)),
      sigma = cv
    ),
    fitted_mu = as.numeric(fitted(fit)),
    fitted_sigma = rep(cv, nrow(df)),
    edf = list(mu = sum(fit$edf)),
    log_likelihood = as.numeric(logLik(fit)),
    aic = AIC(fit),
    error = NULL
  )
}


fit_gaussian_multiple <- function(df) {
  # mu ~ intercept + x1 + x2 + x3
  start_time <- Sys.time()

  fit <- gam(y ~ x1 + x2 + x3, data = df, family = gaussian())

  elapsed_ms <- as.numeric(difftime(Sys.time(), start_time, units = "secs")) * 1000

  list(
    converged = fit$converged,
    iterations = fit$iter,
    fit_time_ms = elapsed_ms,
    coefficients = list(
      mu = as.numeric(coef(fit))
    ),
    fitted_mu = as.numeric(fitted(fit)),
    fitted_sigma = rep(sqrt(fit$sig2), nrow(df)),
    edf = list(mu = sum(fit$edf)),
    log_likelihood = as.numeric(logLik(fit)),
    aic = AIC(fit),
    error = NULL
  )
}


fit_gaussian_large <- function(df) {
  # Same as gaussian_linear, just larger data
  fit_gaussian_linear(df)
}


fit_studentt_smooth <- function(df) {
  # mu ~ s(x), sigma constant, nu constant
  # Using gamlss for Student-t with smooth

  if (!has_gamlss) {
    return(list(
      converged = FALSE,
      iterations = 0,
      fit_time_ms = 0,
      coefficients = list(),
      fitted_mu = list(),
      fitted_sigma = list(),
      edf = list(),
      log_likelihood = NULL,
      aic = NULL,
      error = "gamlss package not installed"
    ))
  }

  start_time <- Sys.time()

  fit <- gamlss(
    y ~ pb(x, df = 10),  # penalized B-spline
    sigma.formula = ~ 1,
    nu.formula = ~ 1,
    data = df,
    family = TF(),
    trace = FALSE
  )

  elapsed_ms <- as.numeric(difftime(Sys.time(), start_time, units = "secs")) * 1000

  list(
    converged = fit$converged,
    iterations = fit$iter,
    fit_time_ms = elapsed_ms,
    coefficients = list(
      mu_smooth = as.numeric(coef(fit, what = "mu")),
      log_sigma = as.numeric(coef(fit, what = "sigma")),
      log_nu = as.numeric(coef(fit, what = "nu"))
    ),
    fitted_mu = as.numeric(fitted(fit, what = "mu")),
    fitted_sigma = as.numeric(exp(predict(fit, what = "sigma"))),
    edf = list(mu = sum(fit$mu.df), sigma = fit$sigma.df, nu = fit$nu.df),
    log_likelihood = as.numeric(logLik(fit)),
    aic = AIC(fit),
    error = NULL
  )
}


fit_negative_binomial_smooth <- function(df) {
  # log(mu) ~ s(x), overdispersion estimated
  start_time <- Sys.time()

  fit <- gam(y ~ s(x, bs = "tp", k = 20), data = df, family = nb(link = "log"))

  elapsed_ms <- as.numeric(difftime(Sys.time(), start_time, units = "secs")) * 1000

  theta <- fit$family$getTheta(TRUE)
  sigma <- 1 / sqrt(theta)

  list(
    converged = fit$converged,
    iterations = fit$iter,
    fit_time_ms = elapsed_ms,
    coefficients = list(
      log_mu_smooth = as.numeric(coef(fit)),
      sigma = sigma
    ),
    fitted_mu = as.numeric(fitted(fit)),
    fitted_sigma = rep(sigma, nrow(df)),
    edf = list(mu = sum(fit$edf)),
    log_likelihood = as.numeric(logLik(fit)),
    aic = AIC(fit),
    error = NULL
  )
}


fit_gaussian_quadratic <- function(df) {
  # mu ~ s(x), sigma constant - fitting smooth to quadratic data
  fit_gaussian_smooth(df)
}


# Dispatch table
FITTERS <- list(
  gaussian_linear = fit_gaussian_linear,
  gaussian_heteroskedastic = fit_gaussian_heteroskedastic,
  gaussian_smooth = fit_gaussian_smooth,
  gaussian_multiple = fit_gaussian_multiple,
  gaussian_large = fit_gaussian_large,
  gaussian_quadratic = fit_gaussian_quadratic,
  poisson_linear = fit_poisson_linear,
  poisson_smooth = fit_poisson_smooth,
  gamma_linear = fit_gamma_linear,
  gamma_smooth = fit_gamma_smooth,
  studentt_linear = fit_studentt_linear,
  studentt_smooth = fit_studentt_smooth,
  negative_binomial_linear = fit_negative_binomial_linear,
  negative_binomial_smooth = fit_negative_binomial_smooth,
  beta_linear = fit_beta_linear
)


main <- function() {
  option_list <- list(
    make_option("--data", type = "character", help = "Path to parquet data file"),
    make_option("--scenario", type = "character", help = "Scenario name"),
    make_option("--output", type = "character", help = "Path to output JSON file")
  )
  
  opt <- parse_args(OptionParser(option_list = option_list))
  
  if (is.null(opt$data) || is.null(opt$scenario) || is.null(opt$output)) {
    stop("Must provide --data, --scenario, and --output arguments")
  }
  
  # Read parquet
  df <- as.data.frame(read_parquet(opt$data))
  
  # Get appropriate fitter
  fitter <- FITTERS[[opt$scenario]]
  if (is.null(fitter)) {
    result <- list(
      converged = FALSE,
      error = paste("Unknown scenario:", opt$scenario)
    )
  } else {
    result <- tryCatch(
      fitter(df),
      error = function(e) {
        list(
          converged = FALSE,
          iterations = 0,
          fit_time_ms = 0,
          coefficients = list(),
          fitted_mu = list(),
          fitted_sigma = list(),
          edf = list(),
          log_likelihood = NULL,
          aic = NULL,
          error = as.character(e)
        )
      }
    )
  }
  
  # Write output
  write_json(result, opt$output, auto_unbox = TRUE, pretty = TRUE)
  
  cat("R fitting complete:", opt$scenario, "\n")
}


if (!interactive()) {
  main()
}
