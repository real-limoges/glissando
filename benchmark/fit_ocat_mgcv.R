#!/usr/bin/env Rscript
# mgcv reference fit for ordered-categorical (ocat) models.
#
# Fits mgcv::gam with family=ocat(R=4) and predicts category probabilities.
# Outputs an (n × R) probability matrix alongside training log-likelihood.
#
# Usage:
#   Rscript benchmark/fit_ocat_mgcv.R \
#     --train  /path/to/ocat_train.parquet \
#     --test   /path/to/ocat_test.parquet  \
#     --output /path/to/mgcv_ocat.json

suppressPackageStartupMessages({
  library(mgcv)
  library(jsonlite)
  library(optparse)
})

# Parquet reader: prefer arrow, fall back to the dependency-free nanoparquet.
read_parquet <- if (requireNamespace("arrow", quietly = TRUE)) {
  arrow::read_parquet
} else if (requireNamespace("nanoparquet", quietly = TRUE)) {
  function(path) as.data.frame(nanoparquet::read_parquet(path))
} else {
  stop("need the 'arrow' or 'nanoparquet' package to read parquet input")
}

opts <- parse_args(OptionParser(option_list = list(
  make_option(c("--train"),         type = "character"),
  make_option(c("--test"),          type = "character"),
  make_option(c("--output"),        type = "character"),
  make_option(c("--intercept-only"), action = "store_true", default = FALSE,
              help = "Fit intercept-only model (for log-likelihood cross-check)")
)))

train_df <- as.data.frame(read_parquet(opts$train))
test_df  <- as.data.frame(read_parquet(opts$test))

# ocat expects y as a positive integer 1..R
train_df$y <- as.integer(round(train_df$y))
test_df$y  <- as.integer(round(test_df$y))

start <- Sys.time()

if (isTRUE(opts[["intercept-only"]])) {
  m <- gam(
    y ~ 1,
    family = ocat(R = 4),
    data   = train_df,
    method = "ML"
  )
} else {
  m <- gam(
    y ~ s(x1, bs = "ps", k = 10) + s(x2, bs = "ps", k = 10),
    family = ocat(R = 4),
    data   = train_df,
    method = "REML"
  )
}

fit_time_ms <- as.numeric(Sys.time() - start, units = "secs") * 1000

# Training log-likelihood: sum of log P(y_i = r_i | model).
train_fitted_probs <- predict(m, newdata = train_df, type = "response")
loglik_train <- sum(log(train_fitted_probs[cbind(seq_len(nrow(train_df)), train_df$y)]))

# Test predictions: predict(type="response") returns an n_test × R matrix.
probs_mat <- predict(m, newdata = test_df, type = "response")

result <- list(
  # list-of-rows so jsonlite encodes as [[p1,p2,p3,p4], ...]
  probs        = lapply(seq_len(nrow(probs_mat)), function(i) as.list(unname(probs_mat[i, ]))),
  loglik_train = loglik_train,
  fit_time_ms  = fit_time_ms,
  converged    = isTRUE(m$converged),
  error        = NA
)

write_json(result, opts$output, auto_unbox = TRUE, pretty = TRUE, na = "null")
cat(sprintf(
  "mgcv ocat done: n_train=%d n_test=%d loglik_train=%.4f time=%.1fms converged=%s\n",
  nrow(train_df), nrow(test_df), loglik_train, fit_time_ms,
  if (isTRUE(m$converged)) "TRUE" else "FALSE"
))
