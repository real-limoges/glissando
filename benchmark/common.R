# Shared helpers for the benchmark fitter scripts (fit_mgcv.R, fit_gamlss.R,
# fit_ocat_mgcv.R). Source with:
#   source(file.path(script_dir(), "common.R"))

# Directory containing the currently executing Rscript (robust to the caller's
# working directory).
script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) == 0) return(".")
  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
}

# Parquet reader: prefer arrow, fall back to the dependency-free nanoparquet.
read_parquet <- if (requireNamespace("arrow", quietly = TRUE)) {
  arrow::read_parquet
} else if (requireNamespace("nanoparquet", quietly = TRUE)) {
  function(path) as.data.frame(nanoparquet::read_parquet(path))
} else {
  stop("need the 'arrow' or 'nanoparquet' package to read parquet input")
}
