#!/usr/bin/env Rscript
# CT score via Dirichlet-Multinomial Empirical Bayes.
# Usage: Rscript ct_score_dm.R input.csv output.csv [--sep=auto|comma|tab|semicolon] [--alpha=yes,no,unknown]
# Input: geneID, yes, no, unknown, total. total must equal yes+no+unknown.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L || any(args %in% c('--help', '-h'))) {
  cat('Usage: Rscript ct_score_dm.R input.csv output.csv [--sep=auto|comma|tab|semicolon] [--alpha=yes,no,unknown]\n',
      'Required input columns: geneID, yes, no, unknown, total\n',
      'By default alpha is estimated by marginal maximum likelihood.\n', sep='')
  quit(status = if (length(args) < 2L) 1L else 0L)
}
input <- args[1]; output <- args[2]
opts <- if (length(args) > 2L) args[-(1:2)] else character()
if (any(!grepl('^--(sep|alpha)=', opts))) stop('Unknown option: ', paste(opts[!grepl('^--(sep|alpha)=', opts)], collapse=', '))
opt_value <- function(key, default=NULL) {
  x <- grep(paste0('^--', key, '='), opts, value=TRUE)
  if (length(x)>1L) stop('Duplicate option: ', key)
  if (!length(x)) return(default)
  sub(paste0('^--',key,'='), '', x)
}
sep_arg <- opt_value('sep', 'auto')
if (!sep_arg %in% c('auto','comma','tab','semicolon')) stop('Invalid --sep option')
if (!file.exists(input)) stop('Input file does not exist: ', input)
if (normalizePath(input, mustWork=TRUE) == normalizePath(output, mustWork=FALSE)) stop('Output cannot overwrite input')
if (sep_arg == 'auto') {
  first <- readLines(input, n=1L, warn=FALSE, encoding='UTF-8')
  if (!length(first)) stop('Input file is empty')
  counts <- c(comma=length(strsplit(first, ',', fixed=TRUE)[[1]]),
              tab=length(strsplit(first, '\t', fixed=TRUE)[[1]]),
              semicolon=length(strsplit(first, ';', fixed=TRUE)[[1]]))
  sep_arg <- names(which.max(counts))
}
separator <- switch(sep_arg, comma=',', tab='\t', semicolon=';')
d <- read.table(input, header=TRUE, sep=separator, quote='"', comment.char='',
                stringsAsFactors=FALSE, check.names=FALSE, fileEncoding='UTF-8',
                na.strings=c('', 'NA'), fill=FALSE)
required <- c('geneID','yes','no','unknown','total')
if (!all(required %in% names(d))) stop('Missing columns: ', paste(setdiff(required, names(d)), collapse=', '))
if (!nrow(d)) stop('No data rows')
if (anyNA(d$geneID) || any(!nzchar(trimws(as.character(d$geneID))))) stop('Missing geneID')
if (anyDuplicated(d$geneID)) stop('Duplicated geneID: ', d$geneID[anyDuplicated(d$geneID)])
for (nm in c('yes','no','unknown','total')) {
  v <- suppressWarnings(as.numeric(d[[nm]]))
  if (anyNA(v) || any(!is.finite(v)) || any(v < 0) || any(v != floor(v)))
    stop('Column ', nm, ' must contain finite nonnegative integers without missing values')
  d[[nm]] <- v
}
if (any(d$total != d$yes + d$no + d$unknown)) {
  bad <- which(d$total != d$yes + d$no + d$unknown)
  stop('total != yes + no + unknown (first invalid gene: ', d$geneID[bad[1]], ')')
}
if (any(d$total == 0)) stop('total must be >= 1 for every gene')
X <- as.matrix(d[, c('yes','no','unknown')]); storage.mode(X) <- 'double'
alpha_fixed <- opt_value('alpha')
if (!is.null(alpha_fixed)) {
  alpha <- suppressWarnings(as.numeric(strsplit(alpha_fixed, ',', fixed=TRUE)[[1]]))
  if (length(alpha) != 3L || anyNA(alpha) || any(!is.finite(alpha)) || any(alpha <= 0))
    stop('--alpha requires three strictly positive comma-separated numeric values')
  fit <- list(value=NA_real_, convergence=NA_integer_)
  mode <- 'provided'
} else {
  # Omit multinomial coefficients (constant with respect to alpha).
  dm_nll <- function(log_alpha, X) {
    a <- exp(log_alpha)
    n <- rowSums(X)
    ll <- lgamma(sum(a)) - lgamma(n + sum(a))
    for (k in seq_along(a)) ll <- ll + lgamma(X[, k] + a[k]) - lgamma(a[k])
    value <- -sum(ll)
    if (is.finite(value)) value else .Machine$double.xmax / 100
  }
  mu0 <- (colSums(X) + 0.5) / (sum(X) + 1.5)
  starts <- c(0.01, 0.1, 1, 10, 100, 1000)
  fits <- lapply(starts, function(k) {
    tryCatch(optim(log(pmax(mu0*k, 1e-8)), dm_nll, X=X,
                   method='L-BFGS-B', lower=rep(log(1e-10),3),
                   upper=rep(log(1e8),3), control=list(maxit=10000)),
             error=function(e) NULL)
  })
  good <- Filter(function(z) !is.null(z) && is.finite(z$value) && z$convergence == 0L, fits)
  if (!length(good)) stop('No successful model convergence; inspect the input or supply --alpha=...')
  fit <- good[[which.min(vapply(good, function(z) z$value, numeric(1)))]]
  alpha <- exp(fit$par)
  mode <- 'estimated'
}
names(alpha) <- c('yes','no','unknown')
aY <- d$yes + alpha['yes']; aN <- d$no + alpha['no']; aU <- d$unknown + alpha['unknown']
A <- aY+aN+aU
C <- d$yes + d$no
# Note: p_ct_gt_* are probabilities about the latent conclusive yes fraction,
# not probabilities that a gene is biologically a CT gene.
d$n_conclusive <- C
d$p_yes_eb <- as.numeric(aY/A)
d$p_no_eb <- as.numeric(aN/A)
d$p_unknown_eb <- as.numeric(aU/A)
d$p_conclusive_eb <- as.numeric((aY+aN)/A)
d$ct_score_eb <- as.numeric(aY/(aY+aN))
d$p_ct_gt_05 <- as.numeric(pbeta(0.5,aY,aN,lower.tail=FALSE))
d$p_ct_gt_08 <- as.numeric(pbeta(0.8,aY,aN,lower.tail=FALSE))
d$ct_low95 <- as.numeric(qbeta(0.025,aY,aN))
d$ct_high95 <- as.numeric(qbeta(0.975,aY,aN))
d$evidence_status <- ifelse(C==0, 'no_conclusive_evidence', 'conclusive_evidence')
d$rank_ct_score <- rank(-d$ct_score_eb, ties.method='min')
d$rank_p_ct_gt_08 <- rank(-d$p_ct_gt_08, ties.method='min')
write.csv(d, output, row.names=FALSE, na='')
param_file <- paste0(tools::file_path_sans_ext(output), '_parameters.csv')
params <- data.frame(parameter=c('alpha_yes','alpha_no','alpha_unknown', 'mu_yes','mu_no',
                                'mu_unknown','kappa','log_likelihood_without_constants',
                                'optim_convergence'),
                     value=c(alpha,alpha/sum(alpha),sum(alpha),
                             if (is.finite(fit$value)) -fit$value else NA_real_,
                             fit$convergence))
write.csv(params,param_file,row.names=FALSE,na='')
cat('Genes: ', nrow(d), '\n',
    'Prior mode: ',mode,'\n',
    'alpha (yes,no,unknown): ',paste(signif(alpha,8),collapse=', '),'\n',
    'kappa: ',signif(sum(alpha),8),'\n',
    'Output: ',output,'\n',
    'Parameters: ',param_file,'\n',sep='')
cat('Caution: exploratory CT evidence scores; model calibration should be checked.\n')
