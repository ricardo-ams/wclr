#' Self-balanced Weighted Clusterwise Linear Regression
#'
#' @param X an object used to select a method.
#' @param ... further arguments passed to or from other methods.
#'
#' @seealso \code{\link{swclr.default}}.
#'
#' @export
swclr <- function(X, ...) UseMethod("swclr")

#' Self-balanced Weighted Clusterwise Linear Regression
#'
#' @inheritParams clr.default
#' @param wnorm wnorm type. One of: \code{"epg"}, \code{"epl"},
#'   \code{"qpg"}, \code{"qpl"}.
#' @param balance self-balance type. One of: \code{"pg"}, \code{"pl"}.
#'
#' @return returns an object of class \code{wclr.swclr}.
#'
#' @export
swclr.default <- function(X, y, K, balance, wnorm,
                          m = 1.0,
                          nstart = 1L,
                          iter.max = 100L,
                          algorithm = c("Lloyd"),
                          trace = FALSE, ...)
{
  ## check input ##############################################################

  X <- as.matrix(X)
  y <- as.numeric(y)
  stopifnot(nrow(X) == length(y))

  K <- as.integer(K)
  stopifnot(all(K > 0L) && length(K) == 1)

  stopifnot(balance %in% c("pg", "pl") && length(wnorm) == 1)

  stopifnot(wnorm %in% c("qpl", "qpg", "epl", "epg") && length(wnorm) == 1)

  m <- as.numeric(m)
  stopifnot(all(m >= 1.0) && length(m) == 1)

  nstart <- as.integer(nstart)
  stopifnot(all(nstart > 0L) && length(nstart) == 1)

  iter.max <- as.integer(iter.max)
  stopifnot(all(iter.max > 0L) && length(iter.max) == 1)

  stopifnot(algorithm %in% c("Lloyd"))

  trace <- as.logical(trace)

  ## model fitting ############################################################

  N <- nrow(X)

  model <- list(loss = Inf)

  if (algorithm == "Lloyd")
  {
    for (run in 1:nstart)
    {
      if (m == 1.0)
        U <- hard_kpartition(N, K)
      else
        U <- fuzzy_kpartition(N, K)

      rmodel <- swclr_lloyd_cpp(U, X, y, m, wnorm, balance, iter.max, trace)

      if (rmodel$loss < model$loss)
        model <- rmodel
    }
  }

  ## output ###################################################################

  model$call                    <- match.call()
  rownames(model$coefficients)  <- c("(intercept)", colnames(X))
  colnames(model$coefficients)  <- paste("cluster", 1:model$K, sep = "")
  colnames(model$fitted.values) <- paste("cluster", 1:model$K, sep = "")
  colnames(model$residuals)     <- paste("cluster", 1:model$K, sep = "")
  rownames(model$centers)       <- colnames(X)
  colnames(model$centers)       <- paste("cluster", 1:model$K, sep = "")
  dimnames(model$weights)       <- list(colnames(X), colnames(X),
                                        paste("cluster", 1:model$K, sep = ""))
  rownames(model$alphas)        <- paste("cluster", 1:model$K, sep = "")
  rownames(model$gammas)        <- paste("cluster", 1:model$K, sep = "")
  colnames(model$membership)    <- paste("cluster", 1:model$K, sep = "")
  model$cluster                 <- max.col(model$membership)
  class(model)                  <- c(class(model), "wclr")
  model
}

#' @export
print.wclr.swclr <- function(x, ...)
{
  cat("Self-balanced Weighted Clusterwise Linear Regression\n\n")

  print(x$call)

  cat("\ncoefficients:\n")
  print(x$coefficients)

  cat("\ncenters:\n")
  print(x$centers)

  cat("\nweights:\n")
  print(x$weights)

  cat("\nalphas:\n")
  print(x$alphas)

  cat("\ngammas:\n")
  print(x$gammas)

  cat("\nnames:\n")
  print(names(x))

  invisible(x)
}

#' Predict
#'
#' Predicted values based on \code{\link{swclr.default}} model object.
#'
#' @param object object of class inheriting from "wclr.swclr".
#' @param newdata a data matrix in which to look for variables with
#' which to predict.
#' @param ... not used.
#'
#' @return produces a vector of predictions.
#'
#' @export
predict.wclr.swclr <- function(object,
                               newdata, ...)
{
  X <- as.matrix(newdata)

  N <- nrow(X)
  K <- object$K

  if (object$m == 1.0)
  {
    predicted.values <- sapply(1:N, function(n) {
      h <- which.min(sapply(1:K, function(k) {
        # squared Mahalanobis distance
        stats::mahalanobis(
          X[n, ],
          object$centers[, k],
          object$weights[, , k],
          inverted = TRUE)
      }))

      sum(c(1, X[n,]) * object$coefficients[,h])
    })
  }
  else
  {
    d <- matrix(rep(0, N * object$K), nrow = N, ncol = K)
    for (n in 1:N)
      for (k in 1:K)
        # squared Mahalanobis distance
        d[n,k] <- stats::mahalanobis(
          X[n, ],
          object$centers[, k],
          object$weights[, , k],
          inverted = TRUE)

    expm <- 1.0 / (object$m - 1.0)

    predicted.values <- sapply(1:N, function(n) {
      sum(sapply(1:K, function(k) {
        unk <- 1.0 / sum((d[n,k] / d[n,])^expm)

        unk * sum(c(1, X[n,]) * object$coefficients[,k])
      }))
    })
  }

  predicted.values
}
