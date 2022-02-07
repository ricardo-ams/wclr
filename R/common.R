#' Fuzzy K-partition
#'
#' Makes a random fuzzy k-partition membership matrix of dimension
#' \code{N} by \code{K}.
#'
#' @param N number of observations
#' @param K number of clusters
#'
#' @seealso \code{\link{hard_kpartition}} for a hard (crisp) k-partition.
#'
#' @export
fuzzy_kpartition <- function(N, K)
{
  U <- matrix(stats::runif(N * K), nrow = N, ncol = K)

  for (n in 1:N)
    U[n,] <- U[n,] / sum(U[n,])

  U
}

#' Hard K-partition
#'
#' Makes a random hard (crisp) k-partition membership matrix of dimension
#' \code{N} by \code{K}.
#'
#' @param N number of observations
#' @param K number of clusters
#'
#' @seealso \code{\link{fuzzy_kpartition}} for a fuzzy k-partition.
#'
#' @export
hard_kpartition <- function(N, K)
{
  cluster <- sample(rep(1:K, (N / K) + 1)[1:N])

  U <- matrix(rep(.Machine$double.eps, N * K), nrow = N, ncol = K)

  for (n in 1:N)
    U[n, cluster[n]] <- 1.0 - ((K - 1) * .Machine$double.eps)

  U
}

#' @export
print.wclr.trace.info <- function(x, ...)
{
  cat(sprintf("%5d %25s %25.5f\n", x$iter, x$step, x$loss))
}

#' @export
print.wclr.trace.error <- function(x, ...)
{
  cat(sprintf("ERROR: %s\n", x$error))
}

#' @export
print.wclr.trace <- function(x, ...)
{
  cat(sprintf("%5s %25s %25s\n", "iter", "step", "loss"))
  for (log in x)
    print(log)
}

#' Predict
#'
#' Predicted values based on \code{\link{wclr.default}} model object.
#'
#' @param object object of class inheriting from "wclr".
#' @param newdata a data matrix in which to look for variables with
#' which to predict.
#' @param ... not used.
#'
#' @return produces a vector of predictions.
#'
#' @export
predict.wclr <- function(object,
                         newdata, ...)
{
  if (!is.null(object$formula))
  {
    dataset <- model.frame(formula = object$formula, data = newdata)
    X <- as.matrix(dataset[, 2:ncol(dataset), drop = FALSE])
  }
  else
  {
    X <- as.matrix(newdata)
  }

  N <- nrow(X)
  K <- object$K

  if (object$m == 1.0)
  {
    predicted.values <- sapply(1:N, function(n) {
      h <- which.min(sapply(1:K, function(k) {
        # squared Euclidean distance
        sum((X[n,] - object$centers[,k])^2.0)
      }))

      sum(c(1, X[n,]) * object$coefficients[,h])
    })
  }
  else
  {
    d <- matrix(rep(0, N * object$K), nrow = N, ncol = K)
    for (n in 1:N)
      for (k in 1:K)
        # squared Euclidean distance
        d[n,k] <- sum((X[n,] - object$centers[,k])^2.0)

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
