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

  U <- matrix(rep(0.0, N * K), nrow = N, ncol = K)

  for (n in 1:N)
    U[n, cluster[n]] <- 1.0

  U
}

#' Predict values
#'
#' @param object object of class inheriting from "WCLR".
#' @param newdata a data matrix in which to look for variables with
#' which to predict.
#' @param ... not used.
#'
#' @return produces a vector of predictions.
#'
#' @export
predict.WCLR <- function(object,
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

  predict_euclidean_cpp(X, object$centers, object$coefficients, object$m)
}
