#' Clusterwise Linear Regression
#'
#' @param X an object used to select a method.
#' @param ... further arguments passed to or from other methods.
#'
#' @seealso \code{\link{clr.default}}.
#'
#' @export
clr <- function(X, ...) UseMethod("clr")

#' Clusterwise Linear Regression
#'
#' @param X data matrix of dimension \code{N} by \code{P}.
#' @param y response vector of length \code{N}.
#' @param K number of clusters.
#' @param m fuzzier exponent. Gives a hard k-partition if \code{m = 1.0} and
#'   a fuzzy k-partition when \code{m > 1.0}.
#' @param nstart number of random runs.
#' @param iter.max the maximum number of iterations allowed in Lloyd algorithm.
#' @param algorithm optimization algorithm. One of: \code{"Lloyd"}.
#' @param ... not used.
#'
#' @return returns an object of class \code{wclr.clr}.
#'
#' @export
clr.default <- function(X, y, K, m = 1.0,
                        nstart = 1L,
                        iter.max = 100L,
                        algorithm = c("Lloyd"), ...)
{
  ## check input ##############################################################

  stopifnot(is.matrix(X))
  stopifnot(is.numeric(y))

  N <- nrow(X) # number of observations
  P <- ncol(X) # number of predictors

  stopifnot(N > 0 && P > 0 && N > P && N == length(y))

  K <- as.integer(K)
  stopifnot(all(K > 0L) && length(K) == 1)

  m <- as.numeric(m)
  stopifnot(all(m >= 1.0) && length(m) == 1)

  nstart <- as.integer(nstart)
  stopifnot(all(nstart > 0L) && length(nstart) == 1)

  iter.max <- as.integer(iter.max)
  stopifnot(all(iter.max > 0L) && length(iter.max) == 1)

  stopifnot(algorithm %in% c("Lloyd"))

  ## model fitting ############################################################

  model <- list(loss = Inf)

  if (algorithm == "Lloyd")
  {
    for (run in 1:nstart)
    {
      if (m == 1.0)
        U <- hard_kpartition(N, K)
      else
        U <- fuzzy_kpartition(N, K)

      rmodel <- clr_lloyd_cpp(U, X, y, m, iter.max)

      if (rmodel$loss < model$loss)
        model <- rmodel
    }
  }

  ## output ###################################################################
  # ensure the correct layout
  model$coefficients  <- as.matrix(model$coefficients)
  model$fitted.values <- as.matrix(model$fitted.values)
  model$residuals     <- as.matrix(model$residuals)
  model$centers       <- as.matrix(model$centers)
  model$membership    <- as.matrix(model$membership)

  # name columns and rows
  if (is.null(colnames(X)))
    var_names <- paste("X", 1:P, sep = "")
  else
    var_names <- colnames(X)

  rownames(model$coefficients) <- c("(Intercept)", var_names)
  rownames(model$centers)      <- var_names
  colnames(model$coefficients)  <- paste("cluster", 1:model$K, sep = "")
  colnames(model$fitted.values) <- paste("cluster", 1:model$K, sep = "")
  colnames(model$residuals)     <- paste("cluster", 1:model$K, sep = "")
  colnames(model$centers)       <- paste("cluster", 1:model$K, sep = "")
  colnames(model$membership)    <- paste("cluster", 1:model$K, sep = "")

  model$call <- match.call()
  model$cluster <- max.col(model$membership)
  class(model) <- c(class(model), "WCLR")
  model
}

#' @export
clr.formula <- function(formula, data, K,
                        m = 1.0,
                        nstart = 1L,
                        iter.max = 100L,
                        algorithm = c("Lloyd"), ...)
{
  dataset <- model.frame(formula = formula, data = data)
  X <- as.matrix(dataset[, 2:ncol(dataset), drop = FALSE])
  y <- as.vector(dataset[, 1])

  model <- clr.default(X, y, K,
                       m = m,
                       nstart = nstart,
                       iter.max = iter.max,
                       algorithm = algorithm, ...)

  model$call <- match.call()
  model$formula <- formula
  model
}

#' @export
print.WCLR.clr <- function(x, ...)
{
  cat("Clusterwise Linear Regression\n\n")

  print(x$call)

  cat("\ncoefficients:\n")
  print(x$coefficients)

  cat("\ncenters:\n")
  print(x$centers)

  cat("\nnames:\n")
  print(names(x))

  invisible(x)
}

#' @export
predict.WCLR.clr <- function(object,
                              newdata, ...)
{
  predict.WCLR(object, newdata, ...)
}
