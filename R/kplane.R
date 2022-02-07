#' K-plane Regression
#'
#' @param X an object used to select a method.
#' @param ... further arguments passed to or from other methods.
#'
#' @seealso \code{\link{kplane.default}}.
#'
#' @export
kplane <- function(X, ...) UseMethod("kplane")

#' K-plane Regression
#'
#' @inheritParams clr.default
#' @param gamma numeric balancing value.
#'
#' @return returns an object of class \code{wclr.kmeans}.
#'
#' @export
kplane.default <- function(X, y, K, gamma,
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

  gamma <- as.numeric(gamma)
  stopifnot(all(gamma > 0.0) && length(gamma) == 1)

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

      rmodel <- kplane_lloyd_cpp(U, X, y, gamma, m, iter.max, trace)

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
  colnames(model$membership)    <- paste("cluster", 1:model$K, sep = "")
  model$cluster                 <- max.col(model$membership)
  class(model)                  <- c(class(model), "wclr")
  model
}

#' @export
kplane.formula <- function(formula, data, K, gamma,
                           m = 1.0,
                           nstart = 1L,
                           iter.max = 100L,
                           algorithm = c("Lloyd"),
                           trace = FALSE, ...)
{
  dataset <- model.frame(formula = formula, data = data)
  X <- as.matrix(dataset[, 2:ncol(dataset), drop = FALSE])
  y <- as.vector(dataset[, 1])

  model <- kplane.default(X, y, K, gamma,
                          m = m,
                          nstart = nstart,
                          iter.max = iter.max,
                          algorithm = algorithm,
                          trace = trace, ...)

  model$call <- match.call()
  model$formula <- formula
  model
}

#' @export
print.wclr.kplane <- function(x, ...)
{
  cat("K-plane Regression\n\n")

  print(x$call)

  cat("\ncoefficients:\n")
  print(x$coefficients)

  cat("\ncenters:\n")
  print(x$centers)

  cat("\nnames:\n")
  print(names(x))

  invisible(x)
}
