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

  gamma <- as.numeric(gamma)
  stopifnot(all(gamma > 0.0) && length(gamma) == 1)

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

      run_model <- kplane_lloyd_cpp(U, X, y, gamma, m, iter.max)

      if (is.finite(run_model$loss) && run_model$loss < model$loss)
        model <- run_model
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
kplane.formula <- function(formula, data, K, gamma,
                           m = 1.0,
                           nstart = 1L,
                           iter.max = 100L,
                           algorithm = c("Lloyd"), ...)
{
  dataset <- model.frame(formula = formula, data = data)
  X <- as.matrix(dataset[, 2:ncol(dataset), drop = FALSE])
  y <- as.vector(dataset[, 1])

  model <- kplane.default(X, y, K, gamma,
                          m = m,
                          nstart = nstart,
                          iter.max = iter.max,
                          algorithm = algorithm, ...)

  model$call <- match.call()
  model$formula <- formula
  model
}

#' @export
print.WCLR.kplane <- function(x, ...)
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

#' @export
predict.WCLR.kplane <- function(object,
                                newdata, ...)
{
  predict.WCLR(object, newdata, ...)
}
