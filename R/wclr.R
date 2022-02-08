#' Weighted Clusterwise Linear Regression
#'
#' @param X an object used to select a method.
#' @param ... further arguments passed to or from other methods.
#'
#' @seealso \code{\link{wclr.default}}.
#'
#' @export
wclr <- function(X, ...) UseMethod("wclr")

#' Weighted Clusterwise Linear Regression
#'
#' @inheritParams clr.default
#' @param alpha numeric balancing value.
#' @param wnorm wnorm type. One of: \code{"epg"}, \code{"epl"},
#'   \code{"qpg"}, \code{"qpl"}.
#'
#' @return returns an object of class \code{wclr.wclr}.
#'
#' @export
wclr.default <- function(X, y, K, alpha, wnorm,
                         m = 1.0,
                         nstart = 1L,
                         iter.max = 100L,
                         algorithm = c("Lloyd"),
                         trace = FALSE, ...)
{
  ## check input ##############################################################

  stopifnot(is.matrix(X))
  stopifnot(is.numeric(y))

  N <- nrow(X) # number of observations
  P <- ncol(X) # number of predictors

  stopifnot(N > 0 && P > 0 && N > P && N == length(y))

  K <- as.integer(K)
  stopifnot(all(K > 0L) && length(K) == 1)

  alpha <- as.numeric(alpha)
  stopifnot(all(alpha > 0.0) && length(alpha) == 1)

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

  model <- list(loss = Inf)

  if (algorithm == "Lloyd")
  {
    for (run in 1:nstart)
    {
      if (m == 1.0)
        U <- hard_kpartition(N, K)
      else
        U <- fuzzy_kpartition(N, K)

      rmodel <- wclr_lloyd_cpp(U, X, y, alpha, m, wnorm, iter.max, trace)

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
  dimnames(model$weights)      <- list(var_names, var_names,
                                       paste("cluster", 1:model$K, sep = ""))
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
wclr.formula <- function(formula, data, K, alpha, wnorm,
                         m = 1.0,
                         nstart = 1L,
                         iter.max = 100L,
                         algorithm = c("Lloyd"),
                         trace = FALSE, ...)
{
  dataset <- model.frame(formula = formula, data = data)
  X <- as.matrix(dataset[, 2:ncol(dataset), drop = FALSE])
  y <- as.vector(dataset[, 1])

  model <- wclr.default(X, y, K, alpha, wnorm,
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
print.WCLR.wclr <- function(x, ...)
{
  cat("Weighted Clusterwise Linear Regression\n\n")

  print(x$call)

  cat("\ncoefficients:\n")
  print(x$coefficients)

  cat("\ncenters:\n")
  print(x$centers)

  cat("\nweights:\n")
  print(x$weights)

  cat("\nnames:\n")
  print(names(x))

  invisible(x)
}

#' Predict
#'
#' Predicted values based on \code{\link{swclr.default}} model object.
#'
#' @param object object of class inheriting from "WCLR.wclr".
#' @param newdata a data matrix in which to look for variables with
#' which to predict.
#' @param ... not used.
#'
#' @return produces a vector of predictions.
#'
#' @export
predict.WCLR.wclr <- function(object,
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

  predict_quadratic_cpp(X, object$centers, object$weights,
                        object$coefficients, object$m)
}
