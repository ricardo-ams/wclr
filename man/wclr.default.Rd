% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/wclr.R
\name{wclr.default}
\alias{wclr.default}
\title{Weighted Clusterwise Linear Regression}
\usage{
\method{wclr}{default}(
  X,
  y,
  K,
  alpha,
  wnorm,
  m = 1,
  nstart = 1L,
  iter.max = 100L,
  algorithm = c("Lloyd"),
  ...
)
}
\arguments{
\item{X}{data matrix of dimension \code{N} by \code{P}.}

\item{y}{response vector of length \code{N}.}

\item{K}{number of clusters.}

\item{alpha}{numeric balancing value.}

\item{wnorm}{wnorm type. One of: \code{"epg"}, \code{"epl"},
\code{"qpg"}, \code{"qpl"}.}

\item{m}{fuzzier exponent. Gives a hard k-partition if \code{m = 1.0} and
a fuzzy k-partition when \code{m > 1.0}.}

\item{nstart}{number of random runs.}

\item{iter.max}{the maximum number of iterations allowed in Lloyd algorithm.}

\item{algorithm}{optimization algorithm. One of: \code{"Lloyd"}.}

\item{...}{not used.}
}
\value{
returns an object of class \code{wclr.wclr}.
}
\description{
Weighted Clusterwise Linear Regression
}
