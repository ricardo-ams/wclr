// print only critical warnings about arguments and/or data which are likely
// to lead to incorrect results.
#define ARMA_WARN_LEVEL 1
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

void
  update_coefficients(arma::mat &R,
                      const arma::mat &U,
                      const arma::mat &X,
                      const arma::colvec &y)
  {
    const int K = U.n_cols;

    for (int k = 0; k < K; k++)
    {
      try
      {
        arma::mat W = arma::diagmat(U.col(k));

        arma::colvec coefs = arma::solve(X.t() * W * X, X.t() * W * y);

        if (coefs.is_finite())
          R.col(k) = coefs;
      }
      catch (...) { }
    }
  }

void
  update_centers(arma::mat &C,
                 const arma::mat &U,
                 const arma::mat &X)
  {
    const int N = U.n_rows;
    const int K = U.n_cols;
    const int P = X.n_cols;

    for (int k = 0; k < K; k++)
    {
      try
      {
        // Compute the weighted arithmetic mean M of a data set using the
        // recurrence relation
        //   M(n) = M(n-1) + (data[n] - M(n-1)) (w(n)/(W(n-1) + w(n)))
        //   W(n) = W(n-1) + w(n)
        // Reference:
        //   GNU GSL 2.6 source code - https://www.gnu.org/software/gsl/doc/html/statistics.html?highlight=weighted%20mean#c.gsl_stats_wmean

        arma::colvec centroid(P, arma::fill::zeros);

        double W = 0.0;

        for (int n = 0; n < N; n++)
        {
          const double unk = U(n, k);

          if (unk > 0.0)
          {
            W += unk;
            centroid += (X.row(n).t() - centroid) * (unk / W);
          }
        }

        if (centroid.is_finite())
          C.col(k) = centroid;
      }
      catch (...) {}
    }
  }

template <typename F>
bool
  update_membership(arma::mat &U,
                    const double m,
                    const F &diss)
  {
    const int N = U.n_rows;
    const int K = U.n_cols;
    const double exp = 1.0 / (m - 1.0);

    bool converged = true;

    for (int n = 0; n < N; n++)
    {
      try
      {
        arma::rowvec membership(K, arma::fill::zeros);

        for (int k = 0; k < K; k++)
        {
          double unk = 0.0;

          for (int h = 0; h < K; h++)
            unk += std::pow(diss(n, k) / diss(n, h), exp);

          unk = 1.0 / unk;

          if (std::fabs(U(n, k) - unk) > 0.01)
            converged = false;

          membership(k) = unk;
        }

        if (membership.is_finite())
          U.row(n) = membership;
      }
      catch (...) {}
    }

    return converged;
  }

template <typename F>
bool
  update_membership(arma::mat &U,
                    const F &diss)
  {
    const int N = U.n_rows;
    const int K = U.n_cols;

    bool converged = true;

    for (int n = 0; n < N; n++)
    {
      try
      {
        int min_k = -1;
        double min_dnk = std::numeric_limits<double>().max();

        for (int k = 0; k < K; k++)
        {
          const double dnk = diss(n, k);

          if (dnk < min_dnk)
          {
            min_dnk = dnk;
            min_k   = k;
          }
        }

        if (U(n, min_k) !=  1.0)
        {
          converged = false;
          U.row(n).zeros();
          U(n, min_k) = 1.0;
        }
      } catch (...) {}
    }

    return converged;
  }

template <typename F>
double
  J(const arma::mat &U,
    const F &diss)
  {
    const int N = U.n_rows;
    const int K = U.n_cols;

    double loss = 0.0;

    for (int k = 0; k < K; k++)
    {
      for (int n = 0; n < N; n++)
      {
        loss += U(n, k) * diss(n, k);
      }
    }

    return loss;
  }

//' Clusterwise Linear Regression
//'
//' This is a bare-bones function and must NOT be used by an end-user.
//'
//' @param U random membership matrix.
//' @param X data matrix of dimension \code{N} by \code{P}.
//' @param y response vector of length \code{N}.
//' @param m fuzzier exponent. Gives a hard k-partition if \code{m = 1.0} and
//'   a fuzzy k-partition when \code{m > 1.0}.
//' @param itermax the maximum number of iterations allowed in Lloyd algorithm.
//'
//' @return returns an object of class \code{WCLR.clr}.
//'
//' @seealso \code{\link{clr.default}} for a user-friendly version of this method.
//'
//' @export
// [[Rcpp::export]]
Rcpp::List
  clr_lloyd_cpp(arma::mat &U,
                const arma::mat &X,
                const arma::colvec &y,
                const double m,
                const int itermax)
  {
    const int N = X.n_rows; // number of observations
    const int P = X.n_cols; // number of variables
    const int K = U.n_cols; // number of clusters

    // augmented input matrix for regression with intercept
    arma::mat X1(N, P + 1, arma::fill::ones);
    X1.cols(1, P) = X;

    // model
    arma::mat coefficients(P + 1, K, arma::fill::randu);
    arma::mat fitted_values(N, K, arma::fill::zeros);
    arma::mat residuals(N, K, arma::fill::zeros);

    // dissimilarity (error) function
    auto E = [&](int n, int k)
    {
      return residuals(n, k) * residuals(n, k);
    };

    // Lloyd Algorithm
    bool converged = false;
    int iterations = 0;

    while (true)
    {
      Rcpp::checkUserInterrupt();

      // update parameters
      arma::mat Um = arma::pow(U, m);

      update_coefficients(coefficients, Um, X1, y);

      for (int k = 0; k < K; k++)
      {
        fitted_values.col(k) = X1 * coefficients.col(k);
        residuals.col(k)     = y - fitted_values.col(k);
      }

      // check convergence
      if (converged || iterations == itermax)
        break;

      iterations++;

      // update kpartition
      if (m == 1.0)
        converged = update_membership(U, E);
      else
        converged = update_membership(U, m, E);
    }

    // compute centroids
    arma::mat centers(P, K, arma::fill::zeros);
    update_centers(centers, arma::pow(U, m), X);

    // build Rcpp model
    auto rcpp_lloyd = Rcpp::List::create(
      Rcpp::Named("iter.max")   = itermax,
      Rcpp::Named("iterations") = iterations,
      Rcpp::Named("converged")  = converged
    );
    rcpp_lloyd.attr("class") = "WCLR.lloyd";

    double loss = J(arma::pow(U, m), E);
    auto rcpp_model = Rcpp::List::create(
      Rcpp::Named("K")             = K,
      Rcpp::Named("m")             = m,
      Rcpp::Named("loss")          = loss,
      Rcpp::Named("coefficients")  = coefficients,
      Rcpp::Named("fitted.values") = fitted_values,
      Rcpp::Named("residuals")     = residuals,
      Rcpp::Named("centers")       = centers,
      Rcpp::Named("membership")    = U,
      Rcpp::Named("algorithm")     = rcpp_lloyd
    );
    rcpp_model.attr("class") = "WCLR.clr";

    return rcpp_model;
  }

//' K-means Linear Regression
//'
//' This is a bare-bones function and must NOT be used by an end-user.
//'
//' @inheritParams clr_lloyd_cpp
//'
//' @return returns an object of class \code{WCLR.kmeans}.
//'
//' @seealso \code{\link{kmeans.default}} for a user-friendly version of this method.
//'
//' @export
// [[Rcpp::export]]
Rcpp::List
  kmeans_lloyd_cpp(arma::mat &U,
                   const arma::mat &X,
                   const arma::colvec &y,
                   const double m,
                   const int itermax)
  {
    const int N = X.n_rows; // number of observations
    const int P = X.n_cols; // number of variables
    const int K = U.n_cols; // number of clusters

    // augmented input matrix for regression with intercept
    arma::mat X1(N, P + 1, arma::fill::ones);
    X1.cols(1, P) = X;

    // model
    arma::mat centers(P, K, arma::fill::randu);

    // dissimilarity (error) function
    auto E = [&](int n, int k)
    {
      // squared Euclidean distance
      return dot(X.row(n).t() - centers.col(k),
                 X.row(n).t() - centers.col(k));
    };

    // Lloyd Algorithm
    bool converged = false;
    int iterations = 0;

    while (true)
    {
      Rcpp::checkUserInterrupt();

      // update parameters
      arma::mat Um = arma::pow(U, m);

      update_centers(centers, Um, X);

      // check convergence
      if (converged || iterations == itermax)
        break;

      iterations++;

      // update kpartition
      if (m == 1.0)
        converged = update_membership(U, E);
      else
        converged = update_membership(U, m, E);
    }

    // compute coefficients
    arma::mat coefficients(P + 1, K, arma::fill::zeros);
    arma::mat fitted_values(N, K, arma::fill::zeros);
    arma::mat residuals(N, K, arma::fill::zeros);

    update_coefficients(coefficients, arma::pow(U, m), X1, y);

    for (int k = 0; k < K; k++)
    {
      fitted_values.col(k) = X1 * coefficients.col(k);
      residuals.col(k)     = y - fitted_values.col(k);
    }

    // build Rcpp model
    auto rcpp_lloyd = Rcpp::List::create(
      Rcpp::Named("iter.max")   = itermax,
      Rcpp::Named("iterations") = iterations,
      Rcpp::Named("converged")  = converged
    );
    rcpp_lloyd.attr("class") = "WCLR.lloyd";

    double loss = J(arma::pow(U, m), E);
    auto rcpp_model = Rcpp::List::create(
      Rcpp::Named("K")             = K,
      Rcpp::Named("m")             = m,
      Rcpp::Named("loss")          = loss,
      Rcpp::Named("coefficients")  = coefficients,
      Rcpp::Named("fitted.values") = fitted_values,
      Rcpp::Named("residuals")     = residuals,
      Rcpp::Named("centers")       = centers,
      Rcpp::Named("membership")    = U,
      Rcpp::Named("algorithm")     = rcpp_lloyd
    );
    rcpp_model.attr("class") = "WCLR.kmeans";

    return rcpp_model;
  }

//' K-plane Regression
//'
//' This is a bare-bones function and must NOT be used by an end-user.
//'
//' @inheritParams clr_lloyd_cpp
//' @param gamma numeric balancing value.
//'
//' @return returns an object of class \code{WCLR.kplane}.
//'
//' @seealso \code{\link{kplane.default}} for a user-friendly version of this method.
//'
//' @export
// [[Rcpp::export]]
Rcpp::List
  kplane_lloyd_cpp(arma::mat &U,
                   const arma::mat &X,
                   const arma::colvec &y,
                   const double gamma,
                   const double m,
                   const int itermax)
  {
    const int N = X.n_rows; // number of observations
    const int P = X.n_cols; // number of variables
    const int K = U.n_cols; // number of clusters

    // augmented input matrix for regression with intercept
    arma::mat X1(N, P + 1, arma::fill::ones);
    X1.cols(1, P) = X;

    // model
    arma::mat coefficients(P + 1, K, arma::fill::randu);
    arma::mat fitted_values(N, K, arma::fill::zeros);
    arma::mat residuals(N, K, arma::fill::zeros);
    arma::mat centers(P, K, arma::fill::randu);

    // dissimilarity (error) function
    auto Ex = [&](int n, int k)
    {
      // squared Euclidean distance
      return dot(X.row(n).t() - centers.col(k),
                 X.row(n).t() - centers.col(k));
    };

    auto Ey = [&](int n, int k)
    {
      // squared residual
      return residuals(n, k) * residuals(n, k);
    };

    auto E = [&](int n, int k)
    {
      return Ey(n, k) + (gamma * Ex(n, k));
    };

    // Lloyd Algorithm
    bool converged = false;
    int iterations = 0;

    while (true)
    {
      Rcpp::checkUserInterrupt();

      // update parameters
      arma::mat Um = arma::pow(U, m);

      update_coefficients(coefficients, Um, X1, y);

      for (int k = 0; k < K; k++)
      {
        fitted_values.col(k) = X1 * coefficients.col(k);
        residuals.col(k)     = y - fitted_values.col(k);
      }

      update_centers(centers, Um, X);

      // check convergence
      if (converged || iterations == itermax)
        break;

      iterations++;

      // update kpartition
      if (m == 1.0)
        converged = update_membership(U, E);
      else
        converged = update_membership(U, m, E);
    }

    // build Rcpp model
    auto rcpp_lloyd = Rcpp::List::create(
      Rcpp::Named("iter.max")   = itermax,
      Rcpp::Named("iterations") = iterations,
      Rcpp::Named("converged")  = converged
    );
    rcpp_lloyd.attr("class") = "WCLR.lloyd";

    double loss = J(arma::pow(U, m), E);
    auto rcpp_model = Rcpp::List::create(
      Rcpp::Named("K")             = K,
      Rcpp::Named("gamma")         = gamma,
      Rcpp::Named("m")             = m,
      Rcpp::Named("loss")          = loss,
      Rcpp::Named("coefficients")  = coefficients,
      Rcpp::Named("fitted.values") = fitted_values,
      Rcpp::Named("residuals")     = residuals,
      Rcpp::Named("centers")       = centers,
      Rcpp::Named("membership")    = U,
      Rcpp::Named("algorithm")     = rcpp_lloyd
    );
    rcpp_model.attr("class") = "WCLR.kplane";

    return rcpp_model;
  }

void
  update_weights_qpl(arma::cube &W,
                     const arma::mat &U,
                     const arma::mat &C,
                     const arma::mat &X)
  {
    const int N = U.n_rows;
    const int K = U.n_cols;
    const int P = C.n_rows;

    for (int k = 0; k < K; k++)
    {
      try {
        arma::mat Q(P, P, arma::fill::zeros);

        for (int n = 0; n < N; n++)
        {
          arma::colvec xc = X.row(n).t() - C.col(k);
          Q += U(n, k) * (xc * xc.t());
        }

        arma::mat weights = std::pow(arma::det(Q), (1.0 / P)) * arma::inv(Q);

        if (weights.is_finite() && weights.is_symmetric())
          W.slice(k) = weights;
      }
      catch (...) {}
    }
  }

void
  update_weights_qpg(arma::cube &W,
                     const arma::mat &U,
                     const arma::mat &C,
                     const arma::mat &X)
  {
    const int N = U.n_rows;
    const int K = U.n_cols;
    const int P = C.n_rows;

    try
    {
      arma::mat Q(P, P, arma::fill::zeros);

      for (int k = 0; k < K; k++)
      {
        for (int n = 0; n < N; n++)
        {
          arma::colvec xc = X.row(n).t() - C.col(k);
          Q += U(n, k) * (xc * xc.t());
        }
      }

      const arma::mat weights = std::pow(arma::det(Q), (1.0 / P)) * arma::inv(Q);

      if (weights.is_finite() && weights.is_symmetric())
        for (int k = 0; k < K; k++)
          W.slice(k) = weights;
    }
    catch (...) {}
  }

void
  update_weights_epl(arma::cube &W,
                     const arma::mat &U,
                     const arma::mat &C,
                     const arma::mat &X)
  {
    const int N = U.n_rows;
    const int K = U.n_cols;
    const int P = C.n_rows;

    for (int k = 0; k < K; k++)
    {
      try
      {
        arma::colvec aux(P, arma::fill::zeros);

        for (int n = 0; n < N; n++)
        {
          arma::colvec t = arma::pow(X.row(n).t() - C.col(k), 2.0);
          aux += U(n, k) * t;
        }

        const arma::colvec w = arma::as_scalar(arma::prod(arma::pow(aux, 1.0 / P))) / aux;

        const arma::mat weights = arma::diagmat(w);

        if (weights.is_finite() && weights.is_symmetric())
          W.slice(k) = weights;
      }
      catch (...) {}
    }
  }

void
  update_weights_epg(arma::cube &W,
                     const arma::mat &U,
                     const arma::mat &C,
                     const arma::mat &X)
  {
    const int N = U.n_rows;
    const int K = U.n_cols;
    const int P = C.n_rows;

    try
    {
      arma::colvec aux(P, arma::fill::zeros);

      for (int k = 0; k < K; k++)
      {
        for (int n = 0; n < N; n++)
        {
          const double unk = U(n, k);

          arma::colvec t = arma::pow(X.row(n).t() - C.col(k), 2.0);
          aux += unk * t;
        }
      }

      const arma::colvec w = arma::as_scalar(arma::prod(arma::pow(aux, 1.0 / P))) / aux;

      const arma::mat weights = arma::diagmat(w);

      if (weights.is_finite() && weights.is_symmetric())
        for (int k = 0; k < K; k++)
          W.slice(k) = weights;
    }
    catch (...) {}
  }

//' Weighted Clusterwise Linear Regression
//'
//' This is a bare-bones function and must NOT be used by an end-user.
//'
//' @inheritParams clr_lloyd_cpp
//' @param alpha numeric balancing value.
//' @param wnorm wnorm type. One of: \code{"epg"}, \code{"epl"},
//'   \code{"qpg"}, \code{"qpl"}.
//'
//' @return returns an object of class \code{WCLR.wclr}.
//'
//' @seealso \code{\link{wclr.default}} for a user-friendly version of this method.
//'
//' @export
// [[Rcpp::export]]
Rcpp::List
  wclr_lloyd_cpp(arma::mat &U,
                 const arma::mat &X,
                 const arma::colvec &y,
                 const double alpha,
                 const double m,
                 const std::string wnorm,
                 const int itermax)
  {
    const int N = X.n_rows; // number of observations
    const int P = X.n_cols; // number of variables
    const int K = U.n_cols; // number of clusters

    // augmented input matrix for regression with intercept
    arma::mat X1(N, P + 1, arma::fill::ones);
    X1.cols(1, P) = X;

    // model
    arma::mat coefficients(P + 1, K, arma::fill::randu);
    arma::mat fitted_values(N, K, arma::fill::zeros);
    arma::mat residuals(N, K, arma::fill::zeros);
    arma::mat centers(P, K, arma::fill::randu);
    arma::cube weights(P, P, K, arma::fill::ones);

    // dissimilarity (error) function
    auto Ex = [&](int n, int k)
    {
      arma::colvec xc = X.row(n).t() - centers.col(k);
      return arma::as_scalar(xc.t() * weights.slice(k) * xc);
    };

    auto Ey = [&](int n, int k)
    {
      // squared residual
      return residuals(n, k) * residuals(n, k);
    };

    auto E = [&](int n, int k)
    {
      return Ex(n, k) + (alpha * Ey(n, k));
    };

    // Lloyd Algorithm
    bool converged = false;
    int iterations = 0;

    while (true)
    {
      Rcpp::checkUserInterrupt();

      // update parameters
      arma::mat Um = arma::pow(U, m);

      update_coefficients(coefficients, Um, X1, y);

      for (int k = 0; k < K; k++)
      {
        fitted_values.col(k) = X1 * coefficients.col(k);
        residuals.col(k)     = y - fitted_values.col(k);
      }

      update_centers(centers, Um, X);

      if (wnorm == "epg")
        update_weights_epg(weights, Um, centers, X);
      else if (wnorm == "epl")
        update_weights_epl(weights, Um, centers, X);
      else if (wnorm == "qpg")
        update_weights_qpg(weights, Um, centers, X);
      else if (wnorm == "qpl")
        update_weights_qpl(weights, Um, centers, X);

      // check convergence
      if (converged || iterations == itermax)
        break;

      iterations++;

      // update kpartition
      if (m == 1.0)
        converged = update_membership(U, E);
      else
        converged = update_membership(U, m, E);
    }

    // build Rcpp model
    auto rcpp_lloyd = Rcpp::List::create(
      Rcpp::Named("iter.max")   = itermax,
      Rcpp::Named("iterations") = iterations,
      Rcpp::Named("converged")  = converged
    );
    rcpp_lloyd.attr("class") = "WCLR.lloyd";

    double loss = J(arma::pow(U, m), E);
    auto rcpp_model = Rcpp::List::create(
      Rcpp::Named("K")             = K,
      Rcpp::Named("alpha")         = alpha,
      Rcpp::Named("m")             = m,
      Rcpp::Named("wnorm")         = wnorm,
      Rcpp::Named("loss")          = loss,
      Rcpp::Named("coefficients")  = coefficients,
      Rcpp::Named("fitted.values") = fitted_values,
      Rcpp::Named("residuals")     = residuals,
      Rcpp::Named("centers")       = centers,
      Rcpp::Named("weights")       = weights,
      Rcpp::Named("membership")    = U,
      Rcpp::Named("algorithm")     = rcpp_lloyd
    );
    rcpp_model.attr("class") = "WCLR.wclr";

    return rcpp_model;
  }

template <typename F1, typename F2>
void
  update_balance_pg(arma::vec &alphas,
                    arma::vec &gammas,
                    const arma::mat &U,
                    const F1 &Ey,
                    const F2 &Ex)
  {
    const int N = U.n_rows;
    const int K = U.n_cols;

    try
    {
      double dx = 0.0;
      double dy = 0.0;

      for (int k = 0; k < K; k++)
      {
        for (int n = 0; n < N; n++)
        {
          const double unk = U(n, k);

          dy += unk * Ey(n, k);
          dx += unk * Ex(n, k);
        }
      }

      double a = std::sqrt(dy / dx);
      double g = std::sqrt(dx / dy);

      if (std::isfinite(a) && std::isfinite(g))
      {
        alphas.fill(a);
        gammas.fill(g);
      }
    }
    catch (...) {}
  }

template <typename F1, typename F2>
void
  update_balance_pl(arma::vec &alphas,
                    arma::vec &gammas,
                    const arma::mat &U,
                    const F1 &Ey,
                    const F2 &Ex)
  {
    const int N = U.n_rows;
    const int K = U.n_cols;

    for (int k = 0; k < K; k++)
    {
      try
      {
        double dx = 0.0;
        double dy = 0.0;

        for (int n = 0; n < N; n++)
        {
          const double unk = U(n, k);

          dy += unk * Ey(n, k);
          dx += unk * Ex(n, k);
        }

        double a = std::sqrt(dy / dx);
        double g = std::sqrt(dx / dy);

        if (std::isfinite(a) && std::isfinite(g))
        {
          alphas(k) = a;
          gammas(k) = g;
        }
      }
      catch (...) {}
    }
  }

//' Self-balanced Weighted Clusterwise Linear Regression
//'
//' This is a bare-bones function and must NOT be used by an end-user.
//'
//' @inheritParams clr_lloyd_cpp
//' @param wnorm wnorm type. One of: \code{"epg"}, \code{"epl"},
//'   \code{"qpg"}, \code{"qpl"}.
//' @param balance self-balance type. One of: \code{"pg"}, \code{"pl"}.
//'
//' @return returns an object of class \code{WCLR.swclr}.
//'
//' @seealso \code{\link{swclr.default}} for a user-friendly version of this method.
//'
//' @export
// [[Rcpp::export]]
Rcpp::List
  swclr_lloyd_cpp(arma::mat &U,
                  const arma::mat &X,
                  const arma::colvec &y,
                  const double m,
                  const std::string wnorm,
                  const std::string balance,
                  const int itermax)
  {
    const int N = X.n_rows; // number of observations
    const int P = X.n_cols; // number of variables
    const int K = U.n_cols; // number of clusters

    // augmented input matrix for regression with intercept
    arma::mat X1(N, P + 1, arma::fill::ones);
    X1.cols(1, P) = X;

    // model
    arma::mat  coefficients(P + 1, K, arma::fill::randu);
    arma::mat  fitted_values(N, K, arma::fill::zeros);
    arma::mat  residuals(N, K, arma::fill::zeros);
    arma::mat  centers(P, K, arma::fill::randu);
    arma::cube weights(P, P, K, arma::fill::ones);
    arma::vec  alphas(K, arma::fill::ones);
    arma::vec  gammas(K, arma::fill::ones);

    // dissimilarity (error) function
    auto Ex = [&](int n, int k)
    {
      arma::colvec xc = X.row(n).t() - centers.col(k);
      return arma::as_scalar(xc.t() * weights.slice(k) * xc);
    };

    auto Ey = [&](int n, int k)
    {
      // squared residual
      return residuals(n, k) * residuals(n, k);
    };

    auto E = [&](int n, int k)
    {
      return (alphas(k) * Ex(n, k)) + (gammas(k) * Ey(n, k));
    };

    // Lloyd Algorithm
    bool converged = false;
    int iterations = 0;

    while (true)
    {
      Rcpp::checkUserInterrupt();

      // update parameters
      arma::mat Um = arma::pow(U, m);

      update_coefficients(coefficients, Um, X1, y);

      for (int k = 0; k < K; k++)
      {
        fitted_values.col(k) = X1 * coefficients.col(k);
        residuals.col(k)     = y - fitted_values.col(k);
      }

      update_centers(centers, Um, X);

      if (wnorm == "epg")
        update_weights_epg(weights, Um, centers, X);
      else if (wnorm == "epl")
        update_weights_epl(weights, Um, centers, X);
      else if (wnorm == "qpg")
        update_weights_qpg(weights, Um, centers, X);
      else if (wnorm == "qpl")
        update_weights_qpl(weights, Um, centers, X);

      if (balance == "pg")
        update_balance_pg(alphas, gammas, Um, Ey, Ex);
      else if (balance == "pl")
        update_balance_pl(alphas, gammas, Um, Ey, Ex);

      // check convergence
      if (converged || iterations == itermax)
        break;

      iterations++;

      // update kpartition
      if (m == 1.0)
        converged = update_membership(U, E);
      else
        converged = update_membership(U, m, E);
    }

    // build Rcpp model
    auto rcpp_lloyd = Rcpp::List::create(
      Rcpp::Named("iter.max")   = itermax,
      Rcpp::Named("iterations") = iterations,
      Rcpp::Named("converged")  = converged
    );
    rcpp_lloyd.attr("class") = "WCLR.lloyd";

    double loss = J(arma::pow(U, m), E);
    auto rcpp_model = Rcpp::List::create(
      Rcpp::Named("K")             = K,
      Rcpp::Named("m")             = m,
      Rcpp::Named("wnorm")         = wnorm,
      Rcpp::Named("loss")          = loss,
      Rcpp::Named("coefficients")  = coefficients,
      Rcpp::Named("fitted.values") = fitted_values,
      Rcpp::Named("residuals")     = residuals,
      Rcpp::Named("centers")       = centers,
      Rcpp::Named("weights")       = weights,
      Rcpp::Named("alphas")        = alphas,
      Rcpp::Named("gammas")        = gammas,
      Rcpp::Named("membership")    = U,
      Rcpp::Named("algorithm")     = rcpp_lloyd
    );
    rcpp_model.attr("class") = "WCLR.swclr";

    return rcpp_model;
  }

template <typename F>
arma::vec
  predict(const arma::mat &X,
          const arma::mat &R,
          const double m,
          const F &dist)
  {
    const int N = X.n_rows;
    const int P = X.n_cols;
    const int K = R.n_cols;

    arma::mat U(N, K, arma::fill::zeros);

    if (m == 1.0)
      update_membership(U, dist);
    else
      update_membership(U, m, dist);

    arma::vec predicted_values(N, arma::fill::zeros);

    for (int n = 0; n < N; n++)
    {
      predicted_values(n) = 0.0;

      for (int k = 0; k < K; k++)
      {
        predicted_values(n) += U(n, k) *
          (R.col(k)(0) + dot(X.row(n), R.col(k).tail(P)));
      }
    }

    return predicted_values;
  }

//' Predict
//'
//' Predict values by nearest Euclidean distance.
//'
//' @param X newdata matrix
//' @param C centers matrix
//' @param R coefficients matrix
//' @param m fuzzier exponent
//'
//' @return a vector of predictions.
//'
//' @export
// [[Rcpp::export]]
arma::vec
  predict_euclidean_cpp(const arma::mat &X,
                        const arma::mat &C,
                        const arma::mat &R,
                        const double m)
  {
    auto dist = [&](int n, int k)
    {
      // squared Euclidean distance
      return dot(X.row(n).t() - C.col(k),
                 X.row(n).t() - C.col(k));
    };

    return predict(X, R, m, dist);
  }

//' Predict
//'
//' Predict values by nearest quadratic distance.
//'
//' @param X newdata matrix
//' @param C centers matrix
//' @param W weights cube
//' @param R coefficients matrix
//' @param m fuzzier exponent
//'
//' @return a vector of predictions.
//'
//' @export
// [[Rcpp::export]]
arma::vec
  predict_quadratic_cpp(const arma::mat  &X,
                        const arma::mat  &C,
                        const arma::cube &W,
                        const arma::mat  &R,
                        const double m)
  {
    auto dist = [&](int n, int k)
    {
      // quadratic dissimilarity
      arma::colvec xc = X.row(n).t() - C.col(k);
      return arma::as_scalar(xc.t() * W.slice(k) * xc);
    };

    return predict(X, R, m, dist);
  }
