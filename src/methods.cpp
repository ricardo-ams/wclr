#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

void update_coefficients(arma::mat &R,
                         const arma::mat &U,
                         const arma::mat &X,
                         const arma::colvec &y,
                         Rcpp::List &log)
{
  const int K = U.n_cols;

  for (int k = 0; k < K; k++)
  {
    try
    {
      arma::mat W = arma::diagmat(U.col(k));
      R.col(k) = arma::solve(X.t() * W * X, X.t() * W * y);
    }
    catch (const std::exception &ex)
    {
      Rcpp::List error = Rcpp::List::create(
        Rcpp::Named("error") = ex.what());
      error.attr("class") = "wclr.trace.error";
      log.push_back(error);
    }
    catch (...)
    {
      Rcpp::List error = Rcpp::List::create(
        Rcpp::Named("error") = "unknown exception");
      error.attr("class") = "wclr.trace.error";
      log.push_back(error);
    }
  }
}

void update_centers(arma::mat &C,
                    const arma::mat &U,
                    const arma::mat &X)
{
  const int N = U.n_rows;
  const int K = U.n_cols;

  C.zeros();
  for (int k = 0; k < K; k++)
  {
    // Compute the weighted arithmetic mean M of a data set using the
    // recurrence relation
    //   M(n) = M(n-1) + (data[n] - M(n-1)) (w(n)/(W(n-1) + w(n)))
    //   W(n) = W(n-1) + w(n)
    // Reference:
    //   GNU GSL 2.6 source code - https://www.gnu.org/software/gsl/doc/html/statistics.html?highlight=weighted%20mean#c.gsl_stats_wmean

    double W = 0.0;

    for (int n = 0; n < N; n++)
    {
      const double unk = U(n, k);

      if (unk > 0.0)
      {
        W += unk;
        C.col(k) += (X.row(n).t() - C.col(k)) * (unk / W);
      }
    }
  }
}

template <typename F>
bool update_membership(arma::mat &U,
                       const double m,
                       const F &diss)
{
  const int N = U.n_rows;
  const int K = U.n_cols;
  const double exp = 1.0 / (m - 1.0);

  bool converged = true;

  for (int n = 0; n < N; n++)
  {
    for (int k = 0; k < K; k++)
    {
      double unk = 0.0;

      for (int h = 0; h < K; h++)
        unk += std::pow(diss(n, k) / diss(n, h), exp);

      unk = 1.0 / unk;

      if (std::fabs(U(n, k) - unk) > 0.01)
        converged = false;

      U(n, k) = unk;
    }
  }

  return converged;
}

template <typename F>
bool update_membership(arma::mat &U,
                       const F &diss)
{
  const int N = U.n_rows;
  const int K = U.n_cols;
  const double EPS = std::numeric_limits<double>::epsilon();

  bool converged = true;

  for (int n = 0; n < N; n++)
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

    if (U(n, min_k) !=  1.0 - ((K - 1) * EPS))
    {
      converged = false;

      for (int k = 0; k < K; k++)
        // a negligible but non-zero membership value
        U(n, k) = EPS;

      U(n, min_k) = 1.0 - ((K - 1) * EPS);
    }
  }

  return converged;
}

template <typename F>
double J(const arma::mat &U,
         const F &diss)
{
  const int N = U.n_rows;
  const int K = U.n_cols;

  double loss = 0.0;

  for (int k = 0; k < K; k++)
    for (int n = 0; n < N; n++)
      loss += U(n, k) * diss(n, k);

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
//' @param trace logical value. If \code{TRUE}, produce a trace information of
//'   the progress of the algorithm.
//'
//' @return returns an object of class \code{wclr.clr}.
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
                const int itermax,
                const bool trace)
  {
    const int N = X.n_rows; // number of observations
    const int P = X.n_cols; // number of variables
    const int K = U.n_cols; // number of clusters

    Rcpp::List log = Rcpp::List::create();

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

      update_coefficients(coefficients, Um, X1, y, log);

      for (int k = 0; k < K; k++)
      {
        fitted_values.col(k) = X1 * coefficients.col(k);
        residuals.col(k)     = y - fitted_values.col(k);
      }

      if (trace)
      {
        Rcpp::List info = Rcpp::List::create(
            Rcpp::Named("iter") = iterations,
            Rcpp::Named("step") = (iterations == 0)? "initialization" : "update coefficients",
            Rcpp::Named("loss") = J(Um, E));
        info.attr("class") = "wclr.trace.info";
        log.push_back(info);
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

      if (trace)
      {
        Rcpp::List info = Rcpp::List::create(
          Rcpp::Named("iter") = iterations,
          Rcpp::Named("step") = "update kpartition",
          Rcpp::Named("loss") = J(arma::pow(U, m), E));
        info.attr("class") = "wclr.trace.info";
        log.push_back(info);
      }
    }

    // compute centroids
    arma::mat centers(P, K, arma::fill::randu);
    update_centers(centers, arma::pow(U, m), X);

    // build Rcpp model
    log.attr("class") = "wclr.trace";

    auto rcpp_lloyd = Rcpp::List::create(
      Rcpp::Named("iter.max")   = itermax,
      Rcpp::Named("iterations") = iterations,
      Rcpp::Named("converged")  = converged
    );
    rcpp_lloyd.attr("class") = "wclr.lloyd";

    auto rcpp_model = Rcpp::List::create(
      Rcpp::Named("K")             = K,
      Rcpp::Named("m")             = m,
      Rcpp::Named("loss")          = J(arma::pow(U, m), E),
      Rcpp::Named("coefficients")  = coefficients,
      Rcpp::Named("fitted.values") = fitted_values,
      Rcpp::Named("residuals")     = residuals,
      Rcpp::Named("centers")       = centers,
      Rcpp::Named("membership")    = U,
      Rcpp::Named("algorithm")     = rcpp_lloyd,
      Rcpp::Named("trace")         = log
    );
    rcpp_model.attr("class") = "wclr.clr";

    return rcpp_model;
  }

//' K-means Linear Regression
//'
//' This is a bare-bones function and must NOT be used by an end-user.
//'
//' @inheritParams clr_lloyd_cpp
//'
//' @return returns an object of class \code{wclr.kmeans}.
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
                   const int itermax,
                   const bool trace)
  {
    const int N = X.n_rows; // number of observations
    const int P = X.n_cols; // number of variables
    const int K = U.n_cols; // number of clusters

    Rcpp::List log = Rcpp::List::create();

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

      if (trace)
      {
        Rcpp::List info = Rcpp::List::create(
          Rcpp::Named("iter") = iterations,
          Rcpp::Named("step") = (iterations == 0)? "initialization" : "update centers",
          Rcpp::Named("loss") = J(Um, E));
        info.attr("class") = "wclr.trace.info";
        log.push_back(info);
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

      if (trace)
      {
        Rcpp::List info = Rcpp::List::create(
          Rcpp::Named("iter") = iterations,
          Rcpp::Named("step") = "update kpartition",
          Rcpp::Named("loss") = J(arma::pow(U, m), E));
        info.attr("class") = "wclr.trace.info";
        log.push_back(info);
      }
    }

    // compute coefficients
    arma::mat coefficients(P + 1, K, arma::fill::randu);
    arma::mat fitted_values(N, K, arma::fill::zeros);
    arma::mat residuals(N, K, arma::fill::zeros);

    update_coefficients(coefficients, arma::pow(U, m), X1, y, log);

    for (int k = 0; k < K; k++)
    {
      fitted_values.col(k) = X1 * coefficients.col(k);
      residuals.col(k)     = y - fitted_values.col(k);
    }

    // build Rcpp model
    log.attr("class") = "wclr.trace";

    auto rcpp_lloyd = Rcpp::List::create(
      Rcpp::Named("iter.max")   = itermax,
      Rcpp::Named("iterations") = iterations,
      Rcpp::Named("converged")  = converged
    );
    rcpp_lloyd.attr("class") = "wclr.lloyd";

    auto rcpp_model = Rcpp::List::create(
      Rcpp::Named("K")             = K,
      Rcpp::Named("m")             = m,
      Rcpp::Named("loss")          = J(arma::pow(U, m), E),
      Rcpp::Named("coefficients")  = coefficients,
      Rcpp::Named("fitted.values") = fitted_values,
      Rcpp::Named("residuals")     = residuals,
      Rcpp::Named("centers")       = centers,
      Rcpp::Named("membership")    = U,
      Rcpp::Named("algorithm")     = rcpp_lloyd,
      Rcpp::Named("trace")         = log
    );
    rcpp_model.attr("class") = "wclr.kmeans";

    return rcpp_model;
  }

//' K-plane Regression
//'
//' This is a bare-bones function and must NOT be used by an end-user.
//'
//' @inheritParams clr_lloyd_cpp
//' @param gamma numeric balancing value.
//'
//' @return returns an object of class \code{wclr.kplane}.
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
                   const int itermax,
                   const bool trace)
  {
    const int N = X.n_rows; // number of observations
    const int P = X.n_cols; // number of variables
    const int K = U.n_cols; // number of clusters

    Rcpp::List log = Rcpp::List::create();

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

      update_coefficients(coefficients, Um, X1, y, log);

      for (int k = 0; k < K; k++)
      {
        fitted_values.col(k) = X1 * coefficients.col(k);
        residuals.col(k)     = y - fitted_values.col(k);
      }

      if (trace)
      {
        Rcpp::List info = Rcpp::List::create(
          Rcpp::Named("iter") = iterations,
          Rcpp::Named("step") = (iterations == 0)? "initialization" : "update coefficients",
          Rcpp::Named("loss") = J(Um, E));
        info.attr("class") = "wclr.trace.info";
        log.push_back(info);
      }

      update_centers(centers, Um, X);

      if (trace)
      {
        Rcpp::List info = Rcpp::List::create(
          Rcpp::Named("iter") = iterations,
          Rcpp::Named("step") = (iterations == 0)? "initialization" : "update centers",
          Rcpp::Named("loss") = J(Um, E));
        info.attr("class") = "wclr.trace.info";
        log.push_back(info);
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

      if (trace)
      {
        Rcpp::List info = Rcpp::List::create(
          Rcpp::Named("iter") = iterations,
          Rcpp::Named("step") = "update kpartition",
          Rcpp::Named("loss") = J(arma::pow(U, m), E));
        info.attr("class") = "wclr.trace.info";
        log.push_back(info);
      }
    }

    // build Rcpp model
    log.attr("class") = "wclr.trace";

    auto rcpp_lloyd = Rcpp::List::create(
      Rcpp::Named("iter.max")   = itermax,
      Rcpp::Named("iterations") = iterations,
      Rcpp::Named("converged")  = converged
    );
    rcpp_lloyd.attr("class") = "wclr.lloyd";

    auto rcpp_model = Rcpp::List::create(
      Rcpp::Named("K")             = K,
      Rcpp::Named("gamma")         = gamma,
      Rcpp::Named("m")             = m,
      Rcpp::Named("loss")          = J(arma::pow(U, m), E),
      Rcpp::Named("coefficients")  = coefficients,
      Rcpp::Named("fitted.values") = fitted_values,
      Rcpp::Named("residuals")     = residuals,
      Rcpp::Named("centers")       = centers,
      Rcpp::Named("membership")    = U,
      Rcpp::Named("algorithm")     = rcpp_lloyd,
      Rcpp::Named("trace")         = log
    );
    rcpp_model.attr("class") = "wclr.kplane";

    return rcpp_model;
  }


void update_weights_qpl(arma::cube &W,
                        const arma::mat &U,
                        const arma::mat &C,
                        const arma::mat &X)
{
  const int N = U.n_rows;
  const int K = U.n_cols;
  const int P = C.n_rows;

  arma::mat Q(P, P, arma::fill::zeros);

  for (int k = 0; k < K; k++)
  {
    Q.zeros();

    for (int n = 0; n < N; n++)
    {
      const double unk = U(n, k);

      arma::colvec XC = X.row(n).t() - C.col(k);
      Q += unk * (XC * XC.t());
    }

    W.slice(k) = std::pow(arma::det(Q), (1.0 / P)) * arma::inv(Q);
  }
}

void update_weights_qpg(arma::cube &W,
                        const arma::mat &U,
                        const arma::mat &C,
                        const arma::mat &X)
{
  const int N = U.n_rows;
  const int K = U.n_cols;
  const int P = C.n_rows;

  arma::mat Q(P, P, arma::fill::zeros);

  for (int k = 0; k < K; k++)
  {
    for (int n = 0; n < N; n++)
    {
      const double unk = U(n, k);

      arma::colvec XC = X.row(n).t() - C.col(k);
      Q += unk * (XC * XC.t());
    }
  }

  const arma::mat w = std::pow(arma::det(Q), (1.0 / P)) * arma::inv(Q);

  for (int k = 0; k < K; k++)
    W.slice(k) = w;
}

void update_weights_epl(arma::cube &W,
                        const arma::mat &U,
                        const arma::mat &C,
                        const arma::mat &X)
{
  const int N = U.n_rows;
  const int K = U.n_cols;
  const int P = C.n_rows;

  arma::colvec aux(P, arma::fill::zeros);

  for (int k = 0; k < K; k++)
  {
    aux.zeros();

    for (int n = 0; n < N; n++)
    {
      const double unk = U(n, k);

      arma::colvec t = arma::pow(X.row(n).t() - C.col(k), 2.0);
      aux += unk * t;
    }

    const arma::colvec w = arma::as_scalar(arma::prod(arma::pow(aux, 1.0 / P))) / aux;

    W.slice(k) = arma::diagmat(w);
  }
}

void update_weights_epg(arma::cube &W,
                        const arma::mat &U,
                        const arma::mat &C,
                        const arma::mat &X)
{
  const int N = U.n_rows;
  const int K = U.n_cols;
  const int P = C.n_rows;

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

  for (int k = 0; k < K; k++)
    W.slice(k) = arma::diagmat(w);
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
//' @return returns an object of class \code{wclr.wclr}.
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
                 const int itermax,
                 const bool trace)
  {
    const int N = X.n_rows; // number of observations
    const int P = X.n_cols; // number of variables
    const int K = U.n_cols; // number of clusters

    Rcpp::List log = Rcpp::List::create();

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

      update_coefficients(coefficients, Um, X1, y, log);

      for (int k = 0; k < K; k++)
      {
        fitted_values.col(k) = X1 * coefficients.col(k);
        residuals.col(k)     = y - fitted_values.col(k);
      }

      if (trace)
      {
        Rcpp::List info = Rcpp::List::create(
          Rcpp::Named("iter") = iterations,
          Rcpp::Named("step") = (iterations == 0)? "initialization" : "update coefficients",
          Rcpp::Named("loss") = J(Um, E));
        info.attr("class") = "wclr.trace.info";
        log.push_back(info);
      }

      update_centers(centers, Um, X);

      if (trace)
      {
        Rcpp::List info = Rcpp::List::create(
          Rcpp::Named("iter") = iterations,
          Rcpp::Named("step") = (iterations == 0)? "initialization" : "update centers",
          Rcpp::Named("loss") = J(Um, E));
        info.attr("class") = "wclr.trace.info";
        log.push_back(info);
      }

      if (wnorm == "epg")
        update_weights_epg(weights, Um, centers, X);
      else if (wnorm == "epl")
        update_weights_epl(weights, Um, centers, X);
      else if (wnorm == "qpg")
        update_weights_qpg(weights, Um, centers, X);
      else if (wnorm == "qpl")
        update_weights_qpl(weights, Um, centers, X);

      if (trace)
      {
        Rcpp::List info = Rcpp::List::create(
          Rcpp::Named("iter") = iterations,
          Rcpp::Named("step") = (iterations == 0)? "initialization" : "update weights",
          Rcpp::Named("loss") = J(Um, E));
        info.attr("class") = "wclr.trace.info";
        log.push_back(info);
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

      if (trace)
      {
        Rcpp::List info = Rcpp::List::create(
          Rcpp::Named("iter") = iterations,
          Rcpp::Named("step") = "update kpartition",
          Rcpp::Named("loss") = J(arma::pow(U, m), E));
        info.attr("class") = "wclr.trace.info";
        log.push_back(info);
      }
    }

    // build Rcpp model
    log.attr("class") = "wclr.trace";

    auto rcpp_lloyd = Rcpp::List::create(
      Rcpp::Named("iter.max")   = itermax,
      Rcpp::Named("iterations") = iterations,
      Rcpp::Named("converged")  = converged
    );
    rcpp_lloyd.attr("class") = "wclr.lloyd";

    auto rcpp_model = Rcpp::List::create(
      Rcpp::Named("K")             = K,
      Rcpp::Named("alpha")         = alpha,
      Rcpp::Named("m")             = m,
      Rcpp::Named("wnorm")         = wnorm,
      Rcpp::Named("loss")          = J(arma::pow(U, m), E),
      Rcpp::Named("coefficients")  = coefficients,
      Rcpp::Named("fitted.values") = fitted_values,
      Rcpp::Named("residuals")     = residuals,
      Rcpp::Named("centers")       = centers,
      Rcpp::Named("weights")       = weights,
      Rcpp::Named("membership")    = U,
      Rcpp::Named("algorithm")     = rcpp_lloyd,
      Rcpp::Named("trace")         = log
    );
    rcpp_model.attr("class") = "wclr.wclr";

    return rcpp_model;
  }

template <typename F1, typename F2>
void update_balance_pg(arma::vec &alpha,
                       arma::vec &gamma,
                       const arma::mat &U,
                       const F1 &Ey,
                       const F2 &Ex)
{
  const int N = U.n_rows;
  const int K = U.n_cols;

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

  alpha.fill(std::sqrt(dy / dx));
  gamma.fill(std::sqrt(dx / dy));
}

template <typename F1, typename F2>
void update_balance_pl(arma::vec &alpha,
                       arma::vec &gamma,
                       const arma::mat &U,
                       const F1 &Ey,
                       const F2 &Ex)
{
  const int N = U.n_rows;
  const int K = U.n_cols;

  for (int k = 0; k < K; k++)
  {
    double dx = 0.0;
    double dy = 0.0;

    for (int n = 0; n < N; n++)
    {
      const double unk = U(n, k);

      dy += unk * Ey(n, k);
      dx += unk * Ex(n, k);
    }

    alpha(k) = std::sqrt(dy / dx);
    gamma(k) = std::sqrt(dx / dy);
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
//' @return returns an object of class \code{wclr.swclr}.
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
                  const int itermax,
                  const bool trace)
  {
    const int N = X.n_rows; // number of observations
    const int P = X.n_cols; // number of variables
    const int K = U.n_cols; // number of clusters

    Rcpp::List log = Rcpp::List::create();

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

      update_coefficients(coefficients, Um, X1, y, log);

      for (int k = 0; k < K; k++)
      {
        fitted_values.col(k) = X1 * coefficients.col(k);
        residuals.col(k)     = y - fitted_values.col(k);
      }

      if (trace)
      {
        Rcpp::List info = Rcpp::List::create(
          Rcpp::Named("iter") = iterations,
          Rcpp::Named("step") = (iterations == 0)? "initialization" : "update coefficients",
          Rcpp::Named("loss") = J(Um, E));
        info.attr("class") = "wclr.trace.info";
        log.push_back(info);
      }

      update_centers(centers, Um, X);

      if (trace)
      {
        Rcpp::List info = Rcpp::List::create(
          Rcpp::Named("iter") = iterations,
          Rcpp::Named("step") = (iterations == 0)? "initialization" : "update centers",
          Rcpp::Named("loss") = J(Um, E));
        info.attr("class") = "wclr.trace.info";
        log.push_back(info);
      }

      if (wnorm == "epg")
        update_weights_epg(weights, Um, centers, X);
      else if (wnorm == "epl")
        update_weights_epl(weights, Um, centers, X);
      else if (wnorm == "qpg")
        update_weights_qpg(weights, Um, centers, X);
      else if (wnorm == "qpl")
        update_weights_qpl(weights, Um, centers, X);

      if (trace)
      {
        Rcpp::List info = Rcpp::List::create(
          Rcpp::Named("iter") = iterations,
          Rcpp::Named("step") = (iterations == 0)? "initialization" : "update weights",
          Rcpp::Named("loss") = J(Um, E));
        info.attr("class") = "wclr.trace.info";
        log.push_back(info);
      }

      if (balance == "pg")
        update_balance_pg(alphas, gammas, Um, Ey, Ex);
      else if (balance == "pl")
        update_balance_pl(alphas, gammas, Um, Ey, Ex);

      if (trace)
      {
        Rcpp::List info = Rcpp::List::create(
          Rcpp::Named("iter") = iterations,
          Rcpp::Named("step") = (iterations == 0)? "initialization" : "update balance",
          Rcpp::Named("loss") = J(Um, E));
        info.attr("class") = "wclr.trace.info";
        log.push_back(info);
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

      if (trace)
      {
        Rcpp::List info = Rcpp::List::create(
          Rcpp::Named("iter") = iterations,
          Rcpp::Named("step") = "update kpartition",
          Rcpp::Named("loss") = J(arma::pow(U, m), E));
        info.attr("class") = "wclr.trace.info";
        log.push_back(info);
      }
    }

    // build Rcpp model
    log.attr("class") = "wclr.trace";

    auto rcpp_lloyd = Rcpp::List::create(
      Rcpp::Named("iter.max")   = itermax,
      Rcpp::Named("iterations") = iterations,
      Rcpp::Named("converged")  = converged
    );
    rcpp_lloyd.attr("class") = "wclr.lloyd";

    auto rcpp_model = Rcpp::List::create(
      Rcpp::Named("K")             = K,
      Rcpp::Named("m")             = m,
      Rcpp::Named("wnorm")         = wnorm,
      Rcpp::Named("loss")          = J(arma::pow(U, m), E),
      Rcpp::Named("coefficients")  = coefficients,
      Rcpp::Named("fitted.values") = fitted_values,
      Rcpp::Named("residuals")     = residuals,
      Rcpp::Named("centers")       = centers,
      Rcpp::Named("weights")       = weights,
      Rcpp::Named("alphas")        = alphas,
      Rcpp::Named("gammas")        = gammas,
      Rcpp::Named("membership")    = U,
      Rcpp::Named("algorithm")     = rcpp_lloyd,
      Rcpp::Named("trace")         = log
    );
    rcpp_model.attr("class") = "wclr.swclr";

    return rcpp_model;
  }
