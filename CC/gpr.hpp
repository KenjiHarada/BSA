#ifndef GPR_HPP
#define GPR_HPP
/* gpr.hpp
 *
 * Copyright 2013, Kenji Harada
 * Released under the GPLv3.
 *
 * To spread the Bayesian scaling analysis method,
 * I hope you will cite the following original paper:
 *   Kenji Harada, Physical Review E 84 (2011) 056704.
 *
 */

/**
   @file gpr.hpp
   @brief Classes for Gaussian process regression
 */
#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_rng.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>

/** @namespace GPR
    @brief Gaussian process regression
*/
namespace GPR {
  template<class GPR_Data, class GPR_Kernel> class Regression;
  const static std::string version = "0.51";
};

/**
  @class GPR::Regression
  @brief Gaussian process regression.

  Gaussian process regression for a data set.  Data are the set of triples @f$ (X_i, Y_i, E_i) @f$.
  They are written as @f$ (\vec{X}, \vec{Y}, \vec{E}) @f$.
  The stochastic model of data is a Gaussian Process as @f$ \langle
  \vec{Y} \rangle = 0 @f$ and @f$ \langle \vec{Y} \vec{Y}^t\rangle =
  \Sigma(\vec{X}, \vec{E}) @f$, where @f$ (\Sigma)_{ij} \equiv K(i, j,
  \vec{X}, \vec{E}) @f$ and @f$ K @f$ is a kernel function.
  @tparam GPR_Data Class of data
  @tparam GPR_Kernel Class of kernel function
 */
template<class GPR_Data, class GPR_Kernel>
class GPR::Regression {
public:
  /**
     @brief Calculate a log-likelihood.

     Calculate a log-likelihood of data with physical and hyper parameters.
     @param[in] xdata Data set
     @param[in] p_params Physical parameters
     @param[in] h_params Hyper parameters
     @return Value of log-likelihood
   */
  double calc_ll(const GPR_Data& xdata, const std::vector<double>& p_params, const std::vector<double>& h_params) const;

  /**
     @brief Search a maximum of log-likelihood.

     Search a parameter of maximum of log-likelihood by an iterative method
     @param[in] xdata Data set
     @param[in, out] p_params Physical parameters
     @param[in] p_mask Mask for physical parameters. The value 0(1) means fixed(unfixed), respectively.
     @param[in, out] h_params Hyper parameters
     @param[in] h_mask Mask for hyper parameters
     @param[in] stepsize The step size of change of parameters
     @param[in] niter The limit of iterations
     @param[in] tol Tolerance
     @param[in] epsabs The absolute accuracy
     @return The final status of iterative method
   */
  int search_mll(const GPR_Data& xdata, std::vector<double>& p_params, const std::vector<int>& p_mask,
                 std::vector<double>& h_params, const std::vector<int>& h_mask,
                 double stepsize = 5e-4, int niter = 1e5, double tol = 1e-3, double epsabs = 1e-7);

  /**
     @brief Monte Carlo estimation of parameters.

     Do a Monte Carlo estimation of parameters.
     @param[in] xdata Data set
     @param[in, out] p_params Physical parameters. Return the average of a parameter.
     @param[in] p_mask Mask for physical parameters. The value 0(1) means fixed(unfixed), respectively.
     @param[in, out] ep_params Step size of change of a physical parameters in a Monte Carlo iteration. Return the error of a parameter.
     @param[in, out] h_params Hyper parameters. Return the average of a parameter.
     @param[in] h_mask Mask for hyper parameters
     @param[in, out] eh_params Step size of change of a hyper parameters in a Monte Carlo iteration. Return the error of a parameter.
     @param[in] nthermal Number of steps for thermalization
     @param[in] nmcs Number of Monte Carlo Sweeps
   */
  void monte_carlo(const GPR_Data& xdata, std::vector<double>& p_params, const std::vector<int>& p_mask,
                   std::vector<double>& ep_params, std::vector<double>& h_params, const std::vector<int>& h_mask,
                   std::vector<double>& eh_params, int nthermal = 1000, int nmcs = 10000);

  void monte_carlo(const GPR_Data& xdata, const std::vector<double>& p_params, const std::vector<int>& p_mask,
                   const std::vector<double>& ep_params, const std::vector<double>& h_params, const std::vector<int>& h_mask, const std::vector<double>& eh_params,
                   std::vector<double>& average,
                   std::vector<double>& covariance,
                   int nthermal = 1000, int nmcs = 10000);

  /**
     @brief Calculate a mean and variance of Gaussian distribution of @f$ Y(X) @f$

     @param[in] X Value of X
     @param[out] mean Mean @f$ \mu(X) \f$
     @param[out] variance Variance @f$ \sigma^2(X) \f$
     @param[in] xdata Data set
     @param[in] p_params Physical parameters
     @param[in] h_params Hyper parameters
   */
  void conditional_probability_y(double X, double& mean, double& variance, const GPR_Data& xdata, const std::vector<double>& p_params, const std::vector<double>& h_params) const;


  void monte_carlo(const std::vector<GPR_Data> xdatas,
                   const std::vector< std::vector<int> >& index_p_params,
                   const std::vector< std::vector<int> >& index_h_params,
                   const std::vector<double>& params,
                   const std::vector<double>& delta_params,
                   std::vector<double>& average,
                   std::vector<double>& covariance,
                   int nthermal = 1000, int nmcs = 10000);

  int search_mll(const std::vector<GPR_Data>& xdatas,
                 const std::vector<std::vector<int> >& xindex_p_params,
                 const std::vector<std::vector<int> >& xindex_h_params,
                 std::vector<double>& params, const std::vector<int>& mask,
                 double stepsize = 5e-4, int niter = 1e5, double tol = 1e-3, double epsabs = 1e-7);

private:
  static double gsl_f(const gsl_vector* z, void* p){
    return ((GPR::Regression<GPR_Data, GPR_Kernel>*)p)->gsl_v_fdf(z);
  }
  static void gsl_df(const gsl_vector* z, void* p, gsl_vector* value_g){
    ((GPR::Regression<GPR_Data, GPR_Kernel>*)p)->gsl_v_fdf(z, value_g);
  }
  static void gsl_fdf(const gsl_vector* z,  void* p, double* value_f, gsl_vector* value_g){
    *value_f = ((GPR::Regression<GPR_Data, GPR_Kernel>*)p)->gsl_v_fdf(z, value_g);
  }
  double gsl_v_fdf(const gsl_vector* z,  gsl_vector* value_g = 0);

  static double gsl_f_m(const gsl_vector* z_m, void* p){
    return ((GPR::Regression<GPR_Data, GPR_Kernel>*)p)->gsl_v_fdf_m(z_m);
  }
  static void gsl_df_m(const gsl_vector* z_m, void* p, gsl_vector* value_g_m){
    ((GPR::Regression<GPR_Data, GPR_Kernel>*)p)->gsl_v_fdf_m(z_m, value_g_m);
  }
  static void gsl_fdf_m(const gsl_vector* z_m,  void* p, double* value_f_m, gsl_vector* value_g_m){
    *value_f_m = ((GPR::Regression<GPR_Data, GPR_Kernel>*)p)->gsl_v_fdf_m(z_m, value_g_m);
  }
  double gsl_v_fdf_m(const gsl_vector* z_m,  gsl_vector* value_g_m = 0);

  const GPR_Data *data;
  std::vector<int> mask_z;
  const std::vector< GPR_Data > *datas;
  const std::vector< std::vector<int> > *pindex_p_params;
  const std::vector< std::vector<int> > *pindex_h_params;
  std::vector<int> mask_z_m;
};

template<class GPR_Data, class GPR_Kernel>
double GPR::Regression<GPR_Data, GPR_Kernel>::calc_ll(const GPR_Data& xdata, const std::vector<double>& p_params, const std::vector<double>& h_params) const {
  // calculate covariance matrix and data vactor
  int ndata = xdata.npoints();
  std::vector<double> vx(ndata), vy(ndata), ve(ndata);
  xdata.x(vx, p_params);
  xdata.y(vy, p_params);
  xdata.e(ve, p_params);
  gsl_matrix* C = gsl_matrix_alloc(ndata, ndata);
  gsl_vector* t = gsl_vector_alloc(ndata);
  for (int i = 0; i < ndata; ++i) {
    gsl_vector_set(t, i, vy[i]);
    for (int j = 0; j < ndata; ++j)
      gsl_matrix_set(C, i, j, GPR_Kernel::k(i, j, vx, ve, h_params));
  }
  // calculate maximum log-likelihood (MLL)
  gsl_linalg_cholesky_decomp(C);
  gsl_vector* alpha = gsl_vector_alloc(ndata);
  gsl_linalg_cholesky_solve(C, t, alpha); // alpha = C^-1 t
  double mll;
  gsl_blas_ddot(t, alpha, &mll);
  mll *= -0.5;
  for (int i = 0; i < ndata; ++i)
    mll -= log(gsl_matrix_get(C, i, i));
  mll -= (0.5 * ndata * log(2e0 * M_PI));
  // free
  gsl_vector_free(alpha);
  gsl_vector_free(t);
  gsl_matrix_free(C);
  return mll;
}

template<class GPR_Data, class GPR_Kernel>
int GPR::Regression<GPR_Data, GPR_Kernel>::search_mll(const GPR_Data& xdata,
                                                      std::vector<double>& p_params, const std::vector<int>& p_mask,
                                                      std::vector<double>& h_params, const std::vector<int>& h_mask,
                                                      double stepsize, int niter, double tol, double epsabs){
  // initialize
  int nparams = p_params.size() + h_params.size();
  gsl_vector* z = gsl_vector_alloc(nparams);
  mask_z.resize(nparams);
  for (int i = 0; i < p_params.size(); ++i) {
    gsl_vector_set(z, i, p_params[i]);
    mask_z[i] = p_mask[i];
  }
  for (int i = 0; i < h_params.size(); ++i) {
    gsl_vector_set(z, i + p_params.size(), h_params[i]);
    mask_z[i + p_params.size()] = h_mask[i];
  }
  data = &xdata;
  //
  size_t iter = 0;
  const gsl_multimin_fdfminimizer_type* T;
  gsl_multimin_fdfminimizer* s;
  gsl_multimin_function_fdf my_func;
  my_func.n = nparams;
  my_func.f = &gsl_f;
  my_func.df = &gsl_df;
  my_func.fdf = &gsl_fdf;
  my_func.params = this;
  T = gsl_multimin_fdfminimizer_conjugate_fr;
  s = gsl_multimin_fdfminimizer_alloc(T, z->size);
  gsl_multimin_fdfminimizer_set(s, &my_func, z, stepsize, tol);
  int status;
  do {
    iter++;
    status = gsl_multimin_fdfminimizer_iterate(s);
#ifndef GPR_NLOG
    std::cerr << "# ITER=" << iter << ", LL=" <<  -(s->f);
    for (int i = 0; i < p_params.size(); ++i)
      std::cerr << ", p[" << i << "]=" << gsl_vector_get(s->x, i);
    for (int i = 0; i < h_params.size(); ++i)
      std::cerr << ", h[" << i << "]=" << gsl_vector_get(s->x, i + p_params.size());
    std::cerr << std::endl;
#endif
    if (status)
      break;
    status = gsl_multimin_test_gradient(s->gradient, epsabs);
  } while (status == GSL_CONTINUE && iter < niter);
  for (int i = 0; i < p_params.size(); ++i)
    p_params[i] = gsl_vector_get(s->x, i);
  for (int i = 0; i < h_params.size(); ++i)
    h_params[i] = gsl_vector_get(s->x, i + p_params.size());
  gsl_multimin_fdfminimizer_free(s);
  gsl_vector_free(z);
  return status;
}

template<class GPR_Data, class GPR_Kernel>
double GPR::Regression<GPR_Data, GPR_Kernel>::gsl_v_fdf(const gsl_vector* z,  gsl_vector* value_g){
  // initialize parameters
  std::vector<double> p_params(data->nparams());
  for (int i = 0; i < p_params.size(); ++i)
    p_params[i] = gsl_vector_get(z, i);
  std::vector<double> h_params(GPR_Kernel::nparams());
  for (int i = 0; i < h_params.size(); ++i)
    h_params[i] = gsl_vector_get(z, i + data->nparams());
  // calculate covariance matrix and data vactor
  int ndata = data->npoints();
  std::vector<double> vx(ndata), vy(ndata), ve(ndata);
  data->x(vx, p_params);
  data->y(vy, p_params);
  data->e(ve, p_params);
  gsl_matrix* C = gsl_matrix_alloc(ndata, ndata);
  gsl_vector* t = gsl_vector_alloc(ndata);
  for (int i = 0; i < ndata; ++i) {
    gsl_vector_set(t, i, vy[i]);
    for (int j = 0; j < ndata; ++j)
      gsl_matrix_set(C, i, j, GPR_Kernel::k(i, j, vx, ve, h_params));
  }
  // calculate maximum log-likelihood (MLL)
  gsl_linalg_cholesky_decomp(C);
  gsl_vector* alpha = gsl_vector_alloc(ndata);
  gsl_linalg_cholesky_solve(C, t, alpha); // alpha = C^-1 t
  double mll;
  gsl_blas_ddot(t, alpha, &mll);
  mll *= -0.5;
  for (int i = 0; i < ndata; ++i)
    mll -= log(gsl_matrix_get(C, i, i));
  mll -= (0.5 * ndata * log(2e0 * M_PI));
  // calculate gradient of MLL (GMLL)
  if (value_g != 0) {
    gsl_matrix* DC = gsl_matrix_alloc(ndata, ndata);
    gsl_vector* dt = gsl_vector_alloc(ndata);
    gsl_vector* y = gsl_vector_alloc(ndata);
    for (int id = 0; id < z->size; id++) {
      if (mask_z[id] == 1) {
        if (id < p_params.size()) {
          std::vector<double> dx(ndata), dy(ndata), de(ndata);
          data->dx(dx, id, p_params);
          data->dy(dy, id, p_params);
          data->de(de, id, p_params);
          for (int i = 0; i < ndata; ++i) {
            gsl_vector_set(dt, i, dy[i]);
            for (int j = 0; j < ndata; ++j)
              if (i == j)
                gsl_matrix_set(DC, i, i, GPR_Kernel::dkx(i, i, i, vx, ve, h_params) * dx[i] +
                               GPR_Kernel::dke(i, i, i, vx, ve, h_params) * de[i]);
              else
                gsl_matrix_set(DC, i, j, GPR_Kernel::dkx(i, i, j, vx, ve, h_params) * dx[i] +
                               GPR_Kernel::dkx(j, i, j, vx, ve, h_params) * dx[j] +
                               GPR_Kernel::dke(i, i, j, vx, ve, h_params) * de[i] +
                               GPR_Kernel::dke(j, i, j, vx, ve, h_params) * de[j]);
          }
        }else{
          int hid = id - p_params.size();
          for (int i = 0; i < ndata; ++i) {
            gsl_vector_set(dt, i, 0);
            for (int j = 0; j < ndata; ++j)
              gsl_matrix_set(DC, i, j, GPR_Kernel::dk(hid, i, j, vx, ve, h_params));
          }
        }
        for (int i = 0; i < ndata; i++) {
          gsl_vector_view cv = gsl_matrix_column(DC, i);
          gsl_linalg_cholesky_svx(C, &cv.vector);
        }
        gsl_blas_dgemv(CblasNoTrans, 0.5, DC, alpha, 0, y);
        double glml;
        gsl_blas_ddot(t, y, &glml);
        for (int i = 0; i < ndata; i++)
          glml -= 0.5 * gsl_matrix_get(DC, i, i);
        double xxx;
        gsl_blas_ddot(dt, alpha, &xxx);
        glml -= xxx;
        gsl_vector_set(value_g, id, -glml);
      }else
        gsl_vector_set(value_g, id, 0);
    }
    // free
    gsl_vector_free(y);
    gsl_vector_free(dt);
    gsl_matrix_free(DC);
  }
  // free
  gsl_vector_free(alpha);
  gsl_vector_free(t);
  gsl_matrix_free(C);
  // Output
  return -mll;
};

template<class GPR_Data, class GPR_Kernel>
void GPR::Regression<GPR_Data, GPR_Kernel>::monte_carlo(const GPR_Data& xdata,
                                                        std::vector<double>& p_params, const std::vector<int>& p_mask, std::vector<double>& ep_params,
                                                        std::vector<double>& h_params, const std::vector<int>& h_mask, std::vector<double>& eh_params, int nthermal, int nmcs){
  std::vector<double> px(p_params.size(), 0);
  std::vector<double> hx(h_params.size(), 0);
  std::vector<double> px2(p_params.size(), 0);
  std::vector<double> hx2(h_params.size(), 0);
  std::vector<double> px_params(p_params);
  std::vector<double> hx_params(h_params);
  gsl_rng* r = gsl_rng_alloc(gsl_rng_mt19937);
  double o_log_w = calc_ll(xdata, p_params, h_params);
  for (int step = -nthermal; step < nmcs; ++step) {
#ifndef GPR_NLOG
    std::cerr << "# NMC=" << step << ", LL=" << o_log_w;
    for (int i = 0; i < p_params.size(); ++i)
      std::cerr << ", p[" << i << "]=" << p_params[i];
    for (int i = 0; i < h_params.size(); ++i)
      std::cerr << ", h[" << i << "]=" << h_params[i];
    std::cerr << std::endl;
#endif
    for (int i = 0; i < p_params.size(); ++i) {
      if (p_mask[i] == 0) continue;
      px_params[i] += (2 * gsl_rng_uniform(r) - 1) * ep_params[i];
      double log_w = calc_ll(xdata, px_params, h_params);
      if (log_w > o_log_w || exp(log_w - o_log_w) >= gsl_rng_uniform(r)) {
        o_log_w = log_w;
        p_params[i] = px_params[i];
      }else
        px_params[i] = p_params[i];
    }
    for (int i = 0; i < h_params.size(); ++i) {
      if (h_mask[i] == 0) continue;
      hx_params[i] += (2 * gsl_rng_uniform(r) - 1) * eh_params[i];
      double log_w = calc_ll(xdata, p_params, hx_params);
      if (log_w > o_log_w || exp(log_w - o_log_w) >= gsl_rng_uniform(r)) {
        o_log_w = log_w;
        h_params[i] = hx_params[i];
      }else
        hx_params[i] = h_params[i];
    }
    if (step >= 0) {
      for (int i = 0; i < p_params.size(); ++i) {
        if (p_mask[i] != 0) {
          px[i] += p_params[i];
          px2[i] += p_params[i] * p_params[i];
        }
      }
      for (int i = 0; i < h_params.size(); ++i) {
        if (h_mask[i] != 0) {
          hx[i] += h_params[i];
          hx2[i] += h_params[i] * h_params[i];
        }
      }
    }
  }
  for (int i = 0; i < p_params.size(); ++i) {
    if (p_mask[i] != 0 && nmcs > 1) {
      p_params[i] = px[i] / nmcs;
      ep_params[i] = sqrt((px2[i] - px[i] * px[i] / nmcs) / (nmcs - 1));
    }else
      ep_params[i] = 0;
  }
  for (int i = 0; i < h_params.size(); ++i) {
    if (h_mask[i] != 0 && nmcs > 1) {
      h_params[i] = hx[i] / nmcs;
      eh_params[i] = sqrt((hx2[i] - hx[i] * hx[i] / nmcs) / (nmcs - 1));
    }else
      eh_params[i] = 0;
  }
  gsl_rng_free(r);
}

template<class GPR_Data, class GPR_Kernel>
void GPR::Regression<GPR_Data, GPR_Kernel>::monte_carlo(const GPR_Data& xdata, const std::vector<double>& xp_params, const std::vector<int>& p_mask, const std::vector<double>& ep_params,
                                                        const std::vector<double>& xh_params, const std::vector<int>& h_mask, const std::vector<double>& eh_params,
                                                        std::vector<double>& average,
                                                        std::vector<double>& covariance,
                                                        int nthermal, int nmcs){
  std::vector<double> p_params(xp_params);
  std::vector<double> h_params(xh_params);
  std::vector<double> px_params(p_params);
  std::vector<double> hx_params(h_params);
  int nparams = p_params.size() + h_params.size();
  std::vector<double> x_params(nparams);
  average.resize(nparams, 0);
  covariance.resize(nparams * nparams, 0);
  gsl_rng* r = gsl_rng_alloc(gsl_rng_mt19937);
  double o_log_w = calc_ll(xdata, p_params, h_params);
  for (int step = -nthermal; step < nmcs; ++step) {
#ifndef GPR_NLOG
    std::cerr << "# NMC=" << step << ", LL=" << o_log_w;
    for (int i = 0; i < p_params.size(); ++i)
      std::cerr << ", p[" << i << "]=" << p_params[i];
    for (int i = 0; i < h_params.size(); ++i)
      std::cerr << ", h[" << i << "]=" << h_params[i];
    std::cerr << std::endl;
#endif
    for (int i = 0; i < p_params.size(); ++i) {
      if (p_mask[i] == 0) continue;
      px_params[i] += (2 * gsl_rng_uniform(r) - 1) * ep_params[i];
      double log_w = calc_ll(xdata, px_params, h_params);
      if (log_w > o_log_w || exp(log_w - o_log_w) >= gsl_rng_uniform(r)) {
        o_log_w = log_w;
        p_params[i] = px_params[i];
      }else
        px_params[i] = p_params[i];
    }
    for (int i = 0; i < h_params.size(); ++i) {
      if (h_mask[i] == 0) continue;
      hx_params[i] += (2 * gsl_rng_uniform(r) - 1) * eh_params[i];
      double log_w = calc_ll(xdata, p_params, hx_params);
      if (log_w > o_log_w || exp(log_w - o_log_w) >= gsl_rng_uniform(r)) {
        o_log_w = log_w;
        h_params[i] = hx_params[i];
      }else
        hx_params[i] = h_params[i];
    }
    if (step >= 0) {
      for (int i = 0; i < p_params.size(); ++i)
        x_params[i] = p_params[i];
      for (int i = 0; i < h_params.size(); ++i)
        x_params[p_params.size() + i] = h_params[i];
      for (int i = 0; i < x_params.size(); ++i) {
        average[i] += x_params[i];
        for (int j = i; j < x_params.size(); ++j)
          covariance[i * x_params.size() + j] += x_params[i] * x_params[j];
      }
    }
  }

  for (int i = 0; i < nparams; ++i)
    average[i] /= nmcs;
  for (int i = 0; i < nparams; ++i)
    for (int j = i; j < nparams; ++j)
      covariance[i * nparams + j] = (covariance[i * nparams + j] - nmcs * average[i] * average[j]) / (nmcs - 1);
  for (int i = 0; i < nparams; ++i)
    for (int j = 0; j < i; ++j)
      covariance[i * nparams + j] = covariance[j * nparams + i];
  gsl_rng_free(r);
}

template<class GPR_Data, class GPR_Kernel>
void GPR::Regression<GPR_Data, GPR_Kernel>::conditional_probability_y(double X, double& mean, double& variance, const GPR_Data& xdata, const std::vector<double>& p_params, const std::vector<double>& h_params) const {
  int ndata = xdata.npoints();
  std::vector<double> vx(ndata), vy(ndata), ve(ndata);
  xdata.x(vx, p_params);
  xdata.y(vy, p_params);
  xdata.e(ve, p_params);
  gsl_matrix* C = gsl_matrix_alloc(ndata, ndata);
  gsl_vector* t = gsl_vector_alloc(ndata);
  for (int i = 0; i < ndata; ++i) {
    gsl_vector_set(t, i, vy[i]);
    for (int j = 0; j < ndata; ++j)
      gsl_matrix_set(C, i, j, GPR_Kernel::k(i, j, vx, ve, h_params));
  }
  gsl_linalg_cholesky_decomp(C);
  // mu = C^{-1} . y
  gsl_vector* mu = gsl_vector_alloc(ndata);
  gsl_linalg_cholesky_solve(C, t, mu);
  gsl_vector* k = gsl_vector_alloc(ndata);
  for (int i = 0; i < ndata; i++)
    gsl_vector_set(k, i, GPR_Kernel::k(X, vx[i], h_params));
  // Mean
  gsl_blas_ddot(k, mu, &mean);
  // Variance
  gsl_vector* kk = gsl_vector_alloc(ndata);
  gsl_linalg_cholesky_solve(C, k, kk);
  double xxx;
  gsl_blas_ddot(k, kk, &xxx);
  variance = GPR_Kernel::k(X, X, h_params) - xxx;
  // Free
  gsl_vector_free(kk);
  gsl_vector_free(k);
  gsl_vector_free(mu);
  gsl_vector_free(t);
  gsl_matrix_free(C);
}

template<class GPR_Data, class GPR_Kernel>
void GPR::Regression<GPR_Data, GPR_Kernel>::monte_carlo(const std::vector<GPR_Data> xdatas,
                                                        const std::vector< std::vector<int> >& index_p_params,
                                                        const std::vector< std::vector<int> >& index_h_params,
                                                        const std::vector<double>& params,
                                                        const std::vector<double>& delta_params,
                                                        std::vector<double>& average,
                                                        std::vector<double>& covariance,
                                                        int nthermal, int nmcs){
  double o_log_w = 0;
  for (int is = 0; is < xdatas.size(); ++is) {
    std::vector<double> p_params(index_p_params[is].size());
    for (int j = 0; j < p_params.size(); ++j)
      p_params[j] = params[index_p_params[is][j]];
    std::vector<double> h_params(index_h_params[is].size());
    for (int j = 0; j < h_params.size(); ++j)
      h_params[j] = params[index_h_params[is][j]];
    o_log_w += calc_ll(xdatas[is], p_params, h_params);
  }

  average.resize(params.size(), 0);
  covariance.resize(params.size() * params.size(), 0);
  std::vector<double> x_params(params);

  gsl_rng* r = gsl_rng_alloc(gsl_rng_mt19937);

  for (int step = -nthermal; step < nmcs; ++step) {
#ifndef GPR_NLOG
    std::cerr << "# NMC=" << step << ", LL=" << o_log_w;
    for (int i = 0; i < params.size(); ++i)
      std::cerr << ", p[" << i << "]=" << x_params[i];
    std::cerr << std::endl;
#endif
    for (int ip = 0; ip < x_params.size(); ++ip) {
      if (delta_params[ip] == 0) continue;
      double old_x = x_params[ip];
      x_params[ip] += (2 * gsl_rng_uniform(r) - 1) * delta_params[ip];
      double log_w = 0;
      for (int is = 0; is < xdatas.size(); ++is) {
        std::vector<double> p_params(index_p_params[is].size());
        for (int j = 0; j < p_params.size(); ++j)
          p_params[j] = x_params[index_p_params[is][j]];
        std::vector<double> h_params(index_h_params[is].size());
        for (int j = 0; j < h_params.size(); ++j)
          h_params[j] = x_params[index_h_params[is][j]];
        log_w += calc_ll(xdatas[is], p_params, h_params);
      }
      if (log_w > o_log_w || exp(log_w - o_log_w) >= gsl_rng_uniform(r))
        o_log_w = log_w;
      else
        x_params[ip] = old_x;
    }
    if (step >= 0) {
      for (int i = 0; i < x_params.size(); ++i) {
        average[i] += x_params[i];
        for (int j = i; j < x_params.size(); ++j)
          covariance[i * x_params.size() + j] += x_params[i] * x_params[j];
      }
    }
  }

  for (int i = 0; i < params.size(); ++i)
    average[i] /= nmcs;
  for (int i = 0; i < params.size(); ++i)
    for (int j = i; j < params.size(); ++j)
      covariance[i * params.size() + j] = (covariance[i * params.size() + j] - nmcs * average[i] * average[j]) / (nmcs - 1);
  for (int i = 0; i < params.size(); ++i)
    for (int j = 0; j < i; ++j)
      covariance[i * params.size() + j] = covariance[j * params.size() + i];
  gsl_rng_free(r);
}

template<class GPR_Data, class GPR_Kernel>
double GPR::Regression<GPR_Data, GPR_Kernel>::gsl_v_fdf_m(const gsl_vector* z_m,  gsl_vector* value_g_m){
  double xreturn = 0;
  if (value_g_m == 0) {
    for (int is = 0; is < datas->size(); ++is) {
      if ((*datas)[is].npoints() == 0) continue;
      int n = (*pindex_p_params)[is].size() + (*pindex_h_params)[is].size();
      gsl_vector* z = gsl_vector_alloc(n);
      mask_z.resize(n);
      for (int j = 0; j < (*pindex_p_params)[is].size(); ++j) {
        gsl_vector_set(z, j, gsl_vector_get(z_m, (*pindex_p_params)[is][j]));
        mask_z[j] = mask_z_m[(*pindex_p_params)[is][j]];
      }
      for (int j = 0; j < (*pindex_h_params)[is].size(); ++j) {
        gsl_vector_set(z, (*pindex_p_params)[is].size() + j, gsl_vector_get(z_m, (*pindex_h_params)[is][j]));
        mask_z[(*pindex_p_params)[is].size() + j] = mask_z_m[(*pindex_h_params)[is][j]];
      }
      data = &((*datas)[is]);
      xreturn += gsl_v_fdf(z, 0);
      gsl_vector_free(z);
    }
  }else{
    gsl_vector_set_zero(value_g_m);
    for (int is = 0; is < datas->size(); ++is) {
      if ((*datas)[is].npoints() == 0) continue;
      int n = (*pindex_p_params)[is].size() + (*pindex_h_params)[is].size();
      gsl_vector* z = gsl_vector_alloc(n);
      mask_z.resize(n);
      gsl_vector* value_g = gsl_vector_alloc(n);
      for (int j = 0; j < (*pindex_p_params)[is].size(); ++j) {
        gsl_vector_set(z, j, gsl_vector_get(z_m, (*pindex_p_params)[is][j]));
        mask_z[j] = mask_z_m[(*pindex_p_params)[is][j]];
      }
      for (int j = 0; j < (*pindex_h_params)[is].size(); ++j) {
        gsl_vector_set(z, (*pindex_p_params)[is].size() + j, gsl_vector_get(z_m, (*pindex_h_params)[is][j]));
        mask_z[(*pindex_p_params)[is].size() + j] = mask_z_m[(*pindex_h_params)[is][j]];
      }
      data = &((*datas)[is]);
      xreturn += gsl_v_fdf(z, value_g);
      for (int j = 0; j < (*pindex_p_params)[is].size(); ++j)
        gsl_vector_set(value_g_m, (*pindex_p_params)[is][j], gsl_vector_get(value_g_m, (*pindex_p_params)[is][j]) + gsl_vector_get(value_g, j));
      for (int j = 0; j < (*pindex_h_params)[is].size(); ++j)
        gsl_vector_set(value_g_m, (*pindex_h_params)[is][j], gsl_vector_get(value_g_m, (*pindex_h_params)[is][j]) + gsl_vector_get(value_g, (*pindex_p_params)[is].size() + j));
      gsl_vector_free(value_g);
      gsl_vector_free(z);
    }
  }
  return xreturn;
}

template<class GPR_Data, class GPR_Kernel>
int GPR::Regression<GPR_Data, GPR_Kernel>::search_mll(const std::vector<GPR_Data>& xdatas,
                                                      const std::vector<std::vector<int> >& xindex_p_params,
                                                      const std::vector<std::vector<int> >& xindex_h_params,
                                                      std::vector<double>& params, const std::vector<int>& mask,
                                                      double stepsize, int niter, double tol, double epsabs){
  // initialize
  int nparams = params.size();
  gsl_vector* z_m = gsl_vector_alloc(nparams);
  mask_z_m.resize(nparams);
  for (int i = 0; i < params.size(); ++i) {
    gsl_vector_set(z_m, i, params[i]);
    mask_z_m[i] = mask[i];
  }
  datas = &xdatas;
  pindex_p_params = &xindex_p_params;
  pindex_h_params = &xindex_h_params;
  //
  size_t iter = 0;
  const gsl_multimin_fdfminimizer_type* T;
  gsl_multimin_fdfminimizer* s;
  gsl_multimin_function_fdf my_func;
  my_func.n = nparams;
  my_func.f = &gsl_f_m;
  my_func.df = &gsl_df_m;
  my_func.fdf = &gsl_fdf_m;
  my_func.params = this;
  T = gsl_multimin_fdfminimizer_conjugate_fr;
  s = gsl_multimin_fdfminimizer_alloc(T, z_m->size);
  gsl_multimin_fdfminimizer_set(s, &my_func, z_m, stepsize, tol);
  int status;
  do {
    iter++;
    status = gsl_multimin_fdfminimizer_iterate(s);
#ifndef GPR_NLOG
    std::cerr << "# ITER=" << iter << ", LL=" <<  -(s->f);
    for (int i = 0; i < params.size(); ++i)
      std::cerr << ", p[" << i << "]=" << gsl_vector_get(s->x, i);
    std::cerr << std::endl;
#endif
    if (status)
      break;
    status = gsl_multimin_test_gradient(s->gradient, epsabs);
  } while (status == GSL_CONTINUE && iter < niter);
  for (int i = 0; i < params.size(); ++i)
    params[i] = gsl_vector_get(s->x, i);
  gsl_multimin_fdfminimizer_free(s);
  gsl_vector_free(z_m);
  return status;
}

#endif
