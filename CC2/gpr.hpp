/* gpr.hpp
 *
 * Copyright (C) 2014 Kenji Harada
 *
 */
#ifndef GPR_HPP
#define GPR_HPP
/**
   @file gpr.hpp
   @brief Classes for Gaussian process regression and data set.
*/
#define _USE_MATH_DEFINES

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <vector>
#include <string>

#if __cplusplus >= 201103L
#include <random>
#else
#include <cstdlib>
#endif

#ifndef GPR_NLOG
#include <fstream>
#include <iostream>
#include <sstream>
#endif

/** @namespace GPR
    @brief Gaussian process regression.
*/
namespace GPR {
template <class GPR_DataSet> class Regression;
class DataSet;
const static std::string version = "1.00";
};

/**
   @class GPR::DataSet
   @brief Sample class of data set for GPR::Regression class.

   Data set is a set of data units. Data unit is a element of data
   set, which is a set of values.  In a regression process, a data
   unit is converted to a point. Data set also defines a kernel
   function and the derivative. The convertion from a data unit to a
   point and kernel function depends on parameters.
*/
class GPR::DataSet {
public:
  /**
     @brief Number of elements (Size of a set of values) in a data unit.
  */
  int num_e() const;

  /**
     @brief Number of parameters for data set.
  */
  int num_p() const;

  /**
     @brief Number of core parameters for data set.
  */
  int num_p_shared() const;

  /**
     @brief Number of data units.
  */
  int num() const;

  /**
     @brief Add a data unit.

     @param[in] Xdata Vector of values to define a data unit
  */
  void add(const std::vector<double> &Xdata);

  /**
     @brief Get a data unit.

     @param[in] index Index of data unit to get
     @param[out] Xdata Vector of values in the "index"-th data unit
  */
  void get(int index, std::vector<double> &Xdata) const;

  /**
     @brief Get a reference of a data unit.

     @param[in] index Index of data unit to get
     @return Reference to vector for data unit to get (const)
  */
  const std::vector<double> &get(int index) const;

  /**
     @brief Get a reference of a data unit.

     @param[in] index Index of data unit to get
     @return Reference to vector for data unit to get
  */
  std::vector<double> &at(int index);

  /**
     @brief Calculate log-prior with parameters.

     @param[in] Params Parameters for data set
     @return Value of log-prior with parameter "Params"
  */
  double prior(const std::vector<double> &Params) const;

  /**
     @brief Calculate a gradient vector of log-prior with parameters.

     @param[in] Params Parameters for data set
     @param[out] Grad_vec Graduate vector of log-prior by parameter "Params"
  */
  void grad_prior(const std::vector<double> &Params,
                  std::vector<double> &Grad_vec) const;

  /**
     @brief Calculate a gram matrix with parameters.

     @param[in] Params Parameters for data set
     @param[out] Xgram Gram matrix with parameter "Params"
  */
  void gram(const std::vector<double> &Params,
            std::vector<double> &Xgram) const;

  /**
     @brief Calculate a derivative of gram matrix by a parameter.

     @param[in] Params Parameters for data set
     @param[in] p_index Index of parameters for a derivative
     @param[out] Xdgram Derivative of a gram matrix by the "p_index"-th
     parameter
  */
  void differentiated_gram(const std::vector<double> &Params, int p_index,
                           std::vector<double> &Xdgram) const;

  /**
     @brief Calculate a gram vector between data set and a data unit.

     @param[in] Params Parameters for data set and data unit and kernel function
     @param[in] Xdata Vector of data unit
     @param[out] Gram_vec Gram vector between data set and the data unit
     ("Xdata")
  */
  void gram_vector(const std::vector<double> &Params,
                   const std::vector<double> &Xdata,
                   std::vector<double> &Gram_vec) const;

  /**
     @brief Calculate a gram vector between data set and point.

     @param[in] Params Parameters for data set and kernel function
     @param[in] X_regression Vector of point
     @param[out] Gram_vec Gram vector between data set and the data unit
     ("X_regression")
  */
  void gram_vector_regression(const std::vector<double> &Params,
                              const std::vector<double> &X_regression,
                              std::vector<double> &Gram_vec) const;

  /**
     @brief Calculate kernel function between two data units.

     @param[in] Params Parameters for two data units and kernel function
     @param[in] X1 Vector of data unit
     @param[in] X2 Vector of data unit
     @param[in] diagonal If true, use the diagonal part of kernel function
     @return Value of kernel function
  */
  double kernel(const std::vector<double> &Params,
                const std::vector<double> &X1, const std::vector<double> &X2,
                bool diagonal = false) const;

  /**
     @brief Calculate kernel function between two points.

     @param[in] Params Parameters for kernel function
     @param[in] X_regression_1 Vector of point
     @param[in] X_regression_2 Vector of point
     @param[in] diagonal If true, use the diagonal part of kernel function
     @return Value of kernel function
  */
  double kernel_regression(const std::vector<double> &Params,
                           const std::vector<double> &X_regression_1,
                           const std::vector<double> &X_regression_2,
                           bool diagonal = false) const;

  /**
     @brief Derivative of kernel function between two data units.

     @param[in] Params Parameters for two data units and kernel function
     @param[in] p_index Index of parameters for a derivative
     @param[in] X1 Vector of data unit
     @param[in] X2 Vector of data unit
     @param[in] diagonal If true, use the diagonal part of kernel function
     @return Value of derivative of kernel function
  */
  double differentiated_kernel(const std::vector<double> &Params, int p_index,
                               const std::vector<double> &X1,
                               const std::vector<double> &X2,
                               bool diagonal = false) const;

  /**
     @brief Convertion from data unit to point.

     @param[in] Params Parameters for data unit and kernel function
     @param[in] X Vector of data unit
     @param[out] X_regression Vector of point; @f$ X_{regression} = (Y, E, X_1,
     ...). @f$
  */
  void convert(const std::vector<double> &Params, const std::vector<double> &X,
               std::vector<double> &X_regression) const;

  /**
     @brief Differentiate converted data unit.

     @param[in] Params Parameters for data unit and kernel function
     @param[in] p_index Index of parameters for a derivative
     @param[in] X Vector of data unit
     @param[out] Dx_regression Vector of converted data unit by the "p_index"-th
     parameter; @f$ Dx_{regression} = (\partial Y/ \partial \theta_i, \partial E
     / \partial \theta_i, \partial X_1 / \partial \theta_i, ...). @f$
  */
  void differentiated_convert(const std::vector<double> &Params, int p_index,
                              const std::vector<double> &X,
                              std::vector<double> &Dx_regression) const;

  /**
     @brief Reverse convertion from point to data unit.

     @param[in] Params Parameters for data unit and kernel function
     @param[in] X_regression Vector of point
     @param[out] X Vector of data unit
  */
  void reverse_convert(const std::vector<double> &Params,
                       const std::vector<double> &X_regression,
                       std::vector<double> &X) const;

  /**
     @brief Get info about convertion from data unit to point.

     @param[out] info Key is a name of parameter for convertion from data unit
     to point
  */
  void get_info(std::map<std::string, double> &info) const;

  /**
     @brief Initialize parameters.

     @param[in] Params Vector of initial parameters
  */
  void initialize_parameters(std::vector<double> &Params) const;
};

/**
   @class GPR::Regression
   @brief Class of Gaussian process regression.
*/
template <class GPR_DataSet> class GPR::Regression {
public:
  Regression() : mode(-1) {};

  /**
     @brief Calculate a log-likelihood.

     @param[in] xdataset Data set
     @param[in] Params Parameters for data set
     @return Value of log-likelihood
  */
  double log_likelihood(const GPR_DataSet &xdataset,
                        const std::vector<double> &Params) const;

  /**
     @brief Calculate a gradient of a log-likelihood.

     @param[in] xdataset Data set
     @param[in] Params Parameters for data set
     @param[out] Grad_vec Gradient vector of log-likelihood
  */
  void gradient_of_log_likelihood(const GPR_DataSet &xdataset,
                                  const std::vector<double> &Params,
                                  std::vector<double> &Grad_vec) const;

  /**
     @brief Calculate a log-likelihood for datasets.

     @param[in] Datasets Set of Datasets
     @param[in] Index Vector of correspondence of parameters for data set.
     index[m][i] => Position of parameter "i" for data set "m" in Params.
     @param[in] Params Parameters for data set
     @return Value of log-likelihood
  */
  double log_likelihood(const std::vector<GPR_DataSet> &Datasets,
                        const std::vector<std::vector<int> > &Index,
                        const std::vector<double> &Params) const;

  /**
     @brief Calculate a gradient of a log-likelihood for datasets.

     @param[in] xdataset Data set
     @param[in] Index Vector of correspondence of parameters for data set.
     index[m][i] => Position of parameter "i" for data set "m" in Params.
     @param[in] Params Parameters for data set
     @param[out] Grad_vec Gradient vector of log-likelihood
  */
  void gradient_of_log_likelihood(const std::vector<GPR_DataSet> &Datasets,
                                  const std::vector<std::vector<int> > &Index,
                                  const std::vector<double> &Params,
                                  std::vector<double> &Grad_vec) const;

  /**
     @brief Infer points from data set with parameters.

     @param[in] xdataset Data set
     @param[in] Params Parameters for data set and inferred data points
     @param[in, out] *points Data set for inferred data points
  */
  void infer(const GPR_DataSet &xdataset, const std::vector<double> &Params,
             GPR_DataSet *points) const;

  /**
     @brief Infer points from data set with parameters.

     @param[in] xdataset Data set
     @param[in] Params Parameters for data set
     @param[in, out] *Point_regressions Vector of inferred data points
  */
  void
  infer_regression(const GPR_DataSet &xdataset,
                   const std::vector<double> &Params,
                   std::vector<std::vector<double> > *Point_regressions) const;

  /**
     @brief Estimate MAP.

     @param[in] xdataset Data set
     @param[in] Params Initial parameters for data set
     @param[out] Params MAP parameters for data set
     @param[in] Mask Mask of optimizing parameters; {0: unchanged, 1:optimized}
     @param[in] setting Setting parameters to estimate MAP
     @param[out] H Inverse of Hesse matrix
  */
  int map(const GPR_DataSet &xdataset, std::vector<double> &Params,
          const std::vector<int> &Mask,
          const std::map<std::string, double> &setting, std::vector<double> &H);

  /**
     @brief Estimate MAP.

     @param[in] xdatasets Vector of data sets
     @param[in] Index Vector of correspondence of parameters for data set.
     index[m][i] => Position of parameter "i" for data set "m" in Params.
     @param[in] Params Initial parameters for all data sets
     @param[out] Params MAP parameters for all data sets
     @param[in] Mask Mask of optimizing parameters; {0: unchanged, 1:optimized}
     @param[in] setting Setting parameters to estimate MAP
     @param[out] H Inverse of Hesse matrix
  */
  int map(const std::vector<GPR_DataSet> &xdatasets,
          const std::vector<std::vector<int> > &Index,
          std::vector<double> &Params, const std::vector<int> &Mask,
          const std::map<std::string, double> &setting, std::vector<double> &H);

  /**
     @brief Monte Carlo sampling

     @param[in] xdataset Data set
     @param[in] Params Initial parameters for data set
     @param[out] Params Final parameters for data set
     @param[in] Mask Mask of optimizing parameters; {0: unchanged, 1:optimized}
     @param[out] Average Mean of parameters
     @param[out] Covariance Covriance of parameters
     @param[out] setting Setting parameters in Monte Carlo sampling
     @return Ratio of succeed
  */
  double mc(const GPR_DataSet &xdataset, std::vector<double> &Params,
            const std::vector<int> &Mask, std::vector<double> &Average,
            std::vector<double> &Covariance,
            const std::map<std::string, double> &setting);

  /**
     @brief Monte Carlo sampling

     @param[in] xdatasets Vector of data sets
     @param[in] Index Vector of correspondence of parameters for data set.
     index[m][i] => Position of parameter "i" for data set "m" in Params.
     @param[in] Params Initial parameters for all data sets
     @param[out] Params Final parameters for all data sets
     @param[in] Mask Mask of optimizing parameters; {0: unchanged, 1:optimized}
     @param[out] Average Mean of parameters
     @param[out] Covariance Covriance of parameters
     @param[in] setting Setting parameters in Monte Carlo sampling
     @return Ratio of succeed
  */
  double mc(const std::vector<GPR_DataSet> &xdatasets,
            const std::vector<std::vector<int> > &Index,
            std::vector<double> &Params, const std::vector<int> &Mask,
            std::vector<double> &Average, std::vector<double> &Covariance,
            const std::map<std::string, double> &setting);

  /**
     @brief GPR::OPT::ObjectiveFunction::f()
  */
  double f(const std::vector<double> &X) const {
    if (mode == 0)
      return -log_likelihood(*pdataset, X);
    else {
      double sum = 0;
      for (int i = 0; i < pdatasets->size(); ++i) {
        std::vector<double> Params((*pdatasets)[i].num_p());
        for (int j = 0; j < (*pdatasets)[i].num_p(); ++j)
          Params[j] = X[(*pindex)[i][j]];
        sum -= log_likelihood((*pdatasets)[i], Params);
      }
      return sum;
    }
  }

  /**
     @brief GPR::OPT::ObjectiveFunction::df()
  */
  void df(const std::vector<double> &X, std::vector<double> &Grad) const {
    if (mode == 0) {
      gradient_of_log_likelihood(*pdataset, X, Grad);
      for (int i = 0; i < pmask->size(); ++i)
        Grad[i] *= -((*pmask)[i]);
    } else {
      Grad.resize(X.size());
      std::fill(Grad.begin(), Grad.end(), 0);
      for (int t = 0; t < pdatasets->size(); ++t) {
        std::vector<double> Params((*pdatasets)[t].num_p());
        for (int j = 0; j < (*pdatasets)[t].num_p(); ++j)
          Params[j] = X[(*pindex)[t][j]];
        std::vector<double> Xgrad;
        gradient_of_log_likelihood((*pdatasets)[t], Params, Xgrad);
        for (int j = 0; j < (*pdatasets)[t].num_p(); ++j)
          Grad[(*pindex)[t][j]] += Xgrad[j];
      }
      for (int i = 0; i < pmask->size(); ++i)
        Grad[i] *= -((*pmask)[i]);
    }
  }

private:
  /**
     @brief Main function to estimate MAP.

     estimate MAP.
     @param[in] Params Parameters
     @param[in] setting Setting parameters
     @param[out] H Inverse of Hesse matrix
  */
  int map_main(std::vector<double> &Params,
               const std::map<std::string, double> &setting,
               std::vector<double> &H);
  /**
     @brief Main function of Monte Carlo sampling

     @param[in] Params Parameters
     @param[in] Mask Mask {0: unchanged, 1:optimized}
     @param[in] Average Mean of parameters
     @param[in] Covariance Coariance of parameters
     @param[in] setting Setting parameters
     @return Ratio of succeed
  */
  double mc_main(std::vector<double> &Params, const std::vector<int> &Mask,
                 std::vector<double> &Average, std::vector<double> &Covariance,
                 const std::map<std::string, double> &setting);
  /**
     @brief Run a Monte Calro step (Hybrid Monte Carlo)

     @param[in] X State vector
     @param[in] Mask Mask {0: unchanged, 1:optimized}
     @param[in] Stepsize Time step
     @param[in] num_steps Number of time evolution
     @return {0:not succeed, 1:succeed}
  */
  int mc_run(std::vector<double> &X, const std::vector<int> &Mask,
             const std::vector<double> &Stepsize, int num_steps);
  static void check_setting(double &var, std::string key,
                            const std::map<std::string, double> &setting) {
    std::map<std::string, double>::const_iterator it = setting.find(key);
    if (it != setting.end())
      var = it->second;
  }
  static void check_setting(int &var, std::string key,
                            const std::map<std::string, double> &setting) {
    std::map<std::string, double>::const_iterator it = setting.find(key);
    if (it != setting.end())
      var = static_cast<int>(it->second);
  }
  int mode;
  const GPR_DataSet *pdataset;
  const std::vector<int> *pmask;
  const std::vector<GPR_DataSet> *pdatasets;
  const std::vector<std::vector<int> > *pindex;
/* Randam number by Boost */
#if __cplusplus >= 201103L
  std::mt19937 mtgen;
  std::uniform_real_distribution<> uniform;
  std::normal_distribution<> normal;
#else
  double uniform() { return static_cast<double>(rand()) / RAND_MAX; }
  double normal() {
    double u1 = uniform();
    double u2 = uniform();
    return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
  }
#endif
};

/* Codes */
/**
   @file BLAS_LAPACK.hpp
   @brief Header file for some fortran functions in BLAS and LAPACK

   This header file defines some fortran functions in BLAS and LAPACK.
*/
/** BLASS and LAPACK **/
extern "C" {
/// Compute the solution to system of linear equations A * X = B for positive
/// definite symmetric matrices
int dposv_(char *UPLO, int *N, int *NRHS, double *A, int *LDA, double *B,
           int *LDB, int *INFO);
}

#define DPOSV dposv_

#include "optimize.hpp"

template <class GPR_DataSet>
double GPR::Regression<GPR_DataSet>::log_likelihood(
    const GPR_DataSet &xdataset, const std::vector<double> &Params) const {
  std::vector<double> xgram;
  xdataset.gram(Params, xgram);
  char UPLO = 'U';
  int N = xdataset.num();
  int NRHS = 1;
  int LDA = N;
  std::vector<double> Y(N);
  {
    std::vector<double> xc;
    for (int i = 0; i < N; ++i) {
      xdataset.convert(Params, xdataset.get(i), xc);
      Y[i] = xc[0];
    }
  }
  std::vector<double> Y2(Y);
  int LDB = N;
  int INFO;
  DPOSV(&UPLO, &N, &NRHS, &(xgram[0]), &LDA, &(Y[0]), &LDB, &INFO);
  if (INFO != 0) {
    std::cerr << "# INFO in log_likelihood : " << INFO << std::endl;
    return NAN;
  }
  double f_ll = xdataset.prior(Params);
  for (int i = 0; i < N; ++i)
    f_ll -= Y[i] * Y2[i];
  f_ll /= 2;
  for (int i = 0; i < N; ++i)
    f_ll -= log(xgram[i + i * N]);
  f_ll -= N * log(2 * M_PI) / 2;

  return f_ll;
}

template <class GPR_DataSet>
void GPR::Regression<GPR_DataSet>::gradient_of_log_likelihood(
    const GPR_DataSet &xdataset, const std::vector<double> &Params,
    std::vector<double> &Grad_vec) const {
  Grad_vec.resize(Params.size());
  std::vector<double> xgram;
  xdataset.gram(Params, xgram);
  char UPLO = 'U';
  int N = xdataset.num();
  int NRHS = (N + 1) * Params.size() + 1;
  int LDA = N;
  std::vector<double> B(NRHS * N);
  std::vector<double> Y(N);
  {
    std::vector<double> xc;
    for (int i = 0; i < N; ++i) {
      xdataset.convert(Params, xdataset.get(i), xc);
      B[i + (NRHS - 1) * N] = Y[i] = xc[0];
    }
  }
  for (int pi = 0; pi < Params.size(); ++pi) {
    std::vector<double> xdgram;
    xdataset.differentiated_gram(Params, pi, xdgram);
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        B[i + (j + pi * N) * N] = xdgram[i + j * N];
    for (int i = 0; i < N; ++i) {
      std::vector<double> dx_reg;
      xdataset.differentiated_convert(Params, pi, xdataset.get(i), dx_reg);
      B[i + (pi + Params.size() * N) * N] = dx_reg[0];
    }
  }
  int LDB = N;
  int INFO;
  DPOSV(&UPLO, &N, &NRHS, &xgram[0], &LDA, &B[0], &LDB, &INFO);
  if (INFO != 0) {
    std::cerr << "# INFO in gradient_of_log_likelihood : " << INFO << std::endl;
    std::fill(Grad_vec.begin(), Grad_vec.end(), NAN);
  }
  xdataset.grad_prior(Params, Grad_vec);
  for (int pi = 0; pi < Params.size(); ++pi) {
    std::vector<double> X(N);
    for (int i = 0; i < N; ++i) {
      X[i] = -B[i + (pi + Params.size() * N) * N];
      double xp = 0;
      for (int j = 0; j < N; ++j)
        xp += B[i + (j + pi * N) * N] * B[j + (NRHS - 1) * N];
      X[i] += xp / 2;
    }
    double x = 0;
    for (int i = 0; i < N; ++i) {
      x -= B[i + (i + pi * N) * N] / 2;
      x += Y[i] * X[i];
    }
    Grad_vec[pi] += x;
  }
}

template <class GPR_DataSet>
void GPR::Regression<GPR_DataSet>::infer(const GPR_DataSet &xdataset,
                                         const std::vector<double> &Params,
                                         GPR_DataSet *points) const {
  std::vector<double> xgram;
  xdataset.gram(Params, xgram);
  char UPLO = 'U';
  int N = xdataset.num();
  int NRHS = points->num();
  int LDA = N;
  std::vector<double> Y(N);
  {
    std::vector<double> xc;
    for (int i = 0; i < N; ++i) {
      xdataset.convert(Params, xdataset.get(i), xc);
      Y[i] = xc[0];
    }
  }
  std::vector<double> B(NRHS * N);
  for (int pi = 0; pi < NRHS; ++pi) {
    std::vector<double> gv;
    xdataset.gram_vector(Params, points->at(pi), gv);
    for (int i = 0; i < N; ++i)
      B[i + pi * N] = gv[i];
  }
  std::vector<double> B0 = B;
  int LDB = N;
  int INFO;
  DPOSV(&UPLO, &N, &NRHS, &xgram[0], &LDA, &B[0], &LDB, &INFO);
  if (INFO != 0) {
    std::cerr << "# INFO in infer : " << INFO << std::endl;
    exit(-1);
  }
  for (int pi = 0; pi < NRHS; ++pi) {
    std::vector<double> xc;
    xdataset.convert(Params, points->at(pi), xc);
    xc[0] = 0;
    xc[1] = 0;
    double mean = 0;
    double variance = xdataset.kernel_regression(Params, xc, xc, true);
    for (int i = 0; i < N; ++i) {
      mean += Y[i] * B[i + pi * N];
      variance -= B0[i + pi * N] * B[i + pi * N];
    }
    xc[0] = mean;
    xc[1] = sqrt(variance);
    xdataset.reverse_convert(Params, xc, points->at(pi));
  }
}

template <class GPR_DataSet>
void GPR::Regression<GPR_DataSet>::infer_regression(
    const GPR_DataSet &xdataset, const std::vector<double> &Params,
    std::vector<std::vector<double> > *point_regressions) const {
  std::vector<double> xgram;
  xdataset.gram(Params, xgram);
  char UPLO = 'U';
  int N = xdataset.num();
  int NRHS = point_regressions->size();
  int LDA = N;
  std::vector<double> Y(N);
  {
    std::vector<double> xc;
    for (int i = 0; i < N; ++i) {
      xdataset.convert(Params, xdataset.get(i), xc);
      Y[i] = xc[0];
    }
  }
  std::vector<double> B(NRHS * N);
  for (int pi = 0; pi < NRHS; ++pi) {
    std::vector<double> gv;
    xdataset.gram_vector_regression(Params, point_regressions->at(pi), gv);
    for (int i = 0; i < N; ++i)
      B[i + pi * N] = gv[i];
  }
  std::vector<double> B0 = B;
  int LDB = N;
  int INFO;
  DPOSV(&UPLO, &N, &NRHS, &xgram[0], &LDA, &B[0], &LDB, &INFO);
  if (INFO != 0) {
    std::cerr << "# INFO in infer_regression : " << INFO << std::endl;
    exit(-1);
  }
  for (int pi = 0; pi < NRHS; ++pi) {
    double mean = 0;
    double variance = xdataset.kernel_regression(
        Params, point_regressions->at(pi), point_regressions->at(pi), true);
    for (int i = 0; i < N; ++i) {
      mean += Y[i] * B[i + pi * N];
      variance -= B0[i + pi * N] * B[i + pi * N];
    }
    point_regressions->at(pi)[0] = mean;
    point_regressions->at(pi)[1] = sqrt(variance);
  }
}

template <class GPR_DataSet>
int GPR::Regression<GPR_DataSet>::map_main(
    std::vector<double> &Params, const std::map<std::string, double> &setting,
    std::vector<double> &H) {
  double step_size = 1e-4;
  check_setting(step_size, "MAP::STEP_SIZE", setting);
  double epsilon = 1e-8;
  check_setting(epsilon, "MAP::EPSILON", setting);
  double tol = 1e-3;
  check_setting(epsilon, "MAP::TOL", setting);
#ifndef GPR_NLOG
  std::cerr << "# Optimize parameters to do MAP estimate" << std::endl;
#endif
  GPR::OPT::CG_FR<GPR::Regression<GPR_DataSet> > optimizer;
  optimizer.minimize(Params, this, step_size, epsilon, tol);
#ifndef GPR_NLOG
  std::cerr << std::endl << std::endl;
#endif
  return 0;
}

template <class GPR_DataSet>
int GPR::Regression<GPR_DataSet>::map(
    const GPR_DataSet &xdataset, std::vector<double> &Params,
    const std::vector<int> &Mask, const std::map<std::string, double> &setting,
    std::vector<double> &H) {
  mode = 0;
  pdataset = &xdataset;
  pmask = &Mask;
  map_main(Params, setting, H);
  return 0;
}

template <class GPR_DataSet>
int GPR::Regression<GPR_DataSet>::map(
    const std::vector<GPR_DataSet> &xdatasets,
    const std::vector<std::vector<int> > &index, std::vector<double> &Params,
    const std::vector<int> &Mask, const std::map<std::string, double> &setting,
    std::vector<double> &H) {
  mode = 1;
  pdatasets = &xdatasets;
  pindex = &index;
  pmask = &Mask;
  map_main(Params, setting, H);
  return 0;
}

template <class GPR_DataSet>
double GPR::Regression<GPR_DataSet>::mc(
    const GPR_DataSet &xdataset, std::vector<double> &Params,
    const std::vector<int> &Mask, std::vector<double> &Average,
    std::vector<double> &Covariance,
    const std::map<std::string, double> &setting) {
  mode = 0;
  pdataset = &xdataset;
  pmask = &Mask;
  return mc_main(Params, Mask, Average, Covariance, setting);
}

template <class GPR_DataSet>
double GPR::Regression<GPR_DataSet>::mc(
    const std::vector<GPR_DataSet> &xdatasets,
    const std::vector<std::vector<int> > &index, std::vector<double> &Params,
    const std::vector<int> &Mask, std::vector<double> &Average,
    std::vector<double> &Covariance,
    const std::map<std::string, double> &setting) {
  mode = 1;
  pdatasets = &xdatasets;
  pindex = &index;
  pmask = &Mask;
  return mc_main(Params, Mask, Average, Covariance, setting);
}

template <class GPR_DataSet>
double GPR::Regression<GPR_DataSet>::mc_main(
    std::vector<double> &Params, const std::vector<int> &Mask,
    std::vector<double> &Average, std::vector<double> &Covariance,
    const std::map<std::string, double> &setting) {
#ifndef GPR_NLOG
  std::cerr
      << "# Monte Carlo estimation of confidential intervals of parameters"
      << std::endl;
#endif
  int seed = 20140318;
  check_setting(seed, "MC::SEED", setting);
#if __cplusplus >= 201103L
  mtgen.seed(seed);
#else
  srand(seed);
#endif
  int NMCS = 1000;
  check_setting(NMCS, "MC::NMCS", setting);
  int LIMIT = 20000;
  check_setting(LIMIT, "MC::LIMIT", setting);
  int num_steps = 10;
  check_setting(num_steps, "MC::NUM_STEPS", setting);
  double step_size0 = 1e0 / num_steps;
  check_setting(step_size0, "MC::STEP_SIZE", setting);
  const double LOW_AR = 0.4;
  int N = Params.size();
  Average.resize(N);
  Covariance.resize(N * N);
  std::vector<double> Samples(N * NMCS);
  int nsample = 0;
RESTART:
  double rsucceed = 0;
  {
    double step_size = step_size0;
    std::vector<double> Stepsize(N, step_size);
    std::vector<double> Backup_P(Params);
    /* Configure Stepsize */
    {
#ifndef GPR_NLOG
      std::cerr << "## Adjust stepsize" << std::endl;
#endif
      int ic = 0;
      int st = 0;
      int target = -1;
      int status = 0;
      const double NTRY = 50;
      int itime = 0;
      while (++itime) {
        ++nsample;
        if (nsample > LIMIT)
          exit(-1);
        int xst = mc_run(Params, Mask, Stepsize, num_steps);
#ifndef GPR_NLOG
        std::cerr << itime << " " << f(Params);
        for (int i = 0; i < N; ++i)
          std::cerr << " " << Params[i];
        std::cerr << std::endl;
#endif
        ++ic;
        st += xst;
        if (target == -1) {
          if (ic == 0.2 * NTRY && st == 0) {
#ifndef GPR_NLOG
            std::cerr << "### Frozen" << std::endl;
#endif
            status = 1;
            step_size *= 0.5;
            for (int i = 0; i < N; ++i)
              Stepsize[i] = step_size;
#ifndef GPR_NLOG
            std::cerr << "#### Change stepsize : " << step_size << std::endl;
            std::cerr << "#### Status : " << status << std::endl;
#endif
            st = 0;
            ic = 0;
            std::copy(Backup_P.begin(), Backup_P.end(), Params.begin());
          } else if (ic == NTRY) {
#ifndef GPR_NLOG
            std::cerr << "### Ratio of succeed : " << st / NTRY << std::endl;
#endif
            if (st < LOW_AR * NTRY) {
              status = 1;
              step_size *= 0.5;
              std::copy(Backup_P.begin(), Backup_P.end(), Params.begin());
            } else if (status == 0) {
              step_size *= 2e0;
              std::copy(Backup_P.begin(), Backup_P.end(), Params.begin());
            } else {
              status = 2;
              std::copy(Params.begin(), Params.end(), Backup_P.begin());
            }
            if (status != 2) {
              for (int i = 0; i < N; ++i)
                Stepsize[i] = step_size;
#ifndef GPR_NLOG
              std::cerr << "#### Change stepsize : " << step_size << std::endl;
#endif
            }
#ifndef GPR_NLOG
            std::cerr << "#### Status : " << status << std::endl;
#endif
            st = 0;
            ic = 0;
          }
        } else {
          if (ic == 0.2 * NTRY && st == 0) {
#ifndef GPR_NLOG
            std::cerr << "### Frozen" << std::endl;
#endif
            status = 1;
            Stepsize[target] *= 0.5e0;
#ifndef GPR_NLOG
            std::cerr << "#### Change stepsize : [" << target << "] "
                      << Stepsize[target] << std::endl;
            std::cerr << "#### Status : " << status << std::endl;
#endif
            st = 0;
            ic = 0;
            std::copy(Backup_P.begin(), Backup_P.end(), Params.begin());
            if (Stepsize[target] <= step_size)
              status = 2;
          } else if (ic == NTRY) {
#ifndef GPR_NLOG
            std::cerr << "### Ratio of succeed : " << st / NTRY << std::endl;
#endif
            if (st < LOW_AR * NTRY) {
              status = 1;
              Stepsize[target] *= 0.5e0;
              std::copy(Backup_P.begin(), Backup_P.end(), Params.begin());
#ifndef GPR_NLOG
              std::cerr << "#### Change stepsize : [" << target << "] "
                        << Stepsize[target] << std::endl;
#endif
              if (Stepsize[target] <= step_size)
                status = 2;
            } else if (status == 0) {
              Stepsize[target] *= 2e0;
              std::copy(Backup_P.begin(), Backup_P.end(), Params.begin());
#ifndef GPR_NLOG
              std::cerr << "#### Change stepsize : [" << target << "] "
                        << Stepsize[target] << std::endl;
#endif
            } else {
              status = 2;
              std::copy(Params.begin(), Params.end(), Backup_P.begin());
            }
#ifndef GPR_NLOG
            std::cerr << "#### Status : " << status << std::endl;
#endif
            st = 0;
            ic = 0;
          }
        }
        if (status == 2) {
        RECHECK:
          if (target == (N - 1)) {
#ifndef GPR_NLOG
            std::cerr << "### Adjusted stepsize :";
            for (int i = 0; i < N; ++i)
              std::cerr << " " << Stepsize[i];
            std::cerr << std::endl;
            std::cerr << std::endl << std::endl;
#endif
            break;
          } else {
            ++target;
            if (Mask[target] == 0)
              goto RECHECK;
            status = 0;
          }
        }
      }
    }

/* Sampling */
#ifndef GPR_NLOG
    std::cerr << "## Sampling " << rsucceed / NMCS << std::endl;
#endif
    rsucceed = 0;
    for (int time = 0; time < NMCS; ++time) {
      int xst = mc_run(Params, Mask, Stepsize, num_steps);
      rsucceed += xst;
      for (int i = 0; i < N; ++i)
        Samples[i + time * N] = Params[i];
#ifndef GPR_NLOG
      std::cerr << time << " " << f(Params);
      for (int i = 0; i < N; ++i)
        std::cerr << " " << Params[i];
      std::cerr << std::endl;
#endif
    }
  }

#ifndef GPR_NLOG
  std::cerr << "### Acceptance ratio : " << rsucceed / NMCS << std::endl;
  std::cerr << std::endl << std::endl;
#endif
  /* Check a sampling ratio */
  if (rsucceed / NMCS < LOW_AR)
    goto RESTART;

  /* Summarize */
  for (int i = 0; i < N; ++i) {
    double x = 0;
    for (int time = 0; time < NMCS; ++time)
      x += Samples[i + time * N];
    Average[i] = x / NMCS;
  }
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      double x2 = 0;
      for (int time = 0; time < NMCS; ++time)
        x2 += (Samples[i + time * N] - Average[i]) *
              (Samples[j + time * N] - Average[j]);
      Covariance[i + j * N] = x2 / (NMCS - 1e0);
    }
  return rsucceed / NMCS;
}

template <class GPR_DataSet>
int GPR::Regression<GPR_DataSet>::mc_run(std::vector<double> &X,
                                         const std::vector<int> &Mask,
                                         const std::vector<double> &Stepsize,
                                         int num_steps) {
  int N = X.size();
  std::vector<double> Q(X);
  std::vector<double> P(N);
  std::vector<double> dU(N);
  double current_K = 0;
  for (int i = 0; i < N; ++i) {
#if __cplusplus >= 201103L
    P[i] = normal(mtgen) * Mask[i];
#else
    P[i] = normal() * Mask[i];
#endif
    current_K += P[i] * P[i];
  }
  current_K *= 0.5;
  df(Q, dU);
  for (int i = 0; i < N; ++i)
    P[i] -= Stepsize[i] * dU[i] * 0.5 * Mask[i];
  for (int time = 0; time < num_steps; ++time) {
    for (int i = 0; i < N; ++i)
      Q[i] += Stepsize[i] * P[i];
    df(Q, dU);
    if (time != (num_steps - 1))
      for (int i = 0; i < N; ++i)
        P[i] -= Stepsize[i] * dU[i] * Mask[i];
  }
  double proposed_K = 0;
  for (int i = 0; i < N; ++i) {
    P[i] -= Stepsize[i] * dU[i] * 0.5 * Mask[i];
    proposed_K += P[i] * P[i];
  }
  proposed_K *= 0.5;
  double wdiff = exp(f(X) - f(Q) + current_K - proposed_K);
#if __cplusplus >= 201103L
  if (std::isfinite(wdiff) && uniform(mtgen) < wdiff) {
    std::copy(Q.begin(), Q.end(), X.begin());
    return 1;
  } else
    return 0;
#else
  if (std::isfinite(wdiff) && uniform() < wdiff) {
    std::copy(Q.begin(), Q.end(), X.begin());
    return 1;
  } else
    return 0;
#endif
}

template <class GPR_DataSet>
double GPR::Regression<GPR_DataSet>::log_likelihood(
    const std::vector<GPR_DataSet> &Datasets,
    const std::vector<std::vector<int> > &Index,
    const std::vector<double> &Params) const {
  int NSET = Datasets.size();
  int np = Datasets[0].num_p();
  double f_ll = 0;
  for (int i = 0; i < NSET; ++i) {
    std::vector<double> Params_local(np);
    for (int j = 0; j < np; ++j)
      Params_local[j] = Params[Index[i][j]];
    f_ll += log_likelihood(Datasets[i], Params_local);
  }
  return f_ll;
}

template <class GPR_DataSet>
void GPR::Regression<GPR_DataSet>::gradient_of_log_likelihood(
    const std::vector<GPR_DataSet> &Datasets,
    const std::vector<std::vector<int> > &Index,
    const std::vector<double> &Params, std::vector<double> &Grad_vec) const {
  int NSET = Datasets.size();
  int np = Datasets[0].num_p();
  Grad_vec.resize(Params.size());
  std::fill(Grad_vec.begin(), Grad_vec.end(), 0);
  for (int i = 0; i < NSET; ++i) {
    std::vector<double> Params_local(np);
    std::vector<double> Grad_vec_local;
    for (int j = 0; j < np; ++j)
      Params_local[j] = Params[Index[i][j]];
    gradient_of_log_likelihood(Datasets[i], Params_local, Grad_vec_local);
    for (int j = 0; j < np; ++j)
      Grad_vec[Index[i][j]] += Grad_vec_local[j];
  }
}

#endif
