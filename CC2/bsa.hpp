/* bsa.hpp
 *
 * Copyright (C) 2014, 2015 Kenji Harada
 *
 */
#ifndef BSA_HPP
#define BSA_HPP
/**
   @file bsa.hpp
   @brief Classes for Bayesian scaling analysis.

   This file defines DataSet classes for Bayesian scaling anlysis.
*/
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <numeric>
#include <algorithm>
#include <cassert>
/** @namespace BSA
    @brief Implementation of GPR::DataSet for kernel methods in Bayesian scaling
   analysis
*/
namespace BSA {
class DataSet;
class DataSet_C;
const static std::string version = "1.01";
};
/**
   @class BSA::DataSet
   @brief Data set for GPR::Regression class.

   The finite-size scaling form is written as
   @f[
   A(T, L) = L^{c_2} F[ ( T - T_c ) L^{c_1} ],
   @f] where @f$ A @f$ is an observable. The triplet of a point
   is defined as
   @f[
   X = (T - T_c ) (L/L_{MAX})^{c_1} / R_X, Y = (A / (L/L_{MAX})^{c_2} -
   Y_0)/R_Y, E = \delta A/ (L/L_{MAX})^{c_2}/R_Y,
   @f]
   where @f$ \delta A @f$ is an error of @f$ A @f$ and @f$ L_{MAX} @f$ is the
   largest @f$ L @f$.
   Scaling factor @f$ R_X @f$ is defined so that the width of X for @f$ L_{MAX}
   @f$ is 2.
   Scaling factor @f$ R_Y @f$ and @f$ Y_0 @f$ is defined so that Y for @f$
   L_{MAX} @f$ is in [-1:1].
   The data ansatz is @f[ Y \sim F(X) \pm E. @f]

   Kernel function is written as
   @f[
   k_G(i, j) = \delta_{ij} (E(i)^2 + \theta_0^2)
   + \theta_1^2 \exp( - |X(i)- X(j)|^2 / 2\theta_2^2 ).
   @f]

   Parameters are defined as
   Params[0] = @f$ T_c @f$,
   Params[1] = @f$ c_1 @f$,
   Params[2] = @f$ c_2 @f$,
   Params[3] = @f$ \theta_0 @f$,
   Params[4] = @f$ \theta_1 @f$, and
   Params[5] = @f$ \theta_2 @f$.
   Shared parameters are @f$ T_c @f$ and @f$ c_1 @f$.
*/
class BSA::DataSet {
public:
  /**
     @brief Description
   */
  static std::string description() {
    std::ostringstream osst;
    osst << "[Finite-size scaling form]" << std::endl;
    osst << "  A(T, L) = L^c2 F[ ( T - Tc ) L^c1 ]," << std::endl;
    osst << "[Data ansatz]" << std::endl;
    osst << "  Y ~ F(X) +- E," << std::endl;
    osst << "  X = (T - Tc ) L^c1, Y = A / L^c2, E = dA / L^c2." << std::endl;
    osst << "[Kernel function]" << std::endl;
    osst << "  k_G(i, j) = delta_{ij} (E(i)^2 + theta0^2) + theta1^2 exp( - "
            "|X(i)- X(j)|^2 / 2 theta2^2 )." << std::endl;
    osst << "[Parameter list]" << std::endl;
    osst << "  (Tc, c1, c2, theta0, theta1, theta2)." << std::endl;
    osst << "  Shared : Tc and c1." << std::endl;
    return osst.str();
  }

  int num_e() const { // (L, T, A, Delta A)
    return 4;
  }

  int num_p() const { return 6; }

  int num_p_shared() const { return 2; }

  int num() const { return Datas.size(); }

  /**
     @note The format of data unit is @f$ (L, T, A, \delta A) @f$.
  */
  void add(const std::vector<double> &Xdata) {
    assert(Xdata.size() == static_cast<unsigned int>(num_e()));
    double L = Xdata[0];
    if (first_time.size() == 0)
      LMIN = LMAX = L;
    if (LMAX < L)
      LMAX = L;
    if (LMIN > L)
      LMIN = L;
    if (first_time.find(L) == first_time.end()) {
      tmax[L] = tmin[L] = Xdata[1];
      amax[L] = amin[L] = Xdata[2];
      first_time[L] = true;
    } else {
      if (tmax[L] < Xdata[1])
        tmax[L] = Xdata[1];
      if (tmin[L] > Xdata[1])
        tmin[L] = Xdata[1];
      if (amax[L] < Xdata[2])
        amax[L] = Xdata[2];
      if (amin[L] > Xdata[2])
        amin[L] = Xdata[2];
    }
    RX = (tmax[LMAX] - tmin[LMAX]) / 2;
    Y0 = (amax[LMAX] + amin[LMAX]) / 2;
    RY = amax[LMAX] - Y0;
    Datas.push_back(Xdata);
  }

  void get(int index, std::vector<double> &Xdata) const {
    Xdata.resize(num_e());
    std::copy(Datas.at(index).begin(), Datas.at(index).end(), Xdata.begin());
  }

  const std::vector<double> &get(int index) const { return Datas.at(index); }

  std::vector<double> &at(int index) { return Datas.at(index); }

  double prior(const std::vector<double> &Params) const { return 0; }

  void grad_prior(const std::vector<double> &Params,
                  std::vector<double> &Grad_vec) const {
    Grad_vec.resize(Params.size());
    std::fill(Grad_vec.begin(), Grad_vec.end(), 0);
  }

  void gram(const std::vector<double> &Params,
            std::vector<double> &Xgram) const {
    int M = Datas.size();
    Xgram.resize(M * M);
    for (int i = 0; i < M; ++i) {
      Xgram[i + i * M] = kernel(Params, get(i), get(i), true);
      for (int j = i + 1; j < M; ++j)
        Xgram[i + j * M] = Xgram[j + i * M] =
            kernel(Params, get(i), get(j), false);
    }
  }

  void differentiated_gram(const std::vector<double> &Params, int p_index,
                           std::vector<double> &Xdgram) const {
    int M = Datas.size();
    Xdgram.resize(M * M);
    for (int i = 0; i < M; ++i) {
      Xdgram[i + i * M] =
          differentiated_kernel(Params, p_index, get(i), get(i), true);
      for (int j = i + 1; j < M; ++j)
        Xdgram[i + j * M] = Xdgram[j + i * M] =
            differentiated_kernel(Params, p_index, get(i), get(j), false);
    }
  }

  void gram_vector(const std::vector<double> &Params,
                   const std::vector<double> &Xdata,
                   std::vector<double> &Gram_vec) const {
    int M = Datas.size();
    Gram_vec.resize(M);
    for (int i = 0; i < M; ++i)
      Gram_vec[i] = kernel(Params, Xdata, get(i), false);
  }

  void gram_vector_regression(const std::vector<double> &Params,
                              const std::vector<double> &X_regression,
                              std::vector<double> &Gram_vec) const {
    int M = Datas.size();
    Gram_vec.resize(M);
    for (int i = 0; i < M; ++i) {
      std::vector<double> xc;
      convert(Params, get(i), xc);
      Gram_vec[i] = kernel_regression(Params, X_regression, xc, false);
    }
  }

  double kernel(const std::vector<double> &Params,
                const std::vector<double> &X1, const std::vector<double> &X2,
                bool diagonal = false) const {
    double x = Params[4] * Params[4] *
               std::exp(-(value_x(Params, X1) - value_x(Params, X2)) *
                        (value_x(Params, X1) - value_x(Params, X2)) /
                        (2 * Params[5] * Params[5]));
    if (diagonal)
      return x + value_e(Params, X1) * value_e(Params, X1) +
             Params[3] * Params[3];
    else
      return x;
  }

  double kernel_regression(const std::vector<double> &Params,
                           const std::vector<double> &X_regression_1,
                           const std::vector<double> &X_regression_2,
                           bool diagonal = false) const {
    double x = Params[4] * Params[4] *
               std::exp(-(X_regression_1[2] - X_regression_2[2]) *
                        (X_regression_1[2] - X_regression_2[2]) /
                        (2 * Params[5] * Params[5]));
    if (diagonal)
      return x + X_regression_1[1] * X_regression_1[1] + Params[3] * Params[3];
    else
      return x;
  }

  double differentiated_kernel(const std::vector<double> &Params, int p_index,
                               const std::vector<double> &X1,
                               const std::vector<double> &X2,
                               bool diagonal = false) const {
    const double &THETA0 = Params[3];
    const double &THETA1 = Params[4];
    const double &THETA2 = Params[5];
    if (p_index <= 2) {
      if (diagonal)
        return 2 * value_e(Params, X1) * value_de(Params, p_index, X1);
      else
        return kernel(Params, X1, X2, false) *
               (-(value_x(Params, X1) - value_x(Params, X2)) /
                (THETA2 * THETA2)) *
               (value_dx(Params, p_index, X1) - value_dx(Params, p_index, X2));
    } else {
      if (p_index == 3 && diagonal)
        return 2 * THETA0;
      else if (p_index == 4)
        return kernel(Params, X1, X2, false) * 2 / THETA1;
      else if (p_index == 5)
        return kernel(Params, X1, X2, false) *
               ((value_x(Params, X1) - value_x(Params, X2)) *
                (value_x(Params, X1) - value_x(Params, X2)) /
                (THETA2 * THETA2 * THETA2));
      else
        return 0;
    }
  }

  /**
     @note @f$ x_{regression} = (Y, E, X). @f$
  */
  void convert(const std::vector<double> &Params, const std::vector<double> &X,
               std::vector<double> &X_regression) const {
    assert(X.size() == static_cast<unsigned int>(num_e()));
    X_regression.resize(3);
    X_regression[0] = value_y(Params, X);
    X_regression[1] = value_e(Params, X);
    X_regression[2] = value_x(Params, X);
  };

  void differentiated_convert(const std::vector<double> &Params, int p_index,
                              const std::vector<double> &X,
                              std::vector<double> &Dx_regression) const {
    assert(X.size() == static_cast<unsigned int>(num_e()));
    Dx_regression.resize(3);
    Dx_regression[0] = value_dy(Params, p_index, X);
    Dx_regression[1] = value_de(Params, p_index, X);
    Dx_regression[2] = value_dx(Params, p_index, X);
  }

  void reverse_convert(const std::vector<double> &Params,
                       const std::vector<double> &X_regression,
                       std::vector<double> &X) const {
    assert(X_regression.size() == 3);
    assert(X.size() == static_cast<unsigned int>(num_e()));
    X[2] = (X_regression[0] * RY + Y0) * (std::pow(X[0] / LMAX, Params[2]));
    X[3] = (X_regression[1] * RY) * (std::pow(X[0] / LMAX, Params[2]));
  };

  void get_info(std::map<std::string, double> &info) const {
    info["LMIN"] = LMIN;
    info["LMAX"] = LMAX;
    info["RX"] = RX;
    info["RY"] = RY;
    info["Y0"] = Y0;
    info["TMIN"] = tmin.find(LMAX)->second;
    info["TMAX"] = tmax.find(LMAX)->second;
  }

  void initialize_parameters(std::vector<double> &Params) const {
    Params.resize(num_p());
    Params[0] = 0;
    Params[1] = 0;
    Params[2] = 0;
    Params[3] = 1;
    Params[4] = 1;
    Params[5] = 1;
  }

private:
  double value_x(const std::vector<double> &Params,
                 const std::vector<double> &X) const {
    return (X[1] - Params[0]) * std::pow(X[0] / LMAX, Params[1]) / RX;
  }

  double value_dx(const std::vector<double> &Params, int p_index,
                  const std::vector<double> &X) const {
    if (p_index == 0)
      return -std::pow(X[0] / LMAX, Params[1]) / RX;
    else if (p_index == 1)
      return (X[1] - Params[0]) * std::pow(X[0] / LMAX, Params[1]) / RX *
             std::log(X[0] / LMAX);
    else
      return 0;
  }

  double value_y(const std::vector<double> &Params,
                 const std::vector<double> &X) const {
    return (X[2] * std::pow(LMAX / X[0], Params[2]) - Y0) / RY;
  }

  double value_dy(const std::vector<double> &Params, int p_index,
                  const std::vector<double> &X) const {
    if (p_index == 2)
      return X[2] * std::pow(LMAX / X[0], Params[2]) / RY *
             std::log(LMAX / X[0]);
    else
      return 0;
  }

  double value_e(const std::vector<double> &Params,
                 const std::vector<double> &X) const {
    return X[3] * std::pow(LMAX / X[0], Params[2]) / RY;
  }

  double value_de(const std::vector<double> &Params, int p_index,
                  const std::vector<double> &X) const {
    if (p_index == 2)
      return X[3] * std::pow(LMAX / X[0], Params[2]) / RY *
             std::log(LMAX / X[0]);
    else
      return 0;
  }

  std::vector<std::vector<double> > Datas;
  double LMIN, LMAX, RX, RY, Y0;
  std::map<int, bool> first_time;
  std::map<int, double> tmax, tmin, amax, amin;
};

/**
   @class BSA::DataSet_C
   @brief Data set for GPR::Regression class with correction to scaling

   The finite-size scaling form is written as
   @f[
   A(T, L) = L^{c_2} F[ ( T - T_c ) L^{c_1}, L^{-c_3} ],
   @f] where @f$ A @f$ is an observable. The triplet of a point
   is defined as
   @f[
   X_1 = (T - T_c ) (L/L_{MAX})^{c_1} / R_X, X_2 = (L/L_{MIN})^{-c_3}, Y = (A /
   (L/L_{MAX})^{c_2} - Y_0)/R_Y, E = \delta A/ (L/L_{MAX})^{c_2}/R_Y,
   @f]
   where @f$ \delta A @f$ is an error of @f$ A @f$ and @f$ L_{MAX} @f$ is the
   largest @f$ L @f$.
   Scaling factor @f$ R_X @f$ is defined so that the width of X for @f$ L_{MAX}
   @f$ is 2.
   Scaling factor @f$ R_Y @f$ and @f$ Y_0 @f$ is defined so that Y for @f$
   L_{MAX}@f$ is in [-1:1].
   The data ansatz is @f[ Y \sim F(X_1, X_2) \pm E. @f]

   Kernel function is written as
   @f[
   k_G(i, j) = \delta_{ij} (E(i)^2 + \theta_0^2)
   + \theta_1^2 \exp\left[ - \frac{|X_1(i)- X_1(j)|^2}{2\theta_2^2}\right]
   + \theta_3^2 \exp\left[ - \frac{|X_1(i)- X_1(j)|^2}{2\theta_4^2}\right]
   X_2(i) X_2(j).
   @f]

   Parameters are defined as
   Params[0] = @f$ T_c @f$,
   Params[1] = @f$ c_1 @f$,
   Params[2] = @f$ c_3 @f$,
   Params[3] = @f$ c_2 @f$,
   Params[4] = @f$ \theta_0 @f$,
   Params[5] = @f$ \theta_1 @f$,
   Params[6] = @f$ \theta_2 @f$,
   Params[7] = @f$ \theta_3 @f$, and
   Params[8] = @f$ \theta_4 @f$.
   Shared parameters are @f$ T_c, c_1 @f$ and @f$ c_3 @f$.
*/
class BSA::DataSet_C {
public:
  /**
     @brief Description
   */
  static std::string description() {
    std::ostringstream osst;
    osst << "[Finite-size scaling form]" << std::endl;
    osst << "  A(T, L) = L^c2 F[ ( T - Tc ) L^c1, 1 / L^c3]," << std::endl;
    osst << "[Data ansatz]" << std::endl;
    osst << "  Y ~ F(X1, X2) +- E," << std::endl;
    osst << "  X1 = (T - Tc) L^c1, X2 = 1 / L^c3, Y = A / L^c2, E = dA / L^c2."
         << std::endl;
    osst << "[Kernel function]" << std::endl;
    osst << "  k_G(i, j) = delta_{ij} (E(i)^2 + theta0^2) + theta1^2 exp( - "
            "|X1(i)- X1(j)|^2 / 2 theta2^2 )"
            "+ theta3^2 exp( - |X1(i)- X1(j)|^2 / 2 theta4^2 ) X2(i) X2(j)."
         << std::endl;
    osst << "[Parameter list]" << std::endl;
    osst << "  (Tc, c1, c3, c2, theta0, theta1, theta2, theta3, theta4)."
         << std::endl;
    osst << "  Shared : Tc, c1, and c3." << std::endl;
    return osst.str();
  }

  int num_e() const { // (L, T, A, Delta A)
    return 4;
  }

  int num_p() const { return 9; }

  int num_p_shared() const { return 3; }

  int num() const { return Datas.size(); }

  /**
     @note The format of data unit is @f$ (L, T, A, \delta A) @f$.
  */
  void add(const std::vector<double> &Xdata) {
    assert(Xdata.size() == static_cast<unsigned int>(num_e()));
    double L = Xdata[0];
    if (first_time.size() == 0)
      LMIN = LMAX = L;
    if (LMAX < L)
      LMAX = L;
    if (LMIN > L)
      LMIN = L;
    if (first_time.find(L) == first_time.end()) {
      tmax[L] = tmin[L] = Xdata[1];
      amax[L] = amin[L] = Xdata[2];
      first_time[L] = true;
    } else {
      if (tmax[L] < Xdata[1])
        tmax[L] = Xdata[1];
      if (tmin[L] > Xdata[1])
        tmin[L] = Xdata[1];
      if (amax[L] < Xdata[2])
        amax[L] = Xdata[2];
      if (amin[L] > Xdata[2])
        amin[L] = Xdata[2];
    }
    RX = (tmax[LMAX] - tmin[LMAX]) / 2;
    Y0 = (amax[LMAX] + amin[LMAX]) / 2;
    RY = amax[LMAX] - Y0;
    Datas.push_back(Xdata);
  }

  void get(int index, std::vector<double> &Xdata) const {
    Xdata.resize(num_e());
    std::copy(Datas.at(index).begin(), Datas.at(index).end(), Xdata.begin());
  }

  const std::vector<double> &get(int index) const { return Datas.at(index); }

  std::vector<double> &at(int index) { return Datas.at(index); }

  double prior(const std::vector<double> &Params) const { return 0; }

  void grad_prior(const std::vector<double> &Params,
                  std::vector<double> &Grad_vec) const {
    Grad_vec.resize(Params.size());
    std::fill(Grad_vec.begin(), Grad_vec.end(), 0);
  }

  void gram(const std::vector<double> &Params,
            std::vector<double> &Xgram) const {
    int M = Datas.size();
    Xgram.resize(M * M);
    for (int i = 0; i < M; ++i) {
      Xgram[i + i * M] = kernel(Params, get(i), get(i), true);
      for (int j = i + 1; j < M; ++j)
        Xgram[i + j * M] = Xgram[j + i * M] =
            kernel(Params, get(i), get(j), false);
    }
  }

  void differentiated_gram(const std::vector<double> &Params, int p_index,
                           std::vector<double> &Xdgram) const {
    int M = Datas.size();
    Xdgram.resize(M * M);
    for (int i = 0; i < M; ++i) {
      Xdgram[i + i * M] =
          differentiated_kernel(Params, p_index, get(i), get(i), true);
      for (int j = i + 1; j < M; ++j)
        Xdgram[i + j * M] = Xdgram[j + i * M] =
            differentiated_kernel(Params, p_index, get(i), get(j), false);
    }
  }

  void gram_vector(const std::vector<double> &Params,
                   const std::vector<double> &Xdata,
                   std::vector<double> &Gram_vec) const {
    int M = Datas.size();
    Gram_vec.resize(M);
    for (int i = 0; i < M; ++i)
      Gram_vec[i] = kernel(Params, Xdata, get(i), false);
  }

  void gram_vector_regression(const std::vector<double> &Params,
                              const std::vector<double> &X_regression,
                              std::vector<double> &Gram_vec) const {
    int M = Datas.size();
    Gram_vec.resize(M);
    for (int i = 0; i < M; ++i) {
      std::vector<double> xc;
      convert(Params, get(i), xc);
      Gram_vec[i] = kernel_regression(Params, X_regression, xc, false);
    }
  }

  double kernel(const std::vector<double> &Params,
                const std::vector<double> &X1, const std::vector<double> &X2,
                bool diagonal = false) const {
    double x = 0;
    x = Params[5] * Params[5] *
            std::exp(-(value_x(Params, X1) - value_x(Params, X2)) *
                     (value_x(Params, X1) - value_x(Params, X2)) /
                     (2 * Params[6] * Params[6])) +
        Params[7] * Params[7] *
            std::exp(-(value_x(Params, X1) - value_x(Params, X2)) *
                     (value_x(Params, X1) - value_x(Params, X2)) /
                     (2 * Params[8] * Params[8])) *
            value_x2(Params, X1) * value_x2(Params, X2);
    if (diagonal)
      return x + value_e(Params, X1) * value_e(Params, X1) +
             Params[4] * Params[4];
    else
      return x;
  }

  double kernel_regression(const std::vector<double> &Params,
                           const std::vector<double> &X_regression_1,
                           const std::vector<double> &X_regression_2,
                           bool diagonal = false) const {
    double x = 0;
    x = Params[5] * Params[5] *
            std::exp(-(X_regression_1[2] - X_regression_2[2]) *
                     (X_regression_1[2] - X_regression_2[2]) /
                     (2 * Params[6] * Params[6])) +
        Params[7] * Params[7] *
            std::exp(-(X_regression_1[2] - X_regression_2[2]) *
                     (X_regression_1[2] - X_regression_2[2]) /
                     (2 * Params[8] * Params[8])) *
            X_regression_1[3] * X_regression_2[3];
    if (diagonal)
      return x + X_regression_1[1] * X_regression_1[1] + Params[4] * Params[4];
    else
      return x;
  }

  double differentiated_kernel(const std::vector<double> &Params, int p_index,
                               const std::vector<double> &X1,
                               const std::vector<double> &X2,
                               bool diagonal = false) const {
    const double &THETA0 = Params[4];
    const double &THETA1 = Params[5];
    const double &THETA2 = Params[6];
    const double &THETA3 = Params[7];
    const double &THETA4 = Params[8];
    const double x1a = value_x(Params, X1);
    const double x2a = value_x2(Params, X1);
    const double x1b = value_x(Params, X2);
    const double x2b = value_x2(Params, X2);
    if (p_index <= 1 || p_index == 3) {
      if (diagonal)
        return 2 * value_e(Params, X1) * value_de(Params, p_index, X1);
      else
        return THETA1 * THETA1 * std::exp(-(x1a - x1b) * (x1a - x1b) /
                                          (2 * THETA2 * THETA2)) *
                   (-(x1a - x1b) / (THETA2 * THETA2)) *
                   (value_dx(Params, p_index, X1) -
                    value_dx(Params, p_index, X2)) +
               THETA3 * THETA3 * std::exp(-(x1a - x1b) * (x1a - x1b) /
                                          (2 * THETA4 * THETA4)) *
                   x2a * x2b * (-(x1a - x1b) / (THETA4 * THETA4)) *
                   (value_dx(Params, p_index, X1) -
                    value_dx(Params, p_index, X2));
    } else {
      if (p_index == 5)
        return 2 * THETA1 *
               std::exp(-(x1a - x1b) * (x1a - x1b) / (2 * THETA2 * THETA2));
      else if (p_index == 6)
        return THETA1 * THETA1 *
               std::exp(-(x1a - x1b) * (x1a - x1b) / (2 * THETA2 * THETA2)) *
               ((x1a - x1b) * (x1a - x1b) / (THETA2 * THETA2 * THETA2));
      else if (p_index == 4 && diagonal)
        return 2 * THETA0;
      else if (p_index == 2)
        return THETA3 * THETA3 *
               std::exp(-(x1a - x1b) * (x1a - x1b) / (2 * THETA4 * THETA4)) *
               (value_dx2(Params, p_index, X1) * x2b +
                x2a * value_dx2(Params, p_index, X2));
      else if (p_index == 7)
        return 2 * THETA3 *
               std::exp(-(x1a - x1b) * (x1a - x1b) / (2 * THETA4 * THETA4)) *
               (x2a * x2b);
      else if (p_index == 8)
        return THETA3 * THETA3 *
               std::exp(-(x1a - x1b) * (x1a - x1b) / (2 * THETA4 * THETA4)) *
               (x2a * x2b) *
               ((x1a - x1b) * (x1a - x1b) / (THETA4 * THETA4 * THETA4));
      else
        return 0;
    }
  };

  /**
     @note @f$ x_{regression} = (Y, E, X_1, X_2). @f$
  */
  void convert(const std::vector<double> &Params, const std::vector<double> &X,
               std::vector<double> &X_regression) const {
    assert(X.size() == static_cast<unsigned int>(num_e()));
    X_regression.resize(4);
    X_regression[0] = value_y(Params, X);
    X_regression[1] = value_e(Params, X);
    X_regression[2] = value_x(Params, X);
    X_regression[3] = value_x2(Params, X);
  };

  void differentiated_convert(const std::vector<double> &Params, int p_index,
                              const std::vector<double> &X,
                              std::vector<double> &Dx_regression) const {
    assert(X.size() == static_cast<unsigned int>(num_e()));
    Dx_regression.resize(4);
    Dx_regression[0] = value_dy(Params, p_index, X);
    Dx_regression[1] = value_de(Params, p_index, X);
    Dx_regression[2] = value_dx(Params, p_index, X);
    Dx_regression[3] = value_dx2(Params, p_index, X);
  }

  void reverse_convert(const std::vector<double> &Params,
                       const std::vector<double> &X_regression,
                       std::vector<double> &X) const {
    assert(X_regression.size() == 4);
    assert(X.size() == static_cast<unsigned int>(num_e()));
    X[2] = (X_regression[0] * RY + Y0) * (std::pow(X[0] / LMAX, Params[3]));
    X[3] = (X_regression[1] * RY) * (std::pow(X[0] / LMAX, Params[3]));
  };

  void get_info(std::map<std::string, double> &info) const {
    info["LMIN"] = LMIN;
    info["LMAX"] = LMAX;
    info["RX"] = RX;
    info["RY"] = RY;
    info["Y0"] = Y0;
    info["TMIN"] = tmin.find(LMAX)->second;
    info["TMAX"] = tmax.find(LMAX)->second;
  }

  void initialize_parameters(std::vector<double> &Params) const {
    Params.resize(num_p());
    Params[0] = 0;
    Params[1] = 0;
    Params[2] = 1;
    Params[3] = 0;
    Params[4] = 1;
    Params[5] = 1;
    Params[6] = 1;
    Params[7] = 1;
    Params[8] = 1;
  }

private:
  double value_x(const std::vector<double> &Params,
                 const std::vector<double> &X) const {
    return (X[1] - Params[0]) * std::pow(X[0] / LMAX, Params[1]) / RX;
  }

  double value_dx(const std::vector<double> &Params, int p_index,
                  const std::vector<double> &X) const {
    if (p_index == 0)
      return -std::pow(X[0] / LMAX, Params[1]) / RX;
    else if (p_index == 1)
      return (X[1] - Params[0]) * std::pow(X[0] / LMAX, Params[1]) / RX *
             std::log(X[0] / LMAX);
    else
      return 0;
  }

  double value_x2(const std::vector<double> &Params,
                  const std::vector<double> &X) const {
    return std::pow(LMIN / X[0], Params[2]);
  }

  double value_dx2(const std::vector<double> &Params, int p_index,
                   const std::vector<double> &X) const {
    if (p_index == 2)
      return std::pow(LMIN / X[0], Params[2]) * std::log(LMIN / X[0]);
    else
      return 0;
  }

  double value_y(const std::vector<double> &Params,
                 const std::vector<double> &X) const {
    return (X[2] * std::pow(LMAX / X[0], Params[3]) - Y0) / RY;
  }

  double value_dy(const std::vector<double> &Params, int p_index,
                  const std::vector<double> &X) const {
    if (p_index == 3)
      return X[2] * std::pow(LMAX / X[0], Params[3]) / RY *
             std::log(LMAX / X[0]);
    else
      return 0;
  }

  double value_e(const std::vector<double> &Params,
                 const std::vector<double> &X) const {
    return X[3] * std::pow(LMAX / X[0], Params[3]) / RY;
  }

  double value_de(const std::vector<double> &Params, int p_index,
                  const std::vector<double> &X) const {
    if (p_index == 3)
      return X[3] * std::pow(LMAX / X[0], Params[3]) / RY *
             std::log(LMAX / X[0]);
    else
      return 0;
  }

  std::vector<std::vector<double> > Datas;
  double LMIN, LMAX, RX, RY, Y0;
  std::map<int, bool> first_time;
  std::map<int, double> tmax, tmin, amax, amin;
};

#endif
