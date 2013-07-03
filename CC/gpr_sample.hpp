#ifndef GPR_SAMPLE_H
#define GPR_SAMPLE_H
/* gpr_sample.hpp
 *
 * Copyright 2013, Kenji Harada
 * Released under the MIT and GPLv3 licenses.
 *
 * To spread the Bayesian scaling analysis method,
 * I hope you will cite the following original paper:
 *   Kenji Harada, Physical Review E 84 (2011) 056704.
 */
/**
   @file gpr_sample.hpp
   @brief Sample classes for Gaussian process regression
*/
#include <vector>
#include <string>

namespace GPR {
  class Data;
  class Kernel;
};

/**
   @class GPR::Data
   @brief Sample class of data for Gaussian process regressioin.

  Store data for Bayesian scaling analysis.
  The triplet of a data point is written as @f$ (X_i, Y_i, E_i).@f$
  The scaling ansatz is written as @f[ Y \sim F(X) \pm E. @f]
*/
class GPR::Data {
public:
  //! Return the number of data points
  int npoints() const;
  //! Return the number of physical parameters
  int nparams() const;
  /** @brief Calculate @f$ X_i @f$ for physical parameters
      @param[in] i Index of @f$ X_i@f$
      @param[in] p_params Physical parameters
      @return @f$ X_i @f$
  */
  double x(int i, const std::vector<double> &p_params) const;
  /** @brief Calculate @f$ Y_i @f$ for physical parameters
      @param[in] i Index of @f$ Y_i@f$
      @param[in] p_params Physical parameters
      @return @f$ Y_i @f$
  */
  double y(int i, const std::vector<double> &p_params) const;
  /** @brief Calculate @f$ E_i @f$ for physical parameters
      @param[in] i Index of @f$ E_i@f$
      @param[in] p_params Physical parameters
      @return @f$ E_i @f$
  */
  double e(int i, const std::vector<double> &p_params) const;
  /** @brief Calculate @f$ \vec{X} @f$ for physical parameters
      @param[out] vx @f$ \vec{X} @f$
      @param[in] p_params Physical parameters
  */
  void x(std::vector<double> &vx, const std::vector<double> &p_params) const;
  /** @brief Calculate @f$ \vec{Y} @f$ for physical parameters
      @param[out] vy @f$ \vec{Y} @f$
      @param[in] p_params Physical parameters
  */
  void y(std::vector<double> &vy, const std::vector<double> &p_params) const;
  /** @brief Calculate @f$ \vec{E} @f$ for physical parameters
      @param[out] ve @f$ \vec{E} @f$
      @param[in] p_params Physical parameters
  */
  void e(std::vector<double> &ve, const std::vector<double> &p_params) const;
  /** @brief Calculate @f$ d\vec{X}/d\theta @f$ for a physical parameter @f$ \theta @f$
      @param[out] dx @f$ d\vec{X}/d\theta @f$
      @param[in] pi Index of physical parameter @f$ \theta @f$
      @param[in] p_params Physical parameters
  */
  void dx(std::vector<double> &dx, int pi, const std::vector<double> &p_params) const;
  /** @brief Calculate @f$ d\vec{Y}/d\theta @f$ for a physical parameter @f$ \theta @f$
      @param[out] dy @f$ d\vec{Y}/d\theta @f$
      @param[in] pi Index of physical parameter @f$ \theta @f$
      @param[in] p_params Physical parameters
  */
  void dy(std::vector<double> &dy, int pi, const std::vector<double> &p_params) const;
  /** @brief Calculate @f$ d\vec{E}/d\theta @f$ for a physical parameter @f$ \theta @f$
      @param[out] de @f$ d\vec{E}/d\theta @f$
      @param[in] pi Index of physical parameter @f$ \theta @f$
      @param[in] p_params Physical parameters
  */
  void de(std::vector<double> &de, int pi, const std::vector<double> &p_params) const;
  //! Description of data
  std::string description(std::string header) const;
};

/** @class GPR::Kernel
   @brief Sample class of kernel function for Gaussical process regression

   This class defines a kernel function @f$ k(i, j) @f$ and the derivatives.
*/
class GPR::Kernel {
public:
  //! Return the number of hyper parameters
  static int nparams();
  /** @brief Calculate the kernel function @f$ k(i, j) @f$

      @param[in] i @f$ i @f$
      @param[in] j @f$ j @f$
      @param[in] vx @f$ \vec{X} @f$
      @param[in] ve @f$ \vec{E} @f$
      @param[in] h_params Hyper parameters
      @return The value of @f$ k(i,j) @f$
  */
  static double k(int i, int j, const std::vector<double> &vx, const std::vector<double> &ve, const std::vector<double> &h_params);

  /** @brief Calculate @f$ k(x_1, x_2) @f$

      @param[in] x1 @f$ x_1 @f$
      @param[in] x2 @f$ x_2 @f$
      @param[in] h_params Hyper parameters
      @return The value of @f$ k(x_1, x_2) @f$
   */
  static double k(double x1, double x2, const std::vector<double>& h_params);

  /** @brief Calculate @f$ dk(i,j)/d\theta @f$

      @param[in] pi Index of hyper parameter @f$ \theta @f$
      @param[in] i @f$ i @f$
      @param[in] j @f$ j @f$
      @param[in] vx @f$ \vec{X} @f$
      @param[in] ve @f$ \vec{E} @f$
      @param[in] h_params Hyper parameters
      @return The value of @f$ dk(i,j)/d\theta @f$
  */
  static double dk(int pi, int i, int j, const std::vector<double> &vx, const std::vector<double> &ve, const std::vector<double> &h_params);
  /** @brief Calculate @f$ dk(i,j)/dx_k @f$

      @param[in] k Index of @f$ x_k @f$
      @param[in] i @f$ i @f$
      @param[in] j @f$ j @f$
      @param[in] vx @f$ \vec{X} @f$
      @param[in] ve @f$ \vec{E} @f$
      @param[in] h_params Hyper parameters
      @return The value of @f$ dk(i,j)/dx_k @f$
  */
  static double dkx(int k, int i, int j, const std::vector<double> &vx, const std::vector<double> &ve, const std::vector<double> &h_params);
  /** @brief Calculate @f$ dk(i,j)/de_k @f$

      @param[in] k Index of @f$ e_k @f$
      @param[in] i @f$ i @f$
      @param[in] j @f$ j @f$
      @param[in] vx @f$ \vec{X} @f$
      @param[in] ve @f$ \vec{E} @f$
      @param[in] h_params Hyper parameters
      @return The value of @f$ dk(i,j)/de_k @f$
  */
  static double dke(int k, int i, int j, const std::vector<double> &vx, const std::vector<double> &ve, const std::vector<double> &h_params);
  //! Description of kernel function
  static std::string description(std::string header);
};

// Dumy definitions
int GPR::Data::npoints() const {
  return 0;
}
int GPR::Data::nparams() const {
  return 0;
}
double GPR::Data::x(int i, const std::vector<double> &p_params) const {
  return 0e0;
}
double GPR::Data::y(int i, const std::vector<double> &p_params) const {
  return 0e0;
}
double GPR::Data::e(int i, const std::vector<double> &p_params) const {
  return 0e0;
}
void GPR::Data::x(std::vector<double> &vx, const std::vector<double> &p_params) const {
};
void GPR::Data::y(std::vector<double> &vy, const std::vector<double> &p_params) const {
};
void GPR::Data::e(std::vector<double> &ve, const std::vector<double> &p_params) const {
};
void GPR::Data::dx(std::vector<double> &dx, int pi, const std::vector<double> &p_params) const {
};
void GPR::Data::dy(std::vector<double> &dy, int pi, const std::vector<double> &p_params) const {
};
void GPR::Data::de(std::vector<double> &de, int pi, const std::vector<double> &p_params) const {
};
std::string GPR::Data::description(std::string header) const {
  std::string x;
  return x;
}
int GPR::Kernel::nparams(){
  return 0;
}
double GPR::Kernel::k(int i, int j, const std::vector<double> &vx, const std::vector<double> &ve, const std::vector<double> &h_params){
  return 0e0;
}
double GPR::Kernel::k(double x1, double x2, const std::vector<double>& h_params){
  return 0e0;
}
double GPR::Kernel::dk(int pi, int i, int j, const std::vector<double> &vx, const std::vector<double> &ve, const std::vector<double> &h_params){
  return 0e0;
}
double GPR::Kernel::dkx(int k, int i, int j, const std::vector<double> &vx, const std::vector<double> &ve, const std::vector<double> &h_params){
  return 0e0;
}
double GPR::Kernel::dke(int k, int i, int j, const std::vector<double> &vx, const std::vector<double> &ve, const std::vector<double> &h_params){
  return 0e0;
}
std::string GPR::Kernel::description(std::string header){
  std::string x;
  return x;
}
#endif
