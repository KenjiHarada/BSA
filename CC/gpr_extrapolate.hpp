#ifndef GPR_EXTRAPOLATE_HPP
#define GPR_EXTRAPOLATE_HPP
/* gpr_extrapolate.hpp
 *
 * Copyright 2013, Kenji Harada
 * Released under the MIT and GPLv3 licenses.
 *
 * To spread the Bayesian scaling analysis method,
 * I hope you will cite the following original paper:
 *   Kenji Harada, Physical Review E 84 (2011) 056704.
 */
/** @file gpr_extrapolate.hpp
    @brief Classes for extrapolation
*/
#include <vector>
#include <cmath>
#include <string>
#include <sstream>

/**
   @namespace GPR::EXTRAPOLATE
   @brief Extrapolation
 */
namespace GPR {
  namespace EXTRAPOLATE {
    class Data;
    class Gaussian_Kernel;
    const static std::string version = "0.1";
  }
}

/**
  @class GPR::EXTRAPOLATE::Data
  @brief Data for Bayesian extrapolation
*/
class GPR::EXTRAPOLATE::Data {
private:
  std::vector<double> data_x;
  std::vector<double> data_y;
  std::vector<double> data_e;
  int ndata;
  double xmin, xmax;
  double ymin, ymax;
  double xmid, xw;
  double ymid, yw;
public:
  //! Make an empty data set
  Data(){
    reset();
  }
  //! add new data
  void set(double x, double y, double e){
    if (ndata == 0) {
      xmin = xmax = x;
      ymin = ymax = y;
    }else{
      if (xmax < x) xmax = x;
      if (xmin > x) xmin = x;
      if (ymax < y) ymax = y;
      if (ymin > y) ymin = y;
    }
    data_x.push_back(x);
    data_y.push_back(y);
    data_e.push_back(e);
    xmid = (xmax + xmin) / 2;
    xw = (xmax - xmin);
    ymid = (ymax + ymin) / 2;
    yw = (ymax - ymin);
    ++ndata;
  }
  //! reset stored data
  void reset(){
    ndata = 0;
    data_x.clear();
    data_y.clear();
    data_e.clear();
  }
  //! Return the number of data points
  int npoints() const {
    return ndata;
  }
  //! Return the number of physical parameters
  int nparams() const {
    return 0;
  }
  /** @brief Calculate @f$ X_i @f$ for physical parameters
      @param[in] i Index of @f$ X_i@f$
      @param[in] p_params Physical parameters
      @return @f$ X_i @f$
  */
  double x(int i, const std::vector<double> &p_params) const {
    return (data_x[i] - xmid) / xw;
  }
  /** @brief Calculate @f$ Y_i @f$ for physical parameters
      @param[in] i Index of @f$ Y_i@f$
      @param[in] p_params Physical parameters
      @return @f$ Y_i @f$
  */
  double y(int i, const std::vector<double> &p_params) const {
    return (data_y[i] - ymid) / yw;
  }
  /** @brief Calculate @f$ E_i @f$ for physical parameters
      @param[in] i Index of @f$ E_i@f$
      @param[in] p_params Physical parameters
      @return @f$ E_i @f$
  */
  double e(int i, const std::vector<double> &p_params) const {
    return data_e[i] / yw;
  }
  /** @brief Calculate @f$ \vec{X} @f$ for physical parameters
      @param[out] vx @f$ \vec{X} @f$
      @param[in] p_params Physical parameters
  */
  void x(std::vector<double> &vx, const std::vector<double> &p_params) const {
    for (int i = 0; i < ndata; ++i)
      vx[i] = (data_x[i] - xmid) / xw;
  }
  /** @brief Calculate @f$ \vec{Y} @f$ for physical parameters
      @param[out] vy @f$ \vec{Y} @f$
      @param[in] p_params Physical parameters
  */
  void y(std::vector<double> &vy, const std::vector<double> &p_params) const {
    for (int i = 0; i < ndata; ++i)
      vy[i] = (data_y[i] - ymid) / yw;
  }
  /** @brief Calculate @f$ \vec{E} @f$ for physical parameters
      @param[out] ve @f$ \vec{E} @f$
      @param[in] p_params Physical parameters
  */
  void e(std::vector<double> &ve, const std::vector<double> &p_params) const {
    for (int i = 0; i < ndata; ++i)
      ve[i] = data_e[i] / yw;
  }
  /** @brief Calculate @f$ d\vec{X}/d\theta @f$ for a physical parameter @f$ \theta @f$
      @param[out] dx @f$ d\vec{X}/d\theta @f$
      @param[in] pi Index of physical parameter @f$ \theta @f$
      @param[in] p_params Physical parameters
  */
  void dx(std::vector<double> &dx, int pi, const std::vector<double> &p_params) const {
    for (int i = 0; i < ndata; ++i)
      dx[i] = 0;
  }
  /** @brief Calculate @f$ d\vec{Y}/d\theta @f$ for a physical parameter @f$ \theta @f$
      @param[out] dy @f$ d\vec{Y}/d\theta @f$
      @param[in] pi Index of physical parameter @f$ \theta @f$
      @param[in] p_params Physical parameters
  */
  void dy(std::vector<double> &dy, int pi, const std::vector<double> &p_params) const {
    for (int i = 0; i < ndata; ++i)
      dy[i] = 0;
  }
  /** @brief Calculate @f$ d\vec{E}/d\theta @f$ for a physical parameter @f$ \theta @f$
      @param[out] de @f$ d\vec{E}/d\theta @f$
      @param[in] pi Index of physical parameter @f$ \theta @f$
      @param[in] p_params Physical parameters
  */
  void de(std::vector<double> &de, int pi, const std::vector<double> &p_params) const {
    for (int i = 0; i < ndata; ++i)
      de[i] = 0;
  }
  //! Description of data
  std::string description(std::string header) const {
    std::stringstream ostr;
    ostr << header << "This class stores extrapolated data." << std::endl;
    return ostr.str();
  };
  //! Convert an original x to a normalized X
  double xtoX(double x) const {
    return (x - xmid) / xw;
  }
  //! Convert an original y to a normalized Y
  double ytoY(double y) const {
    return (y - ymid) / yw;
  }
  //! Convert an original e to a normalized E
  double etoE(double e) const {
    return e / yw;
  }
  //! Conver a normalized X to an original x
  double Xtox(double X) const {
    return X * xw + xmid;
  }
  //! Conver a normalized Y to an original y
  double Ytoy(double Y) const {
    return Y * yw + ymid;
  }
  //! Conver a normalized E to an original e
  double Etoe(double E) const {
    return E * yw;
  }
};

/**
   @class GPR::EXTRAPOLATE::Gaussian_Kernel
   @brief Gaussian kernel function for Bayesian extrapolation by Gaussian process regression

   This class defines the Gaussian kernel function @f$ k_G(i, j) @f$ and the derivatives.
   Kernel function is written as
   @f[
   k_G(i, j) = \delta_{ij} (E_i^2 + \theta_2^2)
   + \theta_0^2 \exp( - |X_i- X_j|^2 / 2\theta_1^2 ).
   @f]
   The hyper parameters are defined as
   h_params[0] = @f$ \theta_0 @f$,
   h_params[1] = @f$ \theta_1 @f$, and
   h_params[2] = @f$ \theta_2 @f$.
*/
class GPR::EXTRAPOLATE::Gaussian_Kernel {
public:
  //! Description of kernel function
  static std::string description(std::string header){
    std::stringstream ostr;
    ostr << header << "Gaussian kernel function:" << std::endl
         << header << "  k_G(i, j) = \\delta_{ij} (E_i^2 + \\theta_2^2)" << std::endl
         << header << "  + \\theta_0^2 \\exp( - |X_i- X_j|^2 / 2\\theta_1^2 )" << std::endl
         << header << "Hyper parameters:" << std::endl
         << header << "  h_params[0] = \\theta_0" << std::endl
         << header << "  h_params[1] = \\theta_1" << std::endl
         << header << "  h_params[2] = \\theta_2" << std::endl;
    return ostr.str();
  }

  //! Return the number of hyper parameters
  static int nparams() {
    return 3;
  };

  /** @brief Calculate the Gaussian kernel function @f$ k_G(i, j) @f$

      @param[in] i @f$ i @f$
      @param[in] j @f$ j @f$
      @param[in] vx @f$ \vec{X} @f$
      @param[in] ve @f$ \vec{E} @f$
      @param[in] h_params Hyper parameters
      @return The value of @f$ k_G(i,j) @f$
  */
  static double k(int i, int j, const std::vector<double> &vx, const std::vector<double> &ve, const std::vector<double> &h_params){
    if (i == j)
      return ve[i] * ve[i] + h_params[2] * h_params[2] + h_params[0] * h_params[0];
    else
      return h_params[0] * h_params[0] * exp((-0.5e0) * (vx[i] - vx[j]) * (vx[i] - vx[j]) / (h_params[1] * h_params[1]));
  };

  /** @brief Calculate @f$ k_G(x_1, x_2) @f$

      @param[in] x1 @f$ x_1 @f$
      @param[in] x2 @f$ x_2 @f$
      @param[in] h_params Hyper parameters
      @return The value of @f$ k_G(x_1, x_2) @f$
   */
  static double k(double x1, double x2, const std::vector<double>& h_params){
    return h_params[0] * h_params[0] * exp((-0.5e0) * (x1 - x2) * (x1 - x2) / (h_params[1] * h_params[1]));
  }

  /** @brief Calculate @f$ dk_G(i,j)/d\theta @f$

      @param[in] pi Index of hyper parameter @f$ \theta @f$
      @param[in] i @f$ i @f$
      @param[in] j @f$ j @f$
      @param[in] vx @f$ \vec{X} @f$
      @param[in] ve @f$ \vec{E} @f$
      @param[in] h_params Hyper parameters
      @return The value of @f$ dk_G(i,j)/d\theta @f$
  */
  static double dk(int pi, int i, int j, const std::vector<double> &vx, const std::vector<double> &ve, const std::vector<double> &h_params){
    switch (pi) {
    case 0:
      if (i == j)
        return 2 * h_params[0];
      else
        return 2 * h_params[0] * exp((-0.5e0) * (vx[i] - vx[j]) * (vx[i] - vx[j]) / (h_params[1] * h_params[1]));
    case 1:
      if (i == j)
        return 0e0;
      else
        return h_params[0] * h_params[0] * exp((-0.5e0) * (vx[i] - vx[j]) * (vx[i] - vx[j]) / (h_params[1] * h_params[1]))
               * ((vx[i] - vx[j]) * (vx[i] - vx[j]) / (h_params[1] * h_params[1] * h_params[1]));
    case 2:
      if (i == j)
        return 2 * h_params[2];
      else
        return 0e0;
    default:
      return 0e0;
    }
  };

  /** @brief Calculate @f$ dk_G(i,j)/dx_k @f$

      @param[in] k Index of @f$ x_k @f$
      @param[in] i @f$ i @f$
      @param[in] j @f$ j @f$
      @param[in] vx @f$ \vec{X} @f$
      @param[in] ve @f$ \vec{E} @f$
      @param[in] h_params Hyper parameters
      @return The value of @f$ dk_G(i,j)/dx_k @f$
  */
  static double dkx(int k, int i, int j, const std::vector<double> &vx, const std::vector<double> &ve, const std::vector<double> &h_params){
    if ((i == j) || (i != k && j != k))
      return 0e0;
    else if (i == k)
      return h_params[0] * h_params[0] * exp((-0.5e0) * (vx[i] - vx[j]) * (vx[i] - vx[j]) / (h_params[1] * h_params[1]))
             * (-(vx[i] - vx[j]) / (h_params[1] * h_params[1]));
    else
      return h_params[0] * h_params[0] * exp((-0.5e0) * (vx[i] - vx[j]) * (vx[i] - vx[j]) / (h_params[1] * h_params[1]))
             * ((vx[i] - vx[j]) / (h_params[1] * h_params[1]));
  };

  /** @brief Calculate @f$ dk_G(i,j)/de_k @f$

      @param[in] k Index of @f$ e_k @f$
      @param[in] i @f$ i @f$
      @param[in] j @f$ j @f$
      @param[in] vx @f$ \vec{X} @f$
      @param[in] ve @f$ \vec{E} @f$
      @param[in] h_params Hyper parameters
      @return The value of @f$ dk_G(i,j)/de_k @f$
  */
  static double dke(int k, int i, int j, const std::vector<double> &vx, const std::vector<double> &ve, const std::vector<double> &h_params){
    if (i == j && i == k)
      return 2 * ve[i];
    else
      return 0e0;
  }
};

#endif
