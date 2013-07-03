#ifndef GPR_BSA_HPP
#define GPR_BSA_HPP
/* gpr_bsa.hpp
 *
 * Copyright 2013, Kenji Harada
 * Released under the MIT and GPLv3 licenses.
 *
 * To spread the Bayesian scaling analysis method,
 * I hope you will cite the following original paper:
 *   Kenji Harada, Physical Review E 84 (2011) 056704.
 */

/**
   @mainpage C++ classes and applications for Gaussian process regression and Bayesian scaling analysis

   This toolkit consists of C++ classes for Gaussian process
   regression(GPR) and Bayesian scaling analysis(BSA) and the
   application codes for the finite-size scaling analysis.

   The BSA is a new method of statistical inference in the scaling
   analysis of critical phenomena. It is based on Bayesian statistics,
   most specifically, the GPR. The BSA assumes only the smoothness of
   a scaling function, and it does not need a form. Thus, it may be
   more effective than conventional approaches in the scaling analyses
   of critical phenomena. We find the detail of BSA in the paper (<A
   HREF="http://hdl.handle.net/10.1103/PhysRevE.84.056704"> Kenji
   Harada, Physical Review E 84 (2011) 056704</A>).

   Because the BSA is a new technique, this toolkit is the reference
   code, but this toolkit is alpha version. Your comment and
   suggestion are welcome. In particular, there are only C++ codes. I
   hope the other language codes as python or perl. Your contribution
   is welcome.

   I hope that the BSA helps your study.

   July 2013,@n
   Kenji Harada

   Graduate school of informatics, Kyoto University@n
   E-mail: harada@acs.i.kyoto-u.ac.jp@n
   URL: http://www-fcs.acs.i.kyoto-u.ac.jp/~harada/index-en.html

   @section Note
   To spread the Bayesian scaling analysis method,
   I hope you will cite the paper
   (<A HREF="http://hdl.handle.net/10.1103/PhysRevE.84.056704"> Kenji
   Harada, Physical Review E 84 (2011) 056704</A>) in your report or paper.

   @section application_Codes Application codes

   This toolkit gives applications for finite-size scaling analysis,
   extrapolation of a data set and finding a cross of two data set.
   <DL>
   <DT> bfss </DT>
   <DD> Maximum likelihood estimation of scaling parameters. </DD>
   <DT> bfss_mc </DT>
   <DD> Monte carlo estimation of mean and confidence interval of scaling parameters. </DD>
   <DT> bfss_c </DT>
   <DD> Maximum likelihood estimation of scaling parameters with correction to scaling. </DD>
   <DT> bfss_c_mc </DT>
   <DD> Monte carlo estimation of confidence intervals of scaling parameters with correction to scaling. </DD>
   <DT> bext </DT>
   <DD> Baysian extrapolation by GPR.</DD>
   <DT> bcross </DT>
   <DD> Find a cross of two data set by Baysian extrapolation.</DD>
   </DL>

   @note
   The documents of the first four applications are written in main_fss.cc.

   @section Classes Classes
   There are three classes in this toolkit. The class
   (GPR::Regression) is for Gaussian process regression on which
   Bayesian scaling analysis is based. The class
   (GPR::BSA::Gaussian_Kernel) defines the Gaussian kernel function for
   Bayesian scaling analysis. The class (GPR::BSA::FSS_Data) defines the
   data of finite-size scaling analysis. The class (GPR::BSA::FSS_Data_C) defines the
   data of finite-size scaling analysis with correction to scaling.  The two classes (GPR::Data
   and GPR::Kernel) are samples to make new classes of Gaussian process regression.
   The details of these classes can be written in gpr.hpp and gpr_bsa.hpp. The following is a skeleton code with these classes.
   @subsection Skeleton Skeleton code of applications
   @code
   #include "gpr.hpp"
   #include "gpr_sample.hpp"

   void setup(int argc, char** argv, GPR::Data& data, std::vector<double>& p_params, std::vector<int>& p_mask, std::vector<double>& h_params, std::vector<int>& h_mask);
   void output(const GPR::Data& data, const GPR::Regression<GPR::Data, GPR::Kernel>& bsa, const std::vector<double>& p_params, const std::vector<double>& h_params);

   int main(int argc, int argv){
   // Declare classes and variables
   GPR::Data data;
   GPR::Regression<GPR::Data, GPR::Kernel> bsa;
   std::vector<double> p_params, h_params;
   std::vector<int> p_mask, h_mask;

   // setup
   setup(argc, argv, data, p_params, p_mask, h_params, h_mask);
   // find a parameter set of maximum log-likelihood
   bsa.search_mll(data, p_params, p_mask, h_params, h_mask);
   // output
   output(data, bsa, p_params, h_params);

   return 0;
   }
   @endcode

*/

/** @file gpr_bsa.hpp
    @brief Classes for Bayesian scaling analysis
 */
#include <vector>
#include <cmath>
#include <string>
#include <sstream>

/**
   @namespace GPR::BSA
   @brief Bayesian scaling analysis
 */
namespace GPR {
  namespace BSA {
    class Gaussian_Kernel;
    class FSS_Data;
    class FSS_Data_C;
    class GFSS_Data;
    const static std::string version = "0.51";
  }
};

/**
   @class GPR::BSA::Gaussian_Kernel
   @brief Gaussian kernel function for Bayesian scaling analysis

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
class GPR::BSA::Gaussian_Kernel {
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

/**
  @class GPR::BSA::FSS_Data
  @brief Data for Bayesian finite-size scaling analysis

  Store data for Bayesian finite-size scaling analysis.
  The finite-size scaling form is written as
  @f[
  A(T, L) = L^{c_2} F[ ( T - T_c ) L^{c_1} ],
  @f] where @f$ A @f$ is an observable. The triplet of a data point
  is defined as @f[ X = (T - T_c ) (L/L_m)^{c_1} / R_X, Y = (A / (L/L_m)^{c_2} - Y_0)/R_Y, E = \delta A/ (L/L_m)^{c_2}/R_Y,@f]
  where @f$ \delta A @f$ is an error of @f$ A @f$ and @f$ L_m @f$ is the largest @f$ L @f$.
  Scaling factor @f$ R_X @f$ is defined so that the width of X for @f$ L_m @f$ is 2.
  Scaling factor @f$ R_Y @f$ and @f$ Y_0 @f$ is defined so that Y for @f$ L_m @f$ is in [-1:1].
  The data ansatz is @f[ Y \sim F(X) \pm E. @f]
  The physical parameters are defined as
  p_params[0] = @f$ T_c @f$,
  p_params[1] = @f$ c_1 @f$, and p_params[2] = @f$ c_2 @f$.
*/
class GPR::BSA::FSS_Data {
public:
  //! Make an empty data set
  FSS_Data(){
    reset();
  }

  /**
     @brief Set a data point

     Set a data point @f$ (L, T, A(T, L), \delta A(T, L)) @f$.
     @param[in] xl @f$L@f$
     @param[in] xt @f$T@f$
     @param[in] xa @f$A(T, L)@f$
     @param[in] xea @f$\delta A(T, L)@f$
  */
  void set(double xl, double xt, double xa, double xea){
    if (lmax < xl) {
      lmax = xl;
      tmax = tmin = xt;
      amid = amax = amin = xa;
    }else if (lmax == xl) {
      if (xt > tmax) tmax = xt;
      if (xt < tmin) tmin = xt;
      if (xa > amax) amax = xa;
      if (xa < amin) amin = xa;
    }
    tmid = (tmax + tmin) / 2;
    dt = (tmax - tmin) / 2;
    amid = (amax + amin) / 2;
    da = (amax - amin) / 2;
    l.push_back(xl);
    t.push_back(xt);
    a.push_back(xa);
    ea.push_back(xea);
    st = scale_x * dt;
    sa = scale_y * da;
  }

  //! Reset
  void reset(){
    lmax = 0;
    scale_x = scale_y = 1e0;
    l.clear();
    t.clear();
    a.clear();
    ea.clear();
  }

  //! Return the number of data points
  int npoints() const {
    return l.size();
  };

  //! Return the number of physical parameters
  int nparams() const {
    return 3;
  };

  /** @brief Calculate @f$ X_i @f$ for physical parameters
      @param[in] i Index of @f$ X_i@f$
      @param[in] p_params Physical parameters
      @return @f$ X_i @f$
  */
  double x(int i, const std::vector<double> &p_params) const {
    return (t[i] - p_params[0]) * pow(l[i] / lmax, p_params[1]) / st;
  };

  /** @brief Calculate @f$ Y_i @f$ for physical parameters
      @param[in] i Index of @f$ Y_i@f$
      @param[in] p_params Physical parameters
      @return @f$ Y_i @f$
  */
  double y(int i, const std::vector<double> &p_params) const {
    return (a[i] / pow(l[i] / lmax, p_params[2]) - amid) / sa;
  };

  /** @brief Calculate @f$ E_i @f$ for physical parameters
      @param[in] i Index of @f$ E_i@f$
      @param[in] p_params Physical parameters
      @return @f$ E_i @f$
  */
  double e(int i, const std::vector<double> &p_params) const {
    return ea[i] / pow(l[i] / lmax, p_params[2]) / sa;
  };

  /** @brief Calculate @f$ \vec{X} @f$ for physical parameters
      @param[out] vx @f$ \vec{X} @f$
      @param[in] p_params Physical parameters
  */
  void x(std::vector<double> &vx, const std::vector<double> &p_params) const {
    for (int i = 0; i < l.size(); ++i)
      vx[i] = x(i, p_params);
  };

  /** @brief Calculate @f$ \vec{Y} @f$ for physical parameters
      @param[out] vy @f$ \vec{Y} @f$
      @param[in] p_params Physical parameters
  */
  void y(std::vector<double> &vy, const std::vector<double> &p_params) const {
    for (int i = 0; i < l.size(); ++i)
      vy[i] = y(i, p_params);
  };

  /** @brief Calculate @f$ \vec{E} @f$ for physical parameters
      @param[out] ve @f$ \vec{E} @f$
      @param[in] p_params Physical parameters
  */
  void e(std::vector<double> &ve, const std::vector<double> &p_params) const {
    for (int i = 0; i < l.size(); ++i)
      ve[i] = e(i, p_params);
  };

  /** @brief Calculate @f$ d\vec{X}/d\theta @f$ for a physical parameter @f$ \theta @f$
      @param[out] dx @f$ d\vec{X}/d\theta @f$
      @param[in] pi Index of physical parameter @f$ \theta @f$
      @param[in] p_params Physical parameters
  */
  void dx(std::vector<double> &dx, int pi, const std::vector<double> &p_params) const {
    switch (pi) {
    case 0:
      for (int i = 0; i < l.size(); ++i)
        dx[i] = -pow(l[i] / lmax, p_params[1]) / st;
      break;
    case 1:
      for (int i = 0; i < l.size(); ++i)
        dx[i] = (t[i] - p_params[0]) / st * pow(l[i] / lmax, p_params[1]) * log(l[i] / lmax);
      break;
    default:
      for (int i = 0; i < l.size(); ++i)
        dx[i] = 0e0;
    }
  };

  /** @brief Calculate @f$ d\vec{Y}/d\theta @f$ for a physical parameter @f$ \theta @f$
      @param[out] dy @f$ d\vec{Y}/d\theta @f$
      @param[in] pi Index of physical parameter @f$ \theta @f$
      @param[in] p_params Physical parameters
  */
  void dy(std::vector<double> &dy, int pi, const std::vector<double> &p_params) const {
    if (pi == 2)
      for (int i = 0; i < l.size(); ++i)
        dy[i] = a[i] / pow(l[i] / lmax, p_params[2]) * (-log(l[i] / lmax)) / sa;
    else
      for (int i = 0; i < l.size(); ++i)
        dy[i] = 0e0;
  };

  /** @brief Calculate @f$ d\vec{E}/d\theta @f$ for a physical parameter @f$ \theta @f$
      @param[out] de @f$ d\vec{E}/d\theta @f$
      @param[in] pi Index of physical parameter @f$ \theta @f$
      @param[in] p_params Physical parameters
  */
  void de(std::vector<double> &de, int pi, const std::vector<double> &p_params) const {
    if (pi == 2)
      for (int i = 0; i < l.size(); ++i)
        de[i] = ea[i] / pow(l[i] / lmax, p_params[2]) * (-log(l[i] / lmax)) / sa;
    else
      for (int i = 0; i < l.size(); ++i)
        de[i] = 0e0;
  };

  //! Description of data
  std::string description(std::string header) const {
    std::stringstream ostr;
    ostr << header << "Observable: A, Error of Observable : \\delta A" << std::endl
         << header << "Finite-size scaling law: A(T, L) = L^c2 F[ ( T - T_c ) L^c1 ]" << std::endl
         << header << "Data: (X, Y, E)" << std::endl
         << header << "  X = ( T - T_c ) (L/L_m)^c1 / R_X" << std::endl
         << header << "  Y = ( A / (L/L_m)^c2 - Y_0 ) / R_Y" << std::endl
         << header << "  E = \\delta A / (L/L_m)^c2 / R_Y" << std::endl
         << header << "  L_m is the largest system size" << std::endl
         << header << "Physical parameters:" << std::endl
         << header << "  p_params[0] = T_c" << std::endl
         << header << "  p_params[1] = c_1" << std::endl
         << header << "  p_params[2] = c_2" << std::endl
         << header << "Scaling factors:" << std::endl
         << header << "  R_X is defined so that the width of X for L_m is 2. R_Y and Y_0 is defined so that Y for L_m is in [-1:1]." << std::endl
         << header << "Ansatz: Y ~ F(X) +- E" << std::endl;
    return ostr.str();
  }

  //! Return a value of @f$ L_i @f$
  double get_L(int i) const {
    return l[i];
  }

  //! Return a value of @f$ T_i @f$
  double get_T(int i) const {
    return t[i];
  }

  //! Return a value of @f$ A_i @f$
  double get_A(int i) const {
    return a[i];
  }

  //! Return a value of @f$ \delta A_i @f$
  double get_EA(int i) const {
    return ea[i];
  }

  //! Return a value of @f$ T @f$ which corresponds to @f$X(L_m)@f$
  double get_T(double X, const std::vector<double> &p_params) const {
    return X * st + p_params[0];
  }

  //! Return a value of @f$ A @f$ which corresponds to @f$Y(L_m)@f$
  double get_A(double Y) const {
    return Y * sa + amid;
  }

  //! Return a value of @f$ \delta A @f$ which corresponds to @f$E(L_m)@f$
  double get_EA(double E) const {
    return E * sa;
  }

  //! Return a value of @f$ L_m @f$
  double get_Lm() const {
    return lmax;
  }

  /** @brief Change a scaling of X and Y.
      @param[in] sx Additional scaling factor of X
      @param[in] sy Additional scaling factor of Y
   */
  void change_scale(double sx, double sy){
    scale_x = sx;
    scale_y = sy;
    st = scale_x * dt;
    sa = scale_y * da;
  }


private:
  //! Reset
  double lmax, tmax, tmin, tmid, amax, amin, amid, dt, da;
  double st, sa, scale_x, scale_y;
  std::vector<double> l;
  std::vector<double> t;
  std::vector<double> a;
  std::vector<double> ea;
};

/**
  @class GPR::BSA::FSS_Data_C
  @brief Data for Bayesian finite-size scaling analysis with correction to scaling

  Store data for Bayesian finite-size scaling analysis.
  The finite-size scaling form is written as
  @f[
  A(T, L) = L^{c_2} F[ ( T - T_c ) L^{c_1} ] ( 1 + a L^{-w} ),
  @f] where @f$ A @f$ is an observable. The triplet of a data point
  is defined as @f[ X = (T - T_c ) (L/L_m)^{c_1} / R_X, Y = (A / (L/L_m)^{c_2} \times [( 1 + a L_m^{-w}) / ( 1 + a L^{-w})] - Y_0)/R_Y, E = \delta A/ (L/L_m)^{c_2} \times [( 1 + a L_m^{-w}) / ( 1 + a L^{-w})] /R_Y,@f]
  where @f$ \delta A @f$ is an error of @f$ A @f$ and @f$ L_m @f$ is the largest @f$ L @f$.
  Scaling factor @f$ R_X @f$ is defined so that the width of X for @f$ L_m @f$ is 2.
  Scaling factor @f$ R_Y @f$ and @f$ Y_0 @f$ is defined so that Y for @f$ L_m @f$ is in [-1:1].
  The data ansatz is @f[ Y \sim F(X) \pm E. @f]
  The physical parameters are defined as
  p_params[0] = @f$ T_c @f$,
  p_params[1] = @f$ c_1 @f$,
  p_params[2] = @f$ c_2 @f$,
  p_params[3] = @f$ a @f$,
  and p_params[4] = @f$ w @f$.
*/
class GPR::BSA::FSS_Data_C {
public:
  //! Make an empty data set
  FSS_Data_C(){
    reset();
  }

  /**
     @brief Set a data point

     Set a data point @f$ (L, T, A(T, L), \delta A(T, L)) @f$.
     @param[in] xl @f$L@f$
     @param[in] xt @f$T@f$
     @param[in] xa @f$A(T, L)@f$
     @param[in] xea @f$\delta A(T, L)@f$
  */
  void set(double xl, double xt, double xa, double xea){
    if (lmax < xl) {
      lmax = xl;
      tmax = tmin = xt;
      amid = amax = amin = xa;
    }else if (lmax == xl) {
      if (xt > tmax) tmax = xt;
      if (xt < tmin) tmin = xt;
      if (xa > amax) amax = xa;
      if (xa < amin) amin = xa;
    }
    tmid = (tmax + tmin) / 2;
    dt = (tmax - tmin) / 2;
    amid = (amax + amin) / 2;
    da = (amax - amin) / 2;
    l.push_back(xl);
    t.push_back(xt);
    a.push_back(xa);
    ea.push_back(xea);
    st = scale_x * dt;
    sa = scale_y * da;
  }

  //! Reset
  void reset(){
    lmax = 0;
    scale_x = scale_y = 1e0;
    l.clear();
    t.clear();
    a.clear();
    ea.clear();
  }

  //! Return the number of data points
  int npoints() const {
    return l.size();
  };

  //! Return the number of physical parameters
  int nparams() const {
    return 5;
  };

  /** @brief Calculate @f$ X_i @f$ for physical parameters
      @param[in] i Index of @f$ X_i@f$
      @param[in] p_params Physical parameters
      @return @f$ X_i @f$
  */
  double x(int i, const std::vector<double> &p_params) const {
    return (t[i] - p_params[0]) * pow(l[i] / lmax, p_params[1]) / st;
  };

  /** @brief Calculate @f$ Y_i @f$ for physical parameters
      @param[in] i Index of @f$ Y_i@f$
      @param[in] p_params Physical parameters
      @return @f$ Y_i @f$
  */
  double y(int i, const std::vector<double> &p_params) const {
    return (a[i] / pow(l[i] / lmax, p_params[2])
            * ( 1e0 + p_params[3] * pow(lmax, -p_params[4]) )
            / ( 1e0 + p_params[3] * pow(l[i], -p_params[4]) )
            - amid) / sa;
  };

  /** @brief Calculate @f$ E_i @f$ for physical parameters
      @param[in] i Index of @f$ E_i@f$
      @param[in] p_params Physical parameters
      @return @f$ E_i @f$
  */
  double e(int i, const std::vector<double> &p_params) const {
    return ea[i] / pow(l[i] / lmax, p_params[2])
           * ( 1e0 + p_params[3] * pow(lmax, -p_params[4]) )
           / ( 1e0 + p_params[3] * pow(l[i], -p_params[4]) )
           / sa;
  };

  /** @brief Calculate @f$ \vec{X} @f$ for physical parameters
      @param[out] vx @f$ \vec{X} @f$
      @param[in] p_params Physical parameters
  */
  void x(std::vector<double> &vx, const std::vector<double> &p_params) const {
    for (int i = 0; i < l.size(); ++i)
      vx[i] = x(i, p_params);
  };

  /** @brief Calculate @f$ \vec{Y} @f$ for physical parameters
      @param[out] vy @f$ \vec{Y} @f$
      @param[in] p_params Physical parameters
  */
  void y(std::vector<double> &vy, const std::vector<double> &p_params) const {
    for (int i = 0; i < l.size(); ++i)
      vy[i] = y(i, p_params);
  };

  /** @brief Calculate @f$ \vec{E} @f$ for physical parameters
      @param[out] ve @f$ \vec{E} @f$
      @param[in] p_params Physical parameters
  */
  void e(std::vector<double> &ve, const std::vector<double> &p_params) const {
    for (int i = 0; i < l.size(); ++i)
      ve[i] = e(i, p_params);
  };

  /** @brief Calculate @f$ d\vec{X}/d\theta @f$ for a physical parameter @f$ \theta @f$
      @param[out] dx @f$ d\vec{X}/d\theta @f$
      @param[in] pi Index of physical parameter @f$ \theta @f$
      @param[in] p_params Physical parameters
  */
  void dx(std::vector<double> &dx, int pi, const std::vector<double> &p_params) const {
    switch (pi) {
    case 0:
      for (int i = 0; i < l.size(); ++i)
        dx[i] = -pow(l[i] / lmax, p_params[1]) / st;
      break;
    case 1:
      for (int i = 0; i < l.size(); ++i)
        dx[i] = (t[i] - p_params[0]) / st * pow(l[i] / lmax, p_params[1]) * log(l[i] / lmax);
      break;
    default:
      for (int i = 0; i < l.size(); ++i)
        dx[i] = 0e0;
    }
  };

  /** @brief Calculate @f$ d\vec{Y}/d\theta @f$ for a physical parameter @f$ \theta @f$
      @param[out] dy @f$ d\vec{Y}/d\theta @f$
      @param[in] pi Index of physical parameter @f$ \theta @f$
      @param[in] p_params Physical parameters
  */
  void dy(std::vector<double> &dy, int pi, const std::vector<double> &p_params) const {
    switch (pi) {
    case 4:
      for (int i = 0; i < l.size(); ++i)
        dy[i] = a[i] / pow(l[i] / lmax, p_params[2]) / sa
                * ( p_params[3] * pow(lmax, -p_params[4]) * (-log(lmax))
                    / (1e0 + p_params[3] * pow(l[i], -p_params[4]))
                    - (1e0 + p_params[3] * pow(lmax, -p_params[4]))
                    / pow((1e0 + p_params[3] * pow(l[i], -p_params[4])), 2e0)
                    * p_params[3] * pow(l[i], -p_params[4]) * (-log(l[i])) );
      break;
    case 3:
      for (int i = 0; i < l.size(); ++i)
        dy[i] = a[i] / pow(l[i] / lmax, p_params[2]) / sa
                * ( pow(lmax, -p_params[4]) / (1e0 + p_params[3] * pow(l[i], -p_params[4]))
                    - (1e0 + p_params[3] * pow(lmax, -p_params[4]))
                    / pow((1e0 + p_params[3] * pow(l[i], -p_params[4])), 2e0)
                    * pow(l[i], -p_params[4]) );
      break;
    case 2:
      for (int i = 0; i < l.size(); ++i)
        dy[i] = a[i] / pow(l[i] / lmax, p_params[2]) * (-log(l[i] / lmax)) / sa
                * ( 1e0 + p_params[3] * pow(lmax, -p_params[4]) )
                / ( 1e0 + p_params[3] * pow(l[i], -p_params[4]) );
      break;
    default:
      for (int i = 0; i < l.size(); ++i)
        dy[i] = 0e0;
    }
  };

  /** @brief Calculate @f$ d\vec{E}/d\theta @f$ for a physical parameter @f$ \theta @f$
      @param[out] de @f$ d\vec{E}/d\theta @f$
      @param[in] pi Index of physical parameter @f$ \theta @f$
      @param[in] p_params Physical parameters
  */
  void de(std::vector<double> &de, int pi, const std::vector<double> &p_params) const {
    switch (pi) {
    case 4:
      for (int i = 0; i < l.size(); ++i)
        de[i] = ea[i] / pow(l[i] / lmax, p_params[2]) / sa
                * ( p_params[3] * pow(lmax, -p_params[4]) * (-log(lmax))
                    / (1e0 + p_params[3] * pow(l[i], -p_params[4]))
                    - (1e0 + p_params[3] * pow(lmax, -p_params[4]))
                    / pow((1e0 + p_params[3] * pow(l[i], -p_params[4])), 2e0)
                    * p_params[3] * pow(l[i], -p_params[4]) * (-log(l[i])) );
      break;
    case 3:
      for (int i = 0; i < l.size(); ++i)
        de[i] = ea[i] / pow(l[i] / lmax, p_params[2]) / sa
                * ( pow(lmax, -p_params[4]) / (1e0 + p_params[3] * pow(l[i], -p_params[4]))
                    - (1e0 + p_params[3] * pow(lmax, -p_params[4]))
                    / pow((1e0 + p_params[3] * pow(l[i], -p_params[4])), 2e0)
                    * pow(l[i], -p_params[4]) );
      break;
    case 2:
      for (int i = 0; i < l.size(); ++i)
        de[i] = ea[i] / pow(l[i] / lmax, p_params[2]) * (-log(l[i] / lmax)) / sa
                * ( 1e0 + p_params[3] * pow(lmax, -p_params[4]) )
                / ( 1e0 + p_params[3] * pow(l[i], -p_params[4]) );
      break;
    default:
      for (int i = 0; i < l.size(); ++i)
        de[i] = 0e0;
    }
  };

  //! Description of data
  std::string description(std::string header) const {
    std::stringstream ostr;
    ostr << header << "Observable: A, Error of Observable : \\delta A" << std::endl
         << header << "Finite-size scaling law: A(T, L) = L^c2 F[ ( T - T_c ) L^c1 ] ( 1 + a L^{-w} )" << std::endl
         << header << "Data: (X, Y, E)" << std::endl
         << header << "  X = ( T - T_c ) (L/L_m)^c1 / R_X" << std::endl
         << header << "  Y = ( A / (L/L_m)^c2 * (1 + a L_m^{-w}) / (1 + a L^{-w}) - Y_0 ) / R_Y" << std::endl
         << header << "  E = \\delta A / (L/L_m)^c2 * (1 + a L_m^{-w}) / (1 + a L^{-w}) / R_Y" << std::endl
         << header << "  L_m is the largest system size" << std::endl
         << header << "Physical parameters:" << std::endl
         << header << "  p_params[0] = T_c" << std::endl
         << header << "  p_params[1] = c_1" << std::endl
         << header << "  p_params[2] = c_2" << std::endl
         << header << "  p_params[3] = a" << std::endl
         << header << "  p_params[4] = w" << std::endl
         << header << "Scaling factors:" << std::endl
         << header << "  R_X is defined so that the width of X for L_m is 2. R_Y and Y_0 is defined so that Y for L_m is in [-1:1]." << std::endl
         << header << "Ansatz: Y ~ F(X) +- E" << std::endl;
    return ostr.str();
  }

  //! Return a value of @f$ L_i @f$
  double get_L(int i) const {
    return l[i];
  }

  //! Return a value of @f$ T_i @f$
  double get_T(int i) const {
    return t[i];
  }

  //! Return a value of @f$ A_i @f$
  double get_A(int i) const {
    return a[i];
  }

  //! Return a value of @f$ \delta A_i @f$
  double get_EA(int i) const {
    return ea[i];
  }

  //! Return a value of @f$ T @f$ which corresponds to @f$X(L_m)@f$
  double get_T(double X, const std::vector<double> &p_params) const {
    return X * st + p_params[0];
  }

  //! Return a value of @f$ A @f$ which corresponds to @f$Y(L_m)@f$
  double get_A(double Y) const {
    return Y * sa + amid;
  }

  //! Return a value of @f$ \delta A @f$ which corresponds to @f$E(L_m)@f$
  double get_EA(double E) const {
    return E * sa;
  }

  //! Return a value of @f$ L_m @f$
  double get_Lm() const {
    return lmax;
  }

  /** @brief Change a scaling of X and Y.
      @param[in] sx Additional scaling factor of X
      @param[in] sy Additional scaling factor of Y
   */
  void change_scale(double sx, double sy){
    scale_x = sx;
    scale_y = sy;
    st = scale_x * dt;
    sa = scale_y * da;
  }


private:
  //! Reset
  double lmax, tmax, tmin, tmid, amax, amin, amid, dt, da;
  double st, sa, scale_x, scale_y;
  std::vector<double> l;
  std::vector<double> t;
  std::vector<double> a;
  std::vector<double> ea;
};

/**
  @class GPR::BSA::GFSS_Data
  @brief Data for Bayesian finite-size scaling analysis

  Store data for Bayesian finite-size scaling analysis.
  The finite-size scaling form is written as
  @f[
  A(T, L) = L^{c_2} F[ ( T - T_c ) L^{c_1} ],
  @f] where @f$ A @f$ is an observable. The triplet of a data point
  is defined as @f[ X = (T - T_c ) (L/L_m)^{c_1} / R_X, Y = (A / (L/L_m)^{c_2} - Y_0)/R_Y, E = \delta A/ (L/L_m)^{c_2}/R_Y,@f]
  where @f$ \delta A @f$ is an error of @f$ A @f$ and @f$ L_m @f$ is the largest @f$ L @f$.
  Scaling factor @f$ R_X @f$ is defined so that the width of X for @f$ L_m @f$ is 2.
  Scaling factor @f$ R_Y @f$ and @f$ Y_0 @f$ is defined so that Y for @f$ L_m @f$ is in [-1:1].
  The data ansatz is @f[ Y \sim F(X) \pm E. @f]
  The physical parameters are defined as
  p_params[0] = @f$ T_c @f$,
  p_params[1] = @f$ c_1 \quad (T \le T_c) @f$, p_params[2] = @f$ c_2 \quad (T \le T_c)@f$.
  and p_params[3] = @f$ c_1' \quad (T > T_c) @f$.
*/
class GPR::BSA::GFSS_Data {
public:
  //! Make an empty data set
  GFSS_Data(){
    reset();
  }

  /**
     @brief Set a data point

     Set a data point @f$ (L, T, A(T, L), \delta A(T, L)) @f$.
     @param[in] xl @f$L@f$
     @param[in] xt @f$T@f$
     @param[in] xa @f$A(T, L)@f$
     @param[in] xea @f$\delta A(T, L)@f$
  */
  void set(double xl, double xt, double xa, double xea){
    if (lmax < xl) {
      lmax = xl;
      tmax = tmin = xt;
      amid = amax = amin = xa;
    }else if (lmax == xl) {
      if (xt > tmax) tmax = xt;
      if (xt < tmin) tmin = xt;
      if (xa > amax) amax = xa;
      if (xa < amin) amin = xa;
    }
    tmid = (tmax + tmin) / 2;
    dt = (tmax - tmin) / 2;
    amid = (amax + amin) / 2;
    da = (amax - amin) / 2;
    l.push_back(xl);
    t.push_back(xt);
    a.push_back(xa);
    ea.push_back(xea);
    st = scale_x * dt;
    sa = scale_y * da;
  }

  //! Reset
  void reset(){
    lmax = 0;
    scale_x = scale_y = 1e0;
    l.clear();
    t.clear();
    a.clear();
    ea.clear();
  }

  //! Return the number of data points
  int npoints() const {
    return l.size();
  };

  //! Return the number of physical parameters
  int nparams() const {
    return 4;
  };

  /** @brief Calculate @f$ X_i @f$ for physical parameters
      @param[in] i Index of @f$ X_i@f$
      @param[in] p_params Physical parameters
      @return @f$ X_i @f$
  */
  double x(int i, const std::vector<double> &p_params) const {
    if (t[i] <= p_params[0])
      return (t[i] - p_params[0]) * pow(l[i] / lmax, p_params[1]) / st;
    else
      return (t[i] - p_params[0]) * pow(l[i] / lmax, p_params[3]) / st;
  };

  /** @brief Calculate @f$ Y_i @f$ for physical parameters
      @param[in] i Index of @f$ Y_i@f$
      @param[in] p_params Physical parameters
      @return @f$ Y_i @f$
  */
  double y(int i, const std::vector<double> &p_params) const {
    return (a[i] / pow(l[i] / lmax, p_params[2]) - amid) / sa;
  };

  /** @brief Calculate @f$ E_i @f$ for physical parameters
      @param[in] i Index of @f$ E_i@f$
      @param[in] p_params Physical parameters
      @return @f$ E_i @f$
  */
  double e(int i, const std::vector<double> &p_params) const {
    return ea[i] / pow(l[i] / lmax, p_params[2]) / sa;
  };

  /** @brief Calculate @f$ \vec{X} @f$ for physical parameters
      @param[out] vx @f$ \vec{X} @f$
      @param[in] p_params Physical parameters
  */
  void x(std::vector<double> &vx, const std::vector<double> &p_params) const {
    for (int i = 0; i < l.size(); ++i)
      vx[i] = x(i, p_params);
  };

  /** @brief Calculate @f$ \vec{Y} @f$ for physical parameters
      @param[out] vy @f$ \vec{Y} @f$
      @param[in] p_params Physical parameters
  */
  void y(std::vector<double> &vy, const std::vector<double> &p_params) const {
    for (int i = 0; i < l.size(); ++i)
      vy[i] = y(i, p_params);
  };

  /** @brief Calculate @f$ \vec{E} @f$ for physical parameters
      @param[out] ve @f$ \vec{E} @f$
      @param[in] p_params Physical parameters
  */
  void e(std::vector<double> &ve, const std::vector<double> &p_params) const {
    for (int i = 0; i < l.size(); ++i)
      ve[i] = e(i, p_params);
  };

  /** @brief Calculate @f$ d\vec{X}/d\theta @f$ for a physical parameter @f$ \theta @f$
      @param[out] dx @f$ d\vec{X}/d\theta @f$
      @param[in] pi Index of physical parameter @f$ \theta @f$
      @param[in] p_params Physical parameters
  */
  void dx(std::vector<double> &dx, int pi, const std::vector<double> &p_params) const {
    switch (pi) {
    case 0:
      for (int i = 0; i < l.size(); ++i)
        if (t[i] <= p_params[0])
          dx[i] = -pow(l[i] / lmax, p_params[1]) / st;
        else
          dx[i] = -pow(l[i] / lmax, p_params[3]) / st;
      break;
    case 1:
      for (int i = 0; i < l.size(); ++i)
        if (t[i] <= p_params[0])
          dx[i] = (t[i] - p_params[0]) / st * pow(l[i] / lmax, p_params[1]) * log(l[i] / lmax);
        else
          dx[i] = 0;
      break;
    case 3:
      for (int i = 0; i < l.size(); ++i)
        if (t[i] > p_params[0])
          dx[i] = (t[i] - p_params[0]) / st * pow(l[i] / lmax, p_params[3]) * log(l[i] / lmax);
        else
          dx[i] = 0;
      break;
    default:
      for (int i = 0; i < l.size(); ++i)
        dx[i] = 0e0;
    }
  };

  /** @brief Calculate @f$ d\vec{Y}/d\theta @f$ for a physical parameter @f$ \theta @f$
      @param[out] dy @f$ d\vec{Y}/d\theta @f$
      @param[in] pi Index of physical parameter @f$ \theta @f$
      @param[in] p_params Physical parameters
  */
  void dy(std::vector<double> &dy, int pi, const std::vector<double> &p_params) const {
    if (pi == 2) {
      for (int i = 0; i < l.size(); ++i)
        if (t[i] <= p_params[0])
          dy[i] = a[i] / pow(l[i] / lmax, p_params[2]) * (-log(l[i] / lmax)) / sa;
        else
          dy[i] = 0;
    }else
      for (int i = 0; i < l.size(); ++i)
        dy[i] = 0e0;
  };

  /** @brief Calculate @f$ d\vec{E}/d\theta @f$ for a physical parameter @f$ \theta @f$
      @param[out] de @f$ d\vec{E}/d\theta @f$
      @param[in] pi Index of physical parameter @f$ \theta @f$
      @param[in] p_params Physical parameters
  */
  void de(std::vector<double> &de, int pi, const std::vector<double> &p_params) const {
    if (pi == 2) {
      for (int i = 0; i < l.size(); ++i)
        if (t[i] <= p_params[0])
          de[i] = ea[i] / pow(l[i] / lmax, p_params[2]) * (-log(l[i] / lmax)) / sa;
        else
          de[i] = 0;
    }else
      for (int i = 0; i < l.size(); ++i)
        de[i] = 0e0;
  };

  //! Description of data
  std::string description(std::string header) const {
    std::stringstream ostr;
    ostr << header << "Observable: A, Error of Observable : \\delta A" << std::endl
         << header << "Finite-size scaling law: A(T, L) = L^c2 F[ ( T - T_c ) L^c1 ]" << std::endl
         << header << "Data: (X, Y, E)" << std::endl
         << header << "  X = ( T - T_c ) (L/L_m)^c1 / R_X" << std::endl
         << header << "  Y = ( A / (L/L_m)^c2 - Y_0 ) / R_Y" << std::endl
         << header << "  E = \\delta A / (L/L_m)^c2 / R_Y" << std::endl
         << header << "  L_m is the largest system size" << std::endl
         << header << "Physical parameters:" << std::endl
         << header << "  p_params[0] = T_c" << std::endl
         << header << "  p_params[1] = c_1 (T <= T_c)" << std::endl
         << header << "  p_params[2] = c_2 (T <= T_c)" << std::endl
         << header << "  p_params[3] = c_1\' (T > T_c)" << std::endl
         << header << "Scaling factors:" << std::endl
         << header << "  R_X is defined so that the width of X for L_m is 2. R_Y and Y_0 is defined so that Y for L_m is in [-1:1]." << std::endl
         << header << "Ansatz: Y ~ F(X) +- E" << std::endl;
    return ostr.str();
  }

  //! Return a value of @f$ L_i @f$
  double get_L(int i) const {
    return l[i];
  }

  //! Return a value of @f$ T_i @f$
  double get_T(int i) const {
    return t[i];
  }

  //! Return a value of @f$ A_i @f$
  double get_A(int i) const {
    return a[i];
  }

  //! Return a value of @f$ \delta A_i @f$
  double get_EA(int i) const {
    return ea[i];
  }

  //! Return a value of @f$ T @f$ which corresponds to @f$X(L_m)@f$
  double get_T(double X, const std::vector<double> &p_params) const {
    return X * st + p_params[0];
  }

  //! Return a value of @f$ A @f$ which corresponds to @f$Y(L_m)@f$
  double get_A(double Y) const {
    return Y * sa + amid;
  }

  //! Return a value of @f$ \delta A @f$ which corresponds to @f$E(L_m)@f$
  double get_EA(double E) const {
    return E * sa;
  }

  //! Return a value of @f$ L_m @f$
  double get_Lm() const {
    return lmax;
  }

  /** @brief Change a scaling of X and Y.
      @param[in] sx Additional scaling factor of X
      @param[in] sy Additional scaling factor of Y
   */
  void change_scale(double sx, double sy){
    scale_x = sx;
    scale_y = sy;
    st = scale_x * dt;
    sa = scale_y * da;
  }


private:
  //! Reset
  double lmax, tmax, tmin, tmid, amax, amin, amid, dt, da;
  double st, sa, scale_x, scale_y;
  std::vector<double> l;
  std::vector<double> t;
  std::vector<double> a;
  std::vector<double> ea;
};

#endif
