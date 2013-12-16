/* main_fss.cc
 *
 * Copyright (C) 2011, 2012, 2013 Kenji Harada
 *
 */
/**
   @file main_fss.cc
   @brief Application code of Bayesian finite-size scaling

   This file is an application code of Bayesian finite-size scaling.
   @section app Applications

   This toolkit gives four applications for finite-size scaling analysis.
   <DL>
   <DT> bfss </DT>
   <DD> Maximum likelihood estimation of scaling parameters. </DD>
   <DT> bfss_mc </DT>
   <DD> Monte carlo estimation of mean and confidence interval of scaling parameters. </DD>
   <DT> bfss_c </DT>
   <DD> Maximum likelihood estimation of scaling parameters with correction to scaling. </DD>
   <DT> bfss_c_mc </DT>
   <DD> Monte carlo estimation of confidence intervals of scaling parameters with correction to scaling. </DD>
   </DL>
   They are made from a code (main_fss.cc) which uses three classes (GPR::Regression, GPR::BSA::FSS_Data,
   GPR::BSA::FSS_Data_C and GPR::BSA::Gaussian_Kernel).

   @section comp Compile and test
   Firstly, we needs to install the GSL library
   (http://www.gnu.org/software/gsl/) for the class GPR::Regression.
   After that, we needs to change variables in Makefile: GSL_DIR and BLAS_LIB.

   To compile and test, we can use "make"
   command as follows.
   @code
   % make
   % make test
   @endcode

   To see the result of finite-size scaling of Binder ratio of Ising model, one
   can use "gnuplot" as follows.
   @code
   % gnuplot
   gnuplot> plot "test.op" u 1:2:3 i 0 w e
   gnuplot> plot "test_mc.op" u 1:2:3 i 0 w e
   gnuplot> plot "test_c.op" u 1:2:3 i 0 w e
   gnuplot> plot "test_c_mc.op" u 1:2:3 i 0 w e
   @endcode

   @section usage Usage
   @code
   Usage: ./bfss [data_file] [three physical parameter sets] (option)[three hyper parameter sets]
   Usage: ./bfss_mc [data_file] [three physical parameter sets] (option)[three hyper parameter sets]
   Usage: ./bfss_c [data_file] [five physical parameter sets] (option)[three hyper parameter sets]
   Usage: ./bfss_c_mc [data_file] [five physical parameter sets] (option)[three hyper parameter sets]
   @endcode

   The command "bfss" finds the best finite-size scaling
   parameters. The command "bfss_mc" calculates the confidential
   intervals of parameters by Monte Carlo sampling.  The command
   "bfss_c" finds the best finite-size scaling parameters with
   correction to scaling. The command "bfss_c_mc" calculates the
   confidential intervals of parameters by Monte Carlo sampling for
   the case of correction to scaling.

   If [data_file] is '-', data are loaded from STDIN. A parameter set
   is a pair of mask and initial value of parameter.

   If mask = 0(1), the parameter is fixed (unfixed).

   @note Usually, it is better that a hyper parameter starts from 1,
   because all data are renormalized. If you do not give hyper
   parameter sets, the default values are 1.

   @code
   Example.

     % ./bfss Test/Ising-square-Binder.dat 1 0.42 1 0.9 1 0.1 1 1 1 1 1 1 > test.op 2>test.log
     % ./bfss_mc Test/Ising-square-Binder.dat 1 0.42 1 0.9 1 0.1 1 1 1 1 1 1 > test_mc.op 2>test_mc.log
     % ./bfss_c Sample/sample-c.dat 1 0.28 1 0.9 0 0 1 1 1 2 1 1 1 1 1 1 > test_c.op 2>test_c.log
     % ./bfss_c_mc Sample/sample-c.dat 1 0.28 1 0.9 0 0 1 1 1 2 1 1 1 1 1 1 > test_c_mc.op 2>test_c_mc.log

   @endcode

   @section physical_p Physical parameters
   @subsection sclaw Scaling law
   The finite-size scaling form is written as @f[ A(T, L) = L^{c_2} F[
  ( T - T_c ) L^{c_1} ], @f] where @f$ A @f$ is an observable. The
  triplet of a data point is defined as @f[ X = (T - T_c )
  (L/L_m)^{c_1} / R_X, Y = (A / (L/L_m)^{c_2} - Y_0)/R_Y, E = \delta
  A/ (L/L_m)^{c_2}/R_Y,@f] where @f$ \delta A @f$ is an error of @f$ A
  @f$ and @f$ L_m @f$ is the largest @f$ L @f$.  Scaling factor @f$
  R_X @f$ is defined so that the width of X for @f$ L_m @f$ is 2.
  Scaling factor @f$ R_Y @f$ and @f$ Y_0 @f$ is defined so that Y for
  @f$ L_m @f$ is in [-1:1].  The data ansatz is @f[ Y \sim F(X) \pm
  E. @f] The physical parameters are defined as [0] = @f$ T_c @f$, [1]
  = @f$ c_1 @f$, and [2] = @f$ c_2 @f$.

   @subsection sclaw_c Scaling law with correction to scaling
   The finite-size scaling form with correction to scaling is written as @f[ A(T, L) = L^{c_2} F[
   ( T - T_c ) L^{c_1} ] ( 1 + a L^{-w} ), @f] where @f$ A @f$ is an
   observable. The triplet of a data point is defined as @f[ X = (T -
   T_c ) (L/L_m)^{c_1} / R_X, Y = (A / (L/L_m)^{c_2} \times [( 1 + a
   L_m^{-w}) / ( 1 + a L^{-w})] - Y_0)/R_Y, E = \delta A/
   (L/L_m)^{c_2} \times [( 1 + a L_m^{-w}) / ( 1 + a L^{-w})] /R_Y,
   @f] where @f$ \delta A @f$ is an error of @f$ A
  @f$ and @f$ L_m @f$ is the largest @f$ L @f$.  Scaling factor @f$
  R_X @f$ is defined so that the width of X for @f$ L_m @f$ is 2.
  Scaling factor @f$ R_Y @f$ and @f$ Y_0 @f$ is defined so that Y for
  @f$ L_m @f$ is in [-1:1].  The data ansatz is @f[ Y \sim F(X) \pm
  E. @f] The physical parameters are defined as [0] = @f$ T_c @f$, [1]
  = @f$ c_1 @f$, [2] = @f$ c_2 @f$, [3] = @f$ a @f$, and [4] = @f$ w @f$.

   @note
   This code can be applied to a scaling analysis which has the same
   form of the finite-size scaling law.

  @section hyper_p Hyper parameters
   @subsection kernel Kernel function
   Kernel function is written as
   @f[
   k_G(i, j) = \delta_{ij} (E_i^2 + \theta_2^2)
   + \theta_0^2 \exp( - |X_i- X_j|^2 / 2\theta_1^2 ).
   @f]
   The hyper parameters are defined as [0] = @f$ \theta_0 @f$,
   [1] = @f$ \theta_1 @f$, and [2] = @f$ \theta_2 @f$.

   @section data Data file for input
   @subsection fmt Format
   The format of data file is as follows.
   @code
   # L   T            A              Error_of_A
   128   4.200000e-01 6.271240e-02   1.336090e-03
   @endcode
   A line has to be ended with the newline character. Comment lines
   starts with the character '#'. A null line is ignored. Other lines
   contain four values.  The value of @f$ L @f$ is in the first column
   of data file.  The value of @f$ T @f$ is in the second column. The
   value of @f$ A @f$ is in the third column. The value of @f$ \delta
   A @f$ is in the fourth column. If a line is not correctly
   formatted, it will be skipped.

   @section output Output of commands

The value of best parameters and the confidential intervals are
 written in header as comments. The remain consists of three groups.
 These three groups are separated two null lines.

The first group is the scaling result with unnormalized variables as
 @f[ X = (T - T_c ) L^{c_1}, Y = A / L^{c_2}, E = \delta A/
 L^{c_2}.@f] In all groups, the best parameters are used.  In the case
 of commands "bfss_mc" and "bfss_c_mc", the average of parameters are
 used. The line of output contains a list of @f$ X, Y, E, L, T, A,
 \mbox{and}\ \delta A.@f$

 The second one consists of 100 points of the inferred
 scaling function with unnormalized variables. The line
 contains a list of @f$ X, \mu(X), \sqrt{\sigma^2(X)} @f$.

 The third one is the
 scaling result with normalized variables as @f[ X = (T - T_c )
 (L/L_m)^{c_1} / R_X, Y = (A / (L/L_m)^{c_2} - Y_0)/R_Y, E = \delta A/
 (L/L_m)^{c_2}/R_Y.@f] The line contains a list of @f$ X, Y, E,
 L, T, A, \mbox{and}\ \delta A.@f$



   @subsection output_c In the case of correction to scaling
   The first
   group is the scaling result with unnormalized variables as @f[ X =
   (T - T_c ) L^{c_1}, Y = A / L^{c_2} / ( 1 + a L^{-w}), E = \delta A/ L^{c_2} / ( 1 + a L^{-w}).@f]
   The third one is the scaling result with normalized variables as
   @f[ X = (T -
   T_c ) (L/L_m)^{c_1} / R_X, Y = (A / (L/L_m)^{c_2} \times [( 1 + a
   L_m^{-w}) / ( 1 + a L^{-w})] - Y_0)/R_Y, E = \delta A/
   (L/L_m)^{c_2} \times [( 1 + a L_m^{-w}) / ( 1 + a L^{-w})] /R_Y. @f]
 */
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstring>

#include "gpr_bsa.hpp"
#include "gpr.hpp"

#ifdef FSS_C
typedef GPR::BSA::FSS_Data_C FSS_DATA_TYPE;
#else
typedef GPR::BSA::FSS_Data FSS_DATA_TYPE;
#endif
typedef GPR::BSA::Gaussian_Kernel FSS_KERNEL_TYPE;

int load(char *fname, FSS_DATA_TYPE &data);
int setup(int argc, char** argv, FSS_DATA_TYPE& data,
          std::vector<double>& p_params, std::vector<int>& p_mask,
          std::vector<double>& h_params, std::vector<int>& h_mask);
void output(const FSS_DATA_TYPE& data,
            const GPR::Regression<FSS_DATA_TYPE, FSS_KERNEL_TYPE>& bayesian_fss,
            const std::vector<double>& p_params,
            const std::vector<double>& h_params);
void output_mc(const FSS_DATA_TYPE& data,
               const GPR::Regression<FSS_DATA_TYPE, FSS_KERNEL_TYPE>& bayesian_fss,
               const std::vector<double>& p_params,
               const std::vector<double>& ep_params,
               const std::vector<double>& h_params,
               const std::vector<double>& eh_params);

void output_mc(const std::vector<FSS_DATA_TYPE>& datas,
               const GPR::Regression<FSS_DATA_TYPE, FSS_KERNEL_TYPE>& bayesian_fss,
               const std::vector< std::vector<int> >& index_p_params,
               const std::vector< std::vector<int> >& index_h_params,
               const std::vector<double>& params,
               const std::vector<double>& delta_params,
               std::vector<double>& average,
               std::vector<double>& covariance);

int main(int argc, char **argv){
  // Declare classes and variables
  FSS_DATA_TYPE data;
  GPR::Regression<FSS_DATA_TYPE, FSS_KERNEL_TYPE> bayesian_fss;
  std::vector<double> p_params, h_params;
  std::vector<int> p_mask, h_mask;

  // Setup
  int num = setup(argc, argv, data, p_params, p_mask, h_params, h_mask);
  if (num == 0) {
    std::cerr << "No data point" << std::endl;
    return 0;
  }
  std::cout << "# Number of data points= " << num << std::endl;
  // Find maximum log-likelihood
  bayesian_fss.search_mll(data, p_params, p_mask, h_params, h_mask);
#ifndef MAIN_MC
  // Output
  output(data, bayesian_fss, p_params, h_params);
#else
  // Setup for Monte Carlo estimation
  std::vector<double> ep_params;
  for (int i = 0; i < p_params.size(); ++i)
    ep_params.push_back(p_params[i] * 0.001);
  std::vector<double> eh_params;
  for (int i = 0; i < h_params.size(); ++i)
    eh_params.push_back(h_params[i] * 0.001);
#ifndef OUTPUT_COV
  // Monte Carlo estimation
  bayesian_fss.monte_carlo(data, p_params, p_mask, ep_params, h_params, h_mask, eh_params);
  // Output
  output_mc(data, bayesian_fss, p_params, ep_params, h_params, eh_params);
#else
  std::vector<double> average;
  std::vector<double> covariance;
  // Monte Carlo estimation
  bayesian_fss.monte_carlo(data, p_params, p_mask, ep_params, h_params, h_mask, eh_params, average, covariance);
  // Output
  std::vector<double> params(p_params.size() + h_params.size());
  std::vector<double> delta_params(p_params.size() + h_params.size());
  std::vector< std::vector<int> > index_p_params(1);
  for (int i = 0; i < p_params.size(); ++i) {
    index_p_params[0].push_back(i);
    params[i] = p_params[i];
    delta_params[i] = ep_params[i];
  }
  std::vector< std::vector<int> > index_h_params(1);
  for (int i = 0; i < h_params.size(); ++i) {
    index_h_params[0].push_back(p_params.size() + i);
    params[p_params.size() + i] = h_params[i];
    delta_params[p_params.size() + i] = eh_params[i];
  }
  std::vector<FSS_DATA_TYPE> datas(1);
  datas[0] = data;
  output_mc(datas, bayesian_fss, index_p_params, index_h_params, params, delta_params, average, covariance);
#endif
#endif
}

/** @brief Load data

    @param[in] fname Name of data file
    @param[out] data Data
 */
int load(char *fname, FSS_DATA_TYPE &data){
  int num = 0;
  if (*fname == '-') {
    if (!std::cin.good()) {
      std::cerr << "No standard input" << std::endl;
      exit(-1);
    }
    while (1) {
      char str[256];
      std::cin.getline(str, 256);
      if (!std::cin.good()) break;
      if (std::strlen(str) > 0 && str[0] != '#') {
        std::istringstream isst(str);
        double l, t, a, ea;
        isst >> l >> t >> a >> ea;
        if (isst.fail()) {
          std::cerr << "# Skip a line (" << str << ")" << std::endl;
          continue;
        }
        data.set(l, t, a, ea);
        ++num;
      }
    }
  }else{
    std::ifstream fin(fname);
    if (!fin.good()) {
      std::cerr << "Cannot open the file " << fname << std::endl;
      exit(-1);
    }
    while (1) {
      char str[256];
      fin.getline(str, 256);
      if (!fin.good()) break;
      if (strlen(str) > 0 && str[0] != '#') {
        std::istringstream isst(str);
        double l, t, a, ea;
        isst >> l >> t >> a >> ea;
        if (isst.fail()) {
          std::cerr << "# Skip a line (" << str << ")" << std::endl;
          continue;
        }
        data.set(l, t, a, ea);
        ++num;
      }
    }
  }
  return num;
}

/** @brief Setup of data, parameters and masks

    @param[in] argc Number of arguments in command line
    @param[in] argv Arguments in command line
    @param[out] data Data
    @param[out] p_params physical parameters
    @param[out] p_mask mask of physical parameters
    @param[out] h_params hyper parameters
    @param[out] h_mask mask of hyper parameters
 */
int setup(int argc, char** argv, FSS_DATA_TYPE& data,
          std::vector<double>& p_params, std::vector<int>& p_mask,
          std::vector<double>& h_params, std::vector<int>& h_mask){
  // Load data from file or STDIN
  if (argc != ((data.nparams() + FSS_KERNEL_TYPE::nparams()) * 2 + 2)
      && argc != (data.nparams() * 2 + 2)) {
    std::cerr << "### Usage" << std::endl
              << "  " << argv[0] << " [data_file] ["
              << data.nparams() << " physical_parameter_sets] (option)["
              << FSS_KERNEL_TYPE::nparams() << " hyper_parameter_sets]" << std::endl
              << "  If data_file = '-', data are loaded from STDIN" << std::endl
              << "  parameter_set := mask + initial_value" << std::endl
              << "  mask := 0 (fixed) or 1 (unfixed)" << std::endl
              << "    Example." << std::endl << "    " << argv[0] << " test.dat 0 2.3 1 -1.2 ..." << std::endl;
    std::cerr << "### Data" << std::endl;
    std::cerr << data.description("  ");
    std::cerr << "### Kernel" << std::endl;
    std::cerr << FSS_KERNEL_TYPE::description("  ");
    exit(-1);
  }
  int num = load(argv[1], data);

  // Load parameters from command line
  {
    std::string str;
    for (int i = 2; i < argc; ++i) {
      str += argv[i];
      str += " ";
    }
    std::istringstream isst(str);
    int mask;
    double value;
    for (int i = 0; i < data.nparams(); ++i) {
      isst >> mask >> value;
      if (isst.fail()) {
        std::cerr << "Cannot load physical parameters" << std::endl;
        exit(-1);
      }
      p_mask.push_back(mask);
      p_params.push_back(value);
    }
    if ((argc - 2) == (data.nparams() + FSS_KERNEL_TYPE::nparams()) * 2) {
      for (int i = 0; i < FSS_KERNEL_TYPE::nparams(); ++i) {
        isst >> mask >> value;
        if (isst.fail()) {
          std::cerr << "Cannot load hyper parameters" << std::endl;
          exit(-1);
        }
        h_mask.push_back(mask);
        h_params.push_back(value);
      }
    }else{
      for (int i = 0; i < FSS_KERNEL_TYPE::nparams(); ++i) {
        h_mask.push_back(1);
        h_params.push_back(1.0);
      }
    }
    // Initial parameters
    std::cout << "# Initial arguments" << std::endl << "# " << str << std::endl;
  }


#ifdef MAIN_CHECK_LOAD
  // Output data and parameters
  std::cerr << "# Data:" << std::endl;
  for (int i = 0; i < data.npoints(); ++i)
    std::cerr << "# [" << i << "] " << data.get_L(i) << " " << data.get_T(i)
              << " " << data.get_A(i) << " " << data.get_EA(i) << std::endl;
  std::cerr << "# Parameters:" << std::endl;
  for (int i = 0; i < data.nparams(); ++i)
    std::cerr << "# (p_mask[" << i << "], p_params[" << i << "])=(" << p_mask[i] << ", " << p_params[i] << ")" << std::endl;
  for (int i = 0; i < data.nparams(); ++i)
    std::cerr << "# (h_mask[" << i << "], h_params[" << i << "])=(" << h_mask[i] << ", " << h_params[i] << ")" << std::endl;
#endif

  return num;
};

/** @brief Output all results

    @param[in] data Data
    @param[in] bayesian_fss Class of Bayesian inference
    @param[in] p_params physical parameters
    @param[in] h_params hyper parameters
 */
void output(const FSS_DATA_TYPE& data,
            const GPR::Regression<FSS_DATA_TYPE, FSS_KERNEL_TYPE>& bayesian_fss,
            const std::vector<double>& p_params,
            const std::vector<double>& h_params){
  // Inference results
  double ll = bayesian_fss.calc_ll(data, p_params, h_params);
  std::cout << "# Log-likelihood=" << ll << std::endl;
  std::cout << "# Number of data points=" << data.npoints() << std::endl;
  for (int i = 0; i < p_params.size(); ++i)
    std::cout << "# p[" << i << "]=" << p_params[i] << std::endl;
  for (int i = 0; i < h_params.size(); ++i)
    std::cout << "# h[" << i << "]=" << h_params[i] << std::endl;
  // Scaling results by unnormalized variables
  for (int i = 0; i < data.npoints(); ++i)
    std::cout << (data.get_T(i) - p_params[0]) * pow(data.get_L(i), p_params[1]) << " "
      #ifdef FSS_C
              << data.get_A(i) * pow(data.get_L(i), -p_params[2]) / ( 1e0 + p_params[3] * pow(data.get_L(i), -p_params[4])) << " "
              << data.get_EA(i) * pow(data.get_L(i), -p_params[2]) / ( 1e0 + p_params[3] * pow(data.get_L(i), -p_params[4])) << " "
      #else
              << data.get_A(i) * pow(data.get_L(i), -p_params[2]) << " "
              << data.get_EA(i) * pow(data.get_L(i), -p_params[2]) << " "
      #endif
              << data.get_L(i) << " "
              << data.get_T(i) << " "
              << data.get_A(i) << " "
              << data.get_EA(i) << std::endl;
  std::cout << std::endl << std::endl;
  // Inferred scaling function by unnormalized variables
  if (data.npoints() > 0) {
    double xmin, xmax;
    xmin = xmax = data.x(0, p_params);
    for (int i = 1; i < data.npoints(); ++i) {
      if (xmin > data.x(i, p_params)) xmin = data.x(i, p_params);
      if (xmax < data.x(i, p_params)) xmax = data.x(i, p_params);
    }
    double lmax = data.get_Lm();
    for (int i = 0; i <= 100; ++i) {
      double X = xmin + (xmax - xmin) / 100e0 * i;
      double mean, variance;
      bayesian_fss.conditional_probability_y(X, mean, variance, data, p_params, h_params);
      double t = data.get_T(X, p_params);
      double a = data.get_A(mean);
      double ea = data.get_EA(sqrt(variance));
      std::cout << (t - p_params[0]) * pow(lmax, p_params[1]) << " "
        #ifdef FSS_C
                << a * pow(lmax, -p_params[2]) / ( 1e0 + p_params[3] * pow(lmax, -p_params[4])) << " "
                << ea * pow(lmax, -p_params[2]) / ( 1e0 + p_params[3] * pow(lmax, -p_params[4])) << " "
        #else
                << a * pow(lmax, -p_params[2]) << " "
                << ea * pow(lmax, -p_params[2]) << " "
        #endif
                << std::endl;
    }
  }
  std::cout << std::endl << std::endl;
  // Scaling results by normalized variables
  for (int i = 0; i < data.npoints(); ++i)
    std::cout << data.x(i, p_params) << " " << data.y(i, p_params) << " " << data.e(i, p_params) << " "
              << data.get_L(i) << " "
              << data.get_T(i) << " "
              << data.get_A(i) << " "
              << data.get_EA(i) << std::endl;
}

/** @brief Output all results of Monte Carlo estimation

    @param[in] data Data
    @param[in] bayesian_fss Class of Bayesian inference
    @param[in] p_params physical parameters
    @param[in] ep_params Error of physical parameters
    @param[in] h_params hyper parameters
    @param[in] eh_params Error of hyper parameters
 */
void output_mc(const FSS_DATA_TYPE& data,
               const GPR::Regression<FSS_DATA_TYPE, FSS_KERNEL_TYPE>& bayesian_fss,
               const std::vector<double>& p_params,
               const std::vector<double>& ep_params,
               const std::vector<double>& h_params,
               const std::vector<double>& eh_params){
  // Inference results
  double ll = bayesian_fss.calc_ll(data, p_params, h_params);
  std::cout << "# Log-likelihood=" << ll << std::endl;
  std::cout << "# Number of data points=" << data.npoints() << std::endl;
  for (int i = 0; i < p_params.size(); ++i)
    std::cout << "# p[" << i << "]=" << p_params[i] << " " << ep_params[i] << std::endl;
  for (int i = 0; i < h_params.size(); ++i)
    std::cout << "# h[" << i << "]=" << h_params[i] << " " << eh_params[i] << std::endl;
  // Scaling results by unnormalized variables
  for (int i = 0; i < data.npoints(); ++i)
    std::cout << (data.get_T(i) - p_params[0]) * pow(data.get_L(i), p_params[1]) << " "
        #ifdef FSS_C
              << data.get_A(i) * pow(data.get_L(i), -p_params[2]) / ( 1e0 + p_params[3] * pow(data.get_L(i), -p_params[4])) << " "
              << data.get_EA(i) * pow(data.get_L(i), -p_params[2]) / ( 1e0 + p_params[3] * pow(data.get_L(i), -p_params[4])) << " "
        #else
              << data.get_A(i) * pow(data.get_L(i), -p_params[2]) << " "
              << data.get_EA(i) * pow(data.get_L(i), -p_params[2]) << " "
        #endif
              << data.get_L(i) << " "
              << data.get_T(i) << " "
              << data.get_A(i) << " "
              << data.get_EA(i) << std::endl;
  std::cout << std::endl << std::endl;
  // Inferred scaling function by unnormalized variables
  if (data.npoints() > 0) {
    double xmin, xmax;
    xmin = xmax = data.x(0, p_params);
    for (int i = 1; i < data.npoints(); ++i) {
      if (xmin > data.x(i, p_params)) xmin = data.x(i, p_params);
      if (xmax < data.x(i, p_params)) xmax = data.x(i, p_params);
    }
    double lmax = data.get_Lm();
    for (int i = 0; i <= 100; ++i) {
      double X = xmin + (xmax - xmin) / 100e0 * i;
      double mean, variance;
      bayesian_fss.conditional_probability_y(X, mean, variance, data, p_params, h_params);
      double t = data.get_T(X, p_params);
      double a = data.get_A(mean);
      double ea = data.get_EA(sqrt(variance));
      std::cout << (t - p_params[0]) * pow(lmax, p_params[1]) << " "
        #ifdef FSS_C
                << a * pow(lmax, -p_params[2]) / ( 1e0 + p_params[3] * pow(lmax, -p_params[4])) << " "
                << ea * pow(lmax, -p_params[2]) / ( 1e0 + p_params[3] * pow(lmax, -p_params[4])) << " "
        #else
                << a * pow(lmax, -p_params[2]) << " "
                << ea * pow(lmax, -p_params[2]) << " "
        #endif
                << std::endl;
    }
  }
  std::cout << std::endl << std::endl;
  // Scaling results by normalized variables
  for (int i = 0; i < data.npoints(); ++i)
    std::cout << data.x(i, p_params) << " " << data.y(i, p_params) << " " << data.e(i, p_params) << " "
              << data.get_L(i) << " "
              << data.get_T(i) << " "
              << data.get_A(i) << " "
              << data.get_EA(i) << std::endl;
}

void output_mc(const std::vector<FSS_DATA_TYPE>& datas,
               const GPR::Regression<FSS_DATA_TYPE, FSS_KERNEL_TYPE>& bayesian_fss,
               const std::vector< std::vector<int> >& index_p_params,
               const std::vector< std::vector<int> >& index_h_params,
               const std::vector<double>& params,
               const std::vector<double>& delta_params,
               std::vector<double>& average,
               std::vector<double>& covariance){
  // Inference results
  double ll = 0;
  int ndata = 0;
  for (int i = 0; i < datas.size(); ++i) {
    std::vector<double> p_params(index_p_params[i].size());
    for (int j = 0; j < p_params.size(); ++j)
      p_params[j] = params[index_p_params[i][j]];
    std::vector<double> h_params(index_h_params[i].size());
    for (int j = 0; j < h_params.size(); ++j)
      h_params[j] = params[index_h_params[i][j]];
    ll += bayesian_fss.calc_ll(datas[i], p_params, h_params);
    ndata += datas[i].npoints();
  }

  std::cout << "# Log-likelihood=" << ll << std::endl;
  std::cout << "# Number of data points=" << ndata << std::endl;
  std::cout << "# Number of parameters=" << params.size() << std::endl;
  for (int i = 0; i < params.size(); ++i)
    std::cout << "# p[" << i << "]=" << average[i] << " " << sqrt(covariance[i * params.size() + i]) << std::endl;
  for (int i = 0; i < params.size(); ++i)
    for (int j = 0; j < params.size(); ++j)
      std::cout << "# cov[" << i << ", " << j << "]=" << covariance[i * params.size() + j] << std::endl;

  for (int is = 0; is < datas.size(); ++is) {
    std::vector<double> p_params(index_p_params[is].size());
    for (int j = 0; j < p_params.size(); ++j)
      p_params[j] = params[index_p_params[is][j]];
    std::vector<double> h_params(index_h_params[is].size());
    for (int j = 0; j < h_params.size(); ++j)
      h_params[j] = params[index_h_params[is][j]];

    // Scaling results by unnormalized variables
    for (int i = 0; i < datas[is].npoints(); ++i)
      std::cout << (datas[is].get_T(i) - p_params[0]) * pow(datas[is].get_L(i), p_params[1]) << " "
        #ifdef FSS_C
                << datas[is].get_A(i) * pow(datas[is].get_L(i), -p_params[2]) / ( 1e0 + p_params[3] * pow(datas[is].get_L(i), -p_params[4])) << " "
                << datas[is].get_EA(i) * pow(datas[is].get_L(i), -p_params[2]) / ( 1e0 + p_params[3] * pow(datas[is].get_L(i), -p_params[4])) << " "
        #else
                << datas[is].get_A(i) * pow(datas[is].get_L(i), -p_params[2]) << " "
                << datas[is].get_EA(i) * pow(datas[is].get_L(i), -p_params[2]) << " "
        #endif
                << datas[is].get_L(i) << " "
                << datas[is].get_T(i) << " "
                << datas[is].get_A(i) << " "
                << datas[is].get_EA(i) << std::endl;
    std::cout << std::endl << std::endl;
    // Inferred scaling function by unnormalized variables
    if (datas[is].npoints() > 0) {
      double xmin, xmax;
      xmin = xmax = datas[is].x(0, p_params);
      for (int i = 1; i < datas[is].npoints(); ++i) {
        if (xmin > datas[is].x(i, p_params)) xmin = datas[is].x(i, p_params);
        if (xmax < datas[is].x(i, p_params)) xmax = datas[is].x(i, p_params);
      }
      double lmax = datas[is].get_Lm();
      for (int i = 0; i <= 100; ++i) {
        double X = xmin + (xmax - xmin) / 100e0 * i;
        double mean, variance;
        bayesian_fss.conditional_probability_y(X, mean, variance, datas[is], p_params, h_params);
        double t = datas[is].get_T(X, p_params);
        double a = datas[is].get_A(mean);
        double ea = datas[is].get_EA(sqrt(variance));
        std::cout << (t - p_params[0]) * pow(lmax, p_params[1]) << " "
          #ifdef FSS_C
                  << a * pow(lmax, -p_params[2]) / ( 1e0 + p_params[3] * pow(lmax, -p_params[4])) << " "
                  << ea * pow(lmax, -p_params[2]) / ( 1e0 + p_params[3] * pow(lmax, -p_params[4])) << " "
          #else
                  << a * pow(lmax, -p_params[2]) << " "
                  << ea * pow(lmax, -p_params[2]) << " "
          #endif
                  << std::endl;
      }
    }
    std::cout << std::endl << std::endl;
    // Scaling results by normalized variables
    for (int i = 0; i < datas[is].npoints(); ++i)
      std::cout << datas[is].x(i, p_params) << " " << datas[is].y(i, p_params) << " " << datas[is].e(i, p_params) << " "
                << datas[is].get_L(i) << " "
                << datas[is].get_T(i) << " "
                << datas[is].get_A(i) << " "
                << datas[is].get_EA(i) << std::endl;
    std::cout << std::endl << std::endl;
  }
}
