/* main_fss_multi.cc
 *
 * Copyright (C) 2014, 2015 Kenji Harada
 *
 */
/**
   @mainpage Kernel method for the finite-size scaling analysis of critical
phenomena

This package includes a reference code of the new kernel method for
the finite-size scaling analysis of critical phenomena. The new method
is based on the Gaussian process regression, which is called kernel
method. After the advent of kernel methods in the machine learning
community, the method of data analysis was drastically
changed. Because the kernel method is very flexible for complex real
data, the power of the kernel method can also help our scaling
analysis.

The original paper <a
href="http://hdl.handle.net/10.1103/PhysRevE.84.056704">[1]</a> mainly
considered the scaling law without corrections to scaling. The recent
progress resolves the problem of corrections to scaling systematically
<a href="http://arxiv.org/abs/1410.3622">[2]</a>. In fact, this
reference code can deal with a general finite-size scaling law with or
without corrections to scaling as follows: @f[ A(T, L) = L^{c_2} F[ (
T - T_c ) L^{c_1} ], @f] and @f[ A(T, L) = L^{c_2} \left(F[ ( T - T_c
) L^{c_1} ] + G[ ( T - T_c ) L^{c_1} ] L^{-c_3} \right), @f] where @f$
A @f$ is an observable, @f$L@f$ is a system size, @f$T_c@f$ is a
critical point, and @f$c_3 > 0@f$ is an irrelevant critical exponent.

Because we assume only the smoothness of the functions @f$ F @f$ and
@f$ G @f$, we do not need to limit the range of data near a critical
point so that the scaling function is approximated by a
polynomial. Thus, this code can be widely applied to real data in
critical phenomena.  This code also uses a hybrid Monte Carlo to
estimate the confidential intervals of inferred values.



I recommend this code for all users of the finite-size scaling
analysis, because it is very flexible, and it automatically and easily
can estimates the values of critical exponents and a critical point
without the technical knowledge. If this new method is useful to your
study, I hope that you cite my papers to spread this new method.

April, 2015

Kenji Harada

[References]
<ol>
          <li> Kenji Harada:
            <em> Bayesian inference in the scaling analysis of critical
phenomena,</em>
            Physical Review E <strong> 84 </strong> (2011) 056704.
            <br>DOI: <a
href="http://hdl.handle.net/10.1103/PhysRevE.84.056704">10.1103/PhysRevE.84.056704</a>
          </li>

          <li> Kenji Harada:
          <em> Kernel Method for Corrections to Scaling,</em>
          <a href="http://arxiv.org/abs/1410.3622">arXiv:1410.3622</a> (2014).
</ol>
<hr>

@section TOC Table of contents
- @ref Comp "Compile"
- @ref Usage "Usage"
  - @ref TEST "Test"
- @ref Data_format "Format of input data file"
  - @ref CASE1 "For a single observable"
  - @ref CASE2 "For multiple observables simultaneously"
- @ref SF "Scaling form and parameters"
  - @ref C1A "For a standard scaling form"
  - @ref C1B "For a scaling form with corrections to scaling"
- @ref output "Output"

<hr>

@section Comp Compile

   To compile, you can use "make" command as follows.
   @code
   % make
   @endcode

   Before compiling this code, You need to install (FORTRAN-base) BLAS
   and LAPACK libraries: For example, intel MKL, ACML, or various free
   implementations. After that, you need to setup CCLIBS in Makefile
   to link them correctly.

   @section Usage Usage

   @code
COMMAND [Options] [Data file] [Parameters]
  [Option]
    -c                : estimate confidential intervals of parameters by MC (default: off)
    -e MAP::EPSILON   : set an epsilon for FR-CG algorithm (default: 1e-8)
    -f SCALING::FORM  : set a scaling form [0:standard, 1:with correction] (default: 0)
    -h                : help
    -i MC::SEED       : set a seed of random number (default::20140318)
    -l MC::LIMIT      : set the limit to the number of MC samples (default: 20000)
    -m MC::NMCS       : set the number of MC samples (default: 1000)
    -n DATA::N        : set the number of data sets (default: 1)
    -s MAP::STEP_SIZE : set a step size of FR-CG algorithm (default: 1e-4)
    -t MAP::TOL       : set a tolerance of FR-CG algorithm (default: 1e-3)
    -w OUTPUT::XSCALE : set a xscale of outputted scaling function (default: 1)
  [Data file]
    If data_file = '-', data are loaded from STDIN
  [Parameters]
    parameter         := mask [0:fixed, 1:unfixed] + initial_value (default of mask: 1, default of initial_value: automatically initialized)
   @endcode

   @note
If you want good inference results, we recommend the option "-c"
which use a sophisticated Monte Carlo estimation. But the
computational cost of Monte Carlo estimation is high, because this
code carefully adjusts the sampling condition in the early stage.

@note
You can reduce the length of [Parameters] list. If you don't set a value
of a parameter, it is automatically initialized. But, it is better to
start from good initial values to succeed in the inferences of
critical exponents, because we need to solve a non-linear optimization
problem in the first stage.

   @subsection TEST Test
   @code
   % make test
   ./new_bfss Sample/Ising-square-Binder.dat 1 0.42 1 0.9 1 0.1 > test.op 2>test.log
   ./new_bfss -c Sample/Ising-square-Binder.dat 1 0.42 1 0.9 1 0.1 > test_mc.op 2>test_mc.log
   @endcode
   The examples of code's output are in the "Sample" folder.

   @section Data_format Format of input data file
   @subsection CASE1 For a single observable (DATA::N == 1)
   The format of data file is as follows.
   @code
   # L   T            A              Error_of_A
   128   4.200000e-01 6.271240e-02   1.336090e-03
   @endcode
   A line has to be ended with the newline character. Comment lines
   starts with the character '#'. A null line is ignored. There are four values
   in each line. The value of @f$ L @f$ is in the 1st column
   of data file.  The value of @f$ T @f$ is in the 2nd column. The
   value of @f$ A @f$ is in the 3rd column. The value of @f$ \delta
   A @f$ is in the 4th column. If a line is not correctly
   formatted, it will be skipped.

   @subsection CASE2 For multiple observables simultaneously (DATA::N > 1)
   In this case, we will assume an independent scaling function for
   each observable with different critical exponents. But the values
   of @f$T_c@f$ and @f$c_1@f$ and @f$c_3@f$ are shared. We will
   infer values to succeed in all scaling analyses simultaneously.
   The format of data file is as follows.
   @code
   # ID  L   T            A              Error_of_A
   0   128   4.200000e-01 6.271240e-02   1.336090e-03
   @endcode

   A line has to be ended with the newline character. Comment lines
   starts with the character '#'. A null line is ignored. There are
   five values in each line. The value of ID is in the 1st column of data
   file.  It is the identification number of data set. It starts from
   0. The maximum number is (DATA::N - 1).  The value of @f$ L @f$ is
   in the 2nd column.  The value of @f$ T @f$ is in the 3rd
   column. The value of @f$ A @f$ is in the 4th column. The value of
   @f$ \delta A @f$ is in the 5th column. If a line is not correctly
   formatted, it will be skipped.

   @section SF Scaling form and parameters
   @subsection C1A For a standard scaling form
   The finite-size scaling form is written as
   @f[
   A(T, L) = L^{c_2} F[ ( T - T_c ) L^{c_1} ],
   @f] where @f$ A @f$ is an observable. The triplet of a point
   is defined as
   @f[
   X = (T - T_c ) (L/L_{MAX})^{c_1} / R_X, Y = (A / (L/L_{MAX})^{c_2} -
   Y_0)/R_Y,\\
   E = \delta A/ (L/L_{MAX})^{c_2}/R_Y,
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


   @subsection C1B For a scaling form with corrections to scaling
   The finite-size scaling form is written as
   @f[
   A(T, L) = L^{c_2} F[ ( T - T_c ) L^{c_1}, L^{-c_3} ],
   @f] where @f$ A @f$ is an observable. The triplet of a point
   is defined as
   @f[
   X_1 = (T - T_c ) (L/L_{MAX})^{c_1} / R_X, X_2 = (L/L_{MIN})^{-c_3},\\
   Y = (A / (L/L_{MAX})^{c_2} - Y_0)/R_Y,
   E = \delta A/ (L/L_{MAX})^{c_2}/R_Y,
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
   + \theta_1^2 \exp\left[ - \frac{|X_1(i)- X_1(j)|^2}{2\theta_2^2}\right]\\
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

   @note
   In the case of multiple observables, the first part of a parameter list is
for shared parameters.
   The non-shared parameters are put after the shared parameters. For
example,
   @f$ (T_c, c_1, c_3, c_2, \theta_0, \theta_1, \theta_2, \theta_3, \theta_4,
c_2', \theta_0', \theta_1', \theta_2', \theta_3', \theta_4')@f$.

   @section output Output

   The process of the optimization and sampling of parameters is
   reported to a standard err channel.  The inference results of
   parameters is reported to a standard output channel.

   @subsection O1 Header comment
   The value of inferred parameters and the confidential intervals are
   written in header comments as follows.
   @verbatim
# p[0] = 4.4068289283487466e-01 6.5315475997187452e-06 --> Average and standard deviation
...
# cov[0, 0]=4.2661114047391705e-11 --> Value of covariance matrix's element
...
# local p[0] = 4.4068289283487466e-01 --> Value of parameter for scaling data
...
   @endverbatim

   @subsection O2 Results
   The remain part consists of some outputs for each data set.  A output for a
   data set consists of four groups as follows:

   - Scaling results
   - Scaling function
   - Scaling results by normalized variables
   - Scaling function by normalized variables

   These output groups are separated two null lines.

   The line of the first group "Scaling results" contains
   as
   @f$ [ (T - T_c ) L^{c_1}, A / L^{c_2}, \delta A/L^{c_2},L, T, A, \delta A ]
@f$
   for a standard scaling form, and
   as
   @f$ [ (T - T_c ) L^{c_1}, A / L^{c_2}, \delta A/L^{c_2}, L^{-c_3}, L, T, A,
\delta A ] @f$
   for the case of corrections to scaling.

   The second group for "Scaling function" consists of 100 points of the
   inferred scaling function in the thermodynamic limit. The output range of
   x axis is equal to the range of the largest system size. It can be changed
   the option "-w". Three values, @f$ [X, \mu(X), \sqrt{\sigma^2(X)} ]@f$ are
   outputed in each line. @f$\mu(X)@f$ is an inference result for scaling
function @f$ F[X] @f$.
   @f$ \sqrt{\sigma^2(X)} @f$ is a confidential intervale of the inference
result.

   The third and the fourth groups use a normalized variables.
*/
/**
   @file main_fss.cc
   @brief Application code of Bayesian finite-size scaling

   This file is an application code of Bayesian finite-size scaling.
*/
#include <list>
#include <string>
#include <cstring>
#include <cmath>
#include <map>
#include <iostream>
#include <sstream>

#include "gpr.hpp"
#include "bsa.hpp"

// Prototype
void setup_option(std::list<std::string> &ARG,
                  std::map<std::string, double> &setting);

template <class FSS_DATASET>
void fss(std::list<std::string> &ARG,
         const std::map<std::string, double> &setting,
         std::vector<FSS_DATASET> &Datasets,
         GPR::Regression<FSS_DATASET> &bayesian_fss);

template <class FSS_DATASET>
double calculate_chi2(const std::vector<FSS_DATASET> &xdataset,
                      const std::vector<std::vector<int> > Index,
                      const std::vector<double> &Params,
                      std::vector<std::vector<double> > &Eigenvalue_sets);

// Main function
int main(int argc, char **argv) {
  std::list<std::string> ARG;
  for (int i = 1; i < argc; ++i)
    ARG.push_back(argv[i]);

  std::map<std::string, double> setting;
  setup_option(ARG, setting);
  std::cout << "# Scaling form : " << setting["SCALING::FORM"] << std::endl;
  switch (static_cast<int>(setting["SCALING::FORM"])) {
  case 1: {
    std::vector<BSA::DataSet_C> Datasets;
    GPR::Regression<BSA::DataSet_C> bayesian_fss;
    fss(ARG, setting, Datasets, bayesian_fss);
    break;
  }
  default: {
    std::vector<BSA::DataSet> Datasets;
    GPR::Regression<BSA::DataSet> bayesian_fss;
    fss(ARG, setting, Datasets, bayesian_fss);
    break;
  }
  }
  return 0;
}

std::string add_header(std::string str, std::string header) {
  std::istringstream ist(str);
  std::ostringstream ost;
  char strline[256];
  while (ist.getline(strline, 256))
    ost << header << strline << std::endl;
  return ost.str();
}

/**
   @brief Output usage.
*/
std::string output_usage() {
  std::ostringstream osst;
  osst << "### Usage" << std::endl;
  osst << "COMMAND [Options] [Data file] [Parameters]" << std::endl;
  osst << "  [Option]" << std::endl;
  osst << "    -c                : estimate confidential intervals of "
          "parameters by MC (default: off)" << std::endl;
  osst << "    -e MAP::EPSILON   : set an epsilon for FR-CG algorithm "
          "(default: 1e-8)" << std::endl;
  osst << "    -f SCALING::FORM  : set a scaling form [0:standard, "
          "1:with correction] (default: 0)" << std::endl;
  osst << "    -h                : help" << std::endl;
  osst << "    -i MC::SEED       : set a seed of random number (default: "
          "20140318)" << std::endl;
  osst << "    -l MC::LIMIT      : set the limit to the number of MC samples "
          "(default: 20000)" << std::endl;
  osst << "    -m MC::NMCS       : set the number of MC samples (default: 1000)"
       << std::endl;
  osst << "    -n DATA::N        : set the number of datasets (default: 1)"
       << std::endl;
  osst << "    -s MAP::STEP_SIZE : set a step size of FR-CG algorithm "
          "(default: 1e-4)" << std::endl;
  osst << "    -t MAP::TOL       : set a tolerance of FR-CG algorithm "
          "(default: 1e-3)" << std::endl;
  osst << "    -w OUTPUT::XSCALE : set a xscale of outputted scaling function "
          "(default: 1)" << std::endl;
  osst << "  [Data file]" << std::endl;
  osst << "    If data_file = '-', data are loaded from STDIN" << std::endl;
  osst << "  [Parameters]" << std::endl;
  osst << "    parameter         := mask [0:fixed, 1:unfixed] + initial_value "
          "(default of mask: 1, default of initial_value: automatically "
          "initialized)" << std::endl;
  return osst.str();
}

/**
   @brief Load option values.
 */
bool load_option(std::string name, std::string key, std::string error,
                 std::list<std::string>::iterator &it,
                 std::list<std::string> &ARG,
                 std::map<std::string, double> &setting) {
  if (*it == name) {
    it = ARG.erase(it);
    if (it == ARG.end()) {
      std::cerr << error << std::endl;
      exit(-1);
    }
    std::istringstream isst(*it);
    it = ARG.erase(it);
    isst >> setting[key];
    return true;
  } else
    return false;
}

/**
   @brief Setup of options.
 */
void setup_option(std::list<std::string> &ARG,
                  std::map<std::string, double> &setting) {
  setting["SCALING::FORM"] = 0;
  setting["USE_MC"] = 0;
  setting["DATA::N"] = 1;
  setting["OUTPUT::XSCALE"] = 1.0;
  bool need_help = false;
  for (std::list<std::string>::iterator it = ARG.begin(); it != ARG.end();) {
    if (*it == "-h")
      need_help = true;
    if (*it == "-c") {
      setting["USE_MC"] = 1e0;
      it = ARG.erase(it);
      continue;
    }
    if (load_option("-e", "MAP::EPSILON", "Not find a epsilon", it, ARG,
                    setting))
      continue;
    if (load_option("-f", "SCALING::FORM", "Not find a scaling form", it, ARG,
                    setting))
      continue;
    if (load_option("-i", "MC::SEED", "Not find a seed of random number", it,
                    ARG, setting))
      continue;
    if (load_option("-l", "MC::LIMIT",
                    "Not find the limit to the number of MC samples", it, ARG,
                    setting))
      continue;
    if (load_option("-m", "MC::NMCS", "Not find the number of MC samples", it,
                    ARG, setting))
      continue;
    if (load_option("-n", "DATA::N", "Not find the number of datasets", it, ARG,
                    setting))
      continue;
    if (load_option("-s", "MAP::STEP_SIZE", "Not find a step_size", it, ARG,
                    setting))
      continue;
    if (load_option("-t", "MAP::TOL", "Not find a tolerance", it, ARG, setting))
      continue;
    if (load_option("-w", "OUTPUT::XSCALE",
                    "X-scale of outputted scaling function", it, ARG, setting))
      continue;
    ++it;
  }
  if (need_help) {
    std::cerr << output_usage();
    std::cerr << "### Description of scaling form "
              << static_cast<int>(setting["SCALING::FORM"]) << std::endl;
    switch (static_cast<int>(setting["SCALING::FORM"])) {
    case 1: {
      std::cerr << BSA::DataSet_C::description();
      break;
    }
    default: {
      std::cerr << BSA::DataSet::description();
      break;
    }
    }
    exit(0);
  }
  if (ARG.size() == 0) {
    std::cerr << output_usage();
    exit(-1);
  }
}

/**
   @brief Setup of data, parameters and masks.
 */
template <class FSS_DATASET>
void load_data(std::list<std::string> &ARG,
               const std::map<std::string, double> &setting,
               std::vector<FSS_DATASET> &Datasets, std::vector<double> &Params,
               std::vector<int> &Mask) {
  std::string filename = ARG.front();
  ARG.pop_front();
  int NSET = static_cast<int>(setting.find("DATA::N")->second);
  Datasets.resize(NSET);
  if (filename == "-") {
    if (!std::cin.good()) {
      std::cerr << "No standard input" << std::endl;
      exit(-1);
    }
    while (1) {
      char str[256];
      std::cin.getline(str, 256);
      if (!std::cin.good())
        break;
      if (std::strlen(str) > 0 && str[0] != '#') {
        std::istringstream isst(str);
        int id;
        double L, T, A, DA;
        if (NSET > 1) {
          isst >> id >> L >> T >> A >> DA;
        } else {
          id = 0;
          isst >> L >> T >> A >> DA;
        }
        if (isst.fail()) {
          std::cerr << "# Skip a line (" << str << ")" << std::endl;
          continue;
        } else {
          std::vector<double> data;
          data.push_back(L);
          data.push_back(T);
          data.push_back(A);
          data.push_back(DA);
          Datasets[id].add(data);
        }
      }
    }
  } else {
    std::ifstream fin(filename.c_str());
    if (!fin.good()) {
      std::cerr << "Cannot open the file " << filename << std::endl;
      exit(-1);
    }
    while (1) {
      char str[256];
      fin.getline(str, 256);
      if (!fin.good())
        break;
      if (std::strlen(str) > 0 && str[0] != '#') {
        std::istringstream isst(str);
        int id;
        double L, T, A, DA;
        if (NSET > 1) {
          isst >> id >> L >> T >> A >> DA;
        } else {
          id = 0;
          isst >> L >> T >> A >> DA;
        }
        if (isst.fail()) {
          std::cerr << "# Skip a line (" << str << ")" << std::endl;
          continue;
        } else {
          std::vector<double> data;
          data.push_back(L);
          data.push_back(T);
          data.push_back(A);
          data.push_back(DA);
          Datasets[id].add(data);
        }
      }
    }
  }
  std::cout << "# Number of datasets = " << NSET << std::endl;
  for (int i = 0; i < NSET; ++i)
    std::cout << "# Number of data points in dataset[" << i
              << "] = " << Datasets[i].num() << std::endl;
  // Load parameters from command line
  int np_shared = Datasets[0].num_p_shared();
  int np = Datasets[0].num_p();
  int np_local = np - np_shared;
  Params.resize(np_shared + np_local * NSET);
  {
    std::vector<double> Params_ini;
    Datasets[0].initialize_parameters(Params_ini);
    for (int i = 0; i < np_shared; ++i)
      Params[i] = Params_ini[i];
    for (int i = 0; i < NSET; ++i)
      for (int j = 0; j < np_local; ++j)
        Params[np_shared + np_local * i + j] = Params_ini[np_shared + j];
  }
  Mask.resize(np_shared + np_local * NSET, 1);
  std::string str;
  for (std::list<std::string>::iterator it = ARG.begin(); it != ARG.end();
       ++it) {
    str += *it;
    str += " ";
  }
  std::istringstream isst(str);
  for (int i = 0; i < Params.size(); ++i) {
    int xmask;
    double xvalue;
    isst >> xmask >> xvalue;
    if (isst.fail())
      break;
    Mask[i] = xmask;
    Params[i] = xvalue;
  }
  // Initial parameters
  std::cout << "# Initial arguments" << std::endl << "#";
  int nmask = 0;
  for (int i = 0; i < Params.size(); ++i) {
    std::cout << " " << Mask[i] << " " << Params[i];
    if (Mask[i] == 1)
      ++nmask;
  }
  std::cout << std::endl;
  std::cout << "# Number of parameters = " << Params.size() << std::endl;
  std::cout << "# Number of free parameters = " << nmask << std::endl;
}

/**
   @brief Output all results.
*/
template <class FSS_DATASET>
void output(const std::vector<FSS_DATASET> &Datasets,
            const std::vector<std::vector<int> > Index,
            const GPR::Regression<FSS_DATASET> &bayesian_fss,
            const std::vector<double> &Params,
            const std::vector<double> &Average,
            const std::vector<double> &Covariance,
            const std::map<std::string, double> &setting) {
  int NSET = Datasets.size();
  int np = Datasets[0].num_p();
  // Inference results
  double f_ll = bayesian_fss.log_likelihood(Datasets, Index, Params);
  std::cout.precision(16);
  std::cout << std::scientific;
  std::cout << "# Log-likelihood = " << f_ll << std::endl;
  {
    std::vector<std::vector<double> > Eigenvalue_sets;
    std::cout << "# chi^2 in Gaussian = "
              << calculate_chi2(Datasets, Index, Params, Eigenvalue_sets)
              << std::endl;
    double chi2 = 0;
    for (int t = 0; t < NSET; ++t) {
      std::vector<double> Params_local(np);
      for (int j = 0; j < np; ++j)
        Params_local[j] = Params[Index[t][j]];
      std::vector<std::vector<double> > point_regressions;
      for (int i = 0; i < Datasets[t].num(); ++i) {
        std::vector<double> data;
        Datasets[t].get(i, data);
        std::vector<double> xc;
        Datasets[t].convert(Params_local, data, xc);
        point_regressions.push_back(xc);
      }
      bayesian_fss.infer_regression(Datasets[t], Params_local,
                                    &point_regressions);
      for (int i = 0; i < Datasets[t].num(); ++i) {
        std::vector<double> data;
        Datasets[t].get(i, data);
        std::vector<double> xc;
        Datasets[t].convert(Params_local, data, xc);
        double x = (xc[0] - point_regressions[i][0]) / xc[1];
        chi2 += x * x;
      }
    }
    std::cout << "# chi^2 = " << chi2 << std::endl;
  }

  std::cout << add_header(Datasets[0].description(), "# ");

  if (setting.find("USE_MC")->second > 0) {
    for (int i = 0; i < Params.size(); ++i)
      std::cout << "# p[" << i << "] = " << Average[i] << " "
                << std::sqrt(Covariance[i + i * Params.size()]) << std::endl;
    for (int i = 0; i < Params.size(); ++i)
      for (int j = 0; j < Params.size(); ++j)
        std::cout << "# cov[" << i << ", " << j
                  << "]=" << Covariance[i + j * Params.size()] << std::endl;
  } else {
    for (int i = 0; i < Params.size(); ++i)
      std::cout << "# p[" << i << "] = " << Params[i] << std::endl;
  }
  // Scaled data
  for (int i = 0; i < Params.size(); ++i)
    std::cout << "# Global p[" << i << "] = " << Params[i] << std::endl;

  for (int t = 0; t < NSET; ++t) {
    std::cout << "# Dataset[" << t << "]" << std::endl;

    std::map<std::string, double> info;
    Datasets[t].get_info(info);
    std::cout << "# LMIN = " << info["LMIN"] << std::endl;
    std::cout << "# LMAX = " << info["LMAX"] << std::endl;
    std::cout << "# RX = " << info["RX"] << std::endl;
    std::cout << "# RY = " << info["RY"] << std::endl;
    std::cout << "# Y0 = " << info["Y0"] << std::endl;

    std::vector<double> Params_local(np);
    for (int j = 0; j < np; ++j)
      Params_local[j] = Params[Index[t][j]];
    for (int i = 0; i < Params_local.size(); ++i)
      std::cout << "# Local p[" << i << "] = " << Params_local[i] << std::endl;

    /// Scaling results by unnormalized variables
    for (int i = 0; i < Datasets[t].num(); ++i) {
      std::vector<double> data;
      Datasets[t].get(i, data);
      std::vector<double> xc;
      Datasets[t].convert(Params_local, data, xc);
      switch (static_cast<int>(setting.find("SCALING::FORM")->second)) {
      case 1:
        std::cout << (data[1] - Params_local[0]) *
                         std::pow(data[0], Params_local[1]) << " "
                  << data[2] * std::pow(data[0], -Params_local[3]) << " "
                  << data[3] * std::pow(data[0], -Params_local[3]) << " "
                  << std::pow(data[0], -Params_local[2]) << " " << data[0]
                  << " " << data[1] << " " << data[2] << " " << data[3] << " "
                  << std::endl;
        break;
      default:
        std::cout << (data[1] - Params_local[0]) *
                         std::pow(data[0], Params_local[1]) << " "
                  << data[2] * std::pow(data[0], -Params_local[2]) << " "
                  << data[3] * std::pow(data[0], -Params_local[2]) << " "
                  << data[0] << " " << data[1] << " " << data[2] << " "
                  << data[3] << " " << std::endl;
        break;
      }
    }
    std::cout << std::endl << std::endl;
    /// Scaling function by unnormalized variables
    {
      const int NUM_POINTS = 100;
      std::vector<std::vector<double> > point_regressions;
      for (int i = 0; i < NUM_POINTS; ++i) {
        std::vector<double> xc;
        switch (static_cast<int>(setting.find("SCALING::FORM")->second)) {
        case 1:
          xc.resize(4);
          xc[3] = 0;
          break;
        default:
          xc.resize(3);
          break;
        }
        xc[2] = (((info["TMIN"] + info["TMAX"]) * 0.5 +
                  setting.find("OUTPUT::XSCALE")->second *
                      (info["TMAX"] - info["TMIN"]) * 0.5 *
                      ((2.0 * i) / NUM_POINTS - 1.0)) -
                 Params_local[0]) /
                info["RX"];
        point_regressions.push_back(xc);
      }
      bayesian_fss.infer_regression(Datasets[t], Params_local,
                                    &point_regressions);
      for (int i = 0; i < NUM_POINTS; ++i) {
        switch (static_cast<int>(setting.find("SCALING::FORM")->second)) {
        case 1:
          std::cout << point_regressions[i][2] *
                           std::pow(info["LMAX"], Params_local[1]) * info["RX"]
                    << " "
                    << (point_regressions[i][0] * info["RY"] + info["Y0"]) /
                           std::pow(info["LMAX"], Params_local[3]) << " "
                    << point_regressions[i][1] * info["RY"] /
                           std::pow(info["LMAX"], Params_local[3]) << std::endl;
          break;
        default:
          std::cout << point_regressions[i][2] *
                           std::pow(info["LMAX"], Params_local[1]) * info["RX"]
                    << " "
                    << (point_regressions[i][0] * info["RY"] + info["Y0"]) /
                           std::pow(info["LMAX"], Params_local[2]) << " "
                    << point_regressions[i][1] * info["RY"] /
                           std::pow(info["LMAX"], Params_local[2]) << std::endl;
          break;
        }
      }
      std::cout << std::endl << std::endl;
    }
    /// Scaling results by normalized variables
    for (int i = 0; i < Datasets[t].num(); ++i) {
      std::vector<double> data;
      Datasets[t].get(i, data);
      std::vector<double> xc;
      Datasets[t].convert(Params_local, data, xc);
      switch (static_cast<int>(setting.find("SCALING::FORM")->second)) {
      case 1:
        std::cout << xc[2] << " " << xc[0] << " " << xc[1] << " " << xc[3]
                  << " " << data[0] << " " << data[1] << " " << data[2] << " "
                  << data[3] << " " << std::endl;
        break;
      default:
        std::cout << xc[2] << " " << xc[0] << " " << xc[1] << " " << data[0]
                  << " " << data[1] << " " << data[2] << " " << data[3] << " "
                  << std::endl;
        break;
      }
    }
    std::cout << std::endl << std::endl;
    /// Scaling function by normalized variables
    {
      const int NUM_POINTS = 100;
      std::vector<std::vector<double> > point_regressions;
      for (int i = 0; i < NUM_POINTS; ++i) {
        std::vector<double> xc;
        switch (static_cast<int>(setting.find("SCALING::FORM")->second)) {
        case 1:
          xc.resize(4);
          xc[3] = 0;
          break;
        default:
          xc.resize(3);
          break;
        }
        xc[2] = (((info["TMIN"] + info["TMAX"]) * 0.5 +
                  setting.find("OUTPUT::XSCALE")->second *
                      (info["TMAX"] - info["TMIN"]) * 0.5 *
                      ((2.0 * i) / NUM_POINTS - 1.0)) -
                 Params_local[0]) /
                info["RX"];
        point_regressions.push_back(xc);
      }
      bayesian_fss.infer_regression(Datasets[t], Params_local,
                                    &point_regressions);
      for (int i = 0; i < NUM_POINTS; ++i) {
        std::cout << point_regressions[i][2] << " " << point_regressions[i][0]
                  << " " << point_regressions[i][1] << std::endl;
      }
      std::cout << std::endl << std::endl;
    }
  }
  return;
}

/**
   @brief FSS analysis.
*/
template <class FSS_DATASET>
void fss(std::list<std::string> &ARG,
         const std::map<std::string, double> &setting,
         std::vector<FSS_DATASET> &Datasets,
         GPR::Regression<FSS_DATASET> &bayesian_fss) {
  std::vector<double> Params;
  std::vector<int> Mask;

  load_data(ARG, setting, Datasets, Params, Mask);

  std::vector<std::vector<int> > Index(Datasets.size());
  int np_shared = Datasets[0].num_p_shared();
  int np = Datasets[0].num_p();
  int np_local = np - np_shared;
  for (int t = 0; t < Datasets.size(); ++t) {
    Index[t].resize(np);
    for (int i = 0; i < np_shared; ++i)
      Index[t][i] = i;
    for (int i = 0; i < np_local; ++i)
      Index[t][i + np_shared] = np_shared + np_local * t + i;
  }

  std::vector<double> H;
  bayesian_fss.map(Datasets, Index, Params, Mask, setting, H);

  std::vector<double> Average;
  std::vector<double> Covariance;
  if (setting.find("USE_MC")->second > 0) {
    //    bayesian_fss.mc(dataset, Params, Mask, Average, Covariance, setting);
    bayesian_fss.mc(Datasets, Index, Params, Mask, Average, Covariance,
                    setting);
    std::copy(Average.begin(), Average.end(), Params.begin());
    output(Datasets, Index, bayesian_fss, Params, Average, Covariance, setting);
  } else
    output(Datasets, Index, bayesian_fss, Params, Average, Covariance, setting);
}

/** BLASS and LAPACK **/
extern "C" {
int dsyev_(char *JOBZ, char *UPLO, int *N, double *A, int *LDA, double *W,
           double *WORK, int *LWORK, int *INFO);
}
#define DSYEV dsyev_

/**
  @breif Calculate chi^2 of Gaussian process.
*/
template <class FSS_DATASET>
double calculate_chi2(const FSS_DATASET &xdataset,
                      const std::vector<double> &Params,
                      std::vector<double> &Eigenvalues) {
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
  double chi2 = 0;
  for (int i = 0; i < N; ++i)
    chi2 += Y[i] * Y2[i];
  chi2 /= 2;

  Eigenvalues.resize(N);
  xdataset.gram(Params, xgram);
  {
    char JOBZ = 'N';
    char UPLO = 'U';
    int LDA = N;
    int LWORK = -1;
    int INFO;
    double size;
    DSYEV(&JOBZ, &UPLO, &LDA, &(xgram[0]), &LDA, &(Eigenvalues[0]), &size,
          &LWORK, &INFO);
    LWORK = size;
    std::vector<double> WORK(LWORK);
    DSYEV(&JOBZ, &UPLO, &LDA, &(xgram[0]), &LDA, &(Eigenvalues[0]), &(WORK[0]),
          &LWORK, &INFO);
  }
  return chi2;
}

/**
  @breif Calculate chi^2 of Gaussian process for datasets.
*/
template <class FSS_DATASET>
double calculate_chi2(const std::vector<FSS_DATASET> &Datasets,
                      const std::vector<std::vector<int> > Index,
                      const std::vector<double> &Params,
                      std::vector<std::vector<double> > &Eigenvalue_sets) {
  int NSET = Datasets.size();
  int np = Datasets[0].num_p();
  Eigenvalue_sets.resize(NSET);
  double chi2 = 0;
  for (int t = 0; t < NSET; ++t) {
    std::vector<double> Params_local(np);
    for (int j = 0; j < np; ++j)
      Params_local[j] = Params[Index[t][j]];
    chi2 += calculate_chi2(Datasets[t], Params_local, Eigenvalue_sets[t]);
  }
  return chi2;
}
