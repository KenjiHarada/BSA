/* main_extrapolate.cc
 *
 * Copyright (C) 2013, Kenji Harada
 * Released under the MIT and GPLv3 licenses.
 *
 * To spread the Bayesian scaling analysis method,
 * I hope you will cite the following original paper:
 *   Kenji Harada, Physical Review E 84 (2011) 056704.
 */
/**
   @file main_extrapolate.cc
   @brief Application code of Bayesian extrapolation by Gaussian process regression

   @code
   Usage: ./bext [data_file] [x_values]
   @endcode
*/
#include <vector>
#include "gpr.hpp"
#include "gpr_extrapolate.hpp"

typedef GPR::EXTRAPOLATE::Data DATA_TYPE;
typedef GPR::EXTRAPOLATE::Gaussian_Kernel KERNEL_TYPE;

void setup(int argc, char** argv, DATA_TYPE& data, std::vector<double>& x_values);
void output(const DATA_TYPE& data,
            const GPR::Regression<DATA_TYPE, KERNEL_TYPE>& bayesian_extrapolation,
            const std::vector<double>& p_params, const std::vector<double>& h_params,
            const std::vector<double>& x_values);

int main(int argc, char **argv){
  DATA_TYPE data;
  GPR::Regression<DATA_TYPE, KERNEL_TYPE> bayesian_extrapolation;
  std::vector<double> p_params, h_params;
  std::vector<int> p_mask, h_mask;
  std::vector<double> x_values;

  // Initialize hyper parameters
  for (int i = 0; i < KERNEL_TYPE::nparams(); ++i) {
    h_mask.push_back(1);
    h_params.push_back(1);
  }
  // setup
  setup(argc, argv, data, x_values);
  // Find maximum log-likelihood
  bayesian_extrapolation.search_mll(data, p_params, p_mask, h_params, h_mask);
  // Output
  output(data, bayesian_extrapolation, p_params, h_params, x_values);
  return 0;
}

/** @brief Setup of data, parameters and masks

    @param[in] argc Number of arguments in command line
    @param[in] argv Arguments in command line
    @param[out] data Data
    @param[out] x_values X values of extrapolated points
 */
void setup(int argc, char** argv, DATA_TYPE& data, std::vector<double>& x_values){
  // Load data from file or STDIN
  if (argc < 3) {
    std::cerr << "### Usage" << std::endl
              << "  " << argv[0] << " [data_file] [x_values]" << std::endl;
    std::cerr << "### Data" << std::endl;
    std::cerr << data.description("  ");
    std::cerr << "### Kernel" << std::endl;
    std::cerr << KERNEL_TYPE::description("  ");
    exit(-1);
  }
  if (*argv[1] == '-') {
    double x, y, e;
    do {
      char str[256];
      std::cin.getline(str, 256);
      if (str[0] != '#') {
        std::istringstream isst(str);
        isst >> x >> y >> e;
        data.set(x, y, e);
      }
    } while (!std::cin.eof());
  }else{
    std::ifstream fin(argv[1]);
    double x, y, e;
    do {
      char str[256];
      fin.getline(str, 256);
      if (str[0] != '#') {
        std::istringstream isst(str);
        isst >> x >> y >> e;
        data.set(x, y, e);
      }
    } while (!fin.eof());
  }
  // set x_values
  for (int i = 2; i < argc; ++i) {
    std::istringstream isst(argv[i]);
    double x;
    isst >> x;
    x_values.push_back(x);
  }
};

/** @brief Output all results

    @param[in] data Data
    @param[in] bayesian_extrapolation Class of Bayesian inference
    @param[in] p_params physical parameters
    @param[in] h_params hyper parameters
    @param[in] x_values X values of extrapolated points
 */
void output(const DATA_TYPE& data,
            const GPR::Regression<DATA_TYPE, KERNEL_TYPE>& bayesian_extrapolation,
            const std::vector<double>& p_params, const std::vector<double>& h_params,
            const std::vector<double>& x_values){
  // Inference results
  double ll = bayesian_extrapolation.calc_ll(data, p_params, h_params);
  std::cout << "# Log-likelihood=" << ll << std::endl;
  for (unsigned int i = 0; i < p_params.size(); ++i)
    std::cout << "# p[" << i << "]=" << p_params[i] << std::endl;
  for (unsigned int i = 0; i < h_params.size(); ++i)
    std::cout << "# h[" << i << "]=" << h_params[i] << std::endl;
  // Extrapolation results
  for (int i = 0; i < x_values.size(); ++i) {
    double mean, variance;
    bayesian_extrapolation.conditional_probability_y(data.xtoX(x_values[i]), mean, variance, data, p_params, h_params);
    std::cout << x_values[i] << " " << data.Ytoy(mean) << " " << data.Etoe(sqrt(variance)) << std::endl;
  }
}
