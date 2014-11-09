/* main_root.cc
 *
 * Copyright (C) 2012, 2013 Kenji Harada
 * Copyright (C) 2014 Kenji Harada, Yuichi Motoyama
 *
 */
/**
   @file main_root.cc
   @brief Application code of find a root of f(x)=C by Bayesian extrapolation based on Gaussian process regression

   @code
   Usage: ./broot [data_file] [C] [xmin] [xmax]
   @endcode
*/
#include <vector>
#include <cstring>
#include "gpr.hpp"
#include "gpr_extrapolate.hpp"

typedef GPR::EXTRAPOLATE::Data DATA_TYPE;
typedef GPR::EXTRAPOLATE::Gaussian_Kernel KERNEL_TYPE;

int setup(int argc, char** argv, DATA_TYPE& data, double& C, double& xmin, double& xmax);
void find_root(const DATA_TYPE& data, double C,
                const GPR::Regression<DATA_TYPE, KERNEL_TYPE>& bayesian_extrapolation,
                const std::vector<std::vector<double> >& params, double& xmin, double& xmax);

int main(int argc, char **argv){
  DATA_TYPE data;
  GPR::Regression<DATA_TYPE, KERNEL_TYPE> bayesian_extrapolation;
  std::vector<std::vector<double> > params(2); // [1]=p_params
  std::vector<std::vector<int> > masks(2);     // [1]=p_mask

  // Initialize hyper parameters
  for (int i = 0; i < KERNEL_TYPE::nparams(); ++i) {
    masks[0].push_back(1);
    params[0].push_back(1);
  }
  // setup
  double xmin, xmax;
  double C;
  int num = setup(argc, argv, data, C, xmin, xmax);
  std::cout << "# Number of data points= " << num << std::endl;
  // Find maximum log-likelihood
  bayesian_extrapolation.search_mll(data, params[1], masks[1], params[0], masks[0]);
  // Find a cross
  find_root(data, C, bayesian_extrapolation, params, xmin, xmax);
  printf("# [%g, %g]\n", xmin, xmax);
  printf("%g %g\n", (xmin + xmax) / 2, (xmax - xmin) / 2);
  return 0;
}

/** @brief Setup of data, parameters and masks

    @param[in] argc Number of arguments in command line
    @param[in] argv Arguments in command line
    @param[out] data Data
    @param[out] C C
    @param[out] xmin minimum value of x
    @param[out] xmax maximum value of x
 */
int setup(int argc, char** argv, DATA_TYPE& data, double& C, double& xmin, double& xmax){
  // Load data from file
  if (argc < 2) {
    std::cerr << "### Usage" << std::endl
              << "  " << argv[0] << " [data_file] [y] [xmin] [xmax]" << std::endl;
    std::cerr << "### Data" << std::endl;
    std::cerr << data.description("  ");
    std::cerr << "### Kernel" << std::endl;
    std::cerr << KERNEL_TYPE::description("  ");
    exit(-1);
  }
  int num = 0;
  std::ifstream fin(argv[1]);
  if (!fin.good()) {
    std::cerr << "Cannot open the file " << argv[1] << std::endl;
    exit(-1);
  }
  double x, y, e;
  while (1) {
    char str[256];
    fin.getline(str, 256);
    if (!fin.good()) break;
    if (std::strlen(str) > 0 && str[0] != '#') {
      std::istringstream isst(str);
      isst >> x >> y >> e;
      if (isst.fail()) {
        std::cerr << "# Skip a line (" << str << ")" << std::endl;
        continue;
      }
      data.set(x, y, e);
      if (num == 0)
        xmax = xmin = x;
      else{
        if (xmin > x) xmin = x;
        if (xmax < x) xmax = x;
      }
      ++num;
    }
  }
  if (num == 0) {
    std::cerr << "No data point" << std::endl;
    exit(-1);
  }

  if (argc < 3){
    C = 0.0;
  }else{
    std::istringstream isst(argv[2]);
    isst >> C;
    if (isst.fail()) exit(-1);
  }

  if (argc == 5) {
    double x;
    std::istringstream isst(argv[3]);
    isst >> x;
    if (isst.fail()) exit(-1);
    if (xmin < x) xmin = x;
    std::istringstream isst2(argv[4]);
    isst2 >> x;
    if (isst2.fail()) exit(-1);
    if (xmax > x) xmax = x;
  }
  return num;
};

/** @brief Output all results

    @param[in] data Data
    @param[in] C C
    @param[in] bayesian_extrapolation Class of Bayesian inference
    @param[in] params parameters
    @param[in] xmin minimum value of x searched
    @param[in] xmax maximum value of x searched
    @param[out] xmin minimum value of cross interval
    @param[out] xmax maximum value of cross interval
 */
void find_root(const DATA_TYPE& data, double C,
                const GPR::Regression<DATA_TYPE, KERNEL_TYPE>& bayesian_extrapolation,
                const std::vector<std::vector<double> >& params, double& xmin, double& xmax){
  // Extrapolation results
  std::vector<double> xmins(2);
  std::vector<double> xmaxs(2);
  for ( int id = 0; id < 2; ++id) {
    xmins[id] = xmin;
    xmaxs[id] = xmax;
  }
  for ( int level = 0; level <= 8; ++level) {
    for ( int id = 0; id < 2; ++id) {
      std::vector<double> vx(11);
      std::vector<int> vs(11);
      for (int i = 0; i <= 10; ++i) {
        double x = xmins[id] + (xmaxs[id] - xmins[id]) / 10e0 * i;
        vx[i] = x;
        double y, e;
        bayesian_extrapolation.conditional_probability_y(data.xtoX(x), y, e, data, params[1], params[0]);
        y = data.Ytoy(y);
        e = data.Etoe(sqrt(e));
        if (fabs(y - C) > e ) {
          if (y > C)
            vs[i] = 1;
          else
            vs[i] = -1;
        }else{
          std::cerr << "|Y-C|=" << fabs(y - C) << ", E=" << e << std::endl;
          vs[i] = 0;
        }
        std::cerr << "CHECK (" << level << "):(" << id << "):(" << i << "):" << x << ":" << vs[i] << std::endl;
      }
      if (id == 0) {
        int i = 1;
        while (vs[0] != 0 && vs[0] == vs[i])
          ++i;
        xmins[id] = vx[i - 1];
        xmaxs[id] = vx[i];
      }else{
        int i = 9;
        while (vs[10] != 0 && vs[10] == vs[i])
          --i;
        xmins[id] = vx[i];
        xmaxs[id] = vx[i + 1];
      }
    }
  }
  xmin = xmins[0];
  xmax = xmaxs[1];
}
