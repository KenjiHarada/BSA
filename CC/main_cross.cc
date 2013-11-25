/* main_cross.cc
 *
 * Copyright (C) 2012, 2013 Kenji Harada
 *
 */
/**
   @file main_cross.cc
   @brief Application code of find a cross by Bayesian extrapolation based on Gaussian process regression

   @code
   Usage: ./bcross [data_file_1] [data_file_2] [xmin] [xmax]
   @endcode
*/
#include <vector>
#include "gpr.hpp"
#include "gpr_extrapolate.hpp"

typedef GPR::EXTRAPOLATE::Data DATA_TYPE;
typedef GPR::EXTRAPOLATE::Gaussian_Kernel KERNEL_TYPE;

int setup(int argc, char** argv, std::vector<DATA_TYPE>& data, double& xmin, double& xmax);
void find_cross(const std::vector<DATA_TYPE>& data,
                const GPR::Regression<DATA_TYPE, KERNEL_TYPE>& bayesian_extrapolation,
                const std::vector<std::vector<double> >& params, double& xmin, double& xmax);

int main(int argc, char **argv){
  std::vector<DATA_TYPE> data(2);
  GPR::Regression<DATA_TYPE, KERNEL_TYPE> bayesian_extrapolation;
  std::vector<std::vector<double> > params(3); // [2]=p_params
  std::vector<std::vector<int> > masks(3);     // [2]=p_mask

  // Initialize hyper parameters
  for (int i = 0; i < KERNEL_TYPE::nparams(); ++i) {
    for (int j = 0; j < 2; ++j) {
      masks[j].push_back(1);
      params[j].push_back(1);
    }
  }
  // setup
  double xmin, xmax;
  int num = setup(argc, argv, data, xmin, xmax);
  std::cout << "# Number of data points= " << num << std::endl;
  // Find maximum log-likelihood
  for (int i = 0; i < 2; ++i)
    bayesian_extrapolation.search_mll(data[i], params[2], masks[2], params[i], masks[i]);
  // Find a cross
  find_cross(data, bayesian_extrapolation, params, xmin, xmax);
  printf("# [%g, %g]\n", xmin, xmax);
  printf("%g %g\n", (xmin + xmax) / 2, (xmax - xmin) / 2);
  return 0;
}

/** @brief Setup of data, parameters and masks

    @param[in] argc Number of arguments in command line
    @param[in] argv Arguments in command line
    @param[out] data Data
    @param[out] xmin minimum value of x
    @param[out] xmax maximum value of x
 */
int setup(int argc, char** argv, std::vector<DATA_TYPE>& data, double& xmin, double& xmax){
  // Load data from file
  if (argc < 3) {
    std::cerr << "### Usage" << std::endl
              << "  " << argv[0] << " [data_file_1] [data_file_2] [xmin] [xmax]" << std::endl;
    std::cerr << "### Data" << std::endl;
    std::cerr << data[0].description("  ");
    std::cerr << "### Kernel" << std::endl;
    std::cerr << KERNEL_TYPE::description("  ");
    exit(-1);
  }
  std::vector<double> xmins(2);
  std::vector<double> xmaxs(2);
  int num = 0;
  for (int i = 0; i < 2; ++i) {
    std::ifstream fin(argv[1 + i]);
    if (!fin.good()) {
      std::cerr << "Cannot open the file " << argv[1 + i] << std::endl;
      exit(-1);
    }
    double x, y, e;
    while (1) {
      char str[256];
      fin.getline(str, 256);
      if (!fin.good()) break;
      if (str[0] != '#') {
        std::istringstream isst(str);
        isst >> x >> y >> e;
        if (isst.fail()) break;
        data[i].set(x, y, e);
        if (num == 0)
          xmaxs[i] = xmins[i] = x;
        else{
          if (xmins[i] > x) xmins[i] = x;
          if (xmaxs[i] < x) xmaxs[i] = x;
        }
        ++num;
      }
    }
  }
  if (num == 0) {
    std::cerr << "No data point" << std::endl;
    exit(-1);
  }
  if (xmaxs[0] > xmins[1] || xmaxs[1] > xmins[0]) {
    std::vector<double> xs;
    for (int i = 0; i < 2; ++i) {
      xs.push_back(xmins[i]);
      xs.push_back(xmaxs[i]);
    }
    for (int i = 1; i <= 3; ++i)
      for (int j = 1; j <= 4 - i; ++j)
        if (xs[j] < xs[j - 1])
          std::swap(xs[j], xs[j - 1]);
    xmin = xs[1];
    xmax = xs[2];
  }else{
    std::cerr << "No overlapped regions in data." << std::endl;
    exit(-1);
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
    @param[in] bayesian_extrapolation Class of Bayesian inference
    @param[in] params parameters
    @param[in] xmin minimum value of x searched
    @param[in] xmax maximum value of x searched
    @param[out] xmin minimum value of cross interval
    @param[out] xmax maximum value of cross interval
 */
void find_cross(const std::vector<DATA_TYPE>& data,
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
        double ya, ea;
        bayesian_extrapolation.conditional_probability_y(data[0].xtoX(x), ya, ea, data[0], params[2], params[0]);
        double yb, eb;
        bayesian_extrapolation.conditional_probability_y(data[1].xtoX(x), yb, eb, data[1], params[2], params[1]);
        ya = data[0].Ytoy(ya);
        yb = data[1].Ytoy(yb);
        ea = data[0].Etoe(sqrt(ea));
        eb = data[1].Etoe(sqrt(eb));
        if (fabs(ya - yb) > (ea + eb)) {
          if (ya > yb)
            vs[i] = 1;
          else
            vs[i] = -1;
        }else{
          std::cerr << "|YA-YB|=" << fabs(ya - yb) << ", EA=" << ea << ", EB=" << eb << std::endl;
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
