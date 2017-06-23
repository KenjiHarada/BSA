/* main_fss.cc
 *
 * Copyright (C) 2014, 2015 Kenji Harada
 *
 */
/**
   @file main_fss.cc
   @brief Application code of Bayesian finite-size scaling

   This file is an application code of Bayesian finite-size scaling.
*/
#include <cmath>
#include <cstring>
#include <iostream>
#include <list>
#include <map>
#include <sstream>
#include <string>

#include "bsa.hpp"
#include "gpr.hpp"

// Prototype
void setup_option(std::vector<std::string> &ARG,
                  std::map<std::string, double> &setting);

template <class FSS_DATASET>
void fss(std::vector<std::string> &ARG,
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
  std::vector<std::string> ARG;
  for (int i = 1; i < argc; ++i)
    ARG.push_back(argv[i]);

  std::map<std::string, double> setting;
  setup_option(ARG, setting);
  std::vector<BSA::MultiDim_DataSet> Datasets;
  GPR::Regression<BSA::MultiDim_DataSet> bayesian_fss;
  fss(ARG, setting, Datasets, bayesian_fss);
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
          "parameters by MC (default: off)"
       << std::endl;
  osst << "    -e MAP::EPSILON   : set an epsilon for FR-CG algorithm "
          "(default: 1e-8)"
       << std::endl;
  osst << "    -h                : help" << std::endl;
  osst << "    -i MC::SEED       : set a seed of random number (default: "
          "20140318)"
       << std::endl;
  osst << "    -l MC::LIMIT      : set the limit to the number of MC samples "
          "(default: 20000)"
       << std::endl;
  osst << "    -m MC::NMCS       : set the number of MC samples (default: 1000)"
       << std::endl;
  osst << "    -n DATA::N        : set the number of datasets (default: 1)"
       << std::endl;
  osst << "    -s MAP::STEP_SIZE : set a step size of FR-CG algorithm "
          "(default: 1e-4)"
       << std::endl;
  osst << "    -t MAP::TOL       : set a tolerance of FR-CG algorithm "
          "(default: 1e-3)"
       << std::endl;
  osst << "    -w OUTPUT::XSCALE : set a xscale of outputted scaling function "
          "(default: 1)"
       << std::endl;
  osst << "  [Data file]" << std::endl;
  osst << "    If data_file = '-', data are loaded from STDIN" << std::endl;
  osst << "  [Parameters]" << std::endl;
  osst << "    parameter         := mask [0:fixed, 1:unfixed] + initial_value "
          "(default of mask: 1, default of initial_value: automatically "
          "initialized)"
       << std::endl;
  return osst.str();
}

/**
   @brief Load option values.
 */
bool load_option(std::string name, std::string key, std::string error,
                 std::vector<std::string>::iterator &it,
                 std::vector<std::string> &ARG,
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
void setup_option(std::vector<std::string> &ARG,
                  std::map<std::string, double> &setting) {
  setting["USE_MC"] = 0;
  setting["USE_LOCAL_OPTIMIZATION"] = 1;
  setting["CHECK_ANOMALY"] = 1;
  setting["DATA::N"] = 1;
  setting["DATA::SUB_N"] = 1;
  setting["DATA::NDIM"] = -1;
  setting["SCALING::NDIM"] = 1;
  setting["OUTPUT::XSCALE"] = 1.0;
  bool need_help = false;
  for (std::vector<std::string>::iterator it = ARG.begin(); it != ARG.end();) {
    if (*it == "-h")
      need_help = true;
    if (*it == "-c") {
      setting["USE_MC"] = 1e0;
      it = ARG.erase(it);
      continue;
    }
    if (*it == "-L") {
      setting["USE_LOCAL_OPTIMIZATION"] = 0;
      it = ARG.erase(it);
      continue;
    }
    if (*it == "-A") {
      setting["CHECK_ANOMALY"] = 0;
      it = ARG.erase(it);
      continue;
    }
    if (load_option("-D", "DATA::NDIM",
                    "Not find the number of x-values of a data point", it, ARG,
                    setting))
      continue;
    if (load_option("-d", "SCALING::NDIM",
                    "Not find the number of arguments of a scaling function",
                    it, ARG, setting))
      continue;
    if (load_option("-e", "MAP::EPSILON", "Not find a epsilon", it, ARG,
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
    if (load_option("-N", "DATA::SUB_N",
                    "Not find the number of datasets which partially share", it,
                    ARG, setting))
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
    std::cerr << "### Description of scaling form" << std::endl;
    BSA::MultiDim_DataSet Datasets;
    std::cerr << Datasets.description();
    exit(0);
  }
  if (ARG.size() == 0) {
    std::cerr << output_usage();
    exit(-1);
  }
  if (setting.find("DATA::NDIM")->second == -1)
    setting["DATA::NDIM"] = setting["SCALING::NDIM"];
}

/**
   @brief Setup of data, parameters and masks.
 */
template <class FSS_DATASET>
void load_data(std::vector<std::string> &ARG,
               const std::map<std::string, double> &setting,
               std::vector<FSS_DATASET> &Datasets, std::vector<double> &Params,
               std::vector<int> &Mask) {
  std::string filename = ARG.front();
  ARG.erase(ARG.begin());
  int NSUBSET = static_cast<int>(setting.find("DATA::SUB_N")->second);
  int NSET = static_cast<int>(setting.find("DATA::N")->second);
  int NDATASET = NSET * NSUBSET;
  Datasets.resize(NDATASET);
  int ndim = static_cast<int>(setting.find("DATA::NDIM")->second);
  int sndim = static_cast<int>(setting.find("SCALING::NDIM")->second);
  for (FSS_DATASET &x : Datasets)
    x.set_dim(sndim, ndim);
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
        std::vector<double> xdata(ndim + 3);
        int id = 0;
        if (NDATASET > 1)
          isst >> id;
        for (int i = 0; i < (ndim + 3); ++i)
          isst >> xdata[i];
        if (isst.fail()) {
          std::cerr << "# Skip a line (" << str << ")" << std::endl;
          continue;
        } else {
          Datasets[id].add(xdata);
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
        std::vector<double> xdata(ndim + 3);
        int id = 0;
        if (NDATASET > 1)
          isst >> id;
        for (int i = 0; i < (ndim + 3); ++i)
          isst >> xdata[i];
        if (isst.fail()) {
          std::cerr << "# Skip a line (" << str << ")" << std::endl;
          continue;
        } else {
          Datasets[id].add(xdata);
        }
      }
    }
  }
  std::cout << "# Number of datasets = " << NDATASET << std::endl;
  if (NSET > 1) {
    std::cout << "# Number of a group of datasets = " << NSET << std::endl;
    if (NSUBSET > 1)
      std::cout << "# Size of a group of datasets which partially share = "
                << NSUBSET << std::endl;
  }
  for (int i = 0; i < NDATASET; ++i)
    std::cout << "# Number of data points in dataset[" << i
              << "] = " << Datasets[i].num() << std::endl;
  // Load parameters from command line
  int np = Datasets[0].num_p();
  int np_shared = Datasets[0].num_p_shared();
  int np_partial = 0;
  if (NSUBSET > 1)
    np_partial = 1;
  int np_global = np_shared - np_partial;
  int np_local = np - np_shared;
  std::cout << "# Number of parameters of a dataset = " << np << std::endl;
  if (NSET > 1) {
    std::cout << "# Number of global shared parameters of a dataset = "
              << np_global << std::endl;
    if (np_partial > 0)
      std::cout << "# Number of partial shared parameters of a dataset = "
                << np_partial << std::endl;
    std::cout << "# Number of local parameters of a dataset = " << np_local
              << std::endl;
  }

  Params.resize(np_global + (np_partial + np_local * NSUBSET) * NSET);
  {
    std::vector<double> Params_ini;
    Datasets[0].initialize_parameters(Params_ini);
    for (int i = 0; i < np_global; ++i)
      Params[i] = Params_ini[i];
    for (int i = 0; i < NSET; ++i)
      for (int j = 0; j < np_partial; ++j) {
        int ip = np_global + (np_partial + np_local * NSUBSET) * i + j;
        assert(ip < np_global + (np_partial + np_local * NSUBSET) * NSET);
        Params[ip] = Params_ini[np_global + j];
      }
    for (int i = 0; i < NSET; ++i)
      for (int j = 0; j < NSUBSET; ++j)
        for (int k = 0; k < np_local; ++k) {
          int ip = np_global + (np_partial + np_local * NSUBSET) * i +
                   np_partial + np_local * j + k;
          assert(ip < np_global + (np_partial + np_local * NSUBSET) * NSET);
          Params[ip] = Params_ini[np_global + np_partial + k];
        }
  }

  std::string str;
  for (std::string x : ARG) {
    str.append(x);
    str.append(" ");
  }
  Mask.resize(Params.size(), 1);
  std::istringstream isst(str);
  for (unsigned int i = 0; i < Params.size(); ++i) {
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
  for (unsigned int i = 0; i < Params.size(); ++i) {
    std::cout << " " << Mask[i] << " " << Params[i];
    if (Mask[i] == 1)
      ++nmask;
  }
  std::cout << std::endl;
  std::cout << "# Number of total parameters = " << Params.size() << std::endl;
  std::cout << "# Number of total free parameters = " << nmask << std::endl;
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
  int NDATASET = Datasets.size();
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
    for (int t = 0; t < NDATASET; ++t) {
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
    for (unsigned int i = 0; i < Params.size(); ++i)
      std::cout << "# p[" << i << "] = " << Average[i] << " "
                << std::sqrt(Covariance[i + i * Params.size()]) << std::endl;
    for (unsigned int i = 0; i < Params.size(); ++i)
      for (unsigned int j = 0; j < Params.size(); ++j)
        std::cout << "# cov[" << i << ", " << j
                  << "]=" << Covariance[i + j * Params.size()] << std::endl;
  } else {
    for (unsigned int i = 0; i < Params.size(); ++i)
      std::cout << "# p[" << i << "] = " << Params[i] << std::endl;
  }
  // Scaled data
  for (unsigned int i = 0; i < Params.size(); ++i)
    std::cout << "# Global p[" << i << "] = " << Params[i] << std::endl;

  for (int t = 0; t < NDATASET; ++t) {
    std::cout << "# Dataset[" << t << "]" << std::endl;
    int ndim = Datasets[t].get(0).size() - 3;
    int sndim = Datasets[t].num_dim();
    std::map<std::string, double> info;
    Datasets[t].get_info(info);
    std::cout << "# LMIN = " << info["LMIN"] << std::endl;
    std::cout << "# LMAX = " << info["LMAX"] << std::endl;
    std::cout << "# YMID = " << info["YMID"] << std::endl;
    std::cout << "# YR = " << info["YR"] << std::endl;
    std::vector<double> xmid(ndim);
    std::vector<double> xr(ndim);
    for (int i = 0; i < ndim; ++i) {
      std::ostringstream osst;
      osst << "XMID" << i;
      xmid[i] = info[osst.str()];
      std::cout << "# " << osst.str() << " = " << xmid[i] << std::endl;
    }
    for (int i = 0; i < ndim; ++i) {
      std::ostringstream osst;
      osst << "XR" << i;
      xr[i] = info[osst.str()];
      std::cout << "# " << osst.str() << " = " << xr[i] << std::endl;
    }

    std::vector<double> Params_local(np);
    for (int j = 0; j < np; ++j)
      Params_local[j] = Params[Index[t][j]];
    for (unsigned int i = 0; i < Params_local.size(); ++i)
      std::cout << "# Local p[" << i << "] = " << Params_local[i] << std::endl;
    std::vector<double> xc_min(sndim);
    std::vector<double> xc_max(sndim);
    for (int id = 0; id < Datasets[t].num(); ++id) {
      std::vector<double> data;
      Datasets[t].get(id, data);
      std::vector<double> xc;
      Datasets[t].convert(Params_local, data, xc);
      for (int i = 0; i < sndim; ++i) {
        if (id == 0) {
          xc_min[i] = xc[i + 2];
          xc_max[i] = xc[i + 2];
        }
        xc_max[i] = std::max(xc[i + 2], xc_max[i]);
        xc_min[i] = std::min(xc[i + 2], xc_min[i]);
      }
    }
    /// Scaling results by unnormalized variables
    for (int i = 0; i < Datasets[t].num(); ++i) {
      std::vector<double> data;
      Datasets[t].get(i, data);
      for (int i = 0; i < sndim; ++i)
        std::cout << (data[i + 1] - Params_local[2 * i]) *
                         std::pow(data[0], Params_local[2 * i + 1])
                  << " ";
      std::cout << data[ndim + 1] * std::pow(data[0], -Params_local[2 * sndim])
                << " "
                << data[ndim + 2] * std::pow(data[0], -Params_local[2 * sndim])
                << " ";
      for (int i = 0; i <= (ndim + 2); ++i)
        std::cout << data[i] << " ";
      std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
    /// Scaling function by unnormalized variables
    {
      const int NUM_POINTS = 100;
      unsigned int tnum =
          static_cast<unsigned int>(std::pow(NUM_POINTS, sndim));
      std::vector<double> xa(sndim + 2);
      std::vector<std::vector<double> > point_regressions;
      for (unsigned int i = 0; i < tnum; ++i) {
        unsigned int ix = i;
        for (int ip = 0; ip < sndim; ++ip) {
          unsigned int iy = ix % NUM_POINTS;
          double xc_mid = (xc_max[ip] + xc_min[ip]) / 2;
          double xc_r = (xc_max[ip] - xc_min[ip]) / 2;
          xa[ip + 2] = xc_mid +
                       setting.find("OUTPUT::XSCALE")->second * xc_r *
                           ((2e0 * iy) / (NUM_POINTS - 1e0) - 1e0);
          ix /= NUM_POINTS;
        }
        xa[0] = xa[1] = 0;
        point_regressions.push_back(xa);
      }
      bayesian_fss.infer_regression(Datasets[t], Params_local,
                                    &point_regressions);
      for (unsigned int i = 0; i < tnum; ++i) {
        for (int ip = 0; ip < sndim; ++ip)
          std::cout << point_regressions[i][ip + 2] *
                           std::pow(info["LMAX"], Params_local[2 * ip + 1]) *
                           xr[ip]
                    << " ";
        std::cout << (point_regressions[i][0] * info["YR"] + info["YMID"]) *
                         std::pow(info["LMAX"], -Params_local[2 * sndim])
                  << " "
                  << point_regressions[i][1] * info["YR"] *
                         std::pow(info["LMAX"], -Params_local[2 * sndim])
                  << std::endl;
      }
      std::cout << std::endl << std::endl;
    }
    /// Scaling results by normalized variables
    for (int id = 0; id < Datasets[t].num(); ++id) {
      std::vector<double> data;
      Datasets[t].get(id, data);
      std::vector<double> xc;
      Datasets[t].convert(Params_local, data, xc);
      for (int i = 0; i < sndim; ++i)
        std::cout << xc[i + 2] << " ";
      std::cout << xc[0] << " " << xc[1] << " ";
      for (int i = 0; i <= (ndim + 2); ++i)
        std::cout << data[i] << " ";
      std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;
    /// Scaling function by normalized variables
    {
      const int NUM_POINTS = 100;
      unsigned int tnum =
          static_cast<unsigned int>(std::pow(NUM_POINTS, sndim));
      std::vector<double> xa(sndim + 2);
      std::vector<std::vector<double> > point_regressions;
      for (unsigned int i = 0; i < tnum; ++i) {
        unsigned int ix = i;
        for (int ip = 0; ip < sndim; ++ip) {
          unsigned int iy = ix % NUM_POINTS;
          double xc_mid = (xc_max[ip] + xc_min[ip]) / 2;
          double xc_r = (xc_max[ip] - xc_min[ip]) / 2;
          xa[ip + 2] = xc_mid +
                       setting.find("OUTPUT::XSCALE")->second * xc_r *
                           ((2e0 * iy) / (NUM_POINTS - 1e0) - 1e0);
          ix /= NUM_POINTS;
        }
        xa[0] = xa[1] = 0;
        point_regressions.push_back(xa);
      }
      bayesian_fss.infer_regression(Datasets[t], Params_local,
                                    &point_regressions);
      for (unsigned int i = 0; i < tnum; ++i) {
        for (int ip = 0; ip < sndim; ++ip)
          std::cout << point_regressions[i][ip + 2] << " ";
        std::cout << point_regressions[i][0] << " " << point_regressions[i][1]
                  << std::endl;
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
void fss(std::vector<std::string> &ARG,
         const std::map<std::string, double> &setting,
         std::vector<FSS_DATASET> &Datasets,
         GPR::Regression<FSS_DATASET> &bayesian_fss) {
  std::vector<double> Params;
  std::vector<int> Mask;

  load_data(ARG, setting, Datasets, Params, Mask);
  int NSUBSET = static_cast<int>(setting.find("DATA::SUB_N")->second);
  int NSET = static_cast<int>(setting.find("DATA::N")->second);
  int NDATASET = NSET * NSUBSET;
  std::vector<std::vector<int> > Index(NDATASET);
  int np_shared = Datasets[0].num_p_shared();
  int np = Datasets[0].num_p();
  int np_partial = 0;
  if (NSUBSET > 1)
    np_partial = 1;
  int np_global = np_shared - np_partial;
  int np_local = np - np_global;
  for (int t = 0; t < NDATASET; ++t) {
    Index[t].resize(np);
    int id = static_cast<int>(t / NSUBSET);
    for (int i = 0; i < np_global; ++i)
      Index[t][i] = i;
    for (int i = 0; i < np_partial; ++i)
      Index[t][np_global + i] =
          np_global + (np_partial + np_local * NSUBSET) * id + i;
    for (int i = 0; i < np_local; ++i)
      Index[t][np_global + np_partial + i] =
          np_global + (np_partial + np_local * NSUBSET) * id + np_partial +
          np_local * (t % NSUBSET) + i;
  }
  std::vector<double> H;
  bayesian_fss.map(Datasets, Index, Params, Mask, setting, H);
  if (setting.find("CHECK_ANOMALY")->second > 0) {
    for (int t = 0; t < NDATASET; ++t) {
      std::vector<double> xParams(np);
      for (int i = 0; i < np; ++i)
        xParams[i] = Params[Index[t][i]];
      if (!Datasets[t].check_anomaly(xParams)) {
        std::cerr << "# Anomaly :" << t << std::endl;
        exit(-1);
      }
    }
    std::cerr << "# No anomaly" << std::endl;
  }
  if (setting.find("USE_LOCAL_OPTIMIZATION")->second > 0) {
    std::cerr << "# Optimize only local parameters" << std::endl;
    std::vector<int> tMask(Mask.size(), 0);
    for (int t = 0; t < NDATASET; ++t) {
      int id = static_cast<int>(t / NSUBSET);
      for (int i = 0; i < np_local; ++i)
        tMask[np_global + (np_partial + np_local * NSUBSET) * id + np_partial +
              np_local * (t % NSUBSET) + i] = 1;
    }
    bayesian_fss.map(Datasets, Index, Params, tMask, setting, H);
    if (setting.find("CHECK_ANOMALY")->second > 0) {
      for (int t = 0; t < NDATASET; ++t) {
        std::vector<double> xParams(np);
        for (int i = 0; i < np; ++i)
          xParams[i] = Params[Index[t][i]];
        if (!Datasets[t].check_anomaly(xParams)) {
          std::cerr << "# Anomaly :" << t << std::endl;
          exit(-1);
        }
      }
      std::cerr << "# No anomaly" << std::endl;
    }
  }
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
  int NDATASET = Datasets.size();
  int np = Datasets[0].num_p();
  Eigenvalue_sets.resize(NDATASET);
  double chi2 = 0;
  for (int t = 0; t < NDATASET; ++t) {
    std::vector<double> Params_local(np);
    for (int j = 0; j < np; ++j)
      Params_local[j] = Params[Index[t][j]];
    chi2 += calculate_chi2(Datasets[t], Params_local, Eigenvalue_sets[t]);
  }
  return chi2;
}
