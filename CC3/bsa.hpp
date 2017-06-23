/* bsa.hpp
 *
 * Copyright (C) 2016 Kenji Harada
 *
 */
#ifndef BSA_HPP
#define BSA_HPP

#define _USE_MATH_DEFINES
#include "gpr.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace BSA {
class MultiDim_DataSet;
const static std::string version = "0.2";
};

class BSA::MultiDim_DataSet : public GPR::GK_DataSet {
public:
  /*
    Scaling form:
        A(L, x1, x2, ...) = L^c0 f[ (x1-g1)L^c1, ... ]

    Data:
        X[0] = L
        X[1] = x1
        X[2] = x2
        ...
        X[ndim] = x_ndim
        X[ndim+1] = A
        X[ndim+2] = delta_A

    Parameters:
        Params[0]=g1
        Params[1]=c1
        Params[2]=g2
        Params[3]=c2
        ...
        Params[2*(sndim-1)] = g_sndim
        Params[2*(sndim-1)+1] = c_sndim
        Params[2*sndim] = c0
   */
  MultiDim_DataSet() {
    ndim = 0;
    sndim = 0;
  }
  MultiDim_DataSet(int dimension, int data_dimension = -1) {
    set_dim(dimension, data_dimension);
  }
  void set_dim(int dimension, int data_dimension = -1) {
    sndim = dimension;
    if (data_dimension == -1)
      ndim = dimension;
    else
      ndim = data_dimension;
    nphy = sndim * 2 + 1;
    xmax.resize(ndim);
    xmin.resize(ndim);
    xmid.resize(ndim);
    xr.resize(ndim);
  }
  std::string description() const {
    std::ostringstream osst;
    osst << "Bayesian finite size scaling on a multi-dimensional space."
         << std::endl;
    return osst.str();
  }
  int num_p_shared() const { return nphy; }
  int num_dim() const { return sndim; }
  double value_y(const std::vector<double> &Params,
                 const std::vector<double> &X) const {
    return (X[ndim + 1] * std::pow(X[0] / lmax, -Params[2 * sndim]) - ymid) /
           yr;
  }
  double value_e(const std::vector<double> &Params,
                 const std::vector<double> &X) const {
    return X[ndim + 2] * std::pow(X[0] / lmax, -Params[2 * sndim]) / yr;
  }
  double value_x(const std::vector<double> &Params,
                 const std::vector<double> &X, int index) const {
    return (X[index + 1] - Params[index * 2]) *
           std::pow(X[0] / lmax, Params[index * 2 + 1]) / xr[index];
  }
  double value_dy(const std::vector<double> &Params, int p_index,
                  const std::vector<double> &X) const {
    if (static_cast<unsigned int>(p_index) == (2 * sndim))
      return X[ndim + 1] * std::pow(X[0] / lmax, -Params[2 * sndim]) *
             (-std::log(X[0] / lmax)) / yr;
    else
      return 0;
  }
  double value_de(const std::vector<double> &Params, int p_index,
                  const std::vector<double> &X) const {
    if (static_cast<unsigned int>(p_index) == (2 * sndim))
      return X[ndim + 2] * std::pow(X[0] / lmax, -Params[2 * sndim]) *
             (-std::log(X[0] / lmax)) / yr;
    else
      return 0;
  }
  double value_dx(const std::vector<double> &Params, int p_index,
                  const std::vector<double> &X, int index) const {
    if (p_index == (index * 2))
      return -std::pow(X[0] / lmax, Params[index * 2 + 1]) / xr[index];
    else if (p_index == (index * 2 + 1))
      return (X[index + 1] - Params[index * 2]) *
             std::pow(X[0] / lmax, Params[index * 2 + 1]) *
             std::log(X[0] / lmax) / xr[index];
    else
      return 0;
  }
  void reverse_convert(const std::vector<double> &Params,
                       const std::vector<double> &X_regression,
                       std::vector<double> &X) const {
    X[ndim] = (X_regression[0] * yr + ymid) *
              std::pow(X[0] / lmax, Params[2 * sndim]);
    X[ndim + 1] =
        X_regression[1] * yr * std::pow(X[0] / lmax, Params[2 * sndim]);
  }
  void get_info(std::map<std::string, double> &info) const {
    info["LMIN"] = lmin;
    info["LMAX"] = lmax;
    info["YMID"] = ymid;
    info["YR"] = yr;
    for (unsigned int i = 0; i < ndim; ++i) {
      std::ostringstream osst;
      osst << "XMID" << i;
      info[osst.str()] = xmid[i];
    }
    for (unsigned int i = 0; i < ndim; ++i) {
      std::ostringstream osst;
      osst << "XR" << i;
      info[osst.str()] = xr[i];
    }
  }

  void add(const std::vector<double> &Xdata) {
    if (num() == 0 || lmax < Xdata[0]) {
      lmax = Xdata[0];
      for (unsigned int i = 0; i < ndim; ++i)
        xmin[i] = xmax[i] = Xdata[i + 1];
      ymin = ymax = Xdata[ndim + 1];
    } else if (lmax == Xdata[0]) {
      for (unsigned int i = 0; i < ndim; ++i) {
        xmax[i] = std::max(xmax[i], Xdata[i + 1]);
        xmin[i] = std::min(xmin[i], Xdata[i + 1]);
      }
      ymax = std::max(ymax, Xdata[ndim + 1]);
      ymin = std::min(ymin, Xdata[ndim + 1]);
    }
    for (unsigned int i = 0; i < ndim; ++i) {
      xmid[i] = (xmax[i] + xmin[i]) / 2;
      xr[i] = (xmax[i] - xmin[i]) / 2;
    }
    if (num() == 0)
      lmin = Xdata[0];
    else
      lmin = std::min(lmin, Xdata[0]);
    ymid = (ymax + ymin) / 2;
    yr = (ymax - ymin) / 2;
    GK_DataSet::add(Xdata);
  }

  bool check_anomaly(std::vector<double> &Params) { return true; }

private:
  unsigned int ndim, sndim, nphy;
  std::vector<double> xmax;
  std::vector<double> xmin;
  std::vector<double> xmid;
  std::vector<double> xr;
  double ymax, ymin;
  double ymid, yr;
  double lmin, lmax;
};
#endif
