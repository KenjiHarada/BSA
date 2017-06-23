/* optimize.hpp
 *
 * Copyright (C) 2014 Kenji Harada
 *
 */
#ifndef OPTIMIZE_HPP
#define OPTIMIZE_HPP
#include <cmath>
#include <vector>
/**
   @file optimize.hpp
   @brief Classes for optimization of an objective function.
*/

/** @namespace GPR::OPT
    @brief Optimization of parameters to minimize an objective function.
 */
namespace GPR {
namespace OPT {
#ifndef DEFINED_OBJECTIVE_FUNCTION
#define DEFINED_OBJECTIVE_FUNCTION
/** @class GPR::OPT::ObjectiveFunction
    @brief Sample class of objective function to minimize by GPR::OPT::CG_FR.
*/
class ObjectiveFunction {
public:
  /** @brief Calculate a value of objective function.
      @param[in] X Argument of objective function
      @return Value of objective function
   */
  double f(const std::vector<double> &X) const;

  /** @brief Calculate a gradient of objective function.
      @param[in] X Argument of objective function
      @param[out] Grad Gradient vector of objective function
   */
  void df(const std::vector<double> &X, std::vector<double> &Grad) const;
};
#endif

/** @class GPR::OPT::CG_FR
    @brief Fletcher-Reeves conjugate gradient algorithm to minimize an objective
   function.
    */
template <class OPT_ObjectiveFunction> class CG_FR {
public:
  void minimize(std::vector<double> &X, OPT_ObjectiveFunction *obj,
                double &step_size, double epsilon = 1e-8, double tol = 1e-3);

private:
  void new_point(std::vector<double> &X, const std::vector<double> &X0,
                 const std::vector<double> &Direct, double step) {
    for (unsigned int i = 0; i < X0.size(); ++i)
      X[i] = X0[i] + step * Direct[i];
  }
  double golden_section(OPT_ObjectiveFunction *obj,
                        const std::vector<double> &X0,
                        const std::vector<double> &Direct, double factor,
                        double stepa, double stepb, double stepc, double fa,
                        double fb, double fc, double tol, double &step_size,
                        std::vector<double> &Xt, double &ft) {
    const double R = 2 - (1 + std::sqrt(5)) / 2; // ~ 0.38
    double stept;
    std::vector<double> G;

    double x1 = stepa;
    double f1 = fa;
    double x0 = stepb;
    double f0 = fb;
    double x2 = stepc;
    double f2 = fc;
    double pw2 = std::fabs(x2 - x1);
    double pw1 = std::fabs(x1 - x0);

    step_size = stepb;
    for (int time = 0; time < 10; ++time) {
      {
        double dx1 = x1 - x0;
        double df1 = f1 - f0;
        double dx2 = x2 - x0;
        double df2 = f2 - f0;
        double ra = 2 * (df1 * dx2 - df2 * dx1);
        double dx;
        if (ra != 0)
          dx = (df1 * dx2 * dx2 - df2 * dx1 * dx1) /
               ra; // quadratic interpolation
        else
          dx = 0;
        if ((dx > 0 && dx < (stepc - stepb) && std::fabs(dx) < pw2 / 2) ||
            (dx < 0 && dx > (stepa - stepb) && std::fabs(dx) < pw2 / 2))
          stept = x0 + dx;
        else if ((stepc - stepb) > (stepb - stepa))
          stept = R * (stepc - stepb) + stepb;
        else
          stept = stepb - R * (stepb - stepa);
      }
      new_point(Xt, X0, Direct, stept * factor);
      ft = obj->f(Xt);
      if (std::isnan(ft))
        exit(-1);
      if (ft > fb) {
        if (stept < stepb) {
          stepa = stept;
          fa = ft;
        } else {
          stepc = stept;
          fc = ft;
        }
        if (ft < f1) {
          x2 = x1;
          x1 = stept;
          f2 = f1;
          f1 = ft;
        } else if (ft < f2) {
          x2 = stept;
          f2 = ft;
        }
      } else if (ft <= fb) {
        obj->df(Xt, G);
        double p_g =
            std::inner_product(Direct.begin(), Direct.end(), G.begin(), 0e0);
        double g =
            std::sqrt(std::inner_product(G.begin(), G.end(), G.begin(), 0e0));
        step_size = stept;
        if (std::fabs(p_g * factor / g) < tol)
          break;
        if (stept < stepb) {
          stepc = stepb;
          fc = fb;
          stepb = stept;
          fb = ft;
        } else {
          stepa = stepb;
          fa = fb;
          stepb = stept;
          fb = ft;
        }
        pw2 = pw1;
        pw1 = std::fabs(x0 - stept);
        x2 = x1;
        x1 = x0;
        x0 = stept;
        f2 = f1;
        f1 = f0;
        f0 = ft;
      }
    }
    return stept;
  }
};
}
}

/**
   @brief Minimization by Fletcher-Reeves conjugate gradient algorithm.

   @param[in, out] X Argument of objective function
   @param[in] obj Objective function (GPR::OPT::ObjectiveFunction) to minimize
   @param[in] xstep_size Initial step size
   @param[in] epsilon Machine epsilon
   @param[in] tol Tolerance
*/
template <class OPT_ObjectiveFunction>
void GPR::OPT::CG_FR<OPT_ObjectiveFunction>::minimize(
    std::vector<double> &X, OPT_ObjectiveFunction *obj, double &xstep_size,
    double epsilon, double tol) {
  double step_size = xstep_size;
  int N = X.size();
  std::vector<std::vector<double> > XS(2);
  int is = 0;
  XS[is].resize(N);
  std::copy(X.begin(), X.end(), XS[is].begin());
  XS[is ^ 1].resize(N);

  double stepa, stepb, stepc;
  double fa, fb, fc;
  stepa = 0;
  fa = obj->f(XS[is]);
  if (std::isnan(fa))
    exit(-1);
  std::vector<double> G;
  obj->df(XS[is], G);
  std::vector<double> P(G);
  double ga_norm2 = std::inner_product(G.begin(), G.end(), G.begin(), 0e0);
  double pa_norm2 = ga_norm2;

  int nup = 0;
  for (int time = 0;; ++time) {
#ifndef GPR_NLOG
    std::cerr << time << " " << fa;
    for (int i = 0; i < N; ++i)
      std::cerr << " " << XS[is][i];
    std::cerr << std::endl;
#endif
    if (pa_norm2 == 0 || ga_norm2 == 0)
      break;
    double p_g = std::inner_product(P.begin(), P.end(), G.begin(), 0e0);
    double factor;
    if (p_g > 0)
      factor = -1e0 / std::sqrt(pa_norm2);
    else
      factor = 1e0 / std::sqrt(pa_norm2);

    // X(k+1)
    if (std::sqrt(ga_norm2) < epsilon)
      break;
    stepc = step_size;
    new_point(XS[is ^ 1], XS[is], P, stepc * factor);
    fc = obj->f(XS[is ^ 1]);
    if (std::isnan(fc))
      exit(-1);
    if (fc < fa) {
      step_size *= 2;
      fa = fc;
      stepa = 0;
      is = is ^ 1;
      obj->df(XS[is], G);
      continue;
    }

    // Initialize trial point
    double dfa = p_g * factor;
    {
      double stepc0 = stepc;
      double fc0 = fc;
      while (1) {
        // q(z) = fa + dfa * z + (fc-fa-dfa) * z^2 (z in [0,1], fa=f(0),
        // fc=f(1), dfa=df(0))
        stepb = stepa +
                0.5 * (stepc0 - stepa) * (-dfa * (stepc0 - stepa) /
                                          (fc0 - fa - dfa * (stepc0 - stepa)));
        new_point(XS[is ^ 1], XS[is], P, stepb * factor);
        if (stepb == stepa)
          goto end;
        double dx2 = 0;
        for (int i = 0; i < N; ++i) {
          double dx = XS[is][i] - XS[is ^ 1][i];
          dx2 += dx * dx;
        }
        if (dx2 == 0)
          goto end;
        fb = obj->f(XS[is ^ 1]);
        if (std::isnan(fb))
          exit(-1);
        if (fb < fa)
          break;
        stepc0 = stepb;
        fc0 = fb;
      }
    }
    // Golden section search
    golden_section(obj, XS[is], P, factor, stepa, stepb, stepc, fa, fb, fc, tol,
                   step_size, XS[is ^ 1], fa);
    is = is ^ 1;
    // Update search direction
    ++nup;
    if ((nup % N) == 0) {
      obj->df(XS[is], G);
      ga_norm2 = std::inner_product(G.begin(), G.end(), G.begin(), 0e0);
      std::copy(G.begin(), G.end(), P.begin());
      pa_norm2 = ga_norm2;
    } else {
      double gp_norm2 = ga_norm2;
      obj->df(XS[is], G);
      ga_norm2 = std::inner_product(G.begin(), G.end(), G.begin(), 0e0);
      double beta = ga_norm2 / gp_norm2;
      for (int i = 0; i < N; ++i)
        P[i] = beta * P[i] + G[i];
      pa_norm2 = std::inner_product(P.begin(), P.end(), P.begin(), 0e0);
    }
  }
end:
  std::copy(XS[is].begin(), XS[is].end(), X.begin());
}
#endif
