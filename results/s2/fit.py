"""Least-squares trapezoid fit to refine P, T0, depth, total duration and ingress time."""
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from pipeline import fold_phase, robust_std

def trapezoid(dt, depth, T14, T12):
    """dt in days from centre. Flat bottom of width T14-2*T12, linear ingress/egress."""
    a = np.abs(dt)
    T12 = np.clip(T12, 1e-4, T14 / 2)
    m = np.zeros_like(a)
    flat = a <= T14 / 2 - T12
    m[flat] = depth
    ing = (a > T14 / 2 - T12) & (a < T14 / 2)
    m[ing] = depth * (T14 / 2 - a[ing]) / T12
    return 1 - m

def fit(star, P, T0, D, depth, extra_mask=None, fix_P=False):
    d = pd.read_csv(f"work/{star}_lc.csv")
    g = d.good.values.copy()
    if extra_mask is not None: g &= extra_mask(d.t.values)
    t, f = d.t.values[g], d.f.values[g]
    # put reference epoch near the middle of the data (minimises P-T0 covariance)
    n = np.round((np.median(t) - T0) / P); T0m = T0 + n * P
    win = np.abs(fold_phase(t, P, T0m)) < max(2.5 * D, D + 0.1)
    t, f = t[win], f[win]
    sig = robust_std(f[np.abs(fold_phase(t, P, T0m)) > 0.75 * D])
    def model(p):
        Pp = P if fix_P else p[4]
        return trapezoid(fold_phase(t, Pp, p[0]), p[1], p[2], p[3] * p[2] / 2)
    p0 = [T0m, depth, D, 0.3] + ([] if fix_P else [P])
    lb = [T0m - D / 2, 0, D * 0.3, 0.02] + ([] if fix_P else [P * (1 - 1e-3)])
    ub = [T0m + D / 2, 1, D * 2.5, 1.0] + ([] if fix_P else [P * (1 + 1e-3)])
    r = least_squares(lambda p: (f - model(p)) / sig, p0, bounds=(lb, ub), x_scale="jac")
    J = r.jac; cov = np.linalg.pinv(J.T @ J) * max(1, np.sum(r.fun**2) / (len(f) - len(p0)))
    err = np.sqrt(np.diag(cov))
    out = dict(T0=r.x[0], T0_e=err[0], depth=r.x[1], depth_e=err[1], T14=r.x[2], T14_e=err[2],
               ingress_frac=r.x[3], ingress_frac_e=err[3], P=P if fix_P else r.x[4],
               P_e=0 if fix_P else err[4], chi2r=np.sum(r.fun**2) / (len(f) - len(p0)))
    return out
