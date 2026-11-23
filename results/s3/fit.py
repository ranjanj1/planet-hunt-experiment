"""Joint trapezoid fit of a periodic transit with a local polynomial baseline
around every individual transit (fitted to the raw flux, so no detrending bias)."""
import numpy as np
from scipy.optimize import least_squares
from pipeline import load


def trapezoid(t, t0, P, depth, T14, T12):
    ph = np.abs(((t - t0 + 0.5 * P) % P) - 0.5 * P)
    T12 = np.clip(T12, 1e-4, T14 / 2)
    m = np.zeros_like(t)
    full = ph <= T14 / 2 - T12
    ing = (ph > T14 / 2 - T12) & (ph < T14 / 2)
    m[full] = depth
    m[ing] = depth * (T14 / 2 - ph[ing]) / T12
    return 1 - m


def get_windows(star, P, t0, T14, other=(), win=None, deg=2, clip_flares=True):
    d = load(star)
    o = np.argsort(d["time_days"])
    t, f = d["time_days"][o], d["flux"][o]
    hw = win or max(1.5 * T14, T14 / 2 + 0.2)
    ep = np.round((t - t0) / P)
    dt = t - (t0 + ep * P)
    sel = np.abs(dt) < hw
    bad = np.zeros_like(t, bool)
    for (Po, t0o, To) in other:
        pho = ((t - t0o + 0.5 * Po) % Po) - 0.5 * Po
        bad |= np.abs(pho) < 0.6 * To + 0.01
    sel &= ~bad
    groups = []
    for e in np.unique(ep[sel]):
        m = sel & (ep == e)
        tt, ff = t[m], f[m]
        # need data on both sides of transit and inside
        inn = np.abs(tt - (t0 + e * P)) < T14 / 2
        pre = tt < t0 + e * P - T14 / 2
        post = tt > t0 + e * P + T14 / 2
        if inn.sum() < 3 or pre.sum() < 5 or post.sum() < 5:
            continue
        near = np.abs(tt - (t0 + e * P)) < 0.75 * T14
        if near.sum() > 1 and np.max(np.diff(tt[near])) > 0.4 * T14 + 0.02:  # gap across the transit
            continue
        groups.append((e, tt, ff / np.median(ff)))
    return groups, deg


def fit(star, P, t0, T14, depth, other=(), deg=2, win=None, fit_period=True, n_iter=3):
    groups, deg = get_windows(star, P, t0, T14, other, win, deg)
    tall = np.concatenate([g[1] for g in groups])
    fall = np.concatenate([g[2] for g in groups])
    gid = np.concatenate([np.full(len(g[1]), i) for i, g in enumerate(groups)])
    tmid = np.array([t0 + g[0] * P for g in groups])
    keep = np.ones_like(tall, bool)
    # Design matrices for baselines
    def resid(p, return_model=False):
        t0_, P_, dep, T14_, frac = p
        if not fit_period:
            P_ = P
        model = trapezoid(tall, t0_, P_, dep, T14_, frac * T14_ / 2)
        r = np.empty_like(fall)
        base = np.empty_like(fall)
        for i in range(len(groups)):
            m = gid == i
            x = tall[m] - tmid[i]
            A = np.vstack([x**k for k in range(deg + 1)]).T * model[m][:, None]
            k = keep[m]
            c, *_ = np.linalg.lstsq(A[k], fall[m][k], rcond=None)
            base[m] = A @ c / model[m]
            r[m] = fall[m] - A @ c
        if return_model:
            return model, base
        return r[keep]
    p0 = [t0, P, depth, T14, 0.3]
    lb = [t0 - T14 / 2, P * (1 - 1e-3), 0, T14 * 0.3, 0.02]
    ub = [t0 + T14 / 2, P * (1 + 1e-3), 1, T14 * 2.5, 1.0]
    for it in range(n_iter):
        sol = least_squares(resid, p0, bounds=(lb, ub), x_scale=[T14 / 10, P * 1e-5, depth / 10, T14 / 10, 0.1])
        r = np.full_like(fall, np.nan)
        r[keep] = resid(sol.x)
        model, base = resid(sol.x, True)
        rr = fall - model * base
        s = 1.4826 * np.median(np.abs(rr[keep]))
        keep = np.abs(rr) < 5 * s if it < n_iter - 1 else keep
        p0 = sol.x
    # uncertainties
    J = sol.jac
    try:
        cov = np.linalg.inv(J.T @ J) * s**2
        err = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        err = np.full(5, np.nan)
    t0f, Pf, dep, T14f, frac = sol.x
    # move t0 to the epoch nearest the middle of the data, and also report first epoch in data
    out = dict(t0=t0f, P=Pf if fit_period else P, depth=dep, T14=T14f, T12=frac * T14f / 2,
               err=dict(zip(["t0", "P", "depth", "T14", "frac"], err)), sigma=s, ntransits=len(groups),
               snr=dep / err[2] if err[2] > 0 else np.nan)
    out["data"] = (tall, fall, model, base, keep, gid, tmid)
    return out
