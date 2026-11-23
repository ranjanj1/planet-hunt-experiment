"""Shared tools: loading, detrending, BLS search, transit-model helpers."""
import numpy as np
from astropy.timeseries import BoxLeastSquares


# Per-star cleaning decided by eye from the raw/detrended light curves:
#   exclude = time ranges dropped (instrumental ramps); breaks = flux steps where the
#   detrending is restarted.
CONFIG = {
    3: dict(exclude=[(96.0, 97.6), (209.9, 210.9)], breaks=[105.6, 108.57]),
    4: dict(trim_edges=0.2),
    5: dict(trim_edges=0.2),
    # star 7, segment 3: ~1-day bursts of +-10% scatter every ~14.4 d (instrumental)
    7: dict(exclude=[(2523.2, 2524.4), (2537.8, 2538.9), (2552.0, 2553.2), (2566.4, 2567.7)]),
    6: dict(method='spline', knot=0.4),
}


def load(i):
    d = np.genfromtxt(f'data/star_{i:02d}.csv', delimiter=',', names=True)
    m = np.isfinite(d['flux']) & np.isfinite(d['time_days'])
    for lo, hi in CONFIG.get(i, {}).get('exclude', []):
        m &= ~((d['time_days'] > lo) & (d['time_days'] < hi))
    te = CONFIG.get(i, {}).get('trim_edges', 0)
    if te:
        t = d['time_days']
        cuts = np.where(np.diff(t) > 0.5)[0]
        starts = np.r_[t[0], t[cuts + 1]]
        ends = np.r_[t[cuts], t[-1]]
        for s, e in zip(starts, ends):
            m &= ~((t >= s) & (t < s + te)) & ~((t > e - te) & (t <= e))
    out = {k: d[k][m] for k in d.dtype.names}
    out['star'] = i
    return out


def running_median(t, f, window, mask=None):
    """Time-windowed sliding median (robust trend); `mask` = points to ignore."""
    use = np.ones_like(f, bool) if mask is None else ~mask
    tt, ff = t[use], f[use]
    trend = np.empty_like(f)
    lo = np.searchsorted(tt, t - window / 2)
    hi = np.searchsorted(tt, t + window / 2)
    for k in range(len(t)):
        seg = ff[lo[k]:hi[k]]
        trend[k] = np.median(seg) if len(seg) > 5 else np.nan
    return trend


def detrend(d, window=1.0, mask=None, clip_up=3.0):
    """Per-segment (and per-gap) running median; flags upward outliers (flares)."""
    t, f = d['time_days'], d['flux']
    trend = np.full_like(f, np.nan)
    # split on segment and on gaps > 0.5 d
    isbrk = (np.diff(t) > 0.5) | (np.diff(d['segment']) != 0)
    for tb in CONFIG.get(d.get('star'), {}).get('breaks', []):
        isbrk |= (t[:-1] < tb) & (t[1:] >= tb)
    brk = np.where(isbrk)[0] + 1
    for idx in np.split(np.arange(len(t)), brk):
        mk = None if mask is None else mask[idx]
        trend[idx] = running_median(t[idx], f[idx], window, mk)
    flat = f / trend
    resid = flat - 1
    mad = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    good = np.isfinite(flat) & (resid < clip_up * mad)
    return flat, trend, good


def bin_lc(t, f, width):
    edges = np.arange(t.min(), t.max() + width, width)
    idx = np.digitize(t, edges)
    s = np.bincount(idx, f)
    n = np.bincount(idx)
    st = np.bincount(idx, t)
    k = n > 0
    return st[k] / n[k], s[k] / n[k], n[k]


def bls_search(t, f, pmin=0.35, pmax=None, durations=None, oversample=3, max_n=1_500_000):
    """BLS (SNR objective) on a frequency grid sized for the data baseline.

    Frequency step: the accumulated timing error over the baseline, base*P*df,
    must stay below duration/oversample.  We evaluate this at P=10 d (longer
    periods have longer transits, which relaxes the requirement)."""
    base = t.max() - t.min()
    if pmax is None:
        pmax = min(base / 2, 40)
    if durations is None:
        durations = np.array([0.03, 0.05, 0.08, 0.12, 0.17, 0.25, 0.33])
    fmin, fmax = 1 / pmax, 1 / pmin
    step = durations.min() / (oversample * base * 10.0)
    nf = int((fmax - fmin) / step)
    if nf > max_n:
        step = (fmax - fmin) / max_n
        nf = max_n
    periods = np.sort(1 / (fmin + step * np.arange(nf)))
    bls = BoxLeastSquares(t, f)
    r = bls.power(periods, durations, objective='snr')
    return r, bls


def sde(power):
    p = power - np.median(power)
    return p / np.std(p)


def fold(t, P, t0):
    return ((t - t0 + 0.5 * P) % P) - 0.5 * P


# ---------------------------------------------------------------- transit fitting
from scipy.optimize import least_squares


def trapezoid(t, P, t0, depth, T14, T12, cad=None):
    """Trapezoid transit model, optionally supersampled over exposure `cad` (days)."""
    def raw(tt):
        x = np.abs(fold(tt, P, t0))
        T12c = np.clip(T12, 1e-4, T14 / 2)
        m = np.ones_like(tt)
        full = x <= T14 / 2 - T12c
        ing = (x > T14 / 2 - T12c) & (x < T14 / 2)
        m[full] = 1 - depth
        m[ing] = 1 - depth * (T14 / 2 - x[ing]) / T12c
        return m
    if cad is None:
        return raw(t)
    off = np.linspace(-0.5, 0.5, 7)
    out = np.zeros_like(t)
    for o in off:
        out += raw(t + o * cad)
    return out / len(off)


def fit_trapezoid(t, f, P, t0, depth, T14, cad=None, fix_P=False):
    """Least-squares trapezoid fit. Returns dict incl. 1-sigma errors."""
    T12 = T14 / 4
    p0 = [P, t0, depth, T14, T12]
    def resid(p):
        pp = p if not fix_P else [P, *p[1:]]
        return f - trapezoid(t, *pp, cad=cad)
    lo = [P * 0.999, t0 - T14, 0, T14 * 0.3, 1e-4]
    hi = [P * 1.001, t0 + T14, 1, T14 * 3, T14 * 1.5]
    r = least_squares(resid, p0, bounds=(lo, hi), x_scale=[1e-4, 1e-3, depth, T14 / 3, T14 / 10])
    J = r.jac
    s2 = np.sum(r.fun**2) / (len(f) - 5)
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        err = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        err = np.full(5, np.nan)
    P, t0, dep, T14, T12 = r.x
    T12 = min(T12, T14 / 2)
    return dict(P=P, t0=t0, depth=dep, T14=T14, T12=T12, eP=err[0], et0=err[1], edepth=err[2],
                eT14=err[3], rms=np.sqrt(s2), chi2=np.sum(r.fun**2) / s2)


def stellar_density(P, T14, T12, depth):
    """Circular-orbit stellar density (g/cc) from transit shape (Seager & Mallen-Ornelas 2003)."""
    k = np.sqrt(depth)
    TF = max(T14 - 2 * T12, 1e-5)
    sF, sT = np.sin(np.pi * TF / P), np.sin(np.pi * T14 / P)
    num = (1 - k)**2 - (sF / sT)**2 * (1 + k)**2
    den = 1 - (sF / sT)**2
    if den <= 0 or num < 0:
        return np.nan, np.nan
    b2 = num / den
    aR = np.sqrt(((1 + k)**2 - b2 * (1 - sT**2)) / sT**2)
    G = 6.674e-8
    rho = 3 * np.pi / (G * (P * 86400)**2) * aR**3
    return rho, np.sqrt(max(b2, 0))


def refine(i, sigs, window=1.0, prange=0.003, keep=None, verbose=True, method='median'):
    """Re-detrend star i with all signals in `sigs` masked, then refine each signal
    with a narrow full-baseline BLS followed by a trapezoid fit.
    sigs: list of (P, t0, dur_days). keep: optional boolean mask of data to use."""
    d = load(i)
    t = d['time_days']
    tmask = np.zeros_like(t, bool)
    for P, t0, dur in sigs:
        tmask |= np.abs(fold(t, P, t0)) < 1.0 * dur
    if method == 'spline':
        flat, trend, good = detrend_spline(d, knot=window, mask=tmask)
    else:
        flat, trend, good = detrend(d, window=window, mask=tmask)
    if keep is not None:
        good &= keep
    cad = np.median(d['cadence_s']) / 86400
    out = []
    for j, (P, t0, dur) in enumerate(sigs):
        other = np.zeros_like(t, bool)
        for k, (P2, t02, dur2) in enumerate(sigs):
            if k != j:
                other |= np.abs(fold(t, P2, t02)) < 0.75 * dur2
        m = good & ~other
        tt, ff = t[m], flat[m]
        # narrow BLS over full baseline
        base = tt.max() - tt.min()
        step = dur / 10 / base * P
        periods = np.arange(P * (1 - prange), P * (1 + prange), step) if prange > 0 else np.array([P])
        bls = BoxLeastSquares(tt, ff)
        durs = np.array([0.6, 0.8, 1.0, 1.25]) * dur
        durs = durs[durs < periods.min()]
        r = bls.power(periods, durs, objective='likelihood', oversample=20)
        k = np.argmax(r.power)
        Pb, t0b, db = r.period[k], r.transit_time[k], r.duration[k]
        # move t0 to the middle of the data
        n = np.round((np.median(tt) - t0b) / Pb)
        t0b = t0b + n * Pb
        near = np.abs(fold(tt, Pb, t0b)) < 3 * db
        cads = d['cadence_s'][m][near] / 86400
        ccad = None if np.max(cads) < 0.01 else cads  # supersample only for long cadence
        fit = fit_trapezoid(tt[near], ff[near], Pb, t0b, max(r.depth[k], 1e-5), db,
                            cad=None if ccad is None else ccad)
        fit['bls_P'] = Pb
        fit['n_tr'] = len(np.unique(np.round((tt[near] - fit['t0']) / fit['P'])))
        rho, b = stellar_density(fit['P'], fit['T14'], fit['T12'], fit['depth'])
        fit['rho'] = rho; fit['b'] = b
        if verbose:
            print(f"  sig {j}: P={fit['P']:.6f}+-{fit['eP']:.6f} t0={fit['t0']:.5f}+-{fit['et0']:.5f} "
                  f"depth={fit['depth']*1e6:.0f}+-{fit['edepth']*1e6:.0f}ppm T14={fit['T14']*24:.2f}h "
                  f"T12={fit['T12']*24:.2f}h rho*={rho:.2f} b={b:.2f} ntr={fit['n_tr']}")
        out.append(fit)
    return out, (t, flat, good, tmask)


def epoch_depths(t, f, fit, window=3.0):
    """Fixed-shape depth per transit epoch (linear LSQ), with errors from local scatter."""
    P, t0, T14, T12 = fit['P'], fit['t0'], fit['T14'], fit['T12']
    shape = 1 - trapezoid(t, P, t0, 1.0, T14, T12)  # 0..1
    ep = np.round((t - t0) / P).astype(int)
    near = np.abs(fold(t, P, t0)) < window * T14 / 2 + T14
    rows = []
    for e in np.unique(ep[near]):
        m = near & (ep == e)
        s = shape[m]
        if (s > 0.5).sum() < 3:
            continue
        # model: f = c - depth*s
        A = np.vstack([np.ones(m.sum()), -s]).T
        c, *_ = np.linalg.lstsq(A, f[m], rcond=None)
        res = f[m] - A @ c
        cov = np.linalg.inv(A.T @ A) * np.var(res)
        rows.append((e, t0 + e * P, c[1], np.sqrt(cov[1, 1]), (s > 0.5).sum()))
    return np.array(rows)


def odd_even(rows):
    """Weighted mean depths of odd and even epochs and their difference in sigma."""
    out = []
    for par in (0, 1):
        r = rows[rows[:, 0] % 2 == par]
        if len(r) == 0:
            out.append((np.nan, np.nan)); continue
        w = 1 / r[:, 3]**2
        mu = np.sum(w * r[:, 2]) / w.sum()
        # inflate error by sqrt(reduced chi2) if epochs scatter more than errors
        chi = np.sum(w * (r[:, 2] - mu)**2) / max(len(r) - 1, 1)
        out.append((mu, np.sqrt(1 / w.sum()) * np.sqrt(max(chi, 1))))
    (de, ee), (do, eo) = out
    return de, ee, do, eo, abs(de - do) / np.hypot(ee, eo)


from scipy.interpolate import LSQUnivariateSpline


def spline_trend(t, f, knot=0.5, mask=None, nit=5, lo=3.0, hi=2.0):
    """Iteratively clipped cubic spline (asymmetric clip: flares are removed hard)."""
    use = np.ones_like(f, bool) if mask is None else ~mask
    trend = np.full_like(f, np.nan)
    for _ in range(nit):
        tt, ff = t[use], f[use]
        if len(tt) < 20:
            return trend
        knots = np.arange(tt[0] + knot, tt[-1] - knot / 2, knot)
        # drop knots with no data between them
        ok = np.array([np.any((tt > k - knot) & (tt < k)) and np.any((tt > k) & (tt < k + knot))
                       for k in knots]) if len(knots) else np.array([], bool)
        knots = knots[ok]
        try:
            s = LSQUnivariateSpline(tt, ff, knots, k=3)
        except ValueError:
            return trend
        trend = s(t)
        r = f - trend
        sd = 1.4826 * np.median(np.abs(r[use]))
        new = (r < hi * sd) & (r > -lo * sd)
        if mask is not None:
            new &= ~mask
        if np.array_equal(new, use):
            break
        use = new
    return trend


def detrend_spline(d, knot=0.5, mask=None):
    t, f = d['time_days'], d['flux']
    trend = np.full_like(f, np.nan)
    isbrk = (np.diff(t) > 0.2) | (np.diff(d['segment']) != 0)
    for tb in CONFIG.get(d.get('star'), {}).get('breaks', []):
        isbrk |= (t[:-1] < tb) & (t[1:] >= tb)
    for idx in np.split(np.arange(len(t)), np.where(isbrk)[0] + 1):
        if len(idx) < 30:
            continue
        mk = None if mask is None else mask[idx]
        trend[idx] = spline_trend(t[idx], f[idx], knot, mk)
    flat = f / trend
    r = flat - 1
    sd = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))
    good = np.isfinite(flat) & (r < 3 * sd)
    # also drop points right after flares (decay tails): 3 cadences after any >5 sigma point
    big = np.where(np.isfinite(r) & (r > 5 * sd))[0]
    for k in range(1, 6):
        good[np.clip(big + k, 0, len(t) - 1)] = False
    return flat, trend, good
