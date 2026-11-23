"""Transit search pipeline: load, clean, detrend, BLS search, vet."""
import numpy as np
from astropy.timeseries import BoxLeastSquares


def load(star):
    d = np.genfromtxt(f"data/{star}.csv", delimiter=",", names=True)
    return {k: d[k] for k in d.dtype.names}


def split_chunks(t, gap=0.5):
    """Indices of contiguous chunks (split at gaps > gap days)."""
    br = np.where(np.diff(t) > gap)[0] + 1
    return np.split(np.arange(len(t)), br)


def running_biweight(t, f, window, mask=None, c=5.0, n_iter=3):
    """Time-windowed robust location (Tukey biweight) evaluated at each point.
    Points where mask is True are excluded from the estimate (e.g. in-transit)."""
    use = np.ones_like(f, bool) if mask is None else ~mask
    tu, fu = t[use], f[use]
    out = np.full_like(f, np.nan)
    lo = np.searchsorted(tu, t - window / 2)
    hi = np.searchsorted(tu, t + window / 2)
    # evaluate on a decimated grid then interpolate, for speed
    step = max(1, len(t) // 4000)
    idx = np.arange(0, len(t), step)
    if idx[-1] != len(t) - 1:
        idx = np.append(idx, len(t) - 1)
    vals = np.full(len(idx), np.nan)
    for j, i in enumerate(idx):
        seg = fu[lo[i]:hi[i]]
        if len(seg) < 10:
            continue
        m = np.median(seg)
        for _ in range(n_iter):
            mad = np.median(np.abs(seg - m)) + 1e-12
            u = (seg - m) / (c * mad)
            w = (1 - u**2) ** 2
            w[np.abs(u) >= 1] = 0
            if w.sum() == 0:
                break
            m = np.sum(w * seg) / np.sum(w)
        vals[j] = m
    ok = np.isfinite(vals)
    if ok.sum() < 2:
        return np.full_like(f, np.nanmedian(f))
    out = np.interp(t, t[idx][ok], vals[ok])
    return out


def detrend(t, f, window=0.75, mask=None, gap=0.5):
    trend = np.full_like(f, np.nan)
    for ch in split_chunks(t, gap):
        m = None if mask is None else mask[ch]
        trend[ch] = running_biweight(t[ch], f[ch], window, m)
    return f / trend, trend


def clip_upper(f, nsig=4):
    med = np.nanmedian(f)
    mad = 1.4826 * np.nanmedian(np.abs(f - med))
    return f < med + nsig * mad


def bin_lc(t, f, width):
    """Bin to width (days) within the time series; returns t, f, n."""
    edges = np.floor((t - t[0]) / width).astype(int)
    u, inv, cnt = np.unique(edges, return_inverse=True, return_counts=True)
    tb = np.bincount(inv, t) / cnt
    fb = np.bincount(inv, f) / cnt
    return tb, fb, cnt


def period_grid(pmin, pmax, T, dmin, oversample=3, max_nfreq=600000):
    """Frequency grid where a step df shifts the last transit (after T days) by
    at most dmin/oversample: df*T*P <= dmin/os -> df = dmin/(os*T*P) = dmin*f/(os*T).
    That makes the grid geometric in frequency."""
    ratio = 1 + dmin / (oversample * T)
    n = int(np.ceil(np.log(pmax / pmin) / np.log(ratio)))
    if n > max_nfreq:
        n = max_nfreq
    freqs = np.geomspace(1 / pmax, 1 / pmin, n)
    return np.sort(1 / freqs)


def bls_search(t, f, pmin=0.3, pmax=None, durations=None, oversample=3, max_nfreq=600000):
    T = t.max() - t.min()
    if pmax is None:
        pmax = T / 2
    if durations is None:
        durations = np.array([0.02, 0.03, 0.045, 0.065, 0.09, 0.125, 0.17, 0.23, 0.3, 0.4])
    durations = durations[durations < pmin * 0.5] if np.any(durations < pmin * 0.5) else durations[:1]
    periods = period_grid(pmin, pmax, T, durations.min(), oversample, max_nfreq)
    bls = BoxLeastSquares(t, f)
    res = bls.power(periods, durations, objective="snr")
    return bls, res


def sde(power):
    p = power - np.median(power)
    return (p.max()) / np.std(p)
