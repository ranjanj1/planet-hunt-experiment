"""Transit search pipeline: detrend -> quality mask -> iterative BLS -> diagnostics.

Run:  ../.venv/bin/python pipeline.py star_01 [star_02 ...]
"""
import sys, json, os
import numpy as np, pandas as pd
from astropy.timeseries import BoxLeastSquares

OUT = os.environ.get("TESS_OUT", "work")

# Detrending window (days) per star; default 0.75 d.  Star 06 is a fast, high-amplitude
# spotted rotator, so it gets a shorter window.
WINDOW = {"star_06": 0.5}
# Star 06 uses a robust local-quadratic trend instead of the running median.
METHOD = {"star_06": "quadratic"}
# Faster (coarser) BLS settings for the stars with multi-year baselines.
SEARCH_KW = {"star_06": dict(os=1.5, binw=10 / 1440, pmax=40.0),
             "star_07": dict(os=1.5, binw=10 / 1440, pmax=40.0)}

# Time ranges removed by hand after visual inspection (instrumental ramps the
# running median cannot follow).  Everything else is handled automatically.
MANUAL_BAD = {
    "star_03": [(95.3, 97.8), (209.8, 211.8)],
    "star_07": [(2520.0, 2523.0)],
}


def load(star):
    d = pd.read_csv(f"data/{star}.csv")
    d = d[np.isfinite(d.flux) & np.isfinite(d.time_days)]
    return d.sort_values("time_days").reset_index(drop=True)


def chunks(t, seg, gap=0.5):
    brk = np.where((np.diff(t) > gap) | (np.diff(seg) != 0))[0] + 1
    return np.split(np.arange(len(t)), brk)


def robust_std(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def running_median_trend(t, f, seg, window, user_mask=None):
    """Iteratively sigma-clipped running median in a time window, per contiguous chunk.
    Clipping (3 sigma, both sides) keeps transits from dragging the trend down.
    `user_mask` (True = usable) lets known transits be excluded from the trend."""
    trend = np.full_like(f, np.nan)
    um_all = np.ones(len(t), bool) if user_mask is None else user_mask
    for idx in chunks(t, seg):
        tt, ff = t[idx], f[idx]
        um = um_all[idx]
        lo = np.searchsorted(tt, tt - window / 2)
        hi = np.searchsorted(tt, tt + window / 2)
        mask = um.copy()
        for _ in range(3):
            tr = np.empty(len(tt))
            for i in range(len(tt)):
                w = ff[lo[i]:hi[i]][mask[lo[i]:hi[i]]]
                tr[i] = np.median(w) if len(w) > 5 else np.median(ff[lo[i]:hi[i]])
            r = ff - tr
            mask = um & (np.abs(r) < 3 * robust_std(r[mask]))
        trend[idx] = tr
    return trend


def local_quadratic_trend(t, f, seg, window, step=0.02, user_mask=None):
    """Robust local-quadratic (LOWESS-like) trend: at knots every `step` days fit a
    quadratic to points within +-window/2, iteratively clipping (2.5 sigma upward for
    flares, 3 sigma downward for dips), then interpolate between knots.  Follows
    fast, curved stellar variability much better than a running median."""
    trend = np.full_like(f, np.nan)
    keep_all = np.ones(len(t), bool) if user_mask is None else user_mask.copy()
    for idx in chunks(t, seg):
        tt, ff = t[idx], f[idx]
        keep = keep_all[idx].copy()
        for _ in range(4):
            knots = np.arange(tt.min(), tt.max() + step, step)
            vals = np.full(len(knots), np.nan)
            for k, c in enumerate(knots):
                lo, hi = np.searchsorted(tt, [c - window / 2, c + window / 2])
                m = keep[lo:hi]
                if m.sum() < 10:
                    continue
                x = tt[lo:hi][m] - c
                p = np.polyfit(x, ff[lo:hi][m], 2)
                vals[k] = p[-1]
            ok = np.isfinite(vals)
            tr = np.interp(tt, knots[ok], vals[ok])
            r = ff - tr
            s = robust_std(r[keep])
            keep = keep_all[idx] & (r < 2.5 * s) & (r > -3 * s)
        trend[idx] = tr
    return trend


def quality_mask(t, f, seg, star):
    """True = keep.  Removes upward outliers (flares, >4 sigma), time windows where the
    local scatter is > 3x typical (pointing jitter / scattered light), and manual windows."""
    r = f - 1
    sig = robust_std(r)
    good = np.isfinite(f) & (r < 4 * sig)
    # local point-to-point scatter in 0.25-d windows (insensitive to transits)
    loc = np.zeros(len(t))
    for idx in chunks(t, seg):
        tt = t[idx]
        dd = np.abs(np.diff(f[idx], prepend=f[idx][0]))
        lo = np.searchsorted(tt, tt - 0.125); hi = np.searchsorted(tt, tt + 0.125)
        cs = np.concatenate([[0], np.cumsum(dd)])
        loc[idx] = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
        # normalise per chunk (cadence / noise level can differ between segments)
        loc[idx] /= np.median(loc[idx])
    bad = loc > 3
    if bad.any():  # grow bad regions by 0.2 d
        tb = t[bad]
        j = np.searchsorted(tb, t)
        dist = np.minimum(np.abs(t - tb[np.clip(j - 1, 0, len(tb) - 1)]),
                          np.abs(t - tb[np.clip(j, 0, len(tb) - 1)]))
        good &= dist > 0.2
    # chunk edges: drop the first/last 0.1-0.3 d of a chunk if it deviates strongly
    # (ramps / pointing settling at the start or end of an orbit)
    for idx in chunks(t, seg):
        tt, ff = t[idx], f[idx]
        k = np.floor((tt - tt[0]) / 0.1).astype(int)
        u, inv, c = np.unique(k, return_inverse=True, return_counts=True)
        fb = np.bincount(inv, ff) / c
        sb = robust_std(fb)
        for w in (0.1, 0.2, 0.3):
            for side in (tt < tt[0] + w, tt > tt[-1] - w):
                if side.sum() > 3 and abs(np.mean(ff[side]) - np.median(ff)) > 4 * sb * np.sqrt(0.1 / w):
                    good[idx[side]] = False
    for a, b in MANUAL_BAD.get(star, []):
        good &= ~((t > a) & (t < b))
    return good


def bin_lc(t, f, width):
    k = np.floor((t - t.min()) / width).astype(np.int64)
    u, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
    return np.bincount(inv, t) / cnt, np.bincount(inv, f) / cnt, cnt


DURS_H = np.array([0.33, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 13])


def bls_search(t, f, pmin=0.2, pmax=None, os=3, binw=5 / 1440):
    """BLS over a log-uniform frequency grid fine enough that a 1-h transit drifts by
    <1/os of its duration across the full baseline.  Each duration is searched separately
    and only durations with D <= 0.15 P are allowed at each period."""
    base = t.max() - t.min()
    if pmax is None:
        pmax = min(base / 2.0, 60.0)
    tb, fb, cnt = bin_lc(t, f, binw)
    sig = robust_std(np.diff(f)) / np.sqrt(2)
    eb = sig / np.sqrt(cnt)
    dlnf = (1.0 / 24) / (base * os)
    freqs = np.exp(np.arange(np.log(1 / pmax), np.log(1 / pmin), dlnf))
    periods = np.sort(1 / freqs)
    bls = BoxLeastSquares(tb, fb, eb)
    best = np.full(len(periods), -np.inf)
    arg = dict(t0=np.zeros(len(periods)), dur=np.zeros(len(periods)), depth=np.zeros(len(periods)))
    for dh in DURS_H:
        D = dh / 24
        ok = D <= 0.15 * periods
        if not ok.any():
            continue
        r = bls.power(periods[ok], D, objective="likelihood")
        p = np.sqrt(np.maximum(2 * r.power, 0)) * np.sign(r.depth)  # ~ white-noise SNR
        sel = np.where(ok)[0]
        upd = p > best[sel]
        s2 = sel[upd]
        best[s2] = p[upd]
        arg["t0"][s2] = r.transit_time[upd]
        arg["dur"][s2] = D
        arg["depth"][s2] = r.depth[upd]
    return periods, best, arg


def fold_phase(t, P, T0):
    return (t - T0 + 0.5 * P) % P - 0.5 * P  # days from transit centre


def signal_stats(t, f, P, T0, D, good):
    """Depth, red-noise-aware SNR, odd/even, secondary and per-transit depths."""
    ph = fold_phase(t, P, T0)
    intr = np.abs(ph) < 0.5 * D
    near = np.abs(ph) < 1.0 * D
    oot = good & ~near
    epoch = np.round((t - T0) / P).astype(int)
    sig1 = robust_std(f[oot])

    def dep(m):
        m = m & good
        if m.sum() < 3:
            return np.nan, np.nan
        return 1.0 - np.mean(f[m]), sig1 / np.sqrt(m.sum())
    depth, depth_e = dep(intr)
    # red noise: robust std of out-of-transit flux binned to duration D
    k = np.floor((t[oot] - t[oot].min()) / D).astype(np.int64)
    u, inv, c = np.unique(k, return_inverse=True, return_counts=True)
    fb = np.bincount(inv, f[oot]) / c
    okb = c > 0.6 * np.median(c)
    sigD = robust_std(fb[okb])
    n_in = (intr & good).sum()
    ntr_eff = n_in / np.median(c)
    snr = depth / sigD * np.sqrt(ntr_eff)
    odd, odd_e = dep(intr & (epoch % 2 != 0))
    even, even_e = dep(intr & (epoch % 2 == 0))
    ph2 = fold_phase(t, P, T0 + 0.5 * P)
    sec, sec_e = dep(np.abs(ph2) < 0.5 * D)
    per = {}
    for e in np.unique(epoch[intr & good]):
        m = intr & good & (epoch == e)
        if m.sum() >= 0.5 * np.median(c):
            per[int(e)] = float(1.0 - np.mean(f[m]))
    return dict(depth=depth, depth_e=depth_e, snr=snr, sigD=sigD, n_in=int(n_in),
                ntr=len(per), odd=odd, odd_e=odd_e, even=even, even_e=even_e,
                sec=sec, sec_e=sec_e, per_transit=per)


def refine(t, f, good, P, T0, D):
    """Fine local BLS around the peak for better P, T0, D."""
    tt, ff = t[good], f[good]
    bls = BoxLeastSquares(tt, ff)
    base = tt.max() - tt.min()
    dP = P * D / base
    periods = np.linspace(P - 2 * dP, P + 2 * dP, 201)
    durs = np.unique(np.clip(np.linspace(0.5 * D, 1.6 * D, 23), 0.2 / 24, 0.2 * P))
    r = bls.power(periods, durs, objective="likelihood", oversample=30)
    i = np.argmax(r.power)
    T0 = r.transit_time[i]
    # put T0 near the middle of the data
    n = np.round((np.median(tt) - T0) / r.period[i])
    return r.period[i], T0, r.duration[i]


def run(star, nmax=5, snr_min=7.0, verbose=True):
    d = load(star)
    t, f, seg = d.time_days.values, d.flux.values, d.segment.values
    if METHOD.get(star) == "quadratic":
        trend = local_quadratic_trend(t, f, seg, WINDOW.get(star, 0.75))
    else:
        trend = running_median_trend(t, f, seg, WINDOW.get(star, 0.75))
    fd = f / trend
    good = quality_mask(t, fd, seg, star)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(dict(t=t, f=fd, raw=f, trend=trend, good=good, seg=seg)).to_csv(
        f"{OUT}/{star}_lc.csv", index=False)
    if verbose:
        print(star, "masked %.1f%% of points" % (100 * (1 - good.mean())), flush=True)
    search_mask = good.copy()
    sigs = []
    for it in range(nmax):
        periods, power, arg = bls_search(t[search_mask], fd[search_mask], **SEARCH_KW.get(star, {}))
        i = np.argmax(power)
        sde = (power[i] - np.median(power)) / np.std(power)
        P, T0, D = periods[i], arg["t0"][i], arg["dur"][i]
        P, T0, D = refine(t, fd, search_mask, P, T0, D)
        st = signal_stats(t, fd, P, T0, D, search_mask)
        rec = dict(iter=it, P=P, T0=T0, D_h=D * 24, sde=sde, peak=power[i], **st)
        if verbose:
            print(star, it, "P=%.5f T0=%.4f D=%.2fh depth=%.0f snr=%.1f sde=%.1f ntr=%d odd=%.0f±%.0f even=%.0f±%.0f sec=%.0f±%.0f" % (
                P, T0, D * 24, st["depth"] * 1e6, st["snr"], sde, st["ntr"], st["odd"] * 1e6, st["odd_e"] * 1e6,
                st["even"] * 1e6, st["even_e"] * 1e6, st["sec"] * 1e6, st["sec_e"] * 1e6), flush=True)
        np.save(f"{OUT}/{star}_pgram{it}.npy", np.vstack([periods, power]))
        sigs.append(rec)
        if st["snr"] < snr_min:
            break
        search_mask &= ~(np.abs(fold_phase(t, P, T0)) < max(1.0 * D, 0.5 * D + 1 / 24))
    json.dump(sigs, open(f"{OUT}/{star}_signals.json", "w"), default=float, indent=1)
    return sigs


if __name__ == "__main__":
    for s in sys.argv[1:]:
        run(s)
