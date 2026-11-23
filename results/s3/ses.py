"""Single-event search: at each trial mid-time and duration, fit
local polynomial + box to the raw (normalised) flux; record box depth and SNR.
Robust against strong stellar variability (star_06)."""
import numpy as np
from pipeline import split_chunks


def ses(t, f, durs=(0.04, 0.06, 0.08, 0.11, 0.15), step=0.005, half=None, deg=3, clip=4.0):
    f = f.copy()
    out = []
    chunks = split_chunks(t, 0.3)
    for ch in chunks:
        tc, fc = t[ch], f[ch]
        if tc[-1] - tc[0] < 0.3:
            continue
        for dur in durs:
            hw = half or max(2.5 * dur, 0.25)
            centers = np.arange(tc[0] + dur / 2, tc[-1] - dur / 2, step)
            lo = np.searchsorted(tc, centers - hw)
            hi = np.searchsorted(tc, centers + hw)
            for c, a, b in zip(centers, lo, hi):
                x = tc[a:b] - c
                y = fc[a:b]
                inn = np.abs(x) < dur / 2
                nin = inn.sum()
                # need coverage on both sides
                if nin < 5 or (x < -dur / 2).sum() < 10 or (x > dur / 2).sum() < 10:
                    continue
                A = np.vstack([(x / hw) ** k for k in range(deg + 1)] + [inn.astype(float)]).T
                keep = np.ones(len(y), bool)
                for _ in range(2):
                    coef, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
                    r = y - A @ coef
                    s = 1.4826 * np.median(np.abs(r[keep]))
                    keep = r < clip * s  # reject flares (upward)
                depth = -coef[-1]
                cov = np.linalg.pinv(A[keep].T @ A[keep]) * s**2
                err = np.sqrt(cov[-1, -1])
                out.append((c, dur, depth, err, depth / err))
    return np.array(out)


def peaks(res, snr_min=7, sep=0.3):
    """Greedy pick of strongest non-overlapping events."""
    r = res[np.argsort(-res[:, 4])]
    picked = []
    for row in r:
        if row[4] < snr_min:
            break
        if all(abs(row[0] - p[0]) > sep for p in picked):
            picked.append(row)
    return np.array(picked)


def mes_search(res, periods, dur, t_ref, nphase=None, step=0.005, mask_times=None, min_events=2):
    """Fold single-event depths for one duration: for each P, t0 combine
    depth_i with inverse-variance weights. Returns best (P, t0, MES, n) per P."""
    r = res[np.isclose(res[:, 1], dur)]
    tc, d, e = r[:, 0], r[:, 2], r[:, 3]
    if mask_times is not None:
        keep = np.ones(len(tc), bool)
        for (tm, hw) in mask_times:
            keep &= np.abs(tc - tm) > hw
        tc, d, e = tc[keep], d[keep], e[keep]
    w = 1 / e**2
    out = []
    for P in periods:
        nb = int(np.ceil(P / step))
        ph = ((tc - t_ref) / P) % 1.0
        b = np.minimum((ph * nb).astype(int), nb - 1)
        # events from the same transit fall in the same bin repeatedly (grid step) -> fine for ranking
        num = np.bincount(b, d * w, nb)
        den = np.bincount(b, w, nb)
        cnt = np.bincount(b, None, nb)
        # count distinct epochs per bin
        ep = np.floor((tc - t_ref) / P).astype(int)
        mes = np.where(den > 0, num / np.sqrt(np.maximum(den, 1e-30)), 0)
        k = np.argmax(mes)
        nep = len(np.unique(ep[b == k]))
        out.append((P, t_ref + (k + 0.5) / nb * P, mes[k], nep))
    return np.array(out)
