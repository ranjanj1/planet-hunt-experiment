"""Vetting: fit/measure a candidate signal and make a diagnostic plot."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from search import prepare
from pipeline import bin_lc


def phase(t, P, t0):
    return ((t - t0 + 0.5 * P) % P) - 0.5 * P


def binned(x, y, w):
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])
    b =np.floor((x - x.min()) / w).astype(int)
    u, inv, cnt = np.unique(b, return_inverse=True, return_counts=True)
    xb = np.bincount(inv, x) / cnt
    yb = np.bincount(inv, y) / cnt
    eb = np.sqrt(np.maximum(np.bincount(inv, y * y) / cnt - yb**2, 0) / np.maximum(cnt - 1, 1))
    k = cnt >= 3
    return xb[k], yb[k], eb[k]


def depth_at(t, f, P, t0, dur, sig, frac=0.6):
    """Mean depth in the central `frac` of the transit window, with error."""
    ph = phase(t, P, t0)
    inn = np.abs(ph) < frac * dur / 2
    oot = (np.abs(ph) > dur) & (np.abs(ph) < 3 * dur)
    if inn.sum() < 3:
        return np.nan, np.nan, 0
    base = np.median(f[oot]) if oot.sum() > 10 else 1.0
    d = base - np.mean(f[inn])
    return d, sig / np.sqrt(inn.sum()), inn.sum()


def stats(lc, P, t0, dur):
    t, f = lc["t"], lc["f"]
    sig = 1.4826 * np.median(np.abs(f - np.median(f)))
    ph = phase(t, P, t0)
    d, de, n = depth_at(t, f, P, t0, dur, sig)
    ep = np.round((t - t0) / P).astype(int)
    odd = ep % 2 == 1
    do, doe, _ = depth_at(t[odd], f[odd], P, t0, dur, sig)
    dv, dve, _ = depth_at(t[~odd], f[~odd], P, t0, dur, sig)
    ds, dse, _ = depth_at(t, f, P, t0 + P / 2, dur, sig)
    # per-transit depths
    per = []
    for e in np.unique(ep[np.abs(ph) < dur / 2]):
        m = ep == e
        dd, dde, nn = depth_at(t[m], f[m], P, t0, dur, sig)
        if nn >= 5:
            per.append((t0 + e * P, dd, dde))
    return dict(sig=sig, depth=d, depth_err=de, snr=d / de, odd=(do, doe), even=(dv, dve),
                sec=(ds, dse), oddeven_sigma=abs(do - dv) / np.hypot(doe, dve), per=np.array(per))


def plot(star, P, t0, dur, label, out, lc=None, pg=None):
    lc = lc or prepare(star)
    t, f = lc["t"], lc["f"]
    s = stats(lc, P, t0, dur)
    ph = phase(t, P, t0)
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3)
    ax = fig.add_subplot(gs[0, :])
    tb, fb, _ = bin_lc(t, f, 30 / 1440)
    ax.plot(t, f, ",", color="0.6")
    ax.plot(tb, fb, ".", ms=2, color="k")
    intr = np.abs(ph) < dur / 2
    ax.plot(t[intr], f[intr], ",", color="r")
    ax.set_title(f"{star}  {label}: P={P:.5f} d  t0={t0:.4f}  dur={dur*24:.2f} h  depth={s['depth']*1e6:.0f}±{s['depth_err']*1e6:.0f} ppm  SNR={s['snr']:.1f}")
    ax.set_xlabel("time [d]"); ax.set_ylabel("detrended flux")
    # broken axis is messy; just show full span
    w = max(3 * dur, 0.15)
    for col, (cen, name) in enumerate([(0, "primary"), (P / 2, "phase 0.5")]):
        a = fig.add_subplot(gs[1, col])
        x = phase(t, P, t0 + cen) * 24
        m = np.abs(x) < w * 24
        a.plot(x[m], f[m], ".", ms=1, color="0.7")
        xb, yb, eb = binned(x[m], f[m], max(dur * 24 / 12, 0.1))
        a.errorbar(xb, yb, eb, fmt="o", ms=3, color="C0" if col == 0 else "C3")
        a.axvspan(-dur * 12, dur * 12, color="y", alpha=0.15)
        a.set_title(f"{name}" + (f"  depth {s['sec'][0]*1e6:.0f}±{s['sec'][1]*1e6:.0f} ppm" if col else ""))
        a.set_xlabel("hours from mid-transit")
        if col == 0:
            lo = np.percentile(yb, 0) if len(yb) else 0.99
            a.set_ylim(min(lo, 1 - 1.5 * s["depth"]) - 3 * np.median(eb), 1 + 0.8 * s["depth"] + 3 * np.median(eb))
        else:
            a.sharey(fig.axes[1])
    a = fig.add_subplot(gs[1, 2])
    ep = np.round((t - t0) / P).astype(int)
    x = ph * 24
    m = np.abs(x) < w * 24
    for k, (sel, c, nm) in enumerate([(ep % 2 == 0, "C0", "even"), (ep % 2 == 1, "C1", "odd")]):
        xb, yb, eb = binned(x[m & sel], f[m & sel], max(dur * 24 / 10, 0.1))
        a.errorbar(xb, yb, eb, fmt="o", ms=3, color=c, label=f"{nm} {s[nm][0]*1e6:.0f}±{s[nm][1]*1e6:.0f}")
    a.axvspan(-dur * 12, dur * 12, color="y", alpha=0.15)
    a.legend(fontsize=8); a.set_title(f"odd/even diff {s['oddeven_sigma']:.1f} sigma")
    a.sharey(fig.axes[1])
    a = fig.add_subplot(gs[2, 0:2])
    if len(s["per"]):
        a.errorbar(s["per"][:, 0], s["per"][:, 1] * 1e6, s["per"][:, 2] * 1e6, fmt="o")
        a.axhline(s["depth"] * 1e6, color="k", ls=":")
    a.set_title("individual transit depths [ppm]"); a.set_xlabel("time [d]")
    a = fig.add_subplot(gs[2, 2])
    if pg is not None:
        a.plot(pg[0], pg[1], lw=0.5)
        a.axvline(P, color="r", alpha=0.3, lw=3)
        a.set_xscale("log"); a.set_title("BLS periodogram"); a.set_xlabel("period [d]")
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return s


if __name__ == "__main__":
    star, P, t0, dur = sys.argv[1], *map(float, sys.argv[2:5])
    out = sys.argv[5]
    s = plot(star, P, t0, dur / 24, "sig", out)
    print({k: v for k, v in s.items() if k != "per"})
