"""Diagnostic plot for one signal: periodogram, full LC, phase fold, odd/even, secondary."""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pipeline import fold_phase

def binned(x, y, w):
    e = np.arange(x.min(), x.max() + w, w); k = np.digitize(x, e)
    xb = np.array([x[k == i].mean() for i in np.unique(k) if (k == i).sum() > 2])
    yb = np.array([np.median(y[k == i]) for i in np.unique(k) if (k == i).sum() > 2])
    return xb, yb

def plot(star, P, T0, D_h, depth_ppm, title, out, pgram=None, maskout=None):
    d = pd.read_csv(f"work/{star}_lc.csv"); g = d.good.values
    t, f = d.t.values[g], d.f.values[g]
    D = D_h / 24
    fig = plt.figure(figsize=(15, 9))
    ax0 = fig.add_subplot(3, 1, 1)
    if pgram is not None:
        pg = np.load(pgram); ax0.plot(pg[0], pg[1], "k-", lw=0.5); ax0.set_xscale("log")
        ax0.axvline(P, color="r", alpha=0.3, lw=3); ax0.set_xlabel("period [d]"); ax0.set_ylabel("BLS SNR")
    ax0.set_title(title, fontsize=10)
    ax1 = fig.add_subplot(3, 3, 4)
    ph = fold_phase(t, P, T0) * 24
    w = min(max(3 * D_h, 4), P * 24 / 2)
    m = np.abs(ph) < w
    ax1.plot(ph[m], (f[m] - 1) * 1e6, ".", color="0.6", ms=1)
    xb, yb = binned(ph[m], (f[m] - 1) * 1e6, max(D_h / 8, 2 / 60))
    ax1.plot(xb, yb, "o", color="C0", ms=3)
    ax1.axhline(-depth_ppm, color="r", ls=":"); ax1.axvspan(-D_h / 2, D_h / 2, color="r", alpha=0.08)
    lim = max(3 * depth_ppm, 4 * np.std(yb)); ax1.set_ylim(-1.6 * depth_ppm - 0.3 * lim, 0.5 * lim + 0.3*depth_ppm)
    ax1.set_xlabel("hours from mid-transit"); ax1.set_ylabel("ppm"); ax1.set_title("phase-folded")
    ax2 = fig.add_subplot(3, 3, 5)
    ep = np.round((t - T0) / P).astype(int)
    for par, c in [(0, "C0"), (1, "C3")]:
        mm = m & (ep % 2 == par)
        if mm.sum() > 5:
            xb, yb = binned(ph[mm], (f[mm] - 1) * 1e6, max(D_h / 6, 2 / 60))
            ax2.plot(xb, yb, "o-", color=c, ms=3, lw=0.8, label="even" if par == 0 else "odd")
    ax2.axhline(-depth_ppm, color="k", ls=":"); ax2.legend(); ax2.set_ylim(ax1.get_ylim())
    ax2.set_title("odd vs even transits"); ax2.set_xlabel("hours")
    ax3 = fig.add_subplot(3, 3, 6)
    ph2 = fold_phase(t, P, T0 + P / 2) * 24
    m2 = np.abs(ph2) < w
    ax3.plot(ph2[m2], (f[m2] - 1) * 1e6, ".", color="0.6", ms=1)
    xb, yb = binned(ph2[m2], (f[m2] - 1) * 1e6, max(D_h / 6, 2 / 60))
    ax3.plot(xb, yb, "o", color="C2", ms=3); ax3.axvspan(-D_h / 2, D_h / 2, color="r", alpha=0.08)
    ax3.set_ylim(ax1.get_ylim()); ax3.set_title("phase 0.5 (secondary eclipse?)"); ax3.set_xlabel("hours")
    ax4 = fig.add_subplot(3, 1, 3)
    dd = pd.read_csv(f"work/{star}_lc.csv")
    ax4.plot(np.arange(len(dd)), dd.f, ".", color="0.5", ms=0.5)
    ax4.plot(np.arange(len(dd))[~dd.good], dd.f[~dd.good], ".", color="orange", ms=0.5)
    intr = np.abs(fold_phase(dd.t.values, P, T0)) < D / 2
    ax4.plot(np.arange(len(dd))[intr], dd.f[intr], ".", color="r", ms=1)
    lo, hi = np.percentile(dd.f, [0.1, 99.9]); ax4.set_ylim(min(lo, 1 - 1.5e-6 * depth_ppm), hi)
    segb = np.where(np.diff(dd.seg) != 0)[0]
    for b in segb: ax4.axvline(b, color="k", lw=0.5)
    ax4.set_xlabel("data index (segments separated by lines; red = in transit, orange = masked)")
    plt.tight_layout(); plt.savefig(out, dpi=80); plt.close()

if __name__ == "__main__":
    star, P, T0, D_h, dep = sys.argv[1], *map(float, sys.argv[2:6])
    plot(star, P, T0, D_h, dep, sys.argv[6], sys.argv[7], sys.argv[8] if len(sys.argv) > 8 else None)
