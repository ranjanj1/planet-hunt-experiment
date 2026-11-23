"""Single-event (mono-transit) box search on the detrended, quality-masked light curve.
Known periodic signals (from work/<star>_signals.json, accepted ones passed on cmd line) are masked."""
import sys, json, numpy as np, pandas as pd
from pipeline import robust_std, fold_phase
def events(star, known=(), thr=7.0):
    d = pd.read_csv(f"work/{star}_lc.csv"); g = d.good.values.copy()
    t, f = d.t.values, d.f.values
    for P, T0, D in known:
        g &= ~(np.abs(fold_phase(t, P, T0)) < D)
    t, f = t[g], f[g]
    out = []
    for Dh in [1, 2, 3, 4, 6, 8, 12]:
        D = Dh / 24
        cs = np.concatenate([[0], np.cumsum(f - 1)])
        centers = np.arange(t.min(), t.max(), D / 4)
        lo = np.searchsorted(t, centers - D / 2); hi = np.searchsorted(t, centers + D / 2)
        n = hi - lo
        expn = D / np.median(np.diff(t))
        ok = n > 0.7 * expn
        mean = np.where(ok, (cs[hi] - cs[lo]) / np.maximum(n, 1), 0)
        sig = robust_std(mean[ok])          # empirical scatter at this timescale (includes red noise)
        snr = -mean / sig
        for i in np.where(ok & (snr > thr))[0]:
            out.append((centers[i], Dh, -mean[i] * 1e6, snr[i]))
    # merge overlapping
    out.sort(key=lambda x: -x[3]); kept = []
    for e in out:
        if all(abs(e[0] - k[0]) > 0.5 for k in kept): kept.append(e)
    return sorted(kept)
if __name__ == "__main__":
    star = sys.argv[1]; known = json.loads(sys.argv[2]) if len(sys.argv) > 2 else []
    for e in events(star, known):
        print(star, "t=%.3f D=%dh depth=%.0f snr=%.1f" % e)
