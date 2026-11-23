"""Out-of-eclipse phase-curve harmonics from the RAW flux: per-chunk cubic polynomial
(slow systematics) + cos/sin of 1x and 2x orbital phase fitted simultaneously,
with primary and secondary eclipses masked.  cos(2phi) amplitude = ellipsoidal variation."""
import sys, numpy as np, pandas as pd
from pipeline import chunks, fold_phase
def harmonics(star, P, T0, D):
    d = pd.read_csv(f"work/{star}_lc.csv"); t, f = d.t.values, d.raw.values
    g = d.good.values & (np.abs(fold_phase(t, P, T0)) > 0.75 * D) & (np.abs(fold_phase(t, P, T0 + P / 2)) > 0.75 * D)
    ph = 2 * np.pi * (t - T0) / P
    cols, names = [], []
    for k, idx in enumerate(chunks(t, d.seg.values, gap=0.5)):
        if len(idx) < 100: g[idx] = False; continue
        x = (t - t[idx].mean()) / (np.ptp(t[idx]) + 1e-9)
        for p in range(4):
            c = np.zeros(len(t)); c[idx] = x[idx] ** p; cols.append(c)
    harm = [np.cos(ph), np.sin(ph), np.cos(2 * ph), np.sin(2 * ph)]
    A = np.vstack(cols + harm).T[g]; y = f[g]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    cov = np.linalg.pinv(A.T @ A) * np.var(res)
    h = coef[-4:] * 1e6; he = np.sqrt(np.diag(cov))[-4:] * 1e6
    for n, v, e in zip(["cos1 (reflection/phase, -A at transit)", "sin1 (Doppler beaming)", "cos2 (ellipsoidal, negative = minima at conjunctions)", "sin2"], h, he):
        print(f"{star} {n}: {v:.0f} +- {e:.0f} ppm")
    return h, he
if __name__ == "__main__":
    harmonics(sys.argv[1], *map(float, sys.argv[2:5]))
