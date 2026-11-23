"""Second pass: mask known signals, detrend with those transits masked,
trim chunk edges, then BLS across all data (and main group) for residual signals."""
import sys, json, os
import numpy as np
from astropy.timeseries import BoxLeastSquares
from pipeline import load, detrend, clip_upper, bin_lc, sde, split_chunks, period_grid
from search import prepare, groups, bls_run, DURS

SCR = os.environ.get("SCR", ".")

KNOWN = {
    "star_01": [(3.69066, 2.2028, 0.06), (7.45066, 1.2908, 0.05), (2.25315, 0.9068, 0.06)],
    "star_02": [(0.46293, 0.7247, 0.03)],
    "star_03": [(6.26881, 0.4787, 0.13)],
    "star_04": [], "star_05": [],
    "star_06": [(8.46323, 5.36598, 0.15)],
    "star_07": [],
    "star_08": [(0.94145, 0.4558, 0.10)],
    "star_09": [(16.05126, 15.0880, 0.06)],
    "star_10": [(5.66022, 4.5031, 0.08), (11.38036, 4.6751, 0.08), (3.36006, 2.0911, 0.06)],
    "star_11": [(1.56689, 0.8512, 0.06)],
    "star_12": [(5.65194, 1.2253, 0.12)],
}


def known_mask(star, fac=1.5):
    def f(t):
        m = np.zeros_like(t, bool)
        for P, t0, d in KNOWN[star]:
            ph = ((t - t0 + 0.5 * P) % P) - 0.5 * P
            m |= np.abs(ph) < fac * d
        return m
    return f


def run(star, nsig=3, binw=10 / 1440, edge=0.1):
    km = known_mask(star)
    lc = prepare(star, transit_mask=km)
    t, f = lc["t"], lc["f"]
    keep = ~km(t)
    for ch in split_chunks(t, 0.5):
        tc = t[ch]
        keep[ch[(tc < tc[0] + edge) | (tc > tc[-1] - edge)]] = False
    t, f = t[keep], f[keep]
    tb, fb, _ = bin_lc(t, f, binw)
    gr = groups(tb)
    spans = [tb[g].max() - tb[g].min() for g in gr]
    Tg = max(spans)
    gmain = gr[int(np.argmax(spans))]
    Tall = tb.max() - tb.min()
    mask = np.zeros_like(tb, bool)
    out = []
    for k in range(nsig):
        sel = np.zeros_like(mask); sel[gmain] = True; sel &= ~mask
        periods = period_grid(0.3, min(Tg / 1.5, 40), Tg, DURS.min(), 3)
        bls, res = bls_run(tb[sel], fb[sel], periods)
        i = np.argmax(res.power)
        P, t0, dur, dep = res.period[i], res.transit_time[i], res.duration[i], res.depth[i]
        s = sde(res.power)
        ph = ((tb - t0 + 0.5 * P) % P) - 0.5 * P
        inn = np.abs(ph) < dur / 2
        sig = 1.4826 * np.median(np.abs(fb - np.median(fb)))
        snr = dep / (sig / np.sqrt(inn.sum()))
        ntr = len(np.unique(np.round((tb[inn] - t0) / P)))
        print(f"{star} res{k}: P={P:.5f} t0={t0:.4f} dur={dur*24:.2f}h depth={dep*1e6:.0f}ppm SDE={s:.1f} SNR={snr:.1f} ntr={ntr}", flush=True)
        np.save(f"{SCR}/pg2_{star}_{k}.npy", np.vstack([res.period, res.power]))
        out.append(dict(P=float(P), t0=float(t0), dur=float(dur), depth=float(dep), sde=float(s), snr=float(snr), ntr=int(ntr)))
        mask |= np.abs(ph) < 1.0 * dur
    # long-period / single-event check across all data: box search on durations
    return out


if __name__ == "__main__":
    for s in sys.argv[1:]:
        json.dump(run(s), open(f"{SCR}/pass2_{s}.json", "w"), indent=1)
