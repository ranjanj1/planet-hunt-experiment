"""Final step: take the vetted signal list, re-detrend each star with all its transits
masked, fit a trapezoid to every signal, compute vetting statistics, make one plot per
signal (plots/) and write results.json.

Run after pipeline.py (which writes work/<star>_lc.csv, _signals.json, _pgram*.npy).
"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline import (load, running_median_trend, local_quadratic_trend, quality_mask,
                      fold_phase, robust_std, signal_stats, WINDOW, METHOD)
from fit import fit, trapezoid

# ---------------------------------------------------------------------------
# Vetted signal list (initial P, T0, duration[h], depth[ppm] from pipeline.py output;
# all refined below).  pgram = which BLS iteration found it.
# ---------------------------------------------------------------------------
SIGNALS = {
    "star_01": [dict(P=3.69060, T0=2.2036, D=1.15, dep=1630, pgram=0),
                dict(P=7.45086, T0=1.2866, D=0.70, dep=1540, pgram=1),
                dict(P=2.25312, T0=0.9050, D=0.95, dep=560, pgram=2)],
    "star_02": [dict(P=0.46293, T0=0.7257, D=0.47, dep=4100, pgram=0)],
    "star_03": [dict(P=6.26788, T0=0.5022, D=2.85, dep=260, pgram=0)],
    "star_04": [],
    "star_05": [],
    "star_06": [dict(P=8.46319, T0=5.3788, D=3.15, dep=2000, pgram=0)],
    "star_07": [dict(P=14.45, T0=2524.0, D=19.0, dep=0, pgram=None, kind="jitter")],
    "star_08": [dict(P=0.94145, T0=0.4580, D=1.90, dep=10100, pgram=0)],
    "star_09": [dict(P=16.05144, T0=15.0875, D=1.05, dep=2780, pgram=0)],
    "star_10": [dict(P=5.66021, T0=4.5019, D=1.50, dep=3450, pgram=0),
                dict(P=11.38051, T0=4.6762, D=2.00, dep=2560, pgram=1),
                dict(P=3.36013, T0=2.0914, D=1.31, dep=800, pgram=2)],
    # BLS found 1.56689 d; odd/even depths differ hugely -> true period is twice that.
    "star_11": [dict(P=3.13378, T0=0.8510, D=1.00, dep=100000, pgram=0, halfP=True)],
    "star_12": [dict(P=5.65209, T0=1.2284, D=2.50, dep=1220, pgram=0)],
}


def final_lc(star, sigs):
    d = load(star)
    t, f, seg = d.time_days.values, d.flux.values, d.segment.values
    um = np.ones(len(t), bool)
    for s in sigs:
        if s.get("kind") == "jitter":
            continue
        um &= ~(np.abs(fold_phase(t, s["P"], s["T0"])) < 0.75 * s["D"] / 24 + 0.5 / 24)
        if s.get("halfP"):
            um &= ~(np.abs(fold_phase(t, s["P"], s["T0"] + s["P"] / 2)) < 0.75 * s["D"] / 24 + 0.5 / 24)
    if METHOD.get(star) == "quadratic":
        tr = local_quadratic_trend(t, f, seg, WINDOW.get(star, 0.75), user_mask=um)
    else:
        tr = running_median_trend(t, f, seg, WINDOW.get(star, 0.75), user_mask=um)
    fd = f / tr
    good = quality_mask(t, fd, seg, star)
    good |= ~um & np.isfinite(fd) & (fd - 1 < 4 * robust_std(fd[um] - 1))  # never drop in-transit points as "outliers" unless flare-high
    from pipeline import MANUAL_BAD
    for a, b in MANUAL_BAD.get(star, []):
        good &= ~((t > a) & (t < b))
    return t, fd, f, tr, good, seg


def binned(x, y, w):
    if len(x) == 0:
        return np.array([]), np.array([])
    e = np.arange(x.min(), x.max() + w, w)
    k = np.digitize(x, e)
    u = [i for i in np.unique(k) if (k == i).sum() > 2]
    return np.array([x[k == i].mean() for i in u]), np.array([np.mean(y[k == i]) for i in u])


def plot_signal(star, n, t, f, good, seg, fr, st, verdict, reason, pgram, out):
    P, T0, D_h, dep = fr["P"], fr["T0"], fr["T14"] * 24, fr["depth"] * 1e6
    tt, ff = t[good], f[good]
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"{star}  signal {n}:  P = {P:.5f} d   T0 = {T0:.4f}   depth = {dep:.0f} ppm   "
                 f"T14 = {D_h:.2f} h   ->  {verdict.upper()}\n{reason}", fontsize=11)
    ax0 = fig.add_subplot(3, 1, 1)
    if pgram is not None:
        pg = np.load(pgram)
        ax0.plot(pg[0], pg[1], "k-", lw=0.5); ax0.set_xscale("log")
        ax0.axvline(P, color="r", alpha=0.25, lw=4)
        ax0.set_xlabel("period [d]"); ax0.set_ylabel("BLS power (SNR-like)")
        ax0.set_title("BLS periodogram (earlier signals masked); red = this signal", fontsize=9)
    ph = fold_phase(tt, P, T0) * 24
    w = min(max(3 * D_h, 4), P * 24 / 2)
    m = np.abs(ph) < w
    ax1 = fig.add_subplot(3, 3, 4)
    ax1.plot(ph[m], (ff[m] - 1) * 1e6, ".", color="0.7", ms=1)
    xb, yb = binned(ph[m], (ff[m] - 1) * 1e6, max(D_h / 10, 2 / 60))
    ax1.plot(xb, yb, "o", color="C0", ms=3)
    xm = np.linspace(-w, w, 1000)
    ax1.plot(xm, (trapezoid(xm / 24, fr["depth"], fr["T14"], fr["ingress_frac"] * fr["T14"] / 2) - 1) * 1e6, "r-", lw=1.2)
    lim = (-1.6 * dep - 3 * robust_std(yb), 0.6 * dep + 4 * robust_std(yb))
    ax1.set_ylim(*lim); ax1.set_xlabel("hours from mid-transit"); ax1.set_ylabel("flux - 1 [ppm]")
    ax1.set_title("phase-folded (red: trapezoid fit)", fontsize=9)
    ax2 = fig.add_subplot(3, 3, 5)
    ep = np.round((tt - T0) / P).astype(int)
    for par, c, lab in [(0, "C0", "even"), (1, "C3", "odd")]:
        mm = m & (np.abs(ep % 2) == par)
        xb, yb = binned(ph[mm], (ff[mm] - 1) * 1e6, max(D_h / 6, 2 / 60))
        ax2.plot(xb, yb, "o-", color=c, ms=3, lw=0.8,
                 label=f"{lab}: {st[lab] * 1e6:.0f}±{st[lab + '_e'] * 1e6:.0f} ppm")
    ax2.axhline(-dep, color="k", ls=":", lw=0.8); ax2.legend(fontsize=8); ax2.set_ylim(*lim)
    ax2.set_title("odd vs even transits", fontsize=9); ax2.set_xlabel("hours")
    ax3 = fig.add_subplot(3, 3, 6)
    ph2 = fold_phase(tt, P, T0 + P / 2) * 24
    m2 = np.abs(ph2) < w
    ax3.plot(ph2[m2], (ff[m2] - 1) * 1e6, ".", color="0.7", ms=1)
    xb, yb = binned(ph2[m2], (ff[m2] - 1) * 1e6, max(D_h / 6, 2 / 60))
    ax3.plot(xb, yb, "o", color="C2", ms=3)
    ax3.axvspan(-D_h / 2, D_h / 2, color="r", alpha=0.07); ax3.set_ylim(*lim)
    ax3.set_title(f"phase 0.5: secondary = {st['sec'] * 1e6:.0f}±{st['sec_e'] * 1e6:.0f} ppm", fontsize=9)
    ax3.set_xlabel("hours from phase 0.5")
    ax4 = fig.add_subplot(3, 1, 3)
    idx = np.arange(len(t))
    ax4.plot(idx[good], f[good], ".", color="0.55", ms=0.6)
    ax4.plot(idx[~good], f[~good], ".", color="orange", ms=0.6)
    intr = np.abs(fold_phase(t, P, T0)) < fr["T14"] / 2
    ax4.plot(idx[intr & good], f[intr & good], ".", color="r", ms=1.2)
    for b in np.where(np.diff(seg) != 0)[0]:
        ax4.axvline(b, color="k", lw=0.6)
    lo, hi = np.percentile(f[good], [0.05, 99.95])
    ax4.set_ylim(min(lo, 1 - 1.4e-6 * dep), hi)
    ax4.set_xlabel("data point index (black lines = segment boundaries; red = in transit; orange = masked)")
    ax4.set_ylabel("detrended flux")
    plt.tight_layout(rect=(0, 0, 1, 0.94)); plt.savefig(out, dpi=85); plt.close()


def jitter_signal(star, t, fd, raw, good, seg, s):
    """Star 07: periodic bursts of extreme scatter in segment 3.  Locate burst centres from
    the local point-to-point scatter, fit a linear ephemeris, measure mean flux deficit."""
    m = seg == 3
    tt, ff = t[m], raw[m]
    dd = np.abs(np.diff(ff, prepend=ff[0]))
    k = np.floor((tt - tt[0]) / 0.05).astype(int)
    u, inv, c = np.unique(k, return_inverse=True, return_counts=True)
    sc = np.bincount(inv, dd) / c; tb = np.bincount(inv, tt) / c
    hot = sc > 5 * np.median(sc)
    # group hot bins into events
    ev, cur = [], [tb[np.where(hot)[0][0]]]
    for x in tb[hot][1:]:
        if x - cur[-1] < 1.0: cur.append(x)
        else: ev.append(cur); cur = [x]
    ev.append(cur)
    ev = [e for e in ev if len(e) >= 3]
    centres = np.array([0.5 * (min(e) + max(e)) for e in ev])
    widths = np.array([max(e) - min(e) for e in ev])
    n = np.round((centres - centres[0]) / s["P"])
    A = np.vstack([np.ones_like(n), n]).T
    (T0, P), *_ = np.linalg.lstsq(A, centres, rcond=None)
    resid = centres - (T0 + n * P)
    W = np.median(widths) + 0.05
    inwin = np.zeros(len(t), bool)
    for c0 in centres:
        inwin |= np.abs(t - c0) < W / 2
    base = np.median(raw[(seg == 3) & ~inwin])
    deficit = base - np.mean(raw[inwin])
    return dict(P=P, T0=centres[0], T14=W, depth=deficit, centres=centres, resid=resid, widths=widths)


def plot_jitter(star, t, raw, seg, js, reason, out):
    fig, ax = plt.subplots(2, 1, figsize=(15, 8))
    m = seg == 3
    ax[0].plot(t[m], raw[m], "k.", ms=0.7)
    for c0 in js["centres"]:
        ax[0].axvspan(c0 - js["T14"] / 2, c0 + js["T14"] / 2, color="r", alpha=0.15)
    ax[0].set_title(f"{star}: segment 3 raw flux; red = high-scatter episodes (spacing {np.round(np.diff(js['centres']), 2)} d)", fontsize=10)
    ax[0].set_xlabel("time [d]"); ax[0].set_ylabel("raw flux")
    for i, c0 in enumerate(js["centres"]):
        w = np.abs(t - c0) < 1.5
        ax[1].plot(t[w] - c0, raw[w] + 0.25 * i, ".", ms=1.5, label=f"event at {c0:.2f}")
    ax[1].set_xlabel("days from episode centre (offset vertically)"); ax[1].legend(fontsize=8)
    ax[1].set_title("episodes overlaid: chaotic +-10% scatter, not a transit shape", fontsize=10)
    fig.suptitle(f"{star}: P ~ {js['P']:.2f} d  ->  OTHER FALSE POSITIVE\n{reason}", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.93)); plt.savefig(out, dpi=85); plt.close()


def main():
    import os
    os.makedirs("plots", exist_ok=True)
    results, table = {}, {}
    for star, sigs in SIGNALS.items():
        results[star] = []
        if not sigs:
            continue
        t, fd, raw, tr, good, seg = final_lc(star, sigs)
        pd.DataFrame(dict(t=t, f=fd, raw=raw, trend=tr, good=good, seg=seg)).to_csv(f"work/{star}_final_lc.csv", index=False)
        for n, s in enumerate(sigs, 1):
            if s.get("kind") == "jitter":
                js = jitter_signal(star, t, fd, raw, good, seg, s)
                reason = ("Instrumental: recurring ~0.8-d bursts of +-10% point-to-point scatter (pointing jitter/scattered light) "
                          "only in segment 3, quasi-periodic (14.2-14.6 d spacing), no transit shape; no transit signal remains once masked")
                plot_jitter(star, t, raw, seg, js, reason, f"plots/{star}_signal{n}_P{js['P']:.2f}d_jitter.png")
                results[star].append(dict(period_days=round(js["P"], 3), t0_days=round(js["T0"], 3),
                                          depth_ppm=round(js["depth"] * 1e6), duration_hours=round(js["T14"] * 24, 1),
                                          verdict="other false positive", reason=reason))
                table[f"{star}#{n}"] = dict(P=js["P"], centres=list(js["centres"]), resid=list(js["resid"]),
                                            widths=list(js["widths"]), depth=js["depth"])
                continue
            # exclude other signals' transits (overlapping transits bias depths/odd-even)
            good_s = good.copy()
            for o in sigs:
                if o is not s and o.get("kind") != "jitter":
                    good_s &= ~(np.abs(fold_phase(t, o["P"], o["T0"])) < 0.75 * o["D"] / 24 + 0.5 / 24)
            fr = fit_star(star, t, fd, good_s, s)
            st = signal_stats(t, fd, fr["P"], fr["T0"], fr["T14"], good_s)
            extra = dict(oddeven=odd_even_fit(t, fd, good_s, fr))
            good_plot = good_s
            if s.get("halfP"):
                fr2 = fit_star(star, t, fd, good_s, dict(P=fr["P"], T0=fr["T0"] + fr["P"] / 2, D=s["D"], dep=0.08), fix_P=True)
                extra["secondary_fit"] = fr2
            if star == "star_08":
                # secondary eclipse: same shape, centred at phase 0.5 (circular orbit), depth free
                extra["secondary_fit"] = fit_star(star, t, fd, good_s, dict(P=fr["P"], T0=fr["T0"] + fr["P"] / 2, D=fr["T14"] * 24, dep=3e-4), fix_P=True, fix_shape=fr)
            verdict, reason = judge(star, n, fr, st, extra)
            pg = f"work/{star}_pgram{s['pgram']}.npy" if s.get("pgram") is not None else None
            plot_signal(star, n, t, fd, good_plot, seg, fr, st, verdict, reason, pg,
                        f"plots/{star}_signal{n}_P{fr['P']:.4f}d.png")
            results[star].append(dict(period_days=round(fr["P"], 6), t0_days=round(fr["T0"], 5),
                                      depth_ppm=round(fr["depth"] * 1e6, 1), duration_hours=round(fr["T14"] * 24, 3),
                                      verdict=verdict, reason=reason))
            table[f"{star}#{n}"] = dict(fit=fr, stats={k: v for k, v in st.items() if k != "per_transit"},
                                        per_transit_ppm={k: round(v * 1e6) for k, v in st["per_transit"].items()}, **extra)
            oe = extra["oddeven"]
            print(star, n, "odd/even fit: %.0f±%.0f vs %.0f±%.0f (%.1f sig, beta=%.2f)" % (
                oe["odd"] * 1e6, oe["odd_e"] * 1e6, oe["even"] * 1e6, oe["even_e"] * 1e6, oe["sig"], oe["beta"]))
            print(star, n, verdict, "P=%.6f T0=%.5f depth=%.0f T14=%.2fh snr=%.1f odd=%.0f even=%.0f sec=%.0f±%.0f ingress_frac=%.2f" % (
                fr["P"], fr["T0"], fr["depth"] * 1e6, fr["T14"] * 24, st["snr"], st["odd"] * 1e6, st["even"] * 1e6,
                st["sec"] * 1e6, st["sec_e"] * 1e6, fr["ingress_frac"]), flush=True)
    json.dump(results, open("results.json", "w"), indent=1)
    json.dump(table, open("work/final_table.json", "w"), indent=1, default=float)


def fit_star(star, t, fd, good, s, fix_P=False, fix_shape=None):
    if fix_shape is not None:
        # only depth free: linear least squares against the fixed-shape template
        P, T0 = s["P"], s["T0"]
        m = good & (np.abs(fold_phase(t, P, T0)) < 2 * fix_shape["T14"])
        tmpl = 1 - trapezoid(fold_phase(t[m], P, T0), 1.0, fix_shape["T14"], fix_shape["ingress_frac"] * fix_shape["T14"] / 2)
        y = 1 - fd[m]
        A = np.vstack([tmpl, np.ones_like(tmpl)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ coef
        cov = np.linalg.inv(A.T @ A) * np.var(res)
        return dict(depth=coef[0], depth_e=np.sqrt(cov[0, 0]))
    return fit_lc(t, fd, good, s["P"], s["T0"], s["D"] / 24, s["dep"] / 1e6 if s["dep"] > 1 else s["dep"], fix_P)


def fit_lc(t, f, good, P, T0, D, depth, fix_P):
    import fit as F
    import pandas as _pd
    # reuse fit.fit but on in-memory data
    orig = F.pd.read_csv
    F.pd.read_csv = lambda *a, **k: _pd.DataFrame(dict(t=t, f=f, good=good))
    try:
        return F.fit("mem", P, T0, D, depth, fix_P=fix_P)
    finally:
        F.pd.read_csv = orig


def odd_even_fit(t, f, good, fr):
    """Depth of odd and even transits separately, using the fitted trapezoid shape as a
    fixed template (only depth + local offset free).  Errors are inflated by the
    red-noise factor beta measured from out-of-transit data binned on the transit duration."""
    P, T0, T14 = fr["P"], fr["T0"], fr["T14"]
    ph = fold_phase(t, P, T0)
    ep = np.round((t - T0) / P).astype(int)
    oot = good & (np.abs(ph) > T14) & (np.abs(ph) < 4 * T14 + 0.1)
    sig1 = robust_std(f[oot])
    k = np.floor((t[oot] - t[oot].min()) / T14).astype(np.int64)
    u, inv, c = np.unique(k, return_inverse=True, return_counts=True)
    fb = np.bincount(inv, f[oot]) / c
    ok = c > 0.6 * np.median(c)
    beta = max(1.0, robust_std(fb[ok]) / (sig1 / np.sqrt(np.median(c))))
    out = dict(beta=beta)
    for name, sel in [("odd", ep % 2 != 0), ("even", ep % 2 == 0)]:
        m = good & sel & (np.abs(ph) < 2 * T14 + 0.05)
        tmpl = 1 - trapezoid(ph[m], 1.0, T14, fr["ingress_frac"] * T14 / 2)
        A = np.vstack([tmpl, np.ones_like(tmpl)]).T
        coef, *_ = np.linalg.lstsq(A, 1 - f[m], rcond=None)
        cov = np.linalg.inv(A.T @ A) * sig1**2
        out[name] = coef[0]; out[name + "_e"] = np.sqrt(cov[0, 0]) * beta
    out["sig"] = abs(out["odd"] - out["even"]) / np.hypot(out["odd_e"], out["even_e"])
    return out


def judge(star, n, fr, st, extra):
    """Verdict logic (thresholds documented in NOTES.md)."""
    dep = fr["depth"] * 1e6
    oe = extra["oddeven"]
    oe_sig = oe["sig"]
    sec_sig = st["sec"] / st["sec_e"]
    if star == "star_11":
        s2 = extra["secondary_fit"]
        return ("eclipsing binary",
                f"BLS found P/2=1.567 d but alternate eclipses differ ({fr['depth']*1e2:.1f}% vs {s2['depth']*1e2:.1f}%, "
                f">20 sigma); two unequal ~10% eclipses per 3.134-d orbit = EB")
    if star == "star_08":
        s2 = extra["secondary_fit"]
        return ("planet candidate",
                f"1.05% flat-bottomed transit, odd=even; {st['sec']*1e6:.0f}-{s2['depth']*1e6:.0f} ppm secondary eclipse and "
                f"~180 ppm ellipsoidal/~140 ppm phase curve fit a hot (~2800 K) massive planet, too small for a stellar companion")
    if oe_sig > 3 or sec_sig > 4 or dep > 30000:
        return ("eclipsing binary", f"odd/even {oe_sig:.1f} sigma, secondary {sec_sig:.1f} sigma")
    base = (f"{dep:.0f} ppm, {fr['T14']*24:.2f} h transit seen {st['ntr']}x (SNR {st['snr']:.0f}); "
            f"odd/even agree ({oe_sig:.1f} sigma), no secondary ({sec_sig:.1f} sigma)")
    notes = {
        "star_01": "; one of 3 transiting signals in this star (multi-planet system)",
        "star_02": "; ultra-short period, short duration implies small (M-dwarf) host",
        "star_03": "; shallow, consistent across all 3 segments",
        "star_06": "; on a spotted, flaring young star (transits survive a flare-robust detrend)",
        "star_09": "; only 3 transits, V-ish shape poorly constrained",
        "star_10": "; one of 3 transiting signals (5.66 & 11.38 d near 2:1)",
        "star_12": "; somewhat V-shaped (possibly grazing) but depth too small for an undiluted EB",
    }
    return "planet candidate", base + notes.get(star, "")


if __name__ == "__main__":
    main()
