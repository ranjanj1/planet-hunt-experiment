"""Final step: fit each accepted signal, measure vetting statistics, write
plots/<star>_<n>.png and results.json."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fit import fit, trapezoid
from search import prepare
from vet import stats, binned, phase
from pipeline import load, bin_lc

# Accepted signals: (P, t0, T14 guess, depth guess, deg of local baseline)
SIGNALS = {
    # (a 4th, 1.049-d / ~150 ppm signal in star_01 failed the inverted-light-curve
    #  false-alarm test and is not reported; see NOTES.md)
    "star_01": [(3.69060, 2.2039, 0.052, 1700e-6), (7.45083, 1.2869, 0.040, 1600e-6),
                (2.25313, 0.9055, 0.046, 650e-6)],
    "star_02": [(0.462930, 0.72559, 0.022, 4350e-6)],
    "star_03": [(6.267808, 0.50261, 0.132, 300e-6)],
    "star_04": [], "star_05": [],
    "star_06": [(8.463178, 5.37973, 0.157, 2890e-6), (18.85903, 17.2227, 0.178, 1250e-6)],
    "star_07": [(3.182780, 3.6070, 0.065, 480e-6)],
    "star_08": [(0.941452, 0.45791, 0.090, 0.0103)],
    "star_09": [(16.050974, 15.08999, 0.068, 3200e-6)],
    "star_10": [(5.660254, 4.50195, 0.067, 3800e-6), (11.380630, 4.67417, 0.10, 3150e-6),
                (3.360144, 2.09201, 0.056, 980e-6)],
    # eclipsing binary: orbital period = 2 x BLS period, t0 = deeper eclipse
    "star_11": [(2 * 1.566887, 0.85121, 0.064, 0.12)],
    "star_12": [(5.651759, 1.22957, 0.145, 1580e-6)],
}

# Time ranges to exclude from fits (instrumental artefacts), as (huge period, centre, half-width/0.6)
ARTEFACTS = {
    "star_07": [(1e7, c, 1.6) for c in (2523.8, 2538.4, 2552.7, 2567.2)],
}


def secondary_depth(star, P, t0, T14, other, deg):
    """Fit a box-ish eclipse at phase 0.5 with fixed P, T14 (depth only via trapezoid)."""
    try:
        r = fit(star, P, t0 + P / 2, T14, 1e-4, other, deg=deg, fit_period=False)
        return r["depth"], r["err"]["depth"], r["t0"] - (t0 + P / 2)
    except Exception as ex:  # noqa
        return np.nan, np.nan, np.nan


def ellipsoidal(star, P, t0, T14):
    """Amplitude of cos(4*pi*phase) and cos(2*pi*phase) in out-of-eclipse raw flux
    (each segment normalised by its median), eclipses masked."""
    d = load(star)
    t, f, s = d["time_days"], d["flux"], d["segment"]
    fn = np.empty_like(f)
    for k in np.unique(s):
        m = s == k
        fn[m] = f[m] / np.median(f[m])
    ph = ((t - t0) / P) % 1
    oot = (np.abs(phase(t, P, t0)) > T14) & (np.abs(phase(t, P, t0 + P / 2)) > T14)
    A = np.vstack([np.ones(oot.sum()), np.cos(2 * np.pi * ph[oot]), np.sin(2 * np.pi * ph[oot]),
                   np.cos(4 * np.pi * ph[oot]), np.sin(4 * np.pi * ph[oot])]).T
    c, *_ = np.linalg.lstsq(A, fn[oot], rcond=None)
    r = fn[oot] - A @ c
    cov = np.linalg.inv(A.T @ A) * np.var(r)
    return c, np.sqrt(np.diag(cov))


def fit_stats(fitres):
    """Odd/even and per-transit depths from the fit's locally-normalised data
    (flux / local polynomial baseline), central 60% of the transit."""
    P, t0, T14 = fitres["P"], fitres["t0"], fitres["T14"]
    tall, fall, model, base, keep, gid, tmid = fitres["data"]
    y = fall / base
    x = phase(tall, P, t0)
    inn = keep & (np.abs(x) < 0.3 * T14)
    oot = keep & (np.abs(x) > 0.6 * T14)
    sig = 1.4826 * np.median(np.abs(y[oot] - np.median(y[oot])))
    ep = np.round((tall - t0) / P).astype(int)

    def dep(m):
        n = (inn & m).sum()
        if n < 3:
            return (np.nan, np.nan)
        return (1 - np.mean(y[inn & m]), sig / np.sqrt(n))
    odd, even = dep(ep % 2 == 1), dep(ep % 2 == 0)
    per = [(t0 + e * P, *dep(ep == e)) for e in np.unique(ep[inn])]
    per = np.array([p for p in per if np.isfinite(p[1])])
    oe = abs(odd[0] - even[0]) / np.hypot(odd[1], even[1])
    # chi2 of per-transit depths about their weighted mean (depth consistency)
    w = 1 / per[:, 2] ** 2
    mu = np.sum(w * per[:, 1]) / np.sum(w)
    chi2 = float(np.sum(((per[:, 1] - mu) / per[:, 2]) ** 2))
    return dict(odd=odd, even=even, oddeven_sigma=oe, per=per, depth=dep(np.ones_like(ep, bool)),
                chi2_per=chi2, ndof_per=len(per) - 1)


def make_plot(star, n, sig, fitres, lc, extra_title, path):
    P, t0, T14, dep = fitres["P"], fitres["t0"], fitres["T14"], fitres["depth"]
    tall, fall, model, base, keep, gid, tmid = fitres["data"]
    t, f = lc["t"], lc["f"]
    s = stats(lc, P, t0, T14)
    fs = fit_stats(fitres)
    s.update(odd=fs["odd"], even=fs["even"], oddeven_sigma=fs["oddeven_sigma"], per=fs["per"],
             depth=fs["depth"][0], chi2_per=fs["chi2_per"], ndof_per=fs["ndof_per"])
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3)
    # (a) detrended light curve with transit times
    ax = fig.add_subplot(gs[0, :])
    seglist = np.unique(lc["seg"])
    # compress gaps: plot index-ordered segments side by side
    offs, x0, ticks = {}, 0.0, []
    for sg in seglist:
        m = lc["seg"] == sg
        offs[sg] = x0 - t[m].min()
        ticks.append((x0, t[m].min()))
        x0 += t[m].max() - t[m].min() + 2
    xx = t + np.array([offs[v] for v in lc["seg"]])
    tb, fb, _ = bin_lc(xx, f, 30 / 1440)
    ax.plot(xx, f, ",", color="0.65")
    ax.plot(tb, fb, ".", ms=2, color="k")
    for sg in seglist:
        m = lc["seg"] == sg
        ep = np.arange(np.ceil((t[m].min() - t0) / P), np.floor((t[m].max() - t0) / P) + 1)
        for e in ep:
            ax.axvline(t0 + e * P + offs[sg], color="C3", alpha=0.35, lw=0.8)
        ax.text(t[m].min() + offs[sg], ax.get_ylim()[1] if False else 1 + 3 * s["sig"], f"seg {int(sg)}", fontsize=8)
    lo = 1 - 1.6 * dep - 3 * s["sig"] / 3
    ax.set_ylim(max(lo, np.percentile(f, 0.05)) , 1 + 4 * s["sig"])
    ax.set_xlabel("time [d] (segments placed side by side; red lines = predicted transits)")
    ax.set_ylabel("detrended flux")
    ax.set_title(f"{star} signal {n}: {extra_title}")
    # (b) phase-folded transit, raw flux / local baseline, with model
    a = fig.add_subplot(gs[1, 0:2])
    x = phase(tall, P, t0) * 24
    y = fall / base
    a.plot(x[keep], y[keep], ".", ms=1, color="0.7")
    xb, yb, eb = binned(x[keep], y[keep], max(T14 * 24 / 15, 0.05))
    a.errorbar(xb, yb, eb, fmt="o", ms=3, color="C0", label="binned data")
    xm = np.linspace(x.min(), x.max(), 2000)
    a.plot(xm, trapezoid(xm / 24 + t0, t0, P, dep, T14, fitres["T12"]), "C3", lw=2, label="trapezoid fit")
    a.set_xlabel("hours from mid-transit"); a.set_ylabel("relative flux")
    a.set_title(f"P={P:.6f} d  t0={t0:.5f}  depth={dep*1e6:.0f}±{fitres['err']['depth']*1e6:.0f} ppm  "
                f"T14={T14*24:.2f} h  N={fitres['ntransits']}  SNR={fitres['snr']:.1f}", fontsize=10)
    yl = (1 - 1.35 * dep - 2 * np.median(eb), 1 + 0.4 * dep + 3 * np.median(eb))
    a.set_ylim(*yl); a.legend(fontsize=8, loc="lower right")
    # (c) odd/even
    a = fig.add_subplot(gs[1, 2])
    ep = np.round((tall - t0) / P).astype(int)
    for sel, c, nm in [(ep % 2 == 0, "C0", "even"), (ep % 2 == 1, "C1", "odd")]:
        xb, yb, eb = binned(x[keep & sel], y[keep & sel], max(T14 * 24 / 10, 0.05))
        a.errorbar(xb, yb, eb, fmt="o", ms=3, color=c, label=f"{nm}: {s[nm][0]*1e6:.0f}±{s[nm][1]*1e6:.0f} ppm")
    a.set_ylim(*yl); a.legend(fontsize=8, loc="lower right")
    a.set_title(f"odd vs even: {s['oddeven_sigma']:.1f} σ difference"); a.set_xlabel("hours")
    # (d) phase 0.5
    a = fig.add_subplot(gs[2, 0])
    x2 = phase(t, P, t0 + P / 2) * 24
    m = np.abs(x2) < max(3 * T14 * 24, 4)
    xb, yb, eb = binned(x2[m], f[m], max(T14 * 24 / 10, 0.1))
    a.plot(x2[m], f[m], ",", color="0.7")
    a.errorbar(xb, yb, eb, fmt="o", ms=3, color="C2")
    a.axvspan(-T14 * 12, T14 * 12, color="y", alpha=0.2)
    a.set_ylim(*yl)
    a.set_title(f"phase 0.5 (secondary eclipse?): {s['sec'][0]*1e6:.0f}±{s['sec'][1]*1e6:.0f} ppm", fontsize=10)
    a.set_xlabel("hours from phase 0.5")
    # (e) individual transit depths
    a = fig.add_subplot(gs[2, 1])
    if len(s["per"]):
        a.errorbar(s["per"][:, 0], s["per"][:, 1] * 1e6, s["per"][:, 2] * 1e6, fmt="o")
        a.axhline(s["depth"] * 1e6, color="k", ls=":")
    a.set_title(f"individual transit depths [ppm]  (χ²={s['chi2_per']:.0f}/{s['ndof_per']} dof)", fontsize=10)
    a.set_xlabel("time [d]")
    # (f) full phase curve (binned)
    a = fig.add_subplot(gs[2, 2])
    ph = ((t - t0) / P + 0.25) % 1 - 0.25
    xb, yb, eb = binned(ph, f, 0.005 if P < 3 else 0.002)
    a.plot(xb, yb, ".", ms=3, color="k")
    a.set_ylim(1 - 1.3 * dep, 1 + 0.5 * dep)
    a.set_title("full phase-fold (detrended)", fontsize=10); a.set_xlabel("phase")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return s


if __name__ == "__main__":
    allfit = {}
    for star, sigs in SIGNALS.items():
        if not sigs:
            continue
        deg = 3 if star == "star_06" else 2
        lc = prepare(star, transit_mask=lambda t, sigs=sigs: np.any(
            [np.abs(phase(t, P, t0)) < 0.75 * T for P, t0, T, _ in sigs], axis=0))
        allfit[star] = []
        for n, (P, t0, T, dep) in enumerate(sigs):
            other = [(c[0], c[1], c[2]) for j, c in enumerate(sigs) if j != n] + ARTEFACTS.get(star, [])
            r = fit(star, P, t0, T, dep, other, deg=deg)
            sd, sde_, soff = secondary_depth(star, r["P"], r["t0"], r["T14"], other, deg)
            ell, ell_e = ellipsoidal(star, r["P"], r["t0"], r["T14"]) if r["P"] < 4 else (np.zeros(5), np.zeros(5))
            title = f"P={r['P']:.5f} d"
            s = make_plot(star, n + 1, None, r, lc, title, f"plots/{star}_{n+1}.png")
            rec = dict(P=r["P"], P_err=r["err"]["P"], t0=r["t0"], t0_err=r["err"]["t0"], depth=r["depth"],
                       depth_err=r["err"]["depth"], T14=r["T14"], T12=r["T12"], snr=r["snr"], ntr=r["ntransits"],
                       odd=s["odd"], even=s["even"], oddeven_sigma=s["oddeven_sigma"],
                       chi2_per=s["chi2_per"], ndof_per=s["ndof_per"],
                       sec_fit=(sd, sde_), sec_box=s["sec"], ell_cos2=(ell[3], ell_e[3]), refl_cos1=(ell[1], ell_e[1]))
            rec = {k: (tuple(float(x) for x in v) if isinstance(v, tuple) else float(v)) for k, v in rec.items()}
            allfit[star].append(rec)
            print(star, n + 1, json.dumps({k: (round(v, 7) if isinstance(v, float) else [round(x, 7) for x in v]) for k, v in rec.items()}), flush=True)
    json.dump(allfit, open("fits_final.json", "w"), indent=1)
