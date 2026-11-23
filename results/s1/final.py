"""Final step: fit every accepted signal, compute vetting numbers, draw one plot per signal,
and write results.json.  The signal list + verdicts below come from the vetting described in
NOTES.md; all numbers in results.json are recomputed here from the data."""
import json, os
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline import *

os.makedirs('plots', exist_ok=True)
BLUE, ORANGE, GREY, INK = '#2a78d6', '#eb6834', '#b9b8b0', '#3d3c38'
plt.rcParams.update({'axes.edgecolor': GREY, 'axes.labelcolor': INK, 'xtick.color': INK,
                     'ytick.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.size': 9})

# star -> list of (P, t0, T14 guess [d], verdict, reason-template, options)
SIGNALS = {
    1: [(1.04930, 0.7091, 0.045, 'planet candidate', None, {}),
        (2.253165, 93.28406, 0.05, 'planet candidate', None, {}),
        (3.690614, 94.46910, 0.052, 'planet candidate', None, {}),
        (7.450871, 90.69699, 0.035, 'planet candidate', None, {})],
    2: [(0.462930, 720.58105, 0.022, 'planet candidate', None, {})],
    3: [(6.267791, 88.25165, 0.133, 'planet candidate', None, {})],
    6: [(8.463180, 724.74937, 0.158, 'planet candidate', None, {'method': 'spline', 'window': 0.4})],
    7: [(3.182786, 3.60315, 0.078, 'planet candidate', None, {'prange': 0.002})],
    8: [(0.941452, 38.11596, 0.091, 'planet candidate', None, {'prange': 0.0005})],
    9: [(16.050413, 63.24177, 0.059, 'planet candidate', None, {})],
    10: [(3.360117, 42.41372, 0.056, 'planet candidate', None, {}),
         (5.660234, 44.12362, 0.066, 'planet candidate', None, {}),
         (11.380005, 38.81784, 0.097, 'planet candidate', None, {})],
    11: [(3.133773, 41.59027, 0.064, 'eclipsing binary', None, {'prange': 0.0003})],
    12: [(5.651713, 52.09515, 0.145, 'planet candidate', None, {})],
}
REASONS = {
    (1, 1.049): 'shallow (~165 ppm, SNR ~8) but present at equal depth in all 3 segments; odd/even agree, no secondary; duration fits the small host of the other 3 planets',
    (1, 2.253): '30 transits, SNR 16; no secondary; odd/even within 2.3 sigma; one of 4 transiting signals in a compact system',
    (1, 3.691): '16 transits, SNR 40; flat-bottomed, odd/even identical, no secondary; period near 2:1 with the 7.45 d planet',
    (1, 7.451): '8 transits, SNR 24; odd/even agree, no secondary; short 0.85 h duration consistent with an M-dwarf host shared with the 3.69 d planet',
    (2, 0.463): 'ultra-short-period: 152 box-shaped transits, SNR 47; odd/even identical, no secondary (-1+-85 ppm); same depth in segments 700 d apart',
    (3, 6.268): '11 transits, SNR 35 on a very quiet star; odd/even agree, no secondary (-9+-8 ppm); 3.2 h duration fits a Sun-like star',
    (6, 8.463): '7 transits over 3 segments spanning 7 yr on a young spotted/flaring star; consistent depths and timing, no secondary',
    (7, 3.183): 'shallow (~430 ppm, SNR ~7) but found independently in each segment (8.4, 5.1, 4.6 sigma) with the same depth and a linking ephemeris; odd/even agree, no secondary',
    (8, 0.941): '1% flat-bottomed transit + 459 ppm secondary and phase curve; ellipsoidal (~170 ppm) and beaming (~26 ppm) amplitudes imply a few-Jupiter-mass companion, so a hot Jupiter, not an EB',
    (9, 16.051): '3 transits, SNR 15; aliases P/2-P/5 excluded by data at the intermediate epochs; odd/even agree, no secondary',
    (10, 3.360): '17 transits, SNR 15; odd/even agree, no secondary; third planet of a compact system (near 5:3 with 5.66 d)',
    (10, 5.660): '11 transits, SNR 49; flat-bottomed, odd/even identical, no secondary; near 2:1 with the 11.38 d planet',
    (10, 11.380): '6 transits, SNR 37; odd/even agree, no secondary, constant transit times; distinct period from the 5.66 d planet',
    (11, 3.134): 'eclipses alternate 13.5% / 10.3% deep (secondary exactly at phase 0.5, same 1.5 h width, V-shaped): two similar stars; BLS first finds half this period',
    (12, 5.652): '8 transits, SNR 25; odd/even agree once epoch scatter included (1.0 sigma), no secondary; rather V-shaped (high impact parameter) so weakest-vetted candidate',
}
EXTRA = {}  # filled from extra_signals.json if present (second-pass detections)
if os.path.exists('extra_signals.json'):
    for k, v in json.load(open('extra_signals.json')).items():
        SIGNALS.setdefault(int(k), [])
        for s in v:
            SIGNALS[int(k)].append((s['P'], s['t0'], s['T14'], s['verdict'], s.get('reason'),
                                    s.get('opts', {})))


def box_depth(t, f, P, tc, T14, T12):
    """Fixed-shape depth (and error) of a trapezoid centred at phase tc."""
    near = np.abs(fold(t, P, tc)) < 2.5 * T14
    s = 1 - trapezoid(t[near], P, tc, 1.0, T14, T12)
    if (s > 0.5).sum() < 5:
        return np.nan, np.nan
    A = np.vstack([np.ones(near.sum()), -s]).T
    c, *_ = np.linalg.lstsq(A, f[near], rcond=None)
    res = f[near] - A @ c
    # time-correlated noise: scale white error by beta from 30-min binned residuals
    tb, rb, nb = bin_lc(t[near], res, 0.0208)
    beta = max(1.0, np.std(rb[nb > 5]) / (np.std(res) / np.sqrt(np.median(nb)))) if np.sum(nb > 5) > 10 else 1
    cov = np.linalg.inv(A.T @ A) * np.var(res)
    return c[1], np.sqrt(cov[1, 1]) * beta


def first_observed_t0(t, P, t0, T14):
    """Move t0 onto an epoch that actually has in-transit data (the earliest)."""
    ep = np.round((t - t0) / P)
    intr = np.abs(fold(t, P, t0)) < T14 / 2
    e = ep[intr].min()
    return t0 + e * P


results, table = {}, []
for i in range(1, 13):
    results[f'star_{i:02d}'] = []
    sigs = SIGNALS.get(i, [])
    if not sigs:
        continue
    # refine all signals of a star jointly (each masked from the others)
    opts = sigs[0][5]
    fits, (t, flat, good, tmask) = refine(
        i, [(P, t0, T) for P, t0, T, *_ in sigs], window=opts.get('window', 1.0),
        prange=opts.get('prange', 0.002), method=opts.get('method', 'median'), verbose=False)
    for j, ((P0, t00, T0, verdict, reason, o), fit) in enumerate(zip(sigs, fits)):
        P, T14, T12 = fit['P'], fit['T14'], fit['T12']
        other = np.zeros_like(t, bool)
        for k, f2 in enumerate(fits):
            if k != j:
                other |= np.abs(fold(t, f2['P'], f2['t0'])) < 0.75 * f2['T14'] + 0.01
        m = good & ~other
        tt, ff = t[m], flat[m]
        t0 = first_observed_t0(tt, P, fit['t0'], T14)
        dep, edep = box_depth(tt, ff, P, t0, T14, T12)
        rows = epoch_depths(tt, ff, dict(P=P, t0=t0, T14=T14, T12=T12))
        de, ee, do, eo, oe_sig = odd_even(rows)
        sec, esec = box_depth(tt, ff, P, t0 + P / 2, T14, T12)
        snr = dep / edep
        rho, b = stellar_density(P, T14, T12, max(dep, 1e-6))
        info = dict(star=i, P=P, eP=fit['eP'], t0=t0, et0=fit['et0'], depth=dep * 1e6,
                    edepth=edep * 1e6, T14=T14 * 24, T12=T12 * 24, snr=snr, n_tr=len(rows),
                    odd=do * 1e6, eodd=eo * 1e6, even=de * 1e6, eeven=ee * 1e6, oe_sig=oe_sig,
                    sec=sec * 1e6, esec=esec * 1e6, rho=rho, verdict=verdict)
        table.append(info)
        reason = REASONS.get((i, round(P, 3)), reason)
        if reason is None:
            if verdict == 'planet candidate':
                reason = (f"{len(rows)} transits, SNR {snr:.0f}; odd/even depths agree ({oe_sig:.1f} sigma), "
                          f"no secondary at phase 0.5 ({sec*1e6:.0f}+-{esec*1e6:.0f} ppm), U-shaped")
            else:
                reason = ''
        info['reason'] = reason
        results[f'star_{i:02d}'].append(dict(
            period_days=round(float(P), 6), t0_days=round(float(t0), 5),
            depth_ppm=round(float(dep * 1e6), 1), duration_hours=round(float(T14 * 24), 3),
            verdict=verdict, reason=reason))

        # ---------------- plot
        ncol = 4 if i == 8 else 3
        fig, ax = plt.subplots(1, ncol, figsize=(4.7 * ncol, 4),
                               gridspec_kw=dict(width_ratios=[2, 1.3, 1.3, 2][:ncol]))
        w = min(max(3 * T14, 0.12), P / 2)
        ph = fold(tt, P, t0)
        near = np.abs(ph) < w
        binw = max(T14 / 12, 4 / 1440)
        a = ax[0]
        a.plot(ph[near] * 24, (ff[near] - 1) * 1e6, '.', ms=1, color=GREY, rasterized=True)
        tb, fb, nb = bin_lc(ph[near], ff[near], binw)
        a.plot(tb * 24, (fb - 1) * 1e6, 'o', ms=3.5, color=BLUE, label='binned data')
        xm = np.linspace(-w, w, 1000)
        a.plot(xm * 24, (trapezoid(xm, P, 0, dep, T14, T12) - 1) * 1e6, color=ORANGE, lw=2,
               label='trapezoid fit')
        lo = min(np.percentile((fb - 1) * 1e6, 1), -dep * 1e6) * 1.5
        a.set_ylim(lo, -lo * 0.6)
        a.set_xlabel('hours from mid-transit'); a.set_ylabel('relative flux (ppm)')
        a.set_title(f'star {i:02d}  P = {P:.5f} d   depth {dep*1e6:.0f} ppm   T14 {T14*24:.2f} h',
                    loc='left', color=INK)
        a.legend(frameon=False, loc='lower right')
        a.grid(alpha=0.25)
        # odd / even
        a = ax[1]
        ep = np.round((tt - t0) / P).astype(int)
        for par, c, lab in [(0, BLUE, f'even {de*1e6:.0f}±{ee*1e6:.0f}'),
                            (1, ORANGE, f'odd {do*1e6:.0f}±{eo*1e6:.0f}')]:
            mm = near & (ep % 2 == par)
            if mm.sum() > 5:
                tb, fb, _ = bin_lc(ph[mm], ff[mm], binw)
                a.plot(tb * 24, (fb - 1) * 1e6, 'o', ms=3, color=c, label=lab)
        a.set_ylim(*ax[0].get_ylim())
        a.set_title(f'odd vs even ({oe_sig:.1f} σ apart)', loc='left', color=INK)
        a.set_xlabel('hours'); a.legend(frameon=False, loc='lower right'); a.grid(alpha=0.25)
        # secondary
        a = ax[2]
        ph2 = fold(tt, P, t0 + P / 2)
        n2 = np.abs(ph2) < w
        tb, fb, _ = bin_lc(ph2[n2], ff[n2], binw)
        a.plot(tb * 24, (fb - 1) * 1e6, 'o', ms=3, color=BLUE)
        a.axhline(0, color=GREY, lw=1)
        a.set_ylim(*ax[0].get_ylim())
        a.set_title(f'phase 0.5: {sec*1e6:.0f}±{esec*1e6:.0f} ppm', loc='left', color=INK)
        a.set_xlabel('hours from phase 0.5'); a.grid(alpha=0.25)
        if i == 8:
            # full-orbit phase curve from a gentle (5 d) detrend so it is not filtered out
            d8 = load(8)
            tm8 = np.abs(fold(d8['time_days'], P, t0)) < 0.06
            fl8, _, g8 = detrend(d8, window=5.0, mask=tm8)
            x = ((d8['time_days'][g8] - t0) / P + 0.25) % 1 - 0.25
            pb, fb, _ = bin_lc(x, fl8[g8], 0.01)
            a = ax[3]
            a.plot(pb, (fb - 1) * 1e6, 'o', ms=3, color=BLUE)
            ok = (np.abs(pb) > 0.06) & (np.abs(pb - 0.5) > 0.06)
            xx = 2 * np.pi * x
            A = np.vstack([np.ones_like(xx), np.cos(xx), np.sin(xx), np.cos(2 * xx), np.sin(2 * xx)]).T
            oot = (np.abs(x) > 0.06) & (np.abs(x - 0.5) > 0.06)
            c, *_ = np.linalg.lstsq(A[oot], fl8[g8][oot], rcond=None)
            xm = np.linspace(-0.25, 0.75, 400); xr = 2 * np.pi * xm
            mod = c[0] + c[1] * np.cos(xr) + c[2] * np.sin(xr) + c[3] * np.cos(2 * xr) + c[4] * np.sin(2 * xr)
            a.plot(xm, (mod - 1) * 1e6, color=ORANGE, lw=2, label='harmonic fit (out of eclipse)')
            a.set_ylim(-800, 400); a.grid(alpha=0.25); a.legend(frameon=False, loc='lower right')
            a.set_xlabel('orbital phase')
            a.set_title(f'phase curve: ellipsoidal {-c[3]*1e6:.0f}, reflection {-c[1]*1e6:.0f}, '
                        f'beaming {c[2]*1e6:.0f} ppm', loc='left', color=INK, fontsize=8.5)
            info['phase_curve_ppm'] = dict(ellipsoidal=-c[3] * 1e6, cos1=-c[1] * 1e6, beaming=c[2] * 1e6)
        fig.suptitle(f'verdict: {verdict}', x=0.99, ha='right', color=INK, fontsize=10)
        fig.tight_layout()
        fig.savefig(f'plots/star_{i:02d}_P{P:.3f}.png', dpi=110)
        plt.close(fig)

# ---------------- star 7: recurring instrumental bursts in segment 3 (not a transit)
d7 = np.genfromtxt('data/star_07.csv', delimiter=',', names=True)
t7, f7 = d7['time_days'], d7['flux']
s3 = d7['segment'] == 3
t7, f7 = t7[s3], f7[s3]
grid = np.arange(t7.min(), t7.max(), 0.05)
sc = np.array([np.std(f7[np.abs(t7 - x) < 0.05]) if np.sum(np.abs(t7 - x) < 0.05) > 5 else np.nan
               for x in grid])
bad = sc > 5 * np.nanmedian(sc)
groups, cur = [], []
for x, b in zip(grid, bad):
    if b:
        cur.append(x)
    elif cur:
        groups.append((cur[0], cur[-1])); cur = []
groups = [g for g in groups if g[1] - g[0] > 0.3]
cent = np.array([(a + b) / 2 for a, b in groups])
lens = np.array([b - a for a, b in groups])
deps = []
for c, L in zip(cent, lens):
    inn = np.abs(t7 - c) < L / 2; out = (np.abs(t7 - c) > 1) & (np.abs(t7 - c) < 2)
    deps.append(np.median(f7[out]) - np.median(f7[inn]))
n = np.arange(len(cent)); Pb, t0b = np.polyfit(n, cent, 1)[0], cent[0]
oc = (cent - (t0b + n * Pb)) * 24
results['star_07'].append(dict(
    period_days=round(float(Pb), 3), t0_days=round(float(cent[1]), 3),
    depth_ppm=round(float(np.mean(deps) * 1e6), 0), duration_hours=round(float(np.mean(lens) * 24), 1),
    verdict='other false positive',
    reason=(f'instrumental: {len(cent)} ~{np.mean(lens)*24:.0f} h episodes of +-10% chaotic scatter only in segment 3, '
            f'not eclipse-shaped, spacing irregular (O-C up to {np.max(np.abs(oc)):.0f} h); absent from segments 1-2')))
fig, ax = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw=dict(width_ratios=[2.2, 1]))
ax[0].plot(t7, (f7 - 1) * 100, '.', ms=1, color=INK, rasterized=True)
for c in cent:
    ax[0].axvspan(c - 0.4, c + 0.4, color=ORANGE, alpha=0.15, lw=0)
ax[0].set_xlabel('time (days)'); ax[0].set_ylabel('raw flux - 1 (%)')
ax[0].set_title(f'star 07 segment 3: recurring bursts, mean spacing {Pb:.2f} d', loc='left', color=INK)
ax[0].grid(alpha=0.25)
for k, c in enumerate(cent):
    m = np.abs(t7 - c) < 1.2
    ax[1].plot((t7[m] - c) * 24, (f7[m] - 1) * 100 - 25 * k, '.', ms=1, color=BLUE, rasterized=True)
ax[1].set_xlabel('hours from burst centre'); ax[1].set_yticks([])
ax[1].set_title(f'each burst (offset); O-C = {", ".join(f"{o:+.1f}" for o in oc)} h', loc='left',
                color=INK, fontsize=8.5)
fig.suptitle('verdict: other false positive (instrumental)', x=0.99, ha='right', color=INK, fontsize=10)
fig.tight_layout(); fig.savefig(f'plots/star_07_P{Pb:.3f}_instrumental.png', dpi=110); plt.close(fig)

json.dump(results, open('results.json', 'w'), indent=1)
json.dump(table, open('work/final_table.json', 'w'), indent=1, default=float)
for r in table:
    print(f"star {r['star']:2d} P={r['P']:.6f}±{r['eP']:.6f} t0={r['t0']:.5f} depth={r['depth']:.0f}±{r['edepth']:.0f} "
          f"T14={r['T14']:.2f}h T12={r['T12']:.2f}h SNR={r['snr']:.0f} ntr={r['n_tr']} odd={r['odd']:.0f}±{r['eodd']:.0f} "
          f"even={r['even']:.0f}±{r['eeven']:.0f} ({r['oe_sig']:.1f}σ) sec={r['sec']:.0f}±{r['esec']:.0f} rho={r['rho']:.2f} {r['verdict']}")
