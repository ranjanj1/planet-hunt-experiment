"""Transit search: detrend each star, run iterative BLS (mask & repeat) on the
longest group of nearby segments, then refine each period across all data."""
import sys, json, os
import numpy as np
from astropy.timeseries import BoxLeastSquares
from pipeline import load, detrend, clip_upper, bin_lc, sde, split_chunks, period_grid

SCR = os.environ.get("SCR", ".")
WINDOW = {"star_06": 0.4}
DURS = np.array([0.02, 0.03, 0.045, 0.065, 0.09, 0.125, 0.17, 0.23, 0.3, 0.4])


def prepare(star, window=None, transit_mask=None):
    d = load(star)
    o = np.argsort(d["time_days"])
    t, f, e, seg = (d[k][o] for k in ("time_days", "flux", "flux_err", "segment"))
    w = window or WINDOW.get(star, 0.75)
    m = None if transit_mask is None else transit_mask(t)
    fd, trend = detrend(t, f, w, mask=m)
    good = np.isfinite(fd) & clip_upper(fd, 4)
    # flag 0.25-d stretches whose scatter is far above typical (instrumental artefacts)
    resid = fd - 1
    sc = np.zeros_like(fd)
    for ch in split_chunks(t, 0.5):
        tb = np.floor((t[ch] - t[ch][0]) / 0.25).astype(int)
        for b in np.unique(tb):
            mm = tb == b
            r = resid[ch[mm]]
            sc[ch[mm]] = 1.4826 * np.median(np.abs(r - np.median(r)))
    good &= sc < 2.5 * np.median(sc)
    return dict(t=t[good], f=fd[good], e=e[good], seg=seg[good], raw=f[good], trend=trend[good])


def groups(t, gap=100):
    br = np.where(np.diff(t) > gap)[0] + 1
    return np.split(np.arange(len(t)), br)


class _Res:
    pass


def bls_run(t, f, periods, durs=DURS):
    """BLS over `periods`; the period range is split into chunks so that each
    chunk may use every duration shorter than ~0.2 x its shortest period
    (astropy requires max duration < min period)."""
    dy = np.full_like(f, 1.4826 * np.median(np.abs(f - np.median(f))))
    bls = BoxLeastSquares(t, f, dy)
    periods = np.sort(periods)
    edges = [periods.min()]
    for d in durs:
        if d / 0.2 > edges[-1] and d / 0.2 < periods.max():
            edges.append(d / 0.2)
    edges.append(periods.max() * 1.0001)
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        p = periods[(periods >= lo) & (periods < hi)]
        if len(p) == 0:
            continue
        dd = durs[durs <= 0.2 * lo + 1e-9]
        if len(dd) == 0:
            dd = durs[:1]
        parts.append(bls.power(p, dd, objective="snr"))
    res = _Res()
    for k in ("period", "power", "transit_time", "duration", "depth", "depth_snr"):
        setattr(res, k, np.concatenate([getattr(r, k) for r in parts]))
    return bls, res


def run(star, nsig=4, binw=10 / 1440, pmin=0.3):
    lc = prepare(star)
    t, f = lc["t"], lc["f"]
    tb, fb, _ = bin_lc(t, f, binw)
    gr = groups(tb)
    spans = [tb[g].max() - tb[g].min() for g in gr]
    gmain = gr[int(np.argmax(spans))]
    Tg = spans[int(np.argmax(spans))]
    Tall = tb.max() - tb.min()
    print(f"{star}: groups={len(gr)} spans={np.round(spans,1)} Tall={Tall:.1f}", flush=True)
    out = []
    mask = np.zeros_like(tb, bool)
    for k in range(nsig):
        sel = np.zeros_like(mask); sel[gmain] = True; sel &= ~mask
        periods = period_grid(pmin, min(Tg / 2, 40), Tg, DURS.min(), 3)
        bls, res = bls_run(tb[sel], fb[sel], periods)
        i = np.argmax(res.power)
        P, t0, dur, dep = res.period[i], res.transit_time[i], res.duration[i], res.depth[i]
        s = sde(res.power)
        # refine across all groups
        if len(gr) > 1:
            dP = 3 * P * dur / Tg
            per2 = period_grid(P - dP, P + dP, Tall, dur / 2, 5, 2_000_000)
            bls2, res2 = bls_run(tb[~mask], fb[~mask], per2, np.array([dur * 0.7, dur, dur * 1.4]))
            j = np.argmax(res2.power)
            P, t0, dur, dep = res2.period[j], res2.transit_time[j], res2.duration[j], res2.depth[j]
        st = BoxLeastSquares(tb[~mask], fb[~mask]).compute_stats(P, dur, t0)
        print(f"  sig{k}: P={P:.6f} t0={t0:.4f} dur={dur*24:.2f}h depth={dep*1e6:.0f}ppm SDE={s:.1f} "
              f"odd={st['depth_odd'][0]*1e6:.0f}+-{st['depth_odd'][1]*1e6:.0f} even={st['depth_even'][0]*1e6:.0f}+-{st['depth_even'][1]*1e6:.0f} "
              f"half={st['depth_half'][0]*1e6:.0f} phase0.5={st['depth_phased'][0]*1e6:.0f}+-{st['depth_phased'][1]*1e6:.0f} ntr={np.sum(st['per_transit_count']>0)}", flush=True)
        np.save(f"{SCR}/pg_{star}_{k}.npy", np.vstack([res.period, res.power]))
        out.append(dict(P=float(P), t0=float(t0), dur=float(dur), depth=float(dep), sde=float(s)))
        ph = ((tb - t0 + 0.5 * P) % P) - 0.5 * P
        mask |= np.abs(ph) < 1.0 * dur
    return out


if __name__ == "__main__":
    stars = sys.argv[1:]
    for s in stars:
        r = run(s)
        json.dump(r, open(f"{SCR}/pass_{s}.json", "w"), indent=1)
