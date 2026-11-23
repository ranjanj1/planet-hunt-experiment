"""Step 1: detrend every star and run an iterative BLS search.

Usage: python search.py [star numbers...]
Writes work/star_XX_flat.npz and prints the top BLS peaks per iteration."""
import sys, os, json
import numpy as np
from pipeline import load, detrend, bin_lc, bls_search, sde, fold

os.makedirs('work', exist_ok=True)
stars = [int(s) for s in sys.argv[1:]] or range(1, 13)


def dense_cluster(t, seg, span=220):
    """Largest set of consecutive segments spanning <= `span` days (for the search)."""
    segs = np.unique(seg)
    best = None
    for a in range(len(segs)):
        for b in range(a, len(segs)):
            m = (seg >= segs[a]) & (seg <= segs[b])
            if t[m].max() - t[m].min() <= span:
                if best is None or m.sum() > best.sum():
                    best = m
    return best


for i in stars:
    d = load(i)
    flat, trend, good = detrend(d, window=1.0)
    np.savez(f'work/star_{i:02d}_flat.npz', t=d['time_days'], f=flat, good=good,
             seg=d['segment'], raw=d['flux'], err=d['flux_err'], cad=d['cadence_s'])
    t, f = d['time_days'][good], flat[good]
    seg = d['segment'][good]
    cl = dense_cluster(t, seg)
    ts, fs = t[cl], f[cl]
    # bin to 10 min to speed BLS (all cadences <= 30 min)
    tb, fb, nb = bin_lc(ts, fs, 10 / 1440)
    print(f'=== star {i:02d}: search on {ts.min():.1f}-{ts.max():.1f} d, {len(tb)} bins, '
          f'rms={np.std(f):.2e}')
    mask = np.zeros_like(tb, bool)
    found = []
    for it in range(4):
        r, bls = bls_search(tb[~mask], fb[~mask])
        s = sde(r.power)
        k = np.argmax(r.power)
        P, t0, dur, dep = r.period[k], r.transit_time[k], r.duration[k], r.depth[k]
        print(f'  iter {it}: P={P:.5f} t0={t0:.4f} dur={dur*24:.2f}h depth={dep*1e6:.0f}ppm '
              f'SNR={r.depth_snr[k]:.1f} SDE={s[k]:.1f}')
        found.append(dict(P=float(P), t0=float(t0), dur=float(dur), depth=float(dep),
                          snr=float(r.depth_snr[k]), sde=float(s[k])))
        if s[k] < 7:
            break
        mask |= np.abs(fold(tb, P, t0)) < 1.0 * dur
    json.dump(found, open(f'work/star_{i:02d}_bls.json', 'w'), indent=1)
