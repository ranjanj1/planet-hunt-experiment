"""Second-pass search: mask the signals already identified (signals.json), trim 0.2 d at
chunk edges, then BLS each cluster of segments separately and on the inverted light curve
(the inverted run gives the false-alarm SDE level for this star's noise).
Usage: python search2.py STAR"""
import sys, json
import numpy as np
from pipeline import *

i = int(sys.argv[1])
known = json.load(open('signals.json')).get(str(i), [])
d = load(i)
t = d['time_days']
tm = np.zeros_like(t, bool)
for s in known:
    tm |= np.abs(fold(t, s['P'], s['t0'])) < 0.75 * s['T14'] + 0.02
cfg = CONFIG.get(i, {})
if cfg.get('method') == 'spline':
    flat, trend, good = detrend_spline(d, knot=cfg['knot'], mask=tm)
else:
    flat, trend, good = detrend(d, window=1.0, mask=tm)
# edge trim
cuts = np.where(np.diff(t) > 0.5)[0]
for s0, e0 in zip(np.r_[t[0], t[cuts + 1]], np.r_[t[cuts], t[-1]]):
    good &= ~(((t >= s0) & (t < s0 + 0.2)) | ((t > e0 - 0.2) & (t <= e0)))
good &= ~tm
np.savez(f'work/star_{i:02d}_resid.npz', t=t, f=flat, good=good)
tt, ff = t[good], flat[good]
# clusters: split where gap > 300 d
cl = np.split(np.arange(len(tt)), np.where(np.diff(tt) > 300)[0] + 1)
groups = [('all-cluster', c) for c in cl]
for name, idx in groups:
    tb, fb, _ = bin_lc(tt[idx], ff[idx], 10 / 1440)
    for sign, lab in [(1, 'normal'), (-1, 'inverted')]:
        r, _ = bls_search(tb, 1 + sign * (fb - 1))
        s = sde(r.power)
        order = np.argsort(-r.power)
        picks = []
        for k in order:
            if all(abs(r.period[k] / p - round(r.period[k] / p)) > 0.02 and
                   abs(p / r.period[k] - round(p / r.period[k])) > 0.02 for p in picks):
                picks.append(r.period[k])
                if lab == 'normal' or len(picks) == 1:
                    print(f'star {i} {tt[idx][0]:.0f}-{tt[idx][-1]:.0f} {lab:8s} P={r.period[k]:.5f} '
                          f't0={r.transit_time[k]:.4f} dur={r.duration[k]*24:.2f}h '
                          f'depth={r.depth[k]*1e6:.0f}ppm SDE={s[k]:.1f}')
            if len(picks) >= (3 if lab == 'normal' else 1):
                break
