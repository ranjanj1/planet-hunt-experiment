"""Raw + detrended flux around given times: python zoom.py STAR halfwidth_days t1 t2 ... [--out file]"""
import sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pipeline import load, bin_lc
i, hw = int(sys.argv[1]), float(sys.argv[2])
out = None
args = sys.argv[3:]
if '--out' in args:
    out = args[args.index('--out') + 1]; args = args[:args.index('--out')]
ts = [float(x) for x in args]
d = load(i); t, f = d['time_days'], d['flux']
z = np.load(f'work/star_{i:02d}_flat.npz')
fig, ax = plt.subplots(len(ts), 1, figsize=(12, 2.6 * len(ts)), squeeze=False)
for a, tc in zip(ax[:, 0], ts):
    m = np.abs(t - tc) < hw
    if m.sum() == 0:
        a.set_title(f'{tc}: no data'); continue
    a.plot(t[m], f[m], ',k')
    tb, fb, _ = bin_lc(t[m], f[m], 20 / 1440); a.plot(tb, fb, 'r.-', ms=3)
    a.axvline(tc, color='b', alpha=.3)
    lo, hi = np.percentile(f[m], [0.5, 99.5]); a.set_ylim(lo - (hi - lo) * .2, hi + (hi - lo) * .1)
    a.set_title(f't={tc}')
fig.tight_layout(); fig.savefig(out or f'work/zoom_{i:02d}.png', dpi=60)
