"""Quick-look diagnostic for one signal: python inspect_sig.py STAR P T0 DUR_H [out.png]
Panels: phase fold, odd/even, phase 0.5 (secondary), per-transit depths."""
import sys
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline import fold, bin_lc

i, P, t0, dur = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]) / 24
out = sys.argv[5] if len(sys.argv) > 5 else f'work/insp_{i:02d}_{P:.3f}.png'
z = np.load(f'work/star_{i:02d}_flat.npz')
t, f = z['t'][z['good']], z['f'][z['good']]
ph = fold(t, P, t0)
ep = np.round((t - t0) / P).astype(int)
intr = np.abs(ph) < dur / 2
oot = (np.abs(ph) > dur) & (np.abs(ph) < 3 * dur)


def depth(m):
    a, b = m & intr, m & oot
    if a.sum() < 3 or b.sum() < 3:
        return np.nan, np.nan
    return (np.median(f[b]) - np.mean(f[a])) * 1e6, np.std(f[b]) / np.sqrt(a.sum()) * 1e6


all_d = depth(np.ones_like(t, bool))
odd = depth(ep % 2 == 1)
even = depth(ep % 2 == 0)
ph2 = fold(t, P, t0 + P / 2)
sec_in = np.abs(ph2) < dur / 2
sec_oot = (np.abs(ph2) > dur) & (np.abs(ph2) < 3 * dur)
sec = ((np.median(f[sec_oot]) - np.mean(f[sec_in])) * 1e6,
       np.std(f[sec_oot]) / np.sqrt(max(sec_in.sum(), 1)) * 1e6)
print(f'star {i} P={P} depth={all_d[0]:.0f}+-{all_d[1]:.0f} odd={odd[0]:.0f}+-{odd[1]:.0f} '
      f'even={even[0]:.0f}+-{even[1]:.0f} sec={sec[0]:.0f}+-{sec[1]:.0f}')
eps = np.unique(ep[intr])
per = [(e, *depth(ep == e)) for e in eps]
print('n transits with data:', len(eps))

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
w = min(4 * dur, P / 2)
for a, x, lab in [(ax[0, 0], ph, 'primary'), (ax[1, 0], ph2, 'phase 0.5')]:
    m = np.abs(x) < w
    a.plot(x[m] * 24, f[m], ',', color='0.6')
    tb, fb, _ = bin_lc(x[m] * 24, f[m], max(dur * 24 / 10, 0.1))
    a.plot(tb, fb, 'r.')
    a.set_title(lab); a.set_xlabel('hours')
a = ax[0, 1]
for par, c in [(1, 'b'), (0, 'g')]:
    m = (np.abs(ph) < w) & (ep % 2 == par)
    if m.sum() < 5:
        continue
    tb, fb, _ = bin_lc(ph[m] * 24, f[m], max(dur * 24 / 10, 0.1))
    a.plot(tb, fb, '.', color=c, label=['even', 'odd'][par])
a.legend(); a.set_title(f'odd {odd[0]:.0f}  even {even[0]:.0f} ppm')
a = ax[1, 1]
per = np.array(per)
a.errorbar(per[:, 0], per[:, 1], per[:, 2], fmt='o')
a.axhline(all_d[0], color='r')
a.set_title('per-transit depth (ppm) / O-C (min, orange)')
# per-epoch timing: box model of fixed width and depth scanned in time
oc = []
for e in eps:
    m = (ep == e) & (np.abs(ph) < 2.5 * dur)
    if m.sum() < 10:
        continue
    shifts = np.linspace(-dur, dur, 201)
    chi = [np.sum((f[m] - np.where(np.abs(ph[m] - s) < dur / 2, 1 - all_d[0] / 1e6, 1))**2)
           for s in shifts]
    oc.append((e, shifts[np.argmin(chi)] * 1440))
oc = np.array(oc)
if len(oc):
    a2 = a.twinx(); a2.plot(oc[:, 0], oc[:, 1], 's', color='orange')
    print('O-C (min):', ' '.join(f'{int(e)}:{o:.0f}' for e, o in oc))
fig.suptitle(f'star {i} P={P:.5f} t0={t0:.4f} dur={dur*24:.2f}h  sec={sec[0]:.0f}+-{sec[1]:.0f}')
fig.tight_layout(); fig.savefig(out, dpi=70)
