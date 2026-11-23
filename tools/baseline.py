"""The control: the tutorial recipe with no judgement. Each observing segment on its own (as the lightkurve tutorials do one
sector): flatten (~0.75-day window), clip upward outliers, Box Least Squares on 60,000 periods from 0.5 to 15 days, keep the
strongest peak, mask it, search again (3 rounds). Peaks with SDE >= 9 are kept; peaks from different segments within 1% in
period are merged. Every kept peak is called a "planet candidate": nothing is vetted. Writes results/baseline/results.json."""
import json, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, lightkurve as lk
from astropy.timeseries import BoxLeastSquares
ROOT = Path(__file__).resolve().parents[1]
P = 1 / np.linspace(1 / 15, 1 / 0.5, 60000); DUR = [0.04, 0.06, 0.08, 0.12, 0.16]
out = {}
for f in sorted((ROOT / 'task/data').glob('star_*.csv')):
    d = np.loadtxt(f, delimiter=',', skiprows=1); found = []
    for seg in np.unique(d[:, 3]):
        s = d[d[:, 3] == seg]
        lc = lk.LightCurve(time=s[:, 0], flux=s[:, 1]).flatten(window_length=max(5, int(0.75 * 86400 / s[0, 4]) | 1)).remove_outliers(sigma_upper=4, sigma_lower=20)
        t, y = lc.time.value, lc.flux.value; mask = np.ones(len(t), bool)
        for rnd in range(3):
            bls = BoxLeastSquares(t[mask], y[mask]); r = bls.power(P, DUR, objective='snr')
            sde = (r.power.max() - r.power.mean()) / r.power.std(); i = int(np.argmax(r.power))
            if sde < 9: break
            found.append(dict(period_days=float(r.period[i]), t0_days=float(r.transit_time[i]), depth_ppm=float(r.depth[i] * 1e6),
                              duration_hours=float(r.duration[i] * 24), sde=float(sde), segment=int(seg)))
            mask &= ~bls.transit_mask(t, r.period[i], 2 * r.duration[i], r.transit_time[i])
    merged = []
    for x in sorted(found, key=lambda x: -x['sde']):
        if not any(abs(x['period_days'] - m['period_days']) / m['period_days'] < 0.01 for m in merged): merged.append(x)
    out[f.stem] = [{**{k: m[k] for k in ('period_days', 't0_days', 'depth_ppm', 'duration_hours')}, 'verdict': 'planet candidate',
                    'reason': f"BLS peak SDE {m['sde']:.1f} in segment {m['segment']}"} for m in merged]
    print(f.stem, [(round(m['period_days'], 4), round(m['sde'], 1)) for m in merged], flush=True)
(ROOT / 'results/baseline').mkdir(parents=True, exist_ok=True)
(ROOT / 'results/baseline/results.json').write_text(json.dumps(out, indent=1))
