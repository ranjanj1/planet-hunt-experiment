"""Hidden answer key: every known signal of each star (NASA Exoplanet Archive TOI table + confirmed planets,
TESS EB catalog), with how many of its transits fall inside the data Claude was given. Writes key/key.json."""
import csv, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
stars = json.loads((ROOT / 'key/stars.json').read_text())
toi = list(csv.DictReader(open(ROOT / 'key/raw/toi.csv')))
ps = list(csv.DictReader(open(ROOT / 'key/raw/ps.csv')))
EB = {'189639080': {'period': 3.1337762, 'epoch_btjd': 1517.285695, 'src': 'TESS EB catalog (Prsa+ 2022, J/ApJS/258/16)'}}
key = {}
for sid, s in sorted(stars.items()):
    tic, off = s['tic'], s['time_offset_btjd']
    d = np.loadtxt(ROOT / f'task/data/{sid}.csv', delimiter=',', skiprows=1)
    t = d[:, 0]; cad = np.median(np.diff(t))
    sig = []
    for r in toi:
        if r['tid'] != tic: continue
        sig.append({'id': 'TOI-' + r['toi'], 'period': float(r['pl_orbper']), 'epoch': float(r['pl_tranmid']) - 2457000 - off,
                    'depth_ppm': float(r['pl_trandep'] or 0), 'dur_h': float(r['pl_trandurh'] or 0), 'radius_re': float(r['pl_rade'] or 0) if r['pl_rade'] else None,
                    'disp': r['tfopwg_disp'], 'src': 'NASA Exoplanet Archive TOI table'})
    for r in ps:
        if r['tic_id'] != f'TIC {tic}' or not r['pl_trandep']: continue
        P = float(r['pl_orbper'])
        m = [x for x in sig if abs(x['period'] - P) / P < 0.01]
        if m:   # confirmed planet: its archive ephemeris is the better one (the TOI row can be old: AU Mic b's TOI period drifts 3 days over 7 years)
            m[0]['name'] = r['pl_name']
            if r['pl_tranmid']: m[0].update(period=P, epoch=float(r['pl_tranmid']) - 2457000 - off, ephemeris='confirmed planets table')
            continue
        sig.append({'id': r['pl_name'], 'name': r['pl_name'], 'period': P, 'epoch': float(r['pl_tranmid']) - 2457000 - off,
                    'depth_ppm': float(r['pl_trandep']) * 1e4, 'dur_h': float(r['pl_trandur'] or 0), 'radius_re': float(r['pl_rade'] or 0),
                    'disp': 'CP', 'src': 'NASA Exoplanet Archive confirmed planets'})
    if tic in EB:
        e = EB[tic]; sig.append({'id': 'EB', 'period': e['period'], 'epoch': e['epoch_btjd'] - off, 'depth_ppm': None, 'dur_h': 3.0, 'disp': 'EB', 'src': e['src']})
    for x in sig:
        P, E, half = x['period'], x['epoch'], max(x['dur_h'], 1.0) / 48
        n0, n1 = int(np.floor((t.min() - E) / P)) - 1, int(np.ceil((t.max() - E) / P)) + 1
        cov = 0
        for n in range(n0, n1 + 1):
            tc = E + n * P
            got = np.sum(np.abs(t - tc) < half); want = 2 * half / cad
            cov += got >= 0.5 * want
        x['transits_in_data'] = int(cov); x['findable'] = bool(cov >= 2)
    key[sid] = {'tic': tic, 'name': s['name'], 'signals': sig}
    print(sid, s['name'], [(x['id'], round(x['period'], 3), x['disp'], x['transits_in_data']) for x in sig])
(ROOT / 'key/key.json').write_text(json.dumps(key, indent=1))
