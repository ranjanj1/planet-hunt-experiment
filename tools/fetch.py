"""Download the chosen TESS light curves, strip every identifier, write anonymous CSVs.
Writes: task/data/star_XX.csv (what Claude sees) and key/stars.json (hidden: which star is which, time offsets)."""
import json, random, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, lightkurve as lk
ROOT = Path(__file__).resolve().parents[1]
CHOICE = {  # TIC: (common name, [(sector, author)])
 '100100827': ('WASP-18', [(2,'SPOC'),(3,'SPOC'),(29,'SPOC')]),
 '410153553': ('LHS 3844', [(1,'SPOC'),(27,'SPOC'),(28,'SPOC')]),
 '307210830': ('L 98-59', [(2,'SPOC'),(5,'SPOC'),(8,'SPOC')]),
 '259377017': ('TOI-270', [(3,'SPOC'),(4,'SPOC'),(5,'SPOC')]),
 '261136679': ('Pi Mensae', [(1,'SPOC'),(4,'SPOC'),(8,'SPOC')]),
 '441420236': ('AU Microscopii', [(1,'SPOC'),(27,'SPOC'),(95,'SPOC')]),
 '150428135': ('TOI-700', [(1,'SPOC'),(3,'SPOC'),(4,'SPOC')]),
 '4206066':   ('TIC 4206066 (TOI-7930)', [(6,'TESS-SPOC'),(32,'SPOC'),(98,'QLP')]),
 '189639080': ('TIC 189639080 (eclipsing binary, TESS EB catalog)', [(8,'SPOC'),(9,'SPOC'),(35,'SPOC')]),
 '304021498': ('TOI-1031 (NASA: false positive)', [(36,'SPOC'),(38,'SPOC')]),
 '259376992': ('TIC 259376992 (no known signal)', [(3,'SPOC'),(4,'SPOC'),(5,'SPOC')]),
 '259378043': ('TIC 259378043 (no known signal)', [(3,'SPOC'),(5,'SPOC'),(30,'SPOC')]),
}
rng = random.Random(20261122)
tics = list(CHOICE); rng.shuffle(tics)
out = ROOT / 'task/data'; out.mkdir(parents=True, exist_ok=True)
(ROOT / 'key/lc').mkdir(parents=True, exist_ok=True)
key = {}
for i, tic in enumerate(tics, 1):
    sid = f'star_{i:02d}'; name, secs = CHOICE[tic]; parts = []
    for sec, author in secs:
        r = lk.search_lightcurve(f'TIC {tic}', mission='TESS', sector=sec, author=author)
        if author == 'SPOC': r = r[r.exptime.value == 120]
        lc = r[0].download(download_dir=str(ROOT / 'key/lc'), quality_bitmask='default')
        cad = int(round(float(np.nanmedian(np.diff(lc.time.value))) * 86400))
        lc = lc.remove_nans().normalize()
        t, f, e = lc.time.value, lc.flux.value, lc.flux_err.value
        ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e)
        parts.append((t[ok], f[ok], e[ok], np.full(ok.sum(), cad), sec, author))
        print(sid, tic, sec, author, ok.sum(), 'pts', cad, 's', flush=True)
    t0 = float(np.floor(min(p[0].min() for p in parts)))
    rows = ['time_days,flux,flux_err,segment,cadence_s']
    for k, (t, f, e, c, sec, author) in enumerate(parts, 1):
        rows += [f'{a - t0:.6f},{b:.7f},{d:.7f},{k},{cc}' for a, b, d, cc in zip(t, f, e, c)]
    (out / f'{sid}.csv').write_text('\n'.join(rows) + '\n')
    key[sid] = {'tic': tic, 'name': name, 'sectors': [[s, a] for s, a in secs], 'time_offset_btjd': t0,
                'segments': [{'segment': k, 'sector': s, 'author': a, 'n': int(len(p[0])), 'cadence_s': int(p[3][0]),
                              'start': float(p[0].min() - t0), 'end': float(p[0].max() - t0)} for k, (p, (s, a)) in enumerate(zip(parts, secs), 1)]}
(ROOT / 'key/stars.json').write_text(json.dumps(key, indent=1))
print('done', len(key))
