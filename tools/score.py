"""Score a results.json against the hidden key. usage: score.py results/s1/results.json [label]
A reported signal matches a known one when its period is within 1% (or is exactly 2x / 0.5x: an alias, counted separately)
and its transit time lines up (within 3 hours after folding). Writes <results>.score.json next to it."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
key = json.loads((ROOT / 'key/key.json').read_text())
res = json.loads(Path(sys.argv[1]).read_text())
def phase_ok(P, e1, e2, tol_d):
    dphi = ((e1 - e2) / P) % 1.0
    return min(dphi, 1 - dphi) * P < tol_d
out = {'stars': {}, 'totals': {}}
T = dict(found=0, findable=0, alias=0, false_alarms=0, eb_caught=0, eb_total=0, fp_called_planet=0, extra=0)
for sid, k in sorted(key.items()):
    rep = res.get(sid, []) or []
    known = [x for x in k['signals']]
    used, rows = set(), []
    for r in rep:
        P = float(r.get('period_days') or 0); e = float(r.get('t0_days') or 0); v = (r.get('verdict') or '').lower()
        hit, kind = None, None
        # exact matches first across every known signal, then aliases (a 7.45 d planet is not "2 x 3.69 d")
        for mult, name, tol in ((1, 'match', 0.01), (2, 'alias', 0.003), (0.5, 'alias', 0.003), (3, 'alias', 0.003), (1/3, 'alias', 0.003)):
          for i, x in enumerate(known):
                Pk = x['period'] * mult
                if P and abs(P - Pk) / Pk < tol and (phase_ok(x['period'], e, x['epoch'], 0.125) or (x['disp'] == 'EB' and phase_ok(x['period'], e, x['epoch'] + x['period'] / 2, 0.125))):
                    hit, kind = i, name; break
          if hit is not None: break
        rows.append({'reported_period': P, 'verdict': v, 'match': known[hit]['id'] if hit is not None else None, 'kind': kind,
                     'known_disp': known[hit]['disp'] if hit is not None else None})
        if hit is not None: used.add((hit, kind))
    st = {'name': k['name'], 'reported': rows, 'known': []}
    for i, x in enumerate(known):
        m = [kind for (j, kind) in used if j == i]
        got = 'match' in m; al = (not got) and 'alias' in m
        verdicts = [r['verdict'] for r in rows if r['match'] == x['id']]
        st['known'].append({'id': x['id'], 'name': x.get('name'), 'period': x['period'], 'disp': x['disp'], 'transits_in_data': x['transits_in_data'],
                            'findable': x['findable'], 'found': got, 'alias_only': al, 'verdicts': verdicts})
        if x['disp'] in ('CP', 'KP', 'PC') and x['findable']:
            T['findable'] += 1; T['found'] += got; T['alias'] += al
        if x['disp'] == 'EB':
            T['eb_total'] += 1; T['eb_caught'] += any('binary' in v or 'false' in v for v in verdicts)
        if x['disp'] == 'FP':
            T['fp_called_planet'] += any('planet' in v for v in verdicts)
    for r in rows:
        if r['match'] is None and 'planet' in r['verdict']: T['false_alarms'] += 1
        if r['match'] is None: T['extra'] += 1
    out['stars'][sid] = st
out['totals'] = T
Path(sys.argv[1]).with_suffix('.score.json').write_text(json.dumps(out, indent=1))
print(json.dumps(T))
for sid, st in out['stars'].items():
    print(sid, st['name'][:28].ljust(28), ' '.join(f"{x['id']}:{'Y' if x['found'] else ('a' if x['alias_only'] else ('-' if x['findable'] else '.'))}" for x in st['known']),
          '| extra:', [round(r['reported_period'], 3) for r in st['reported'] if r['match'] is None])
