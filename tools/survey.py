import sys, warnings; warnings.filterwarnings('ignore')
import lightkurve as lk
for t in sys.argv[1:]:
    try:
        r = lk.search_lightcurve(f'TIC {t}', mission='TESS')
        rows = [(a, int(e), str(m).replace('TESS Sector ','')) for a, e, m in zip(r.table['author'], r.table['exptime'], r.table['mission'])]
        spoc = sorted({s for a, e, s in rows if a in ('SPOC','TESS-SPOC')})
        print(t, 'SPOC120:', sorted({s for a,e,s in rows if a=='SPOC' and e==120}, key=int), '| TESS-SPOC:', sorted({s for a,e,s in rows if a=='TESS-SPOC'}, key=int), flush=True)
    except Exception as ex:
        print(t, 'ERR', ex, flush=True)
