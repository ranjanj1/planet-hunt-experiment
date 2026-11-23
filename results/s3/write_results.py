"""Assemble results.json from fits_final.json plus verdicts."""
import json
import numpy as np
from pipeline import load

fits = json.load(open("fits_final.json"))

PC, EB, FP = "planet candidate", "eclipsing binary", "other false positive"
VERDICTS = {
    ("star_01", 0): (PC, "Box-shaped 1.7 ppt transits in 18 events; odd/even equal (0.7 sigma), no secondary; one of 3 transiting signals in this star"),
    ("star_01", 1): (PC, "1.7 ppt, 0.8 h transits in 9 events, odd/even equal, no secondary; period ~2.02x the 3.69 d signal (near 2:1 multi-planet chain)"),
    ("star_01", 2): (PC, "Shallow 0.6 ppt transits in 32 events, consistent depths, no secondary; third planet in the same system"),
    ("star_02", 0): (PC, "Ultra-short-period 4.3 ppt, 31-min transits in 152 events; odd/even equal, no secondary, short duration implies a small dense (M-dwarf) host"),
    ("star_03", 0): (PC, "Shallow 300 ppm, 3.2 h flat-bottomed transits in 11 events on a quiet bright star; odd/even equal, no secondary"),
    ("star_06", 0): (PC, "2.9 ppt, 3.8 h box-shaped transits (7 events over 7 yr) on a young spotted/flaring star; no secondary, duration matches a dwarf host"),
    ("star_06", 1): (PC, "Second 1.25 ppt, 4.3 h transit signal (4 events incl. 2 consecutive) on the same active star; no event at half period, no secondary"),
    ("star_07", 0): (PC, "0.4 ppt, 1.9 h transits seen independently in all 3 segments (28 events), odd/even equal, no secondary; found after masking instrumental 14.4 d jitter"),
    ("star_07", "sys"): (FP, "Instrumental: ~12 h stretches of chaotic +-10% flux every ~14.4 d (+-0.1 d jitter) in segment 3 only, tied to spacecraft orbit, not transits"),
    ("star_08", 0): (PC, "1.0% flat, limb-darkened transits; 500 ppm secondary + 180 ppm ellipsoidal + 130 ppm phase curve imply a ~3000 K, few-Jupiter-mass companion (massive hot Jupiter), not a star"),
    ("star_09", 0): (PC, "3 consistent 3.2 ppt, 1.6 h box-shaped transits; no dip at half period or phase 0.5"),
    ("star_10", 0): (PC, "3.8 ppt flat-bottomed transits in 11 events, odd/even equal, no secondary; one of 3 planets (durations scale as P^1/3 for one host)"),
    ("star_10", 1): (PC, "3.1 ppt transits in 6 events, odd/even equal, no secondary, no dip at half period; near 2:1 with the 5.66 d planet"),
    ("star_10", 2): (PC, "1.0 ppt transits in 17 events, consistent depths, no secondary; innermost of 3 planets"),
    ("star_11", 0): (EB, "V-shaped 13.5% primary and 8.3% secondary eclipses at phase 0.5 (BLS found half this period with 21-sigma odd/even depth difference)"),
    ("star_12", 0): (PC, "1.6 ppt, 3.5 h transits in 8 events; odd/even equal (0.8 sigma), no secondary; slightly V-shaped so grazing geometry possible"),
}

EXTRA = {"star_07": [dict(period_days=14.395, t0_days=2523.755, depth_ppm=12000.0, duration_hours=12.5,
                          key="sys")]}


def covered(star, t0, P, T14):
    d = load(star)
    return np.any(np.abs(d["time_days"] - t0) < T14 / 2)


out = {}
for i in range(1, 13):
    st = f"star_{i:02d}"
    out[st] = []
    for n, r in enumerate(fits.get(st, [])):
        P, t0, T14 = r["P"], r["t0"], r["T14"]
        if not covered(st, t0, P, T14):  # make sure t0 is an observed transit
            d = load(st)["time_days"]
            eps = np.arange(np.ceil((d.min() - t0) / P), np.floor((d.max() - t0) / P) + 1)
            cand = [t0 + e * P for e in eps if np.sum(np.abs(d - (t0 + e * P)) < T14 / 2) > 5]
            t0 = cand[0]
        v, why = VERDICTS[(st, n)]
        out[st].append(dict(period_days=round(P, 6), t0_days=round(t0, 5), depth_ppm=round(r["depth"] * 1e6, 1),
                            duration_hours=round(T14 * 24, 3), verdict=v, reason=why))
    for e in EXTRA.get(st, []):
        v, why = VERDICTS[(st, e["key"])]
        out[st].append(dict(period_days=e["period_days"], t0_days=e["t0_days"], depth_ppm=e["depth_ppm"],
                            duration_hours=e["duration_hours"], verdict=v, reason=why))

json.dump(out, open("results.json", "w"), indent=1)
for k, v in out.items():
    print(k, [(x["period_days"], x["t0_days"], x["depth_ppm"], x["duration_hours"], x["verdict"]) for x in v])
