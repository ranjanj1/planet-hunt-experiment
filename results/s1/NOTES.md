# TESS transit search: method and results

## Summary

| star | period (d) | depth (ppm) | dur (h) | verdict | key evidence |
|---|---|---|---|---|---|
| 01 | 1.04927 | 164 | 1.6 | planet candidate (low confidence) | SNR ~5.5–8, seen in all 3 segments |
| 01 | 2.25316 | 653 | 1.17 | planet candidate | 30 transits, no secondary |
| 01 | 3.69062 | 1716 | 1.26 | planet candidate | 16 transits, odd = even |
| 01 | 7.45086 | 1813 | 0.85 | planet candidate | near 2:1 with 3.69 d |
| 02 | 0.46293 | 4349 | 0.53 | planet candidate | ultra-short period, 152 transits |
| 03 | 6.26779 | 295 | 3.20 | planet candidate | very quiet star, SNR 35 |
| 04 | – | – | – | nothing significant | (see "Rejected") |
| 05 | – | – | – | nothing significant | (see "Rejected") |
| 06 | 8.46318 | 2688 | 3.77 | planet candidate | young, spotted, flaring host |
| 07 | 3.18279 | 430 | 1.80 | planet candidate | found independently in each segment |
| 07 | ~14.4 | (~1%) | ~18 | other false positive | instrumental bursts in segment 3 |
| 08 | 0.94145 | 10430 | 2.18 | planet candidate | hot Jupiter: secondary eclipse plus phase curve of planetary mass |
| 09 | 16.05065 | 3244 | 1.71 | planet candidate | 3 transits, aliases excluded |
| 10 | 3.36017 | 950 | 1.38 | planet candidate | 3-planet system |
| 10 | 5.66024 | 3756 | 1.61 | planet candidate | |
| 10 | 11.37988 | 3045 | 2.43 | planet candidate | |
| 11 | 3.13377 | 135000 | 1.54 | eclipsing binary | alternating 13.5% / 10.3% eclipses |
| 12 | 5.65159 | 1587 | 3.41 | planet candidate | V-ish shape; weakest-vetted |

The full numbers are in `results.json`. The per-signal fit and vetting values (errors, odd/even,
secondary, SNR) are in `work/final_table.json`. There is one plot per signal in `plots/`.

## Pipeline (code in this folder)

| file | role |
|---|---|
| `pipeline.py` | loading, per-star cleaning config, detrending, BLS, trapezoid fit, odd/even helpers |
| `search.py` | pass 1: detrend, then iterative BLS with masking |
| `search2.py` | pass 2: mask accepted signals, trim chunk edges, BLS each cluster on normal **and inverted** flux |
| `events.py` | single-event box scan (finds individual dips, useful for long periods and for spotting artefacts) |
| `inspect_sig.py`, `zoom.py` | quick-look vetting plots (output in `work/`) |
| `final.py` | final joint fits, vetting numbers, the plots in `plots/`, and `results.json` |

Run order: `python search.py`, `python search2.py N`, `python final.py`, using `../.venv/bin/python`.

1. **Inspect raw data.** I plotted every segment by eye first. This revealed three problems:
   - instrumental ramps and flux steps in star 3;
   - ±10% bursts in star 7's segment 3;
   - flares and 5% rotational modulation in star 6.
2. **Clean the data** (`CONFIG` in `pipeline.py`):
   - **Star 3:** dropped two ramps (t = 96.0–97.6 and 209.9–210.9) and restarted the detrending at two flux steps (105.6 and 108.57).
   - **Star 7:** dropped the four burst windows in segment 3.
   - **Stars 4 and 5:** trimmed 0.2 d at each chunk edge, because edge ramps were producing spurious BLS peaks.
3. **Detrend.**
   - Default: a 1-day sliding median, restarted at every gap longer than 0.5 d and at every segment boundary. Upward outliers above 3σ (flares) are clipped.
   - Star 6: an iteratively clipped cubic spline with 0.4 d knots, which follows the fast rotational modulation. Points just after big flares are also removed.
   - Final fits: I re-detrend with all known transits masked, so the trend does not eat into transit depth.
4. **Search** (astropy BLS, SNR objective, 10-min bins).
   - Periods: 0.35 d up to the smaller of 40 d and half the baseline.
   - Durations: 0.7–8 h.
   - Frequency grid: sized so the timing drift across the baseline stays below about a third of the shortest duration.
   - Long-gap stars: for stars whose segments are about 700–2500 d apart, I searched the densest group of segments first, then extended the ephemeris to all the data with a narrow, fine period grid. I checked that the cycle count across each gap is unambiguous.
   - Iterations: after each detection the signal is masked and BLS is run again.
   - False-alarm level: in pass 2 the same BLS is run on the inverted light curve, which gives each star's noise ceiling for SDE.
5. **Fit.** I fitted a trapezoid model (P, t0, depth, T14, ingress time) by least squares. Each signal is fitted with the other signals of the same star masked. Depths quoted are the fixed-shape least-squares depths, with errors inflated by a red-noise factor β measured from 30-minute binned residuals. The reported t0 is the earliest transit that actually has data.
6. **Vet each signal.**
   - Odd vs even depth: fixed shape, and errors inflated when transit-to-transit scatter exceeds the formal errors.
   - Secondary eclipse at phase 0.5.
   - Per-transit depths and transit times (O–C).
   - Consistency between segments.
   - Transit shape: U vs V, via the ingress/total-duration ratio.
   - Duration vs period: is the implied stellar density plausible for a main-sequence star?
   - Whether individual events sit on data edges or ramps.
   - For multi-transit long-period signals: whether data at the epochs of P/2, P/3 and so on rule those aliases out.

## Notes on individual stars

- **Star 1.** Four periodic signals; the 2.25, 3.69 and 7.45 d ones are unambiguous. Short durations (0.85–1.3 h) mean a dense, M-dwarf-like host. 7.45/3.69 = 2.019, which is near 2:1 resonance, as is typical of compact multi-planet systems.
  - The 1.049 d signal (164 ± 22 ppm) is the weakest entry in the list:
    - it was found in both search passes (SDE 9.0, against 7.3 for the inverted-flux noise ceiling);
    - it appears in segments 1 and 2 at 4.6–4.8σ each, but only at about 2.6σ in segment 3;
    - its odd/even depths agree and there is no secondary;
    - its 1.6 h duration is consistent with the same small host star.
  - I list it as a planet candidate, with low confidence.
- **Star 2.** P = 0.4629 d, a 0.53 h box-shaped transit. I tested 2P = 0.926 d explicitly: the alternate events have equal depth (4302 ± 112 vs 4417 ± 109 ppm), so this is not an EB at twice the period. It is an ultra-short-period candidate around a small star.
- **Star 3.** A 295 ppm, 3.2 h transit every 6.268 d on a very quiet star (89 ppm per 2-minute point). The strongest raw BLS peaks (27–31 d) were the instrumental ramps; after cleaning, the 6.27 d signal dominates (SDE 15). A weaker 31.4 d peak traced back to a segment start, a ramp edge and a gappy noisy patch, so I rejected it.
- **Star 6.** A young active star: about 5% spot modulation over about 4.8 d, plus many flares. With spline detrending, single-event detection found 7 dips at 8–13σ. All lie on one 8.463 d ephemeris across three segments spanning about 7 years. Per-transit depths scatter more than formal errors (spot crossings and flares), which is expected on such a star. No other significant periodicity remained.
- **Star 7.**
  - Segment 3 contains four episodes of chaotic ±10% flux scatter, each about 18 h long, every ~14.4 d. They sit on slow 1–3% dips.
  - Why instrumental: they are not eclipse-shaped and recur irregularly (O–C up to about 5 h, while a real eclipse would be strictly periodic). Nothing similar appears in segments 1 and 2.
  - Pass-1 BLS flags them at SDE 14, so I report them as **other false positive (instrumental)**.
  - With them masked, the second pass found a 3.18 d, 430 ppm signal in segment 1 (30-minute cadence). A narrow period search in segment 2 independently recovered P = 3.1835 at the same depth. Separate transit-time fits in each segment link with 222 and 575 cycles to give P = 3.18278 d. Segment 3 then also shows the dip (about 4.6σ).
  - Odd/even depths agree and there is no secondary: **planet candidate**, though shallow.
- **Star 8.** The 1.04% flat-bottomed transit comes with a 459 ± 14 ppm secondary eclipse, which could suggest an EB. However, the out-of-eclipse phase curve has:
  - an ellipsoidal amplitude of about 172 ppm;
  - a reflection/thermal term of about 136 ppm, peaking at phase 0.5;
  - Doppler beaming of about 26 ppm.

  The transit-derived a/R* ≈ 3.3 converts these amplitudes into a companion mass of about 5–10 Jupiter masses (ellipsoidal: q ≈ 0.004; beaming: K ≈ 2 km/s). Even the lowest-mass star (about 80 Jupiter masses) would produce an ellipsoidal signal of thousands of ppm. The secondary depth matches a dayside at about 3000 K, as expected for an irradiated planet with a/R* of about 3. Odd and even depths are identical. Verdict: **planet candidate** (a massive hot Jupiter). Radial velocities are needed to confirm the mass.
- **Star 9.** Three transits (t ≈ 15.09, 63.24, 79.29) give P = 16.0507 d. Data exist at the extra epochs that P/2 … P/5 would predict, and show no transit there, so 16.05 d is the true period. Odd/even agree and there is no secondary.
- **Star 10.** Transits arrive in close pairs that slowly drift apart. They are two planets near 2:1 resonance (5.660 d and 11.380 d, ratio 2.0105), plus a third at 3.360 d. All three have constant transit times, matched odd/even depths and no secondaries.
- **Star 11.** BLS first locks on 1.567 d, but the odd and even events differ hugely (13.5% vs 10.3%, more than 20σ). The true period is 3.1338 d, with a secondary eclipse exactly at phase 0.5 and the same 1.5 h width. That means two similar, small stars: **eclipsing binary**. The `depth_ppm` and `t0` in results.json refer to the deeper (primary) eclipse.
- **Star 12.** Eight transits at 5.6516 d, 1590 ppm, 3.4 h. The formal odd/even difference (3σ with white-noise errors) drops to 1.0σ once epoch-to-epoch scatter is included, because one event sits on a segment-start ramp. There is no secondary (43 ± 58 ppm). The shape is fairly V-like (ingress ≈ 30% of the duration), so a grazing or blended EB cannot be excluded. I keep it as a planet candidate but consider it the least secure of the deep signals.

## Rejected detections (not in results.json)

- **Star 4.**
  - A 14.0 d BLS peak (SDE 9) came from three "events" that are downward ramps at the ends of data chunks (t ≈ 37.3, 51.3), so it is instrumental.
  - With edges trimmed, the best remaining peak is P = 4.192 d, about 350 ppm, 0.7 h. It is about 6σ in a random-phase test and has the same depth in both halves of the data. However, its BLS SDE (5.6–6.5) barely exceeds the inverted-flux noise ceiling (5.1), so it falls below my detection threshold. It is worth following up, but I don't claim it.
- **Star 5.** A 13.4 d peak (SDE 7.7) came only from edge ramps (t ≈ 78.7) and gaps. After edge trimming, nothing exceeds the noise ceiling.
- **Star 3, 31.4 d; star 9, single dip at 22.51 d; star 12, dips at 0.87 and 13.89 d.** All of these coincide with segment starts, ramps or gappy noisy stretches.
- **Star 8, second-pass residual at P ≈ 0.94 d with a long duration.** This is the phase curve leaking through the detrending, not a new signal.

## Caveats

- **No stellar parameters.** Densities and host types are inferred only from transit shapes, which are degenerate with impact parameter. I did not convert depths into planet radii.
- **Dilution.** Depths may be diluted by nearby stars; no pixel-level or centroid data were available. This is the main unexplored false-positive route for every candidate, especially stars 12 and 8.
- **Weakest candidates.** Star 1 at 1.05 d and star 7 at 3.18 d are near the detection limit.
- **Trapezoid shape.** The trapezoid is a simple stand-in for a limb-darkened transit model. Depths are the flat-bottom depth of the trapezoid, and duration is first to fourth contact (T14).
