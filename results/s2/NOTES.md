# Transit search in 12 TESS light curves — method and results

All work uses only `data/star_XX.csv` and the Python environment in `../.venv`
(numpy, scipy, astropy, pandas, matplotlib).

## Results

| star | P (d) | T0 | depth (ppm) | T14 (h) | verdict | key evidence |
|---|---|---|---|---|---|---|
| 01 | 3.69061 | 94.4691 | 1670 | 1.24 | planet candidate | SNR 44, 17 transits, odd/even 0.0σ, no secondary |
| 01 | 7.45085 | 90.6972 | 1693 | 0.77 | planet candidate | SNR 25, odd/even 0.9σ (after removing a double transit, see below) |
| 01 | 2.25312 | 93.2836 | 649 | 1.10 | planet candidate | SNR 21, 32 transits, odd/even 1.7σ |
| 02 | 0.462930 | 720.5810 | 4350 | 0.52 | planet candidate | SNR 57, 151 transits, odd/even 1.1σ, no secondary or phase curve |
| 03 | 6.26781 | 88.2519 | 301 | 3.17 | planet candidate | SNR 39, consistent in all 3 segments, secondary −7±5 ppm |
| 04 | — | | | | none | nothing above threshold |
| 05 | — | | | | none | nothing above threshold |
| 06 | 8.46318 | 724.7490 | 2715 | 3.82 | planet candidate | SNR 55, 7 transits across a 7-year baseline, on a spotted, flaring star |
| 07 | ~14.4 | 2523.78 | (24000) | ~14 | other false positive | instrumental jitter episodes in segment 3 |
| 08 | 0.941453 | 38.1160 | 10579 | 2.18 | planet candidate | flat-bottomed 1.06 % transit, 334–439 ppm secondary, 180 ppm ellipsoidal |
| 09 | 16.05065 | 63.2430 | 3316 | 1.71 | planet candidate | SNR 15, 3 transits; P/2 and P/3 ruled out |
| 10 | 5.66025 | 44.1237 | 3702 | 1.61 | planet candidate | SNR 57, odd/even 0.1σ |
| 10 | 11.37985 | 38.8178 | 3115 | 2.36 | planet candidate | SNR 33, odd/even 0.9σ |
| 10 | 3.36012 | 42.4137 | 902 | 1.35 | planet candidate | SNR 17, odd/even 1.0σ |
| 11 | 3.13377 | 41.5903 | 134000 | 1.54 | eclipsing binary | alternating 13.4 % / 10.1 % eclipses, V-shaped |
| 12 | 5.65163 | 52.0956 | 1577 | 3.43 | planet candidate | SNR 22, odd/even 0.5σ, partly V-shaped |

Depth is the depth of the fitted trapezoid's flat bottom (for V-shaped events, the
depth at the deepest point). T14 is the total duration from first to last contact. T0 is
a mid-transit time near the middle of the data, which keeps the period–T0 correlation
small. Each signal has one plot in `plots/`.

## Pipeline

1. **Detrending** (`pipeline.py`). Each contiguous chunk of data (split at gaps over 0.5 d
   and at segment boundaries) is divided by a running median 0.75 d wide. The median is
   clipped at 3σ three times over, so transits do not pull the trend down. Star 06 rotates
   quickly, has ±5 % spot modulation and frequent flares, so a median leaves large
   residuals there. For that star I used a robust local-quadratic (LOWESS-style) trend
   0.5 d wide instead, with asymmetric clipping (2.5σ up for flares, 3σ down for dips).
   In the final fits (`final.py`) the trend is recomputed with every accepted transit
   masked, so the depths are not biased low.
2. **Quality mask.** This step removes:
   - points more than 4σ above the trend (flares);
   - stretches where the local point-to-point scatter is more than 3× the chunk's
     typical value, padded by 0.2 d. This catches the pointing-jitter bursts in star 07;
   - chunk edges that are strongly offset;
   - two hand-chosen windows on star 03 (instrumental ramps at t = 95.3–97.8 and
     209.8–211.8) and one on star 07 (t = 2520–2523, an oscillating, unstable start of
     segment 3).
3. **Box Least Squares search** (astropy). Data are binned to 5 min (10 min for the
   multi-year baselines of stars 06 and 07).
   - Periods run from 0.2 d to min(baseline/2, 60 d), on a log-uniform frequency grid.
     The grid is fine enough that a 1-h transit drifts by less than 1/3 of its duration
     over the whole baseline (oversampling 1.5 for stars 06 and 07).
   - Durations run from 0.33 to 13 h. At each period, only durations ≤ 0.15 P are
     allowed, which stops very long boxes from fitting noise at short periods.
4. **Iteration.** The strongest peak is refined with a fine local BLS. Its transits are
   then masked and the search is repeated, up to 5 times or until the detection SNR
   falls below 7.
   - The SNR accounts for red noise: depth / σ_D × √N_tr, where σ_D is the robust
     scatter of out-of-transit flux binned to the transit duration.
   - The signal-detection efficiency (SDE) of the periodogram peak is also recorded.
5. **Single-event search** (`single_events.py`). Box filters of 1–12 h were slid over the
   residuals with the accepted signals masked, to look for lone transits and
   un-modelled dips.
6. **Fit** (`fit.py`). Each signal gets a trapezoid least-squares fit with free P, T0,
   depth, T14 and ingress fraction. When a star has several signals, the other signals'
   transits are masked during the fit.

## Vetting tests (`final.py`, `phasecurve.py`)

- **Odd/even depths.** Odd and even transits are fitted separately with the fitted
  trapezoid shape as a template. Errors are scaled up by the red-noise factor β. A
  difference above 3σ points to an eclipsing binary (EB) at twice the period.
- **Secondary eclipse.** I measured the depth at phase 0.5 with the same box or
  template. A secondary above 4σ points to an EB, unless its depth fits planetary
  thermal emission.
- **Shape.** The trapezoid ingress fraction runs from 0 (box-shaped) to 1 (fully V-shaped).
  - A V-shape with a large depth suggests an EB.
  - A V-shape with a small depth (under 1 %) is compatible with a grazing planet, or
    with a diluted background EB. Without pixel data I cannot tell those apart.
- **Depth.** A depth above 3 % is treated as stellar-sized (EB).
- **Phase curve.** The raw flux is fitted per chunk with a cubic, plus cos/sin of 1× and
  2× the orbital phase, with eclipses masked. The cos 2φ term is the ellipsoidal
  amplitude, which scales with the companion mass.
- **Per-transit consistency and timing**, and checks that every transit sits on real,
  unmasked data and not at a chunk edge.
- **Period aliases.** I checked P/2, P/3 and 2P directly.

## Notes per star

- **Star 01.** Three independent periodic signals: 2.253, 3.691 and 7.451 d. Their
  T0 values are unrelated, so the system looks like a compact multi-planet system.
  - At first the 7.45 d signal failed the odd/even test at 3.9σ. Fitting each transit on
    its own showed that the whole excess came from one epoch (t ≈ 98.16, depth 3400 ppm).
    The 3.69 d planet transits at the same moment there.
  - With the other signals' transits masked, odd and even depths agree to 0.9σ.
  - A fourth peak (1.049 d, 128 ppm, SNR 6.9) is below threshold and not reported.
- **Star 02.** A 0.463 d, 0.43 % deep, 31-minute transit.
  - The short duration means a small, dense host star; with an M-dwarf host the depth
    corresponds to roughly 2 Earth radii. That makes it an ultra-short-period planet
    candidate.
  - No odd/even difference, no secondary, and no ellipsoidal signal (18 ± 25 ppm).
- **Star 03.** Very bright star (146 ppm per 2-min point). A shallow 300 ppm, 3.2 h
  transit every 6.268 d. It appears consistently in all three segments.
  - The large ramps in segments 2 and 3 are instrumental and were masked.
  - A first-pass "single transit" at t = 85.95 is the ramp at the start of segment 2.
- **Stars 04 and 05.** No significant signal.
  - Star 04's best peak, at 14.02 d (SNR 8.8, SDE 4.8), comes entirely from flux drops
    at the ends of two chunks (t = 37.33 and 51.35). Odd and even depths also disagree,
    so I rejected it as an edge systematic.
  - Star 05's best peak has SNR 4.9.
- **Star 06.** A young, heavily spotted, flaring star.
  - After the local-quadratic detrend, a 0.27 % deep, 3.8 h transit repeats every
    8.46318 d. It appears in all three observing seasons, 7 years apart.
  - Individual depths vary somewhat, which is expected from spot crossings and flares.
  - The phase-0.5 flux is slightly *brighter* (−69 ppm), so there is no secondary
    eclipse.
  - A shallower dip at t ≈ 10.06 sits right before a large flare. It is a single,
    ambiguous event and is not reported.
  - Further BLS peaks (31.4 d and 18.4 d) are built on that flare complex and were
    rejected.
- **Star 07.** No transit signal. The light curve mixes 30-min, 2-min and 200-s cadence.
  - Segment 3 contains four ~0.7-d bursts of ±10 % point-to-point scatter, spaced
    14.2–14.6 d apart. Each comes with a slow, roughly 2 % baseline sag around it.
  - In the unmasked data, BLS picks these up as a deep 14.45 d "transit"
    (≈ 3.8 % box depth). The shape is chaotic rather than a transit, and the spacing is
    not strictly periodic, so I report it as an **other false positive** (instrumental:
    pointing jitter / scattered light).
  - After masking, the remaining BLS peaks (11.1, 15.5, 27.7, 30.1, 31.6, 35.6 d) all
    have SDE < 5.5. Their transits fall on segment edges or steps, or odd and even
    depths disagree, and the folds show nothing coherent. None are reported.
- **Star 08.** A 1.06 % flat-bottomed transit every 0.94145 d, with identical odd and even
  depths. Hunting for a further signal turns up the secondary eclipse at phase 0.5:
  334 ± 9 ppm with a box, 439 ± 11 ppm with a template fit.
  - **Secondary.** Secondary/primary = 0.03–0.04. For Rp/R* ≈ 0.1 that needs a dayside
    brightness temperature of about 2800 K, as expected for a strongly irradiated giant
    planet on a 0.94-d orbit.
  - **Phase curve** (raw flux): ellipsoidal cos 2φ = −180 ± 6 ppm, reflection/thermal
    cos φ = −140 ± 6 ppm, beaming 24 ± 4 ppm.
  - **Mass estimate.** For a Sun-like host, an ellipsoidal amplitude of about 180 ppm
    implies a companion of order 10 Jupiter masses. A stellar companion (≥ 0.1 M☉) would
    give more than 1000 ppm.
  - **Verdict.** All of this fits a massive hot Jupiter, so: **planet candidate**.
- **Star 09.** Three 0.33 % transits, 1.7 h long, at 16.05 d. The fourth predicted epoch
  falls in a data gap.
  - P/2 is excluded: predicted transits at epochs 1 and 9 are flat. P/3 is excluded the
    same way.
  - The shape is poorly constrained (ingress fraction 0.76 ± 0.1), and odd and even
    depths agree.
- **Star 10.** Three periodic signals: 3.360, 5.660 and 11.380 d.
  - The 5.66 and 11.38 d periods are close to a 2:1 ratio but not equal to it. Their
    transits drift past each other: the visible "double dips" are 0.18, 0.22, 0.25 …
    0.42 d apart over 4 cycles.
  - The two periods differ, so these cannot be a primary and secondary of one EB.
  - Each signal passes odd/even and secondary tests.
- **Star 11.** Faint, noisy star (1.8 % per point).
  - BLS finds 1.567 d, but alternate events have very different depths (8.0 vs 10.5 %
    in a 1-h box; 13.4 % primary vs 10.1 % secondary in fits).
  - The true orbit is 3.1338 d, with two V-shaped stellar eclipses: an
    **eclipsing binary**.
  - The secondary sits at phase 0.5, consistent with a circular orbit.
- **Star 12.** A 1580 ppm, 3.4 h transit every 5.6516 d. Odd and even depths agree
  (0.5σ), and there is no secondary (57 ± 35 ppm).
  - The shape is fairly V-like (ingress fraction 0.59): either a grazing planet or a
    diluted EB.
  - The depth is far too small for an undiluted stellar eclipse, so it stays a planet
    candidate, with that caveat.
  - A second peak (28.2 d, SNR 7.1, SDE 3.5) sits on a chunk start and was rejected.

## Other BLS peaks rejected (all are in `work/logs/`)

- **Star 02.** 15.22 d (2 events, SNR 7.0, SDE 4.0) and 3.43 d (SNR 5.8): noise.
- **Star 08.** Iterations 3–4 recover the same 0.94145 d period, with box centres
  2.6 h from the transit. These are leftovers from the detrending around the deep
  transit, not new signals.
- **Star 09.** 9.98 d (SNR 6.2).
- **Star 10.** 27.1 d and 33.8 d (SNR 7.5 and 5.7). Neither has full transits on good data.
- **Star 11.** 34.8 d is one ramp at a chunk start. 0.633 d and 3.195 d (SNR 6–7) are
  0.25-h noise spikes in a 1.8 %-noise light curve.
- **Star 12.** 22.7 d and 28.2 d (SNR 6.8 and 7.1, SDE 3.3–3.5) each rest on 2 events,
  one at a chunk start.

## Caveats

- **What "planet candidate" means here.** Only photometry is available: no pixel or
  centroid data and no stellar parameters. So "planet candidate" means *passes every
  test possible with the light curve alone*. A blended background EB cannot be ruled out
  for any candidate.
- **Effect of detrending on depths.** A 0.75 d median window slightly reduces the depth
  of long transits; masking transits before the final detrend largely fixes this. For
  star 06 the depth still depends on the spot model at the 10–20 % level.
- **Star 08's secondary.** Its depth depends on the method (box vs template; the
  detrending window is close to the orbital period). I give 334–439 ppm as the range.
- **Star 07's depth and duration** describe a noise burst, not an eclipse. They are
  rough numbers.

## Reproduce

```
../.venv/bin/python pipeline.py star_01 ... star_12   # detrend + iterative BLS -> work/
../.venv/bin/python final.py                          # vetted list -> fits, plots/, results.json
../.venv/bin/python phasecurve.py star_08 0.941452 38.11596 0.0908   # phase-curve harmonics
../.venv/bin/python single_events.py star_06 '[[8.46318,724.74862,0.12]]'  # single-event search
```
