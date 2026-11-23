# Transit search in 12 TESS light curves: method and notes

## Summary

| star | period [d] | depth [ppm] | T14 [h] | verdict | plot |
|---|---|---|---|---|---|
| star_01 | 3.690614 | 1712 | 1.25 | planet candidate | plots/star_01_1.png |
| star_01 | 7.450867 | 1712 | 0.83 | planet candidate | plots/star_01_2.png |
| star_01 | 2.253129 | 603 | 1.02 | planet candidate | plots/star_01_3.png |
| star_02 | 0.462929 | 4341 | 0.52 | planet candidate | plots/star_02_1.png |
| star_03 | 6.267809 | 299 | 3.17 | planet candidate | plots/star_03_1.png |
| star_04 | none | | | | |
| star_05 | none | | | | |
| star_06 | 8.463179 | 2864 | 3.76 | planet candidate | plots/star_06_1.png |
| star_06 | 18.859026 | 1252 | 4.26 | planet candidate | plots/star_06_2.png |
| star_07 | 3.182784 | 409 | 1.94 | planet candidate | plots/star_07_1.png |
| star_07 | ~14.4 | (artefact) | ~12 | other false positive | plots/star_07_2_instrumental.png |
| star_08 | 0.941452 | 10274 | 2.16 | planet candidate (massive hot Jupiter) | plots/star_08_1.png |
| star_09 | 16.050975 | 3229 | 1.63 | planet candidate | plots/star_09_1.png |
| star_10 | 5.660238 | 3782 | 1.61 | planet candidate | plots/star_10_1.png |
| star_10 | 11.379854 | 3094 | 2.43 | planet candidate | plots/star_10_2.png |
| star_10 | 3.360119 | 981 | 1.35 | planet candidate | plots/star_10_3.png |
| star_11 | 3.133773 | 135162 (primary) | 1.55 | eclipsing binary | plots/star_11_1.png |
| star_12 | 5.651764 | 1601 | 3.49 | planet candidate | plots/star_12_1.png |

`t0_days` in results.json is the fitted mid-time of an observed transit, on the same time axis as the data.
`depth_ppm` is the depth of the fitted trapezoid. For limb-darkened transits the true central depth is slightly
larger: for star_08, for example, the central depth is ~10,800 ppm against a trapezoid depth of 10,274 ppm.

## Pipeline (code in this folder)

1. **Load and inspect** (`pipeline.load`). I plotted every raw light curve per segment first. Doing this caught
   the problem cases early: star_06 has ~4% rotational modulation and many flares; star_07 segment 3 has
   ~12 h stretches of chaotic ±10% flux about every 14.4 d; star_03 has ramps at 96–97.5 d and 210 d.
2. **Detrending** (`pipeline.detrend`). This is a time-windowed robust Tukey-biweight location filter, run separately on
   each continuous chunk (data split at gaps > 0.5 d). The window is 0.75 d, or 0.4 d for the fast-rotating star_06.
   For the final products, transits already found are masked when the trend is computed.
   Upward outliers (> 4σ, mostly flares) are removed. So are 0.25 d stretches whose scatter is > 2.5× the
   median scatter (instrumental artefacts).
3. **Period search** (`search.py`). Box Least Squares (astropy) on 10-min-binned data, with durations 0.5–9.6 h.
   The frequency grid is chosen so that a grid step moves the last transit by less than 1/3 of the shortest duration.
   Some stars have segments years apart (baselines up to 2,580 d). For those, the search runs on the longest group
   of nearby segments, and the period is then refined with a fine grid over all data, which resolves the cycle count.
   The search is iterative: find the strongest peak, mask its transits, search again.
   (A first version only allowed durations ≤ 1.6 h because of how astropy constrains duration against period. I fixed
   this by splitting the period range so long durations are allowed at longer periods, then reran everything.)
4. **Residual search** (`search2.py`). Detrend again with known transits masked, cut 0.1 d at every chunk edge
   (segment-start ramps), and rerun BLS. A box matched filter for single transits was also run on every star.
   Every single event it flagged turned out to be a ramp, a step, or the edge of a gap.
5. **Single-event + multi-event search for very active stars** (`ses.py`). On star_06, the sliding-window detrending
   leaves strong residuals and BLS fails. Instead, at every trial mid-time (5 min steps) and duration, I fit the raw flux
   with a local cubic polynomial plus a box, rejecting flares. The strongest single events
   (5.375, 22.31, 716.27, 724.735, 733.18, 2561.26, 2578.20 d) fit a linear ephemeris with P = 8.4632 d
   (residuals < 0.025 d over 2,570 d). A second set (17.22, 715.00, 733.86, 2563.18 d) gives P = 18.859 d,
   with residuals < 0.006 d. Two of those transits are consecutive, so the cycle count is unambiguous. The half-period
   epochs (26.65, 724.44) show only noise-level dips (~250 ppm against 1,250 ppm).
6. **Final fit** (`fit.py`, `make_outputs.py`). A joint trapezoid model (t0, P, depth, T14, ingress), fitted with
   an independent local quadratic baseline (cubic for star_06) around each transit on the *raw* flux.
   This avoids depth suppression from the detrending. Other planets' transits in the same star and known artefact
   windows are excluded, and there is 5σ iterative clipping.
7. **Vetting** (`vet.py`, `make_outputs.py`). Each plot shows:
   - the full light curve with predicted transit times;
   - the phase fold with the model;
   - odd vs even depths;
   - flux at phase 0.5 (secondary eclipse);
   - individual transit depths (with χ²);
   - the full phase curve.

   For short periods I also fitted cos(2πφ) and cos(4πφ) terms (reflection/heating and ellipsoidal variation)
   to the raw, un-detrended flux. Further checks: whether the duration makes sense for a dwarf host (T14 against P^(1/3)
   scaling within multi-planet systems), and an **inverted light-curve test** (BLS on 2 − flux) to see how high
   pure-noise peaks get.

## Verdict reasoning

- **star_01 (3 planets, 2.25 / 3.69 / 7.45 d).** All three have consistent depths, odd = even, and no secondary.
  The 7.45 d period is ~2.02× the 3.69 d period. In one transit (t ≈ 98.15 d) the 3.69 d and 7.45 d planets
  transit together, which explains an apparently deep single event; it is excluded from the fits.
  *Not reported:* a 1.049 d, ~150 ppm signal (SNR ≈ 7.8). It shows up in all three segments, but BLS on the
  inverted light curve reaches almost the same peak power (7.4 against 7.9), so its false-alarm probability is not
  small. It is below my reporting threshold.
- **star_02 (0.4629 d).** Ultra-short period, 4.3 ppt, 31 min duration. For a central transit the duration implies
  a stellar density of ~20 ρ☉, i.e. an M dwarf, which makes the ~3 R⊕ companion very plausible. Odd/even agree
  to 0.5σ, there's no secondary (−42 ± 88 ppm), and no ellipsoidal signal.
- **star_03 (6.268 d).** 300 ppm, 3.2 h flat-bottomed transits on a very quiet star (89 ppm per 2 min).
  Odd = even and there's no secondary. The ramps at 96–97.5 d and 210 d are instrumental and were excluded.
- **star_06 (8.463 d and 18.859 d).** A young, spotted, flaring star. Both signals are box-shaped with the right
  duration for a dwarf host and no secondary. Per-transit depths scatter more than the white-noise errors allow
  (χ² 203/6 and 35/3, odd/even 3–5σ). I attribute that to spot crossings, flares and red noise from the large
  stellar variability, not to an eclipsing binary: the depths differ by about ±20%, not by a factor of 2, and there
  is no secondary.
  The highest raw BLS peak (P ≈ 4.86 d) is the rotation period leaking through detrending. It is not reported.
- **star_07 (3.1828 d planet candidate + 14.4 d artefact).** The raw BLS was dominated by the segment-3 events: every
  ~14.4 d (with ±0.1 d irregularity), ~12 h of chaotic ±10% flux. They are present only in segment 3 and look like
  pointing/scattered-light systematics tied to the spacecraft orbit. They're reported as **other false positive**.
  With those windows and chunk edges masked, segments 1 and 2 searched *separately* both peak at the same period
  (3.1842 and 3.1828 d). Folding segment 3 on that ephemeris recovers the dip (523 ± 43 ppm), and segment 3 alone
  also lists 3.182 d among its top peaks. The joint fit gives 409 ± 33 ppm, odd 408 against even 392 ppm,
  per-transit χ² 34/24, and no secondary. This is the weakest planet candidate in the set; segment 3 still has
  residual systematics.
- **star_08 (0.9415 d), planet candidate rather than eclipsing binary.** At first this looked like an eclipsing
  binary, because there is a clear secondary eclipse at phase 0.5 (≈ 390–500 ppm) and ellipsoidal variation. Taken
  quantitatively, though, the evidence points to a planet:
  - The transit is flat-bottomed and limb-darkened, odd and even depths agree (10,444 vs 10,486 ppm), and Rp/R* ≈ 0.10.
  - The ellipsoidal amplitude is 180 ± 8 ppm. With a/R* ≈ 3.3 (from T14), this implies a mass ratio of ~0.5%, i.e.
    ~5–10 Jupiter masses. An M-dwarf companion would produce >1,000 ppm.
  - A secondary depth of ~400–500 ppm with (Rp/R*)² ≈ 0.0108 gives a dayside brightness temperature of ~3,000 K.
    That matches the irradiation temperature expected at a/R* ≈ 3.3 around a ~6,000–6,500 K star.
  - The cos(2πφ) phase-curve term (130 ppm, brightest at phase 0.5) behaves like dayside heating.

  All of this is the signature of a massive ultra-hot Jupiter, not a stellar binary.
- **star_09 (16.05 d).** Only three transits, but they are mutually consistent (χ² 1/2), box-shaped and 3.2 ppt deep.
  Half-period epochs that fall in the data show no dip, and there's no secondary. Planet candidate. The period is
  pinned by consecutive transits 63.24 → 79.29 d.
- **star_10 (3 planets, 3.36 / 5.66 / 11.38 d).** All flat-bottomed, odd = even, no secondaries. The durations
  (1.35 / 1.61 / 2.43 h) scale close to the P^(1/3) expected for planets around one host, which supports a common star.
  The paired-looking dips in the raw light curve are the 5.66 d and 11.38 d planets, which are near 2:1 resonance and
  transit close together in time.
- **star_11 (eclipsing binary, P = 3.1338 d).** BLS finds 1.5669 d, but at that period odd and even depths differ by
  21σ (≈ 9.3% against 12.2%). Folded at the true orbital period of 3.1338 d there is a V-shaped 13.5% primary and an
  8.3% secondary at exactly phase 0.5. Eclipses this deep and V-shaped, with a secondary that large, are stellar.
- **star_12 (5.652 d).** 1.6 ppt, 3.5 h transits, consistent across 8 events, odd = even (0.8σ), no secondary
  (51 ± 46 ppm). The shape is somewhat V-like (ingress ≈ 1 h), so a grazing orbit or a diluted binary can't be
  excluded from photometry alone. With no positive binary evidence, it stays a planet candidate.
- **star_04, star_05: nothing significant.** No BLS peak exceeded SDE ≈ 5, and none rose above the inverted-light-curve
  noise ceiling, in the full data or in any single segment. star_04's first-pass "14.0 d" signal was made of dips at
  the ends of data chunks (37.4 d and 51.3 d, both just before gaps). It disappears when chunk edges are trimmed, so it
  is not reported.

## Limitations

- Depths and durations come from a trapezoid model, not a limb-darkened physical model. Uncertainties are formal
  white-noise values and underestimate the truth on star_06 and star_07.
- Without centroid or pixel data, blended (background) eclipsing binaries can't be ruled out for any candidate.
- Small planets below roughly 150 ppm (bright stars) to ~1,000 ppm (faint stars) could be missed.
