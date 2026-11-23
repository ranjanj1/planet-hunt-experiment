"""Plot for the star_07 instrumental false positive (segment-3 events every ~14.4 d)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pipeline import load, bin_lc

d = load("star_07")
t, f, s = d["time_days"], d["flux"], d["segment"]
P, t0 = 14.395, 2523.755
ev = [2523.755, 2538.355, 2552.556, 2567.006]
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 4)
ax = fig.add_subplot(gs[0, :])
m = s == 3
ax.plot(t[m], f[m], ",", color="0.6")
tb, fb, _ = bin_lc(t[m], f[m], 30 / 1440)
ax.plot(tb, fb, ".", ms=2, color="k")
for e in ev:
    ax.axvline(e, color="C3", alpha=0.4)
for k in range(-1, 5):
    ax.axvline(t0 + k * P, color="C0", ls=":", alpha=0.6)
ax.set_title("star_07 seg 3: ~12 h bursts of ±10% flux every ~14.4 d (red = measured centres, "
             "blue dotted = 14.395 d ephemeris) -> instrumental, not transits")
ax.set_xlabel("time [d]"); ax.set_ylabel("raw flux")
for i, e in enumerate(ev):
    a = fig.add_subplot(gs[1, i])
    mm = m & (np.abs(t - e) < 1.2)
    a.plot(t[mm] - e, f[mm], ".", ms=1, color="k")
    a.set_title(f"event at {e:.2f} d"); a.set_xlabel("days from centre")
fig.text(0.5, 0.005, "Not periodic to better than ±0.1 d, present only in segment 3, symmetric up/down excursions "
         "(not a dip with ingress/egress): spacecraft pointing / scattered-light systematics.", ha="center")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("plots/star_07_2_instrumental.png", dpi=90)
