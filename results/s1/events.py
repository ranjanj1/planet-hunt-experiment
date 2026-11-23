"""Single-event search: slide boxes of several durations along the detrended light curve and
list dips whose depth significance exceeds a threshold.  python events.py STAR [thresh]"""
import sys
import numpy as np
from pipeline import bin_lc


def scan(t, f, durs_h=(1, 2, 3, 5, 8), thresh=6.0, step_min=10):
    # bin to 10 min to make the scan fast, keep per-bin weights
    tb, fb, nb = bin_lc(t, f, step_min / 1440)
    sd = 1.4826 * np.median(np.abs(np.diff(f))) / np.sqrt(2)  # point-to-point white noise
    ev = []
    for dh in durs_h:
        D = dh / 24
        lo = np.searchsorted(tb, tb - D / 2)
        hi = np.searchsorted(tb, tb + D / 2)
        lo2 = np.searchsorted(tb, tb - 1.5 * D - 0.1)
        hi2 = np.searchsorted(tb, tb + 1.5 * D + 0.1)
        cw = np.r_[0, np.cumsum(nb)]
        cf = np.r_[0, np.cumsum(fb * nb)]
        n_in = cw[hi] - cw[lo]
        m_in = (cf[hi] - cf[lo]) / np.maximum(n_in, 1)
        n_out = (cw[hi2] - cw[lo2]) - n_in
        m_out = ((cf[hi2] - cf[lo2]) - (cf[hi] - cf[lo])) / np.maximum(n_out, 1)
        cover = n_in * step_min / 2 / (D * 1440)  # fraction of the box with data (2-min units)
        ok = (n_in > 0) & (n_out > 0) & (cover > 0.6)
        dep = m_out - m_in
        snr = dep / (sd / np.sqrt(np.maximum(n_in, 1)))
        # red-noise correction: scale by robust std of the snr series itself
        s = snr[ok]
        rs = 1.4826 * np.median(np.abs(s - np.median(s)))
        snr = snr / max(rs, 1)
        for k in np.where(ok & (snr > thresh))[0]:
            ev.append((tb[k], dh, dep[k], snr[k]))
    ev.sort(key=lambda x: -x[3])
    keep = []
    for e in ev:  # merge events closer than 0.5 d, keep the most significant
        if all(abs(e[0] - k[0]) > 0.5 for k in keep):
            keep.append(e)
    return sorted(keep)


if __name__ == '__main__':
    i = int(sys.argv[1]); th = float(sys.argv[2]) if len(sys.argv) > 2 else 6
    z = np.load(f'work/star_{i:02d}_flat.npz')
    t, f = z['t'][z['good']], z['f'][z['good']]
    for tc, dh, dep, snr in scan(t, f, thresh=th):
        print(f'  t={tc:9.3f}  dur={dh}h  depth={dep*1e6:7.0f}ppm  snr={snr:5.1f}')
