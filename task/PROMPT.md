The folder data/ holds 12 files, star_01.csv to star_12.csv. Each one is the brightness of one star over time, measured by NASA's TESS space telescope. Columns: time_days (days), flux (brightness, normalised to about 1), flux_err, segment (separate observing runs, with gaps between them), cadence_s (seconds between measurements).

Your job: find every transiting planet signal in these stars, and tell apart signals that are not planets.

Python is available in ../.venv with numpy, scipy, astropy, lightkurve and matplotlib installed. Work only with the files in this folder. Do not use the internet.

When you are done, write results.json in this format:
{"star_01": [{"period_days": 0.0, "t0_days": 0.0, "depth_ppm": 0.0, "duration_hours": 0.0, "verdict": "planet candidate", "reason": "one line"}], "star_02": []}
- one entry per signal you found; an empty list if you found none
- t0_days is the middle of one transit, on the same time axis as the data
- verdict is one of: "planet candidate", "eclipsing binary", "other false positive"

Also write NOTES.md explaining your method, and save one plot per signal in plots/.
