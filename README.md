# Can Claude Code find planets in raw NASA data? (Experiment 6)

The test behind an It's Not Magic film. Twelve real light curves from NASA's TESS telescope, with every name, position and
date removed. The answer key comes from NASA and stayed hidden. Claude Code ran three fresh sittings with the same
one-paragraph prompt, and a textbook search ran on the same data as the control. Run on 22 Nov 2026.

Everything needed to check the numbers is here: the prompt, the exact data Claude saw, the answer key, the scorer, the control,
and each sitting's full transcript and output.

## Results
Scored with `tools/score.py` against `key/key.json`. The data can show 16 known planet signals, meaning signals with at least two
transits inside the data given.

| run | found / 16 | eclipsing binary | TOI-1031 (NASA: false positive) | claims in no list | TOI-7930.02 (3.18 d) | wall time | turns | API-equivalent cost |
|---|---|---|---|---|---|---|---|---|
| sitting 1 | 12 | caught | called planet ("rather V-shaped") | 1 (L 98-59, 1.049 d, low confidence) | found: 3.18279 d, 430 ppm | 16.9 min | 115 | $4.60 |
| sitting 2 | 11 | caught | called planet ("somewhat V-shaped") | 0 | not found ("No transit signal.") | 83.5 min* | 108 | $5.26 |
| sitting 3 | 13 | caught | called planet ("slightly V-shaped") | 0 | found: 3.18278 d, 409 ppm | 27.8 min | 100 | $4.65 |
| textbook search | 6 (+1 at 2× period) | called planet at half the period | called planet | 1 (star 7's 14.4 d instrument bursts) | not found | — | — | — |

NASA lists TOI-7930.02 at 3.1827878 d and 461 ppm.

- **Missed by every sitting:** TOI-700 b (9.98 d), TOI-700 d (37.4 d, two transits) and TOI-7930.01 (11.13 d).
- **AU Mic c:** only sitting 3 found it.
- **L 98-59's 1.049 d bump:** all three sittings saw it. Sittings 2 and 3 left it out as below their threshold. It is in no
  catalogue, and nothing in this test says whether it is real.
- **Star 7's bursts:** every sitting labelled the ±10% bursts every ~14.4 d (QLP, one sector) as instrumental.

\* Sitting 2 overlapped the textbook control (run at `nice 19`) for about 15 minutes, so its wall time is inflated by an unknown amount.

## The stars (answer key)
| star | real name | why it is in |
|---|---|---|
| hot Jupiter | WASP-18 (TOI-185) | easy: a 1% dip every 22.6 h |
| ultra-short | LHS 3844 (TOI-136) | 11-hour orbit around a small M dwarf |
| three planets | L 98-59 (TOI-175) | small planets at 2.25, 3.69 and 7.45 d |
| three planets | TOI-270 | 3.36, 5.66 and 11.38 d |
| four planets | TOI-700 | 9.98, 16.05, 37.4 and 27.8 d (the last two have at most two transits in the data) |
| shallow | Pi Mensae c (TOI-144) | about 300 ppm on a very bright star |
| spotted young star | AU Microscopii | b at 8.46 d and c at 18.86 d, under large spot variability |
| the news star | TIC 4206066 = TOI-7930 | NASA lists two planet candidates (3.18 d, 11.13 d), unconfirmed; TESS re-observed it in Sector 110 (31 Oct – 26 Nov 2026) |
| trap | TIC 189639080 | eclipsing binary from the TESS EB catalog (Prša et al. 2022), P = 3.134 d |
| trap | TOI-1031 | NASA marked the signal a false positive |
| quiet | TIC 259376992, TIC 259378043 | no TOI, no known planet, not in the EB catalog |

## Layout
- **The task:**
  - `task/PROMPT.md` is the exact prompt every sitting got.
  - `task/data/star_01.csv … star_12.csv` is what Claude saw. Columns: `time_days`, `flux`, `flux_err`, `segment`, `cadence_s`.
    Times are shifted so each star starts near day 0. Names, coordinates, dates and FITS headers are removed, and the order is
    shuffled (seed 20261122).
  - `task/settings.json` holds the intended permissions. In an untrusted folder the CLI ignored this file's allow list, so
    `tools/run_sitting.sh` passes the same rules as flags.
- **The answer key** (hidden from the sittings):
  - `key/stars.json` says which star is which, with sectors and time offsets.
  - `key/key.json` lists every known signal and how many of its transits fall inside the data.
  - `key/raw/` holds the NASA Exoplanet Archive TOI table and confirmed-planet rows, plus the TESS EB catalog rows.
- **The tools:**
  - `tools/fetch.py` downloads from MAST and anonymises.
  - `tools/build_key.py` builds the key.
  - `tools/run_sitting.sh` runs one sitting.
  - `tools/score.py` scores a `results.json`.
  - `tools/baseline.py` is the textbook control.
  - `tools/peek.py` and `tools/survey.py` read the transcripts.
- **The results:**
  - `results/sN/` holds what each sitting wrote: `results.json`, `NOTES.md`, its plots and its own scripts. The score is in
    `results.score.json`.
  - `results/sN.stream.jsonl` is the full stream-json transcript, and `sN.start` / `sN.end` are the UTC start and end times.
  - `results/baseline/` is the control.

The original MAST FITS files (about 60 MB) are not included. `tools/fetch.py` downloads them again.

## How a sitting ran
- **The folder:** a fresh folder holding only `data/`, an empty `plots/` and a shared `../.venv` with numpy, scipy, astropy,
  lightkurve and matplotlib. The key lived in a different folder.
- **The command:**
  `claude -p "$(cat task/PROMPT.md)" --model claude-opus-5-5 --effort high --setting-sources project --strict-mcp-config`
- **Tools allowed:** only Bash, Read, Write, Edit, Glob and Grep.
- **Tools denied:** WebFetch, WebSearch, curl, wget, pip and git clone. No MCP servers.
- **One session per sitting:** each sitting was a new session with no memory of the others.

## How a signal is scored
- **Findable:** a known signal is findable when at least two of its transits fall inside the data.
- **Exact match:** a reported signal matches when its period is within 1% and its transit time lines up within 3 hours after
  folding on the key's period. Exact matches are tried across all known signals first.
- **Aliases:** 2×, ½×, 3× and ⅓× periods within 0.3% are counted separately as aliases.
- **Eclipsing binaries:** a binary matches at phase 0 or 0.5.
- **Ephemerides:** confirmed-planet ephemerides are preferred over TOI rows. AU Mic b's TOI period drifts about 3 days over
  seven years.

These rules were fixed while scoring, and every run was rescored after each fix.

## Reproduce the scores
```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tools/score.py results/s1/results.json
.venv/bin/python tools/baseline.py      # the textbook control
```
A new sitting needs Claude Code installed and signed in: run `tools/run_sitting.sh s4`. It writes to `~/Documents/Projects/tess-runs/s4`.

## Caveats
- **Small sample:** twelve stars, most of them with known planets.
- **Light curves only:** no pixel data, which is where many false positives are caught. Finding a dip is not confirming a planet.
- **No key or web access:** no transcript shows access to the key, the web or an archive query.
- **One skill load:** sitting 1 loaded Claude Code's bundled `dataviz` skill for plot styling, because `--setting-sources project`
  does not turn skills off. Sittings 2 and 3 did not load it.

## What was removed from the public transcripts
- account rate-limit events and thinking-token counter ticks;
- the session-start list of the local install's slash commands, skills and plugins (the tool list and model are kept);
- the text of the bundled `dataviz` skill (a one-line note marks where it was).

Home-folder paths were replaced with `~`. Every assistant message, tool call and tool result is otherwise unchanged.

## Data credits
- **Light curves:** NASA TESS mission, from MAST (SPOC 2-min, TESS-SPOC and QLP).
- **Answer key:** NASA Exoplanet Archive (TOI table and Planetary Systems Composite Parameters), operated by Caltech/IPAC under
  contract with NASA.
- **Eclipsing binaries:** Prša et al. 2022, *ApJS* 258, 16 (via VizieR).

## Licence
Code: MIT (see `LICENSE`). The TESS data and NASA tables are public; please credit the sources above if you reuse them.
