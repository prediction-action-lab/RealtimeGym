# RealtimeGym visualization

This directory contains the browser UI and recorded game replays from
https://devesh-maheshwari.github.io/realtimegym-qwen3/.
The snapshot compares Reactive, Planning, and Agile Qwen3-8B agents on Freeway.
It plays recorded episodes; it does not run model inference in the browser.

## Preview the published snapshot

From the repository root:

```bash
python3 -m http.server 8000 --directory chtc/pages/pilot-20260916
```

Open http://localhost:8000/. Choose a token budget and seed, then use the
playback controls to compare the three agents and inspect their reasoning.
The original pilot is available at http://localhost:8000/pilot/.
You can also open `chtc/pages/pilot-20260916/index.html` directly in a browser.

## Files

- `pages/pilot-20260916/`: complete static snapshot, including HTML, replay
  data with embedded game frames, score downloads, plots, and audit records.
- `research.html`: reusable comparison-page template (HTML, CSS, JavaScript).
- `replay.html`: reusable episode replay template.
- `summarize.py`: reads recorded episode directories and generates reports,
  plots, and replay data. Requires NumPy and Matplotlib.
- `prepare_pages.py`: packages a report with an existing passing `audit.json`
  for static hosting. It does not perform the underlying episode audit.
- `check_site.py`: desktop/mobile interaction checks using Playwright Chromium.

The snapshot's `index.html` contains its own styling, scripting, and episode
index. Editing a template does not automatically change the saved snapshot.
Game simulation and frame rendering remain in `src/realtimegym/environments/`.

## Generate a report from existing episode recordings

Each episode directory must contain `summary.json`, `requests.jsonl`,
`steps.jsonl`, and any frame images referenced by the steps.

```bash
python3 -m pip install numpy matplotlib
python3 chtc/summarize.py /path/to/episodes --output /tmp/realtimegym-report
```

The generated `index.html` can be previewed immediately. To package a report
that already has a passing integrity audit from the experiment pipeline:

```bash
python3 chtc/prepare_pages.py /path/to/audited-report --output /tmp/realtimegym-pages
```

Choose a new output directory for each bundle. The original CHTC job runner,
model setup, and audit-generation pipeline are outside this visualization
addition.

## Browser checks

```bash
python3 -m pip install playwright
python3 -m playwright install chromium
python3 chtc/check_site.py chtc/pages/pilot-20260916/index.html \
  --screenshots /tmp/realtimegym-site-screenshots
```

The static bundle can be hosted as a directory on any static server. Pushing
this branch alone does not configure GitHub Pages or update the existing site.
