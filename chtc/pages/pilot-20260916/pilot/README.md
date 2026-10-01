# RealtimeGym: Qwen3-8B Freeway pilot

An audited 12-episode, inference-only pilot comparing Reactive, Planning and
Agile agents at medium difficulty with 8,192 tokens per simulated step.

| Agent | Mean normalized final reward | Mean raw reward |
|---|---:|---:|
| Agile | 0.5309 | 47.25 |
| Reactive | 0.3034 | 27.00 |
| Planning | 0.0000 | 0.00 |

Scores divide raw reward by 89. Two environment seeds and two sampling seeds
per environment make this an exploratory pilot. It does not establish trends
across difficulty, time budgets or games. The 84-episode trend sweep is pending.

Planning used default actions on 351/400 steps while complete plans were often
unavailable at the simulated deadline. The audit checked 671 planner prefixes
and replayed environment transitions. All 12 completed episodes passed.

Open `index.html` for matched game replays, condition selection, scores and
downloadable audit data. Enable GitHub Pages from `main` and `/ (root)`.

The repository contains generated research artifacts. A private source
repository does not make the published GitHub Pages website private.
