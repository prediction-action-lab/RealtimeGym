# Qwen3-8B: completed Freeway pressure comparisons

Interactive meeting snapshot from September 17, 2026. The main page compares
Reactive, Planning, and Agile at 32K, 16K, and 8K tokens per step. All 36
episodes are complete and passed the existing artifact integrity audit.

| Tokens per step | Reactive | Planning | Agile |
|---|---:|---:|---:|
| 32K | 0.3034 | 0.8511 | 0.9157 |
| 16K | 0.3034 | 0.4185 | 0.7809 |
| 8K | 0.3034 | 0.0000 | 0.5309 |

Scores are mean raw reward / 89. Four episodes per cell use two environment
seeds and two sampling seeds, with medium difficulty and a fixed 4K reactive
budget. These are exploratory token-simulation results. The 4K condition and
difficulty sweep are excluded because they are incomplete. This is not the
final 84-episode report or a complete three-game replication.

Open `index.html` for the interactive comparison. The prior 12-episode pilot
remains available at [pilot/](pilot/). The 8K condition repeats the pilot's
configuration and seeds; it is not additional independent evidence.

The repository contains generated site assets, not credentials or model
weights. The Pages site is publicly accessible.
