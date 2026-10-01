"""Aggregate completed episodes using fixed paper bounds; generate plots and replay."""
import argparse
import base64
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MODES = ("reactive", "planning", "agile")
COLORS = {"reactive": "#e5ae00", "planning": "#dc649d", "agile": "#278bc4"}


def interval(rows):
    # Average repetitions within environment first; bootstrap independent
    # environment seeds, not correlated steps or repeated games as new seeds.
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["env_seed"]].append(row["normalized_score"])
    means = np.array([np.mean(values) for values in by_seed.values()])
    mean = float(means.mean())
    if len(means) < 2:
        return mean, None, None
    rng = np.random.default_rng(2026)
    boot = rng.choice(means, size=(5000, len(means)), replace=True).mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return mean, float(low), float(high)


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        if not row["completed"] or row["normalized_score"] is None:
            continue
        key = tuple(row[k] for k in ("sweep", "game", "cognitive_load", "time_pressure", "internal_budget", "mode"))
        grouped[key].append(row)
    result = []
    for key, items in sorted(grouped.items()):
        mean, low, high = interval(items)
        result.append(dict(zip(("sweep", "game", "cognitive_load", "time_pressure", "internal_budget", "mode"), key),
                           mean=mean, ci_low=low, ci_high=high, episodes=len(items),
                           environment_seeds=len({r["env_seed"] for r in items})))
    return result


def plot(rows, output):
    rows = [r for r in rows if r["sweep"] in ("cognitive_load", "time_pressure")]
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for game in sorted({r["game"] for r in rows}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, sweep, field, values, labels in (
            (axes[0], "cognitive_load", "cognitive_load", ["E", "M", "H"], ["Easy", "Medium", "Hard"]),
            (axes[1], "time_pressure", "time_pressure", [32768, 16384, 8192, 4096], ["32K", "16K", "8K", "4K"]),
        ):
            for mode in MODES:
                points = [next((r for r in rows if r["game"] == game and r["sweep"] == sweep
                                and r["mode"] == mode and r[field] == x), None) for x in values]
                means = [r["mean"] if r else float("nan") for r in points]
                ax.plot(range(len(values)), means, "o-", label=mode, color=COLORS[mode])
                for x, row in enumerate(points):
                    if row and row["ci_low"] is not None:
                        ax.vlines(x, row["ci_low"], row["ci_high"], color=COLORS[mode], linewidth=2)
            ax.set_xticks(range(len(values)), labels)
            ax.set_title(sweep.replace("_", " ").title())
            ax.set_ylabel("Score (fixed paper normalization)")
            ax.grid(alpha=0.2)
            ax.legend()
        fig.suptitle(f"Qwen3-8B — {game}; bars: 95% environment-bootstrap intervals")
        fig.savefig(output / f"{game}.png", dpi=170)
        fig.savefig(output / f"{game}.pdf")
        plt.close(fig)


def replay(rows, episodes, output):
    data_dir = output / "replay-data"
    data_dir.mkdir(exist_ok=True)
    index = []
    for row in rows:
        path = episodes / row["id"] / "steps.jsonl"
        if not path.exists():
            continue
        steps = []
        for line in path.read_text().splitlines():
            step = json.loads(line)
            logs = step["agent_logs"]
            item = {"state": step["observation"]["state_string"], "action": step["action"],
                          "reward": step["reward"], "diagnostics": step["diagnostics"],
                          "guidance": logs.get("plan", ""),
                          "response": logs.get("model1_response", logs.get("model2_response", ""))}
            if step.get("frame"):
                frame = episodes / row["id"] / step["frame"]
                if frame.is_file():
                    item["image"] = "data:image/png;base64," + base64.b64encode(frame.read_bytes()).decode()
            steps.append(item)
        # Script files allow local file:// playback without a web server.
        payload = json.dumps({"meta": row, "steps": steps}).replace("<", "\\u003c")
        (data_dir / f"{row['id']}.js").write_text(f"window.receiveEpisode({json.dumps(row['id'])},{payload});\n")
        index.append(row)
    template = Path(__file__).with_name("replay.html").read_text()
    (output / "replay.html").write_text(template.replace("/*EPISODE_INDEX*/[]", json.dumps(index).replace("<", "\\u003c")))
    research_template = Path(__file__).with_name("research.html").read_text()
    (output / "index.html").write_text(research_template.replace("/*EPISODE_INDEX*/[]", json.dumps(index).replace("<", "\\u003c")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(p.read_text()) for p in sorted(args.episodes.glob("*/summary.json"))]
    aggregated = aggregate(rows)
    (args.output / "summary.json").write_text(json.dumps(aggregated, indent=2) + "\n")
    if aggregated:
        with (args.output / "summary.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(aggregated[0]))
            writer.writeheader()
            writer.writerows(aggregated)
    diagnostics = []
    for row in rows:
        requests = [json.loads(x) for x in (args.episodes / row["id"] / "requests.jsonl").read_text().splitlines()]
        for role in ("reactive", "planning", "forcing"):
            rr = [r for r in requests if r["role"] == role]
            if rr:
                diagnostics.append({"episode": row["id"], "role": role, "calls": len(rr),
                                    "output_tokens": sum(r["output_tokens"] for r in rr),
                                    "mean_output_tokens": float(np.mean([r["output_tokens"] for r in rr])),
                                    "length_limited_calls": sum(r["finish_reason"] == "length" for r in rr)})
    (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    plot(aggregated, args.output)
    replay(rows, args.episodes, args.output)
    print(f"{len(rows)} recorded episodes; {sum(r['completed'] for r in rows)} completed games. Report: {args.output}")


if __name__ == "__main__":
    main()
