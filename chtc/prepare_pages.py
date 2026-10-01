"""Create a static GitHub Pages bundle from an audited local report."""
import argparse
import json
import shutil
from pathlib import Path


def prepare(report, output):
    audit = json.loads((report / "audit.json").read_text())
    if audit.get("passed") is not True:
        raise ValueError("A passing integrity audit is required")
    if output.exists():
        raise ValueError("Choose a new output directory to preserve existing bundles")
    files = ["index.html", "summary.json", "summary.csv", "audit.json", "diagnostics.json"]
    for name in files:
        if not (report / name).is_file():
            raise ValueError(f"Missing required artifact: {name}")
    data = report / "replay-data"
    if not data.is_dir() or not list(data.glob("*.js")):
        raise ValueError("Missing replay episode data")
    output.mkdir(parents=True)
    for name in files:
        shutil.copyfile(report / name, output / name)
    for source in report.glob("*.png"):
        shutil.copyfile(source, output / source.name)
    for source in report.glob("*.pdf"):
        shutil.copyfile(source, output / source.name)
    (output / "replay-data").mkdir()
    for source in data.glob("*.js"):
        if source.stat().st_size >= 100_000_000:
            raise ValueError(f"Episode exceeds GitHub file limit: {source.name}")
        shutil.copyfile(source, output / "replay-data" / source.name)
    (output / ".nojekyll").touch()
    print(f"Audited {audit['profile']} report ready for Pages: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.report, args.output)


if __name__ == "__main__":
    main()
