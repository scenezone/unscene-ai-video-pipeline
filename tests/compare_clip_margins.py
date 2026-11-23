"""Compare acceptance margins on saved audit rankings without rerunning models."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

from matching_decision import decide, category_counts

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Existing audit report.json (relative to project root)")
    parser.add_argument("--margins", type=float, nargs="+", default=[0, .05, .1])
    args = parser.parse_args()
    if not all(math.isfinite(value) and 0 <= value <= 2 for value in args.margins):
        parser.error("Margins must be finite numbers between 0 and 2")
    source = args.report if args.report.is_absolute() else ROOT / args.report
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = data.get("matches")
    if not isinstance(rows, list) or any("ranking" not in row for row in rows):
        parser.error("Expected an audit report with matches and saved rankings")
    results = {}
    for margin in sorted(set(args.margins)):
        decisions = [dict(crop=row["crop"], best_product=row["best_product"],
                          **decide(row["ranking"], row["threshold"], margin)) for row in rows]
        results[f"{margin:g}"] = dict(accepted=sum(row["accepted"] for row in decisions),
                                      categories=category_counts(decisions),
                                      total=len(decisions), decisions=decisions)
        print(f"Minimum margin {margin:g}: {results[f'{margin:g}']['accepted']}/{len(decisions)} accepted")
    output = ROOT / "outputs/clip_margin_comparison" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    report = dict(source_report=str(source.resolve()), source_settings=data.get("settings"),
                  note="Same saved scores and similarity thresholds. IDs come from the source catalog; flat catalog filenames are separate candidates. Raw crop decisions only, without replaying tracking/filtering.",
                  comparisons=results)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved comparison: {output / 'comparison.json'}")


if __name__ == "__main__":
    main()
