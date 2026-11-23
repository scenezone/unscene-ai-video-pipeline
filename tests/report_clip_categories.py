"""Add category counts to a saved complete comparison without rerunning models."""
import argparse
import json
from pathlib import Path

from matching_decision import decision_category
from test_clip_matching_comparison import summarize, write_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or "detections" not in report:
        parser.error("Expected a complete CLIP comparison report")
    for row in report["detections"]:
        for result in row["results"].values():
            result["decision_category"] = decision_category(result)
    report["summary"] = summarize(report)
    output = args.report.parent
    report_path = output / "report_categories.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_review(output, report, filename="review_categories.html")
    for key, summary in report["summary"].items():
        print(key, summary["categories"])
    print(f"Review: {output / 'review_categories.html'}")


if __name__ == "__main__":
    main()
