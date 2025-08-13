# scripts/export_metrics.py
from __future__ import annotations
import json, csv, sys
from pathlib import Path

def main(results_json_path: str, out_csv_path: str = "metrics.csv"):
    try:
        data = json.loads(Path(results_json_path).read_text(encoding="utf-8"))
        aggregates = (data.get("metadata") or {}).get("aggregates") or {}
        if not aggregates:
            print("No aggregate metrics found in results.json.")
            return

        rows = []
        for name, metrics in aggregates.items():
            if isinstance(metrics, dict):
                row = {"aggregator": name}
                row.update(metrics)
                rows.append(row)
        if not rows:
            print("No valid aggregate data to write.")
            return

        fieldnames = sorted(list(set(k for r in rows for k in r.keys())))
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} aggregate rows to {out_csv_path}")

    except FileNotFoundError:
        print(f"Error: The file '{results_json_path}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{results_json_path}'.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/export_metrics.py <path_to_results.json> [output_path.csv]")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "metrics.csv")