"""Combine per-coupling-map benchmark JSONs into a single merged file.

Usage:
    python combine_results.py [results_dir]

Defaults to ../../results relative to this script if no argument is given.
"""

import json
import sys
from pathlib import Path


def combine_rand(results_dir: Path) -> dict:
    files = sorted(results_dir.glob("benchmark_rand_*.json"))
    files = [f for f in files if f.stem != "benchmark_rand_combined"]

    combined: dict = {"coupling_maps": {}, "gate_amounts": None, "configs": {}}
    for path in files:
        with open(path) as f:
            data = json.load(f)

        cm_title = data["coupling_map"]
        n_qubits = data["num_qubits"]
        combined["coupling_maps"][cm_title] = n_qubits

        for config_name, gate_data in data["configs"].items():
            if config_name not in combined["configs"]:
                combined["configs"][config_name] = {}
            combined["configs"][config_name][cm_title] = gate_data

        if combined["gate_amounts"] is None:
            combined["gate_amounts"] = [int(k) for k in next(iter(data["configs"].values())).keys()]

    return combined


def combine_mqt(results_dir: Path) -> dict:
    files = sorted(results_dir.glob("benchmark_mqt_*.json"))
    files = [f for f in files if f.stem != "benchmark_mqt_combined"]

    combined: dict = {"coupling_maps": {}, "algorithms": {}}
    for path in files:
        with open(path) as f:
            data = json.load(f)

        cm_title = data["coupling_map"]
        n_qubits = data["num_qubits"]
        combined["coupling_maps"][cm_title] = n_qubits

        for algo, algo_data in data["algorithms"].items():
            if algo not in combined["algorithms"]:
                combined["algorithms"][algo] = {}
            combined["algorithms"][algo][cm_title] = algo_data

    return combined


def main() -> None:
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path(__file__).parent.parent.parent / "results"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    rand_combined = combine_rand(results_dir)
    mqt_combined = combine_mqt(results_dir)

    rand_out = results_dir / "benchmark_rand_combined.json"
    mqt_out = results_dir / "benchmark_mqt_combined.json"

    with open(rand_out, "w") as f:
        json.dump(rand_combined, f, indent=2)
    print(f"Written: {rand_out}")

    with open(mqt_out, "w") as f:
        json.dump(mqt_combined, f, indent=2)
    print(f"Written: {mqt_out}")


if __name__ == "__main__":
    main()
