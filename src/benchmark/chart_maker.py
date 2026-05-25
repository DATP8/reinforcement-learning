import json
import sys
from collections import defaultdict
from pathlib import Path

RESULT_PATH = Path(__file__).parent.parent.parent / "results"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_rand_single(file_name):
    """Load single-coupling-map random benchmark file.

    Returns (coupling_map_title, num_qubits, config_data, metrics) where
    config_data: {config: [(gate_count, {metric: (mean, ci)}), ...]}
    """
    with open(RESULT_PATH / file_name) as f:
        data = json.load(f)
    coupling_map = data["coupling_map"]
    num_qubits = data["num_qubits"]
    all_metrics: set = set()
    config_data: dict = {}
    for config, gate_entries in data["configs"].items():
        points = []
        for gate_str, metrics in gate_entries.items():
            metric_means = {}
            for metric, val in metrics.items():
                metric_means[metric] = (val["mean"], val["ci"])
                all_metrics.add(metric)
            points.append((int(gate_str), metric_means))
        points.sort(key=lambda x: x[0])
        config_data[config] = points
    return coupling_map, num_qubits, config_data, sorted(all_metrics)


def load_rand_combined(file_name):
    """Load multi-coupling-map random benchmark file.

    Returns (coupling_maps, gate_amounts, configs_data, metrics) where
    coupling_maps: {cm_title: num_qubits}
    configs_data:  {config: {cm_title: {gate_str: {metric: {mean, ci}}}}}
    """
    with open(RESULT_PATH / file_name) as f:
        data = json.load(f)
    coupling_maps: dict = data["coupling_maps"]
    gate_amounts: list = sorted(data.get("gate_amounts", []))
    configs_data: dict = data["configs"]
    all_metrics: set = set()
    for cm_data in configs_data.values():
        for gate_data in cm_data.values():
            for m_data in gate_data.values():
                all_metrics.update(m_data.keys())
    return coupling_maps, gate_amounts, configs_data, sorted(all_metrics)


def load_mqt_combined(file_name):
    """Load multi-coupling-map MQT benchmark file.

    Returns (coupling_maps, algorithms, metrics) where
    coupling_maps: {cm_title: num_qubits}
    algorithms:    {algo: {cm_title: {circuit_qubits, configs: {config: {metric: val}}}}}
    """
    with open(RESULT_PATH / file_name) as f:
        data = json.load(f)
    coupling_maps: dict = data["coupling_maps"]
    algorithms: dict = data["algorithms"]
    all_metrics: set = set()
    for algo_data in algorithms.values():
        for cm_data in algo_data.values():
            for metric_vals in cm_data.get("configs", {}).values():
                all_metrics.update(metric_vals.keys())
    return coupling_maps, algorithms, sorted(all_metrics)


# ---------------------------------------------------------------------------
# Typst generators
# ---------------------------------------------------------------------------

def _typst_header():
    return [
        '#import "@preview/lilaq:0.4.0" as lq\n',
        "#set page(width: 297mm, height: 210mm, flipped: true, margin: 1cm)\n",
        "#set text(size: 7pt)\n\n",
    ]


def typst_chart_block(metric, config_data, xlabel, title):
    """Generic chart block.

    config_data: {config: [(x_val, {metric: (mean, ci)}), ...]}
    """
    lines = [
        f'#block(\n'
        f'    width: 100%,\n'
        f'    height: 10cm,\n'
        f'    layout(size => {{\n'
        f'        lq.diagram(\n'
        f'            title: "{title}",\n'
        f'            xlabel: "{xlabel}",\n'
        f'            ylabel: "{metric}",\n'
        f'            width: size.width,\n'
        f'            height: size.height,'
    ]

    for config, points in config_data.items():
        xs = [str(x) for x, _ in points]
        ys = [str(round(m.get(metric, (0, 0))[0], 4)) for _, m in points]
        yerrs = [str(round(m.get(metric, (0, 0))[1], 4)) for _, m in points]
        trailing = "," if len(xs) == 1 else ""
        lines.append(
            f'            lq.plot(\n'
            f'                ({", ".join(xs)}{trailing}),\n'
            f'                ({", ".join(ys)}{trailing}),\n'
            f'                label: "{config}",\n'
            f'                yerr: ({", ".join(yerrs)}{trailing}),\n'
            f'            ),'
        )

    lines.append('        )\n    })\n)\\ \n \\')
    return "\n".join(lines)


def typst_table(metric, config_data, row_label, main_config=None):
    """Generic table.

    config_data: {config: [(row_val, {metric: (mean, ci)}), ...]}
    row_label:   label for the row axis (e.g. "Gate Count", "Num Qubits")
    """
    configs = list(config_data.keys())
    row_values = sorted({rv for points in config_data.values() for rv, _ in points})

    lookup: dict = {}
    for config, points in config_data.items():
        lookup[config] = {rv: m.get(metric, (None, None))[0] for rv, m in points}

    header_cols = ", ".join(f"[*{c}*]" for c in configs)
    if main_config is not None:
        configs_no_main = [c for c in configs if c != main_config]
        compared_header_cols = ", ".join(f"[*{c} vs {main_config}*]" for c in configs_no_main)
        header = f"table.header([*{row_label}*], {header_cols}, {compared_header_cols}),"
        col_widths = ", ".join(["auto"] * (1 + len(configs) + len(configs_no_main)))
    else:
        header = f"table.header([*{row_label}*], {header_cols}),"
        col_widths = ", ".join(["auto"] * (1 + len(configs)))

    rows = []
    for rv in row_values:
        cells = [f"[{rv}]"]
        compare_cells = []
        main_val = lookup.get(main_config, {}).get(rv) if main_config else None

        for config in configs:
            val = lookup.get(config, {}).get(rv)
            if val is None:
                cells.append("[-]")
            elif isinstance(val, float):
                cells.append(f"[{val:.4f}]")
            else:
                cells.append(f"[{val}]")

            if main_config is None or config == main_config:
                continue

            if val is None or main_val is None:
                compare_cells.append("[-]")
            elif val != 0:
                compare_cells.append(f"[{((val - main_val) / val) * 100:.2f}%]")
            else:
                compare_cells.append("[NAN]")

        row = "  " + ", ".join(cells) + ", "
        if main_config is not None:
            row += ", ".join(compare_cells) + ","
        rows.append(row)

    rows_str = "\n".join(rows)
    return f"""=== {metric}

#table(
  columns: ({col_widths}),
  {header}
{rows_str}
)

"""


def typst_tables_section(config_data, metrics, row_label, main_config=None):
    tables = "".join(typst_table(m, config_data, row_label, main_config) for m in metrics)
    return f"== Tables\n\n{tables}"


# ---------------------------------------------------------------------------
# Data reshaping helpers
# ---------------------------------------------------------------------------

def _get_main_config(configs):
    return "receding" if "receding" in configs else None


def _rand_config_data_for_cm(coupling_map_title, configs_data):
    """Reshape combined rand data for a single coupling map (x = gate count)."""
    config_data: dict = {}
    for config_name, cm_gate_data in configs_data.items():
        points = []
        for gate_str, metrics in cm_gate_data.get(coupling_map_title, {}).items():
            metric_means = {m: (v["mean"], v["ci"]) for m, v in metrics.items()}
            points.append((int(gate_str), metric_means))
        points.sort(key=lambda x: x[0])
        config_data[config_name] = points
    return config_data


def _rand_config_data_for_gate(gate_count, configs_data, coupling_maps):
    """Reshape combined rand data for a single gate count (x = num_qubits)."""
    config_data: dict = {}
    gate_str = str(gate_count)
    for config_name, cm_gate_data in configs_data.items():
        points = []
        for cm_title, gate_data in cm_gate_data.items():
            num_qubits = coupling_maps.get(cm_title)
            if num_qubits is None or gate_str not in gate_data:
                continue
            metric_means = {m: (v["mean"], v["ci"]) for m, v in gate_data[gate_str].items()}
            points.append((num_qubits, metric_means))
        points.sort(key=lambda x: x[0])
        config_data[config_name] = points
    return config_data


def _mqt_config_data_for_cm(coupling_map_title, algorithms):
    """Reshape combined MQT data for a single coupling map (x = circuit_qubits)."""
    config_data: dict = {}
    for algo, cm_data in algorithms.items():
        if coupling_map_title not in cm_data:
            continue
        entry = cm_data[coupling_map_title]
        circuit_qubits = entry.get("circuit_qubits", 0)
        for config_name, metric_vals in entry.get("configs", {}).items():
            if config_name not in config_data:
                config_data[config_name] = []
            metric_means = {m: (float(v), 0.0) for m, v in metric_vals.items()}
            config_data[config_name].append((circuit_qubits, metric_means))
    for config_name in config_data:
        config_data[config_name].sort(key=lambda x: x[0])
    return config_data


def _mqt_avg_config_data_across_cm(algorithms, coupling_maps):
    """Average MQT metrics across all algorithms per coupling map (x = num_qubits)."""
    accum: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for algo, cm_data in algorithms.items():
        for cm_title, entry in cm_data.items():
            for config_name, metric_vals in entry.get("configs", {}).items():
                for metric, val in metric_vals.items():
                    accum[config_name][cm_title][metric].append(float(val))

    config_data: dict = {}
    for config_name, cm_metric_data in accum.items():
        points = []
        for cm_title, metric_lists in cm_metric_data.items():
            num_qubits = coupling_maps.get(cm_title)
            if num_qubits is None:
                continue
            metric_means = {m: (sum(vals) / len(vals), 0.0) for m, vals in metric_lists.items()}
            points.append((num_qubits, metric_means))
        points.sort(key=lambda x: x[0])
        config_data[config_name] = points
    return config_data


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def main_rand_single(file_name):
    coupling_map, num_qubits, config_data, metrics = load_rand_single(file_name)
    main_config = _get_main_config(list(config_data.keys()))
    title = f"{coupling_map} ({num_qubits} qubits)"

    typst_lines = _typst_header() + [
        f"= {title}\n\n",
        typst_tables_section(config_data, metrics, "Gate Count", main_config),
        "\n#pagebreak()\n\n",
    ]
    for metric in metrics:
        typst_lines.append(typst_chart_block(metric, config_data, "Gate Count", title))
        typst_lines.append("\n")

    out_file = file_name + ".typ"
    with open(RESULT_PATH / out_file, "w") as f:
        f.write("\n".join(typst_lines))
    print(f"Typst chart file written to {RESULT_PATH / out_file}")


def main_rand_combined(file_name):
    coupling_maps, gate_amounts, configs_data, metrics = load_rand_combined(file_name)
    main_config = _get_main_config(list(configs_data.keys()))

    typst_lines = _typst_header() + ["= Random Circuit Benchmarks\n\n"]

    # --- Coupling map constant (x = gate count) ---
    typst_lines.append("== Coupling Map Constant\n\n")
    for cm_title, num_qubits in coupling_maps.items():
        config_data = _rand_config_data_for_cm(cm_title, configs_data)
        title = f"{cm_title} ({num_qubits} qubits)"
        typst_lines.append(f"=== {title}\n\n")
        typst_lines.append(typst_tables_section(config_data, metrics, "Gate Count", main_config))
        typst_lines.append("\n#pagebreak()\n\n")
        for metric in metrics:
            typst_lines.append(typst_chart_block(metric, config_data, "Gate Count", title))
            typst_lines.append("\n")
        typst_lines.append("\n#pagebreak()\n\n")

    # --- Gate count constant (x = num qubits) ---
    typst_lines.append("== Gate Count Constant\n\n")
    for gate_count in gate_amounts:
        config_data = _rand_config_data_for_gate(gate_count, configs_data, coupling_maps)
        title = f"Gates: {gate_count}"
        typst_lines.append(f"=== {title}\n\n")
        typst_lines.append(typst_tables_section(config_data, metrics, "Num Qubits", main_config))
        typst_lines.append("\n#pagebreak()\n\n")
        for metric in metrics:
            typst_lines.append(typst_chart_block(metric, config_data, "Num Qubits", title))
            typst_lines.append("\n")
        typst_lines.append("\n#pagebreak()\n\n")

    out_file = file_name + ".typ"
    with open(RESULT_PATH / out_file, "w") as f:
        f.write("\n".join(typst_lines))
    print(f"Typst chart file written to {RESULT_PATH / out_file}")


def main_mqt_combined(file_name):
    coupling_maps, algorithms, metrics = load_mqt_combined(file_name)
    all_configs: set = set()
    for algo_data in algorithms.values():
        for cm_data in algo_data.values():
            all_configs.update(cm_data.get("configs", {}).keys())
    main_config = _get_main_config(all_configs)

    typst_lines = _typst_header() + ["= MQT Benchmarks\n\n"]

    # --- Coupling map constant (x = circuit qubits, one point per algorithm) ---
    typst_lines.append("== Coupling Map Constant\n\n")
    for cm_title, num_qubits in coupling_maps.items():
        config_data = _mqt_config_data_for_cm(cm_title, algorithms)
        title = f"{cm_title} ({num_qubits} qubits)"
        typst_lines.append(f"=== {title}\n\n")
        for metric in metrics:
            typst_lines.append(typst_chart_block(metric, config_data, "Circuit Qubits", title))
            typst_lines.append("\n")
        typst_lines.append("\n#pagebreak()\n\n")

    # --- Coupling map varies, averaged across algorithms (x = num qubits) ---
    typst_lines.append("== Coupling Map Varies (Averaged Across Algorithms)\n\n")
    config_data = _mqt_avg_config_data_across_cm(algorithms, coupling_maps)
    if config_data:
        typst_lines.append(typst_tables_section(config_data, metrics, "Num Qubits", main_config))
        typst_lines.append("\n#pagebreak()\n\n")
        for metric in metrics:
            typst_lines.append(typst_chart_block(metric, config_data, "Num Qubits", "All Algorithms (Averaged)"))
            typst_lines.append("\n")

    out_file = file_name + ".typ"
    with open(RESULT_PATH / out_file, "w") as f:
        f.write("\n".join(typst_lines))
    print(f"Typst chart file written to {RESULT_PATH / out_file}")


def main(file_name):
    with open(RESULT_PATH / file_name) as f:
        data = json.load(f)

    if "algorithms" in data and "coupling_maps" in data:
        main_mqt_combined(file_name)
    elif "configs" in data and "coupling_maps" in data:
        main_rand_combined(file_name)
    elif "configs" in data and "coupling_map" in data:
        main_rand_single(file_name)
    else:
        print(f"Unknown file format: {file_name}")


if __name__ == "__main__":
    file_name = sys.argv[1] if len(sys.argv) > 1 else ""
    if not file_name:
        print("Please provide the results file name")
    else:
        main(file_name)
