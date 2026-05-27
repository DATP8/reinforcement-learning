import json
import sys
from pathlib import Path

RESULT_PATH = Path(__file__).parent.parent.parent / "results"

METRIC_KEYS = ["Transpile", "Swap", "CX", "Depth", "Size", "Decomp. Depth", "Decomp. Size", "2Q Size"]

MAIN_CONFIGS =  ["grid ppo", "receding", "bipartite"]  # configs used as baselines in comparison columns

MQT_ALGOS_BLACKLIST = [
    "qwalk", 
    #"ae", 
    #"bmw_quark_cardinality", 
    #"bmw_quark_copula", 
    #"cdkm_ripple_carry_adder", 
    ##"dj", 
    #"draper_qft_adder", 
    #"full_adder", 
    ##"bv",
    ##"ghz",
    ##"ghz_dynamic", 
    ##"graphstate", 
    "grover", 
    #"half_adder", 
    #"hhl", 
    ##"hrs_cumulative_multiplier", 
    #"modular_adder", 
    #"multiplier", 
    #"qaoa", 
    #"qft",
    #"qftentangled", 
    ##"qnn",
    #"qpeexact", 
    #"qpeinexact", 
    ##"randomcircuit", 
    #"rg_qft_multiplier", 
    #"vbe_ripple_carry_adder", 
    #"vqe_real_amp", 
    ##"vqe_su2", 
    #"vqe_two_local", 
    #"wstate",
    #"shor",
    #"shors_nine_qubit_code",
    #"seven_qubit_steane_code"
]

def load_data(file_name):
    with open(RESULT_PATH / file_name) as f:
        data = json.load(f)
    return data["coupling_map"], data["num_qubits"], data["algorithms"]


def load_data_combined(file_name):
    with open(RESULT_PATH / file_name) as f:
        data = json.load(f)
    return data["coupling_maps"], data["algorithms"]


def typst_chart_block(metric, algorithms, configs):
    algo_names = list(algorithms.keys())
    n = len(algo_names)
    algo_labels = '(' + ', '.join(f'"{a}"' for a in algo_names) + ')'

    bars = []
    num_configs = len(configs)
    bar_width = 0.8 / max(1, num_configs)
    offsets = [(-0.4 + bar_width/2) + i*bar_width for i in range(num_configs)]
    for idx, config in enumerate(configs):
        xs, ys = [], []
        for i, (algo, config_metrics) in enumerate(algorithms.items()):
            val = config_metrics.get(config, {}).get(metric)
            xs.append(str(i))
            ys.append(f"{val:.4f}" if isinstance(val, float) else (str(val) if val is not None else "0"))
        label = config.replace('"', '\\"')
        bars.append(
            f'lq.bar(({", ".join(xs)}), ({", ".join(ys)}), offset: {offsets[idx]:.3f}, width: {bar_width:.3f}, label: ["{label}"]),' 
        )
    bars_str = "\n".join(bars)
    return f"""layout(size => {{
lq.diagram(
legend: (position: top + left),
title: "{metric}",
xlabel: "Algorithm",
ylabel: "{metric}",
width: size.width,
xaxis: (
  ticks: {algo_labels}.map(rotate.with(-90deg, reflow: true)).enumerate(),
  subticks: none
),
{bars_str}
)
}})"""

def typst_charts_section(algorithms, configs):
    chart_blocks = [typst_chart_block(m, algorithms, configs) for m in METRIC_KEYS]
    cells = "\n  ".join(f"[#{block}]," for block in chart_blocks)
    return f"""== Charts

#grid(
  columns: (1fr, 1fr),
  gutter: 8pt,
  {cells}
)
"""

def typst_table(metric, algorithms, configs, main_configs=None):
    main_configs = main_configs or []
    configs_non_main = [c for c in configs if c not in main_configs]
    compare_pairs = [(c, m) for c in configs_non_main for m in main_configs if m in configs]

    header_items = (
        ["[*Algorithm*]"]
        + [f"[*{c}*]" for c in configs]
        + [f"[*{c} vs {m}*]" for c, m in compare_pairs]
    )
    header_str = "  (" + ", ".join(header_items) + "),"

    rows = []
    diff_accum = [[] for _ in compare_pairs]

    for algo, config_metrics in algorithms.items():
        cells = [f"[{algo}]"]
        for config in configs:
            val = config_metrics.get(config, {}).get(metric)
            if val is None:
                cells.append("[-]")
            elif isinstance(val, float):
                cells.append(f"[{val:.4f}]")
            else:
                cells.append(f"[{val}]")

        compare_cells = []
        for i, (c, m) in enumerate(compare_pairs):
            val = config_metrics.get(c, {}).get(metric)
            main_val = config_metrics.get(m, {}).get(metric)
            if val is None or main_val is None:
                compare_cells.append("[-]")
            elif val != 0:
                diff = ((val - main_val) / val) * 100
                diff_accum[i].append(diff)
                compare_cells.append(f"[{diff:.4f}%]")
            else:
                compare_cells.append("[NAN]")

        rows.append("  (" + ", ".join(cells + compare_cells) + "),")

    if compare_pairs:
        avg_cells = ["[*Average*]"] + ["[]"] * len(configs)
        for diffs in diff_accum:
            if diffs:
                avg_cells.append(f"[*{sum(diffs) / len(diffs):.4f}%*]")
            else:
                avg_cells.append("[-]")
        rows.append("  (" + ", ".join(avg_cells) + "),")

    rows_str = "\n".join(rows)
    return f"""=== {metric}

#let list = (
{header_str}
{rows_str}
)
#let caption = [{metric}]
#data_to_table(list, caption, none)

"""


def typst_tables_section(algorithms, configs):
    mains = [c for c in MAIN_CONFIGS if c in configs]
    tables = "".join(typst_table(m, algorithms, configs, mains) for m in METRIC_KEYS)
    return f"== Tables\n\n{tables}"


def _cm_section(cm_title, num_qubits, algorithms, heading_level="="):
    """algorithms: {algo: {"circuit_qubits": N, "configs": {config: {metric: val}}}}"""
    filtered = {
        f"{k} ({v['circuit_qubits']}q)": v["configs"]
        for k, v in algorithms.items()
        if k not in MQT_ALGOS_BLACKLIST
    }
    if not filtered:
        return ""
    configs = list(next(iter(filtered.values())).keys())
    return "".join([
        f"{heading_level} {cm_title} ({num_qubits} qubits)\n\n",
        typst_tables_section(filtered, configs),
        "\n#pagebreak()\n\n",
        typst_charts_section(filtered, configs),
        "\n#pagebreak()\n\n",
    ])


def main(file_name):
    with open(RESULT_PATH / file_name) as f:
        data = json.load(f)

    lines = [
        '#import "@preview/lilaq:0.4.0" as lq\n',
        "#set page(width: 297mm, height: 210mm, flipped: true, margin: 1cm)\n",
        "#set text(size: 7pt)\n\n",
        "#let data_to_table(list, cap, place) = {\n"
        "  let header = list.first()\n"
        "  let rows = list.slice(1)\n"
        "  let col_count = header.len()\n\n"
        "  figure(\n"
        "    placement: place,\n"
        "    table(\n"
        "      stroke: none,\n"
        "      columns: (auto,) * col_count,\n"
        "      align: horizon,\n"
        "      table.hline(),\n"
        "      table.vline(x: 0),\n"
        "      table.vline(x: col_count),\n"
        "      table.header(\n"
        "        ..header.map(cell => [#cell])\n"
        "      ),\n"
        "      table.hline(),\n"
        "      ..rows.map(row => row.map(cell => [#cell])).flatten(),\n"
        "      table.hline()\n"
        "    ),\n"
        "    caption: cap\n"
        "  )\n"
        "}\n\n",
    ]

    if "coupling_maps" in data:
        coupling_maps = data["coupling_maps"]
        algorithms = data["algorithms"]
        multi = len(coupling_maps) > 1
        if multi:
            lines.append("= MQT Benchmarks\n\n")
        for cm_title, num_qubits in coupling_maps.items():
            cm_algos = {
                algo: cm_data[cm_title]
                for algo, cm_data in algorithms.items()
                if cm_title in cm_data
            }
            lines.append(_cm_section(cm_title, num_qubits, cm_algos, heading_level="==" if multi else "="))
    else:
        coupling_map = data["coupling_map"]
        num_qubits = data["num_qubits"]
        algorithms = data["algorithms"]
        lines.append(_cm_section(coupling_map, num_qubits, algorithms, heading_level="="))

    out_path = RESULT_PATH / (file_name + ".typ")
    with open(out_path, "w") as f:
        f.write("".join(lines))
    print(f"Typst file written to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1]:
        print("Usage: python mqt_table_maker.py <results_file.json>")
    else:
        main(sys.argv[1])
