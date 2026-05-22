import json
import sys
from pathlib import Path

RESULT_PATH = Path(__file__).parent.parent.parent / "results"

METRIC_KEYS = ["Transpile", "Swap", "CX", "Depth", "Size", "Decomp. Depth", "Decomp. Size", "2Q Size"]

MQT_ALGOS_BLACKLIST = [
    "qwalk", 
    #"ae", 
    "bmw_quark_cardinality", 
    #"bmw_quark_copula", 
    #"cdkm_ripple_carry_adder", 
    #"dj", 
    #"draper_qft_adder", 
    #"full_adder", 
    "ghz",
    "ghz_dynamic", 
    #"graphstate", 
    "grover", 
    #"half_adder", 
    #"hhl", 
    #"hrs_cumulative_multiplier", 
    #"modular_adder", 
    #"multiplier", 
    #"qaoa", 
    #"qft",
    #"qftentangled", 
    "qnn",
    #"qpeexact", 
    #"qpeinexact", 
    #"randomcircuit", 
    #"rg_qft_multiplier", 
    #"vbe_ripple_carry_adder", 
    "vqe_real_amp", 
    "vqe_su2", 
    #"vqe_two_local", 
    "wstate",
    "shor",
    "shors_nine_qubit_code",
    "seven_qubit_steane_code"
]

def load_data(file_name):
    with open(RESULT_PATH / file_name) as f:
        data = json.load(f)
    return data["coupling_map"], data["num_qubits"], data["algorithms"]


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

def typst_table(metric, algorithms, configs, main_config = None):
    header_cols = ", ".join(f"[*{c}*]" for c in configs)
    if main_config is not None:
        configs_no_main = list((set(configs) - {main_config}))
        compared_header_cols = ", ".join(f"[*{c} - {main_config} / {c}*]" for c in configs_no_main)
        header = f"table.header([*Algorithm*], {header_cols}, {compared_header_cols}),"
        col_widths = ", ".join(["auto"] * (1 + len(configs) + len(configs_no_main)))
    else:
        header = f"table.header([*Algorithm*], {header_cols}),"
        col_widths = ", ".join(["auto"] * (1 + len(configs)))

    rows = []
    for algo, config_metrics in algorithms.items():
        cells = [f"[{algo}]"]
        compare_cells = []
        main_val = config_metrics.get(main_config, {}).get(metric)

        for config in configs:
            val = config_metrics.get(config, {}).get(metric)

            if val is None:
                cells.append("[-]")
            elif isinstance(val, float):
                cells.append(f"[{val:.4f}]")
            else:
                cells.append(f"[{val}]")

            if main_config == config:
                continue

            if main_config is not None:    
                if val is None or main_val is None:
                    compare_cells.append(f"[-]")
                elif val != 0: 
                    compare_cells.append(f"[{((val-main_val)/val)*100:.4f}%]")
                else:
                    compare_cells.append(f"[NAN]")
           
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


def typst_tables_section(algorithms, configs):
    tables = "".join(typst_table(m, algorithms, configs, "basic ppo") for m in METRIC_KEYS)
    return f"== Tables\n\n{tables}"


def main(file_name):
    coupling_map, num_qubits, algorithms = load_data(file_name)

    # Flatten to {"{algo} ({n}q)": {config: metrics}} and drop blacklisted entries
    filtered_algorithms = {
        f"{k} ({v['circuit_qubits']}q)": v["configs"]
        for k, v in algorithms.items()
        if k not in MQT_ALGOS_BLACKLIST
    }
    configs = list(next(iter(filtered_algorithms.values())).keys())

    lines = [
        '#import "@preview/lilaq:0.4.0" as lq\n',
        "#set page(width: 297mm, height: 210mm, flipped: true, margin: 1cm)\n",
        "#set text(size: 7pt)\n\n",
        f"= {coupling_map} ({num_qubits} qubits)\n\n",
        typst_tables_section(filtered_algorithms, configs),
        "\n#pagebreak()\n\n",
        typst_charts_section(filtered_algorithms, configs),
        "\n#pagebreak()\n\n",
    ]

    out_path = RESULT_PATH / (file_name + ".typ")
    with open(out_path, "w") as f:
        f.write("".join(lines))
    print(f"Typst file written to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1]:
        print("Usage: python mqt_table_maker.py <results_file.json>")
    else:
        main(sys.argv[1])
