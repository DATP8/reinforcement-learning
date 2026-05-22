import json
import sys
from pathlib import Path

RESULT_PATH = Path(__file__).parent.parent.parent / "results"


def load_data(file_name):
    with open(RESULT_PATH / file_name) as f:
        data = json.load(f)
    # data: {coupling_map, num_qubits, configs: {config: {gate_count: {metric: {mean, ci}}}}}
    coupling_map = data["coupling_map"]
    num_qubits = data["num_qubits"]
    all_metrics = set()
    config_data = {}
    for config, gate_entries in data["configs"].items():
        points = []
        for gate_str, metrics in gate_entries.items():
            gate_count = int(gate_str)
            metric_means = {}
            for metric, val in metrics.items():
                metric_means[metric] = (val["mean"], val["ci"])
                all_metrics.add(metric)
            points.append((gate_count, metric_means))
        points.sort(key=lambda x: x[0])
        config_data[config] = points
    return coupling_map, num_qubits, config_data, sorted(all_metrics)


def typst_chart_block(metric, config_data, coupling_map):
    lines = []
    lines.append(
        f'\
#block(\n\
    width: 100%,\n\
    height: 10cm, \n\
    layout(size => {{ \n\
        lq.diagram(\n\
            title: "{coupling_map}",\n\
            xlabel: "Gate Count",\n\
            ylabel: "{metric}",\n\
            width: size.width,\n\
            height: size.height,'
    )

    for config, points in config_data.items():
        xs = [str(q) for q, _ in points]
        ys = [str(round(m.get(metric, (0, 0))[0], 4)) for _, m in points]
        yerrs = [str(round(m.get(metric, (0, 0))[1], 4)) for _, m in points]
        trailing = "," if len(xs) == 1 else ""
        lines.append(
            f'\
            lq.plot(\n\
                ({", ".join(xs)}{trailing}),\n\
                ({", ".join(ys)}{trailing}),\n\
                label: "{config}",\n\
                yerr: ({", ".join(yerrs)}{trailing}),\n\
            ),'
        )

    lines.append(
        "\
        )\n\
    })\n\
)\\ \n \\"
    )
    return "\n".join(lines)


def main(file_name):
    coupling_map, num_qubits, config_data, metrics = load_data(file_name)

    title = f"{coupling_map} ({num_qubits} qubits)"
    typst_lines = [
        '#import "@preview/lilaq:0.4.0" as lq\n',
        "#set page(width: 22cm, height: 30cm)\n",
    ]

    for metric in metrics:
        typst_lines.append(typst_chart_block(metric, config_data, title))
        typst_lines.append("\n")

    new_file_name = file_name + ".typ"
    with open(RESULT_PATH / new_file_name, "w") as f:
        f.write("\n".join(typst_lines))
    print(f"Typst chart file written to {RESULT_PATH}")


if __name__ == "__main__":
    file_name = sys.argv[1]
    if file_name == "":
        print("Please give the file in results which you want to have made into charts")
    else:
        main(file_name)
