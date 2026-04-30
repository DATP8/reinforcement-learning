import json
import sys
from pathlib import Path

RESULT_PATH = Path(__file__).parent.parent.parent / "results"


def load_data(file_name):
    with open(RESULT_PATH / file_name) as f:
        data = json.load(f)
    # data: {coupling_map: {config: {qubits: {metric: {mean, ci}}}}}
    all_metrics = set()
    # We'll generate charts for each coupling_map separately
    coupling_map_data = {}
    for coupling_map, configs in data.items():
        config_data = {}
        for config, qubit_entries in configs.items():
            points = []
            for qubit_str, metrics in qubit_entries.items():
                try:
                    num_qubits = int(qubit_str)
                except ValueError:
                    continue
                metric_means = {}
                for metric, val in metrics.items():
                    metric_means[metric] = val["mean"]
                    all_metrics.add(metric)
                points.append((num_qubits, metric_means))
            points.sort(key=lambda x: x[0])
            config_data[config] = points
        coupling_map_data[coupling_map] = config_data
    return coupling_map_data, sorted(all_metrics)

def typst_chart_block(metric, config_data, coupling_map):
    lines = []
    lines.append(f"\
#block(\n\
    width: 100%,\n\
    height: 10cm, \n\
    layout(size => {{ \n\
        lq.diagram(\n\
            xlabel: \"Qubits\",\n\
            ylabel: \"{metric}\",\n\
            width: size.width,\n\
            height: size.height,")

    for config, points in config_data.items():
        xs = [str(q) for q, _ in points]
        ys = [str(round(m.get(metric, 0), 4)) for _, m in points]
        lines.append(f"\
            lq.plot(\n\
                ({', '.join(xs)}),\n\
                ({', '.join(ys)}),\n\
                label: \"{config}\",\n\
            ),")

    lines.append("\
        )\n\
    })\n\
)\n")
    return "\n".join(lines)

def main(file_name):
    coupling_map_data, metrics = load_data(file_name)

    typst_lines = [
        "#import \"@preview/lilaq:0.4.0\" as lq\n",
        "#set page(width: 22cm, height: 30cm)\n"
    ]
    
    for coupling_map, config_data in coupling_map_data.items():
        for metric in metrics:
            typst_lines.append(typst_chart_block(metric, config_data, coupling_map))
            typst_lines.append("\n")

    new_file_name = file_name +  ".typ"
    with open(RESULT_PATH / new_file_name, "w") as f:
        f.write("\n".join(typst_lines))
    print(f"Typst chart file written to {RESULT_PATH}")

if __name__ == "__main__":
    file_name = sys.argv[1]
    if file_name == "":
        print("Please give the file in results which you want to have made into charts")
    else :
        main(file_name)