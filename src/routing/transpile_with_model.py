from typing import override
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes import TrivialLayout, VF2Layout, SabreLayout, SabreSwap
from itertools import product
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.routing.bwas_routing import BWASRouting

def collect_metrics(routed_circuit, transpile_time):

    ops = routed_circuit.count_ops()

    swaps = ops.get("swap", 0)
    cx = ops.get("cx", 0)

    metrics = {
        "transpile_time": transpile_time,
        "swap_count": swaps,
        "cx_count": cx,
        "two_qubit_total": swaps * 3 + cx,
        "depth": routed_circuit.depth(),
        "size": routed_circuit.size(),
        "two_qubit_depth": routed_circuit.depth(
            filter_function=lambda inst: inst.operation.name in ["cx","swap"]
        )
    }

    return metrics

class RLInitialLayout(AnalysisPass):

    def __init__(self, coupling_map, model):
        super().__init__()
        self.coupling_map = coupling_map
        self.model = model

    @override
    def run(self, dag):

        candidates = vf2_candidate_layouts(dag, self.coupling_map)

        best_layout = None
        best_score = float("inf")

        for layout in candidates:

            tensor = circuit_to_tensor(dag, layout)
            score = self.model.predict(tensor)

            if score < best_score:
                best_layout = layout
                best_score = score

        self.property_set["layout"] = best_layout

class RLForwardBackward(TransformationPass):
    @override
    def run(self, dag):
        pass


class RLSwapRouter(TransformationPass):

    def __init__(self, coupling_map):
        super().__init__()
        self.coupling_map = coupling_map 
        self.coupling_map.make_symmetric()
        self.model_path = "models/davi/difficulty9_iteration6500.pt"
        self.horizon = 100
        self.device = "cpu"

        self.bwas = BWASRouting(
            coupling_map=self.coupling_map, 
            horizon=self.horizon, 
            model_path=self.model_path, 
            device=self.device
        )

    @override
    def run(self, dag):
        return self.bwas.run(dag)

def build_pass_manager(init, fb, final, coupling_map):

    passes = []

    # ---------- initial layout ----------
    if init == "trivial":
        passes.append(TrivialLayout(coupling_map))

    elif init == "qiskit":
        passes.append(VF2Layout(coupling_map))

    # elif init == "rl":
    #     passes.append(RLInitialLayout(coupling_map, model))

    # ---------- forward/backward ----------
    if fb == "sabre":
        passes.append(SabreLayout(coupling_map))

    # elif fb == "rl":
    #     passes.append(RLForwardBackward(coupling_map, model))

    elif fb == "none":
        pass

    # ---------- final routing ----------
    if final == "sabre":
        passes.append(SabreSwap(coupling_map))

    elif final == "rl":
        passes.append(RLSwapRouter(coupling_map))

    return PassManager(passes)

def plot_heatmap(metric):
    pivot = df.pivot_table(
        index="initial",
        columns="final_router",
        values=metric,
        aggfunc="mean"
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="viridis"
    )

    plt.title(metric)
    plt.show()

def plot_fb_heatmaps(df, metric):

    fbs = df["forward_backward"].unique()

    fig, axes = plt.subplots(1, len(fbs), figsize=(5 * len(fbs), 4))

    for i, fb in enumerate(fbs):

        sub = df[df["forward_backward"] == fb]

        pivot = sub.pivot_table(
            index="initial",
            columns="final_router",
            values=metric
        )

        if pivot.empty:
            return

        sns.heatmap(
        pivot,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            ax=axes[i]
        )

        axes[i].set_title(f"fb={fb}")

    fig.suptitle(metric)
    plt.show()

if __name__ == "__main__":
    # Create combinations
    # initial_layouts = ["trivial", "qiskit", "rl"]
    # forward_backward = ["none", "sabre", "rl"]
    # final_routers = ["sabre", "rl"]

    initial_layouts = ["qiskit"]
    forward_backward = ["none", "sabre"]
    final_routers = ["sabre", "rl"]
    
    configs = list(product(
        initial_layouts,
        forward_backward,
        final_routers
    ))

    qc = QuantumCircuit(6)
    qc.h(0)
    qc.cx(1, 3)
    qc.rz(0.5, 2)
    qc.cz(2, 3)
    qc.h(1)
    qc.cx(0, 2)

    coupling_map = CouplingMap([[0,1],[1,2],[2,3],[3,4],[4,5]])
    
    # Run each combination
    rows = []
    for init, fb, final in configs:

        print("Running with:", init, fb, final)
    
        pm = build_pass_manager(
            init,
            fb,
            final,
            coupling_map,
        )
    
        start = time.perf_counter()
        routed = pm.run(qc)
        end = time.perf_counter()

        print(routed)
    
        transpile_time = end - start
    
        metrics = collect_metrics(
            routed,
            transpile_time
        )
    
        row = {
            "initial": init,
            "forward_backward": fb,
            "final_router": final,
            **metrics
        }
    
        rows.append(row)

    print("Rows:", rows)
    
    # Collect and visualize results
    df = pd.DataFrame(rows)
    df["pipeline"] = (
        df["initial"] +
        " → " +
        df["forward_backward"] +
        " → " +
        df["final_router"])


    # metrics = [
    #     "swap_count",
    #     "two_qubit_total",
    #     "depth",
    #     "two_qubit_depth",
    #     "transpile_time"
    # ]
    
    # for m in metrics:
    #     plot_heatmap(m)   

    baseline = df[
        (df.initial == "qiskit") &
        (df.forward_backward == "sabre") &
        (df.final_router == "sabre")
    ].iloc[0]
    
    df["swap_improvement"] = df["swap_count"] / baseline["swap_count"]
    df["depth_improvement"] = df["depth"] / baseline["depth"]


    plot_fb_heatmaps(df, "swap_improvement")
