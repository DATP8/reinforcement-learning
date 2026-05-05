from qiskit import generate_preset_pass_manager
from qiskit.transpiler import PassManager
from qiskit.transpiler.basepasses import BasePass
from qiskit_ibm_transpiler.ai.routing import AIRouting
from qiskit.transpiler.passes import ApplyLayout, SabreLayout, SabreSwap, TrivialLayout


class Builder:
    def build(self, coupling_map):
        raise NotImplementedError


class BuilderWithLayout:
    def __init__(self, use_sabre_layout=False):
        self.use_sabre_layout = use_sabre_layout
        self.layout_pass: BasePass

    def set_layout_pass(self, coupling_map):
        if self.use_sabre_layout:
            self.layout_pass = SabreLayout(coupling_map=coupling_map, skip_routing=True)
        else:
            self.layout_pass = TrivialLayout(coupling_map)


class IbmRlBuilder(BuilderWithLayout):
    def __init__(self, op_level=1, use_sabre_layout=False):
        super().__init__(use_sabre_layout)
        self.op_level = op_level

    def build(self, coupling_map):
        self.set_layout_pass(coupling_map)
        original_run = AIRouting.run

        def patched_run(self, dag):
            """
            For some reason AIRouting overwrites layout property even with keep
            """
            saved_layout = self.property_set.get("layout", None)
            result_dag = original_run(self, dag)
            if self.layout_mode == "KEEP" and saved_layout is not None:
                self.property_set["layout"] = saved_layout

            return result_dag

        AIRouting.run = patched_run

        return PassManager(
            AIRouting(
                coupling_map=coupling_map,
                optimization_level=self.op_level,
                layout_mode="KEEP",
            )
        )


class SabreBuilder(BuilderWithLayout):
    def build(self, coupling_map):
        self.set_layout_pass(coupling_map)
        return PassManager(
            [self.layout_pass, ApplyLayout(), SabreSwap(coupling_map=coupling_map)]
        )


class QiskitTranspiler(Builder):
    def __init__(self, op_level):
        self.op_level = op_level

    def build(self, coupling_map):
        return generate_preset_pass_manager(
            optimization_level=self.op_level, coupling_map=coupling_map
        )
