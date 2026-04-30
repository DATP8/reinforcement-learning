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


class IbmRlBuilder(Builder):
    def __init__(self, op_level=1, layout_mode="KEEP"):
        self.op_level=op_level
        self.layout_mode=layout_mode

    def build(self, coupling_map):
        return PassManager(AIRouting(
            coupling_map=coupling_map,
            optimization_level=self.op_level,
            layout_mode=self.layout_mode,
        ))

class SabreBuilder(BuilderWithLayout):

    def build(self, coupling_map):
        self.set_layout_pass(coupling_map)
        return PassManager([self.layout_pass, ApplyLayout(), SabreSwap(coupling_map=coupling_map)])

class QiskitTranspiler(Builder):
    def __init__(self, op_level):
        self.op_level = op_level

    def build(self, coupling_map):
        return generate_preset_pass_manager(
            optimization_level=self.op_level, coupling_map=coupling_map
        )