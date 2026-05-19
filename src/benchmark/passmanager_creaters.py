from src.ppo_util import make_env
from sb3_contrib import MaskablePPO
from src.routing.cnot_swap_cancel import CNOTSwapCancelation
from src.routing.agentic_rl_routing_pass import AgenticRlRoutingPass
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

        return PassManager([
            AIRouting(
                coupling_map=coupling_map,
                optimization_level=self.op_level,
                layout_mode="KEEP",
            ), 
            CNOTSwapCancelation()
        ])


class SabreBuilder(BuilderWithLayout):
    def build(self, coupling_map):
        self.set_layout_pass(coupling_map)
        return PassManager(
            [self.layout_pass, ApplyLayout(), SabreSwap(coupling_map=coupling_map), CNOTSwapCancelation()]
        )


class QiskitTranspiler(Builder):
    def __init__(self, op_level):
        self.op_level = op_level

    def build(self, coupling_map):
        return generate_preset_pass_manager(
            optimization_level=self.op_level, coupling_map=coupling_map
        )



class PPOBuilder(BuilderWithLayout):
    def __init__(self, 
                 num_active_swaps: int, 
                 horizon, 
                 initial_difficulty,
                 max_difficulty,
                 diff_slope,
                 layout_exponent,
                 clear_visited_on_stuck,
                 policy_type,
                 seed,
                 model_path,
                 use_sabre_layout):
        super().__init__(use_sabre_layout)
        self.num_active_swaps = num_active_swaps
        self.horizon = horizon
        self.initial_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.diff_slope = diff_slope
        self.layout_exponent = layout_exponent
        self.clear_visited_on_stuck = clear_visited_on_stuck
        self.policy_type = policy_type
        self.seed = seed
        self.model_path = model_path

    def build(self, coupling_map):
        self.set_layout_pass(coupling_map)
        ppo_env = make_env(
            coupling_map,
            num_active_swaps=self.num_active_swaps,
            horizon=self.horizon,
            initial_difficulty=self.initial_difficulty,
            max_difficulty=self.max_difficulty,
            diff_slope=self.diff_slope,
            layout_exponent=self.layout_exponent,
            policy_type=self.policy_type,
            clear_visited_on_stuck=self.clear_visited_on_stuck
        )
        ppo_model = MaskablePPO.load(self.model_path, env=ppo_env, seed=self.seed, device="cpu")

        return PassManager([self.layout_pass, ApplyLayout(), AgenticRlRoutingPass(ppo_model, coupling_map), CNOTSwapCancelation()])