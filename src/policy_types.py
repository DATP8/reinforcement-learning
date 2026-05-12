from enum import Enum, auto

import src.vibed_ppo.hybrid_extractor as vibed
from src.gym_extractor import (
    DenseDagExtractor,
    HybridExtractor,
    SimpleExtractor,
    SimpleExtractor2,
)


class ActorCriticPolicyType(Enum):
    BASIC = auto()
    BASIC_CANCEL = auto()
    SIMPLE_MLP = auto()
    HYBRID_GNN = auto()
    DENSE_GRAPH_GNN = auto()
    VIBE_GRAPH = auto()

    def get_sb3_policy(self) -> str:
        match self.name:
            case self.BASIC.name | self.SIMPLE_MLP.name:
                return "MlpPolicy"
            case _:
                return "MultiInputPolicy"

    def get_feature_extractor(self):
        match self.name:
            case self.BASIC_CANCEL.name:
                return SimpleExtractor2
            case self.SIMPLE_MLP.name:
                return SimpleExtractor
            case self.HYBRID_GNN.name:
                return HybridExtractor
            case self.DENSE_GRAPH_GNN.name:
                print("Using dense dag extractor")
                return DenseDagExtractor
            case _:
                raise ValueError("No feature extractor for policy type")

    def get_policy_kwargs(
        self, features_dim=256, gnn_hidden=64, gnn_heads=2, gnn_out=64, matrix_out=128
    ):
        match self.name:
            case self.BASIC.name:
                return None
            case self.VIBE_GRAPH.name:
                print("Using vibe extractor")
                return dict(
                    features_extractor_class=vibed.HybridExtractor,
                    features_extractor_kwargs=dict(
                        features_dim=features_dim,
                        gnn_hidden=gnn_hidden,
                        gnn_heads=gnn_heads,  # 2 for 6-qubit topology, 4 to torino
                        gnn_out=gnn_out,
                        matrix_out=matrix_out,
                    ),
                    net_arch=[256, 256],  # policy/value MLP after extractor
                )
            case _:
                return dict(
                    features_extractor_class=self.get_feature_extractor(),
                    features_extractor_kwargs=dict(features_dim=128),
                    net_arch=dict(pi=[64, 64], vf=[64, 64]),
                )
