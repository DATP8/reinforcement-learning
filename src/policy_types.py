from enum import Enum, auto

from src.gym_extractor import HybridExtractor, SimpleExtractor, DenseDagExtractor


class ActorCriticPolicyType(Enum):
    BASIC = auto()
    SIMPLE_MLP = auto()
    HYBRID_GNN = auto()
    DENSE_GRAPH_GNN = auto()

    def get_sb3_policy(self) -> str:
        match self.name:
            case self.BASIC.name | self.SIMPLE_MLP.name:
                return "MlpPolicy"
            case _:
                return "MultiInputPolicy"

    def get_feature_extractor(self):
        match self.name:
            case self.SIMPLE_MLP.name:
                return SimpleExtractor
            case self.HYBRID_GNN.name:
                return HybridExtractor
            case self.DENSE_GRAPH_GNN.name:
                return DenseDagExtractor

    def get_policy_kwargs(self):
        if self.name == self.BASIC.name:
            return None
        return dict(
            features_extractor_class=self.get_feature_extractor(),
            features_extractor_kwargs=dict(features_dim=128),
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        )
