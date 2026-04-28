from enum import Enum, auto


class ActorCriticPolicyType(Enum):
    BASIC = auto()
    SIMPLE_GNN = auto()
    HYBRID_GNN = auto()

    def get_sb3_policy(self) -> str:
        match self.name:
            case self.BASIC.name:
                return "MlpPolicy"
            case _:
                return "MultiInputPolicy"
