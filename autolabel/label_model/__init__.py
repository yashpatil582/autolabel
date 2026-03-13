from autolabel.label_model.base import BaseLabelModel
from autolabel.label_model.majority_vote import MajorityVoteLabelModel
from autolabel.label_model.weighted_vote import WeightedVoteLabelModel
from autolabel.label_model.generative import GenerativeLabelModel

__all__ = [
    "BaseLabelModel",
    "MajorityVoteLabelModel",
    "WeightedVoteLabelModel",
    "GenerativeLabelModel",
    "get_label_model",
]


def get_label_model(name: str = "generative") -> BaseLabelModel:
    """Factory. Options: 'majority', 'weighted', 'generative'."""
    models = {
        "majority": MajorityVoteLabelModel,
        "weighted": WeightedVoteLabelModel,
        "generative": GenerativeLabelModel,
    }
    if name not in models:
        raise ValueError(f"Unknown label model '{name}'. Options: {sorted(models)}")
    return models[name]()
