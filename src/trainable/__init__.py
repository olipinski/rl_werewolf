from ray.rllib.models import ModelCatalog

from src.models.PaModel import ParametricActionsModel

ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)
