import logging

from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch

#torch, nn = try_import_torch()

import torch

class ParametricActionsModel(TorchModelV2):
    """
    Parametric action model used to filter out invalid action from environment
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kw):
        num_outputs = 9
        super(ParametricActionsModel, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)

        # get real obs space, discarding action mask
        real_obs_space = obs_space.original_space.spaces['array_obs']

        self.action_embed_model = FullyConnectedNetwork(real_obs_space,
                                                        action_space,
                                                        num_outputs,
                                                        model_config,
                                                        name + "_action_embed")

    def forward(self, input_dict, state, seq_lens):
        """
        Override forward pass to mask out invalid actions

               Arguments:
                   input_dict (dict): dictionary of input tensors, including "obs",
                       "obs_flat", "prev_action", "prev_reward", "is_training"
                   state (list): list of state tensors with sizes matching those
                       returned by get_initial_state + the batch dimension
                   seq_lens (Tensor): 1d tensor holding input sequence lengths

               Returns:
                   (outputs, state): The model output tensor of size
                       [BATCH, num_outputs]

        """
        obs = input_dict['obs']

        # extract action mask  [batch size, num players]
        action_mask = obs['action_mask']
        # extract original observations [batch size, obs size]
        array_obs = obs['array_obs']

        # Compute the predicted action embedding
        # size [batch size, num players * num players]
        action_embed, _ = self.action_embed_model({
            "obs": array_obs
        })

        # Mask out invalid actions (use tf.float32.min for stability)
        # size [batch size, num players * num players]
        inf_mask = torch.maximum(torch.log(action_mask), torch.tensor(torch.finfo(torch.float32).min))
        inf_mask = torch.tensor(inf_mask, dtype=torch.float32)

        masked_actions = action_embed + inf_mask

        log = logging.getLogger(__name__)
        log.warning(masked_actions)

        # return masked action embed and state
        return masked_actions, state

    def value_function(self):
        return self.action_embed_model.value_function()

    def import_from_h5(self, h5_file):
        pass
