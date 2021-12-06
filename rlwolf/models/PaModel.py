from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN, FLOAT_MAX

torch, nn = try_import_torch()


class ParametricActionsModel(TorchModelV2, nn.Module):
    """
    Parametric action model used to filter out invalid action from environment
    """

    # noinspection PyUnusedLocal
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name):
        super(ParametricActionsModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        # This is a hack. TODO Figure out how to fix the num_outputs to have the correct value
        self.num_outputs = sum(self.action_space.nvec)

        # get real obs space, discarding action mask
        real_obs_space = obs_space.original_space.spaces['array_obs']

        self.action_embed_model = FullyConnectedNetwork(real_obs_space,
                                                        action_space,
                                                        self.num_outputs,
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

        # extract action mask  [batch size, num players]
        action_mask = input_dict['obs']['action_mask']
        # extract original observations [batch size, obs size]
        array_obs = input_dict['obs']['array_obs']

        # Compute the predicted action embedding
        # size [batch size, num players * num players]
        action_embed, _ = self.action_embed_model({
            "obs": array_obs
        })

        # Mask out invalid actions
        # size [batch size, num players * num players]
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        masked_actions = action_embed + inf_mask

        return masked_actions, state

    def value_function(self):
        return self.action_embed_model.value_function()

    def import_from_h5(self, h5_file):
        pass
