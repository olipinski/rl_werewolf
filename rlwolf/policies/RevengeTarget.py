from typing import Optional, Tuple, Dict, Union, List

from ray.rllib import Policy, SampleBatch
from ray.rllib.utils.typing import ModelGradients, TensorType

from rlwolf.policies.utils import revenge_target


class RevengeTarget(Policy):
    """Hand-coded policy that returns the id of an agent who chose the
    current one in the last run, if none then random """

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        self.to_kill_list = []

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        """Compute actions on a batch of observations."""

        observations = [elem.get('obs', {}) for elem in info_batch]
        signal_conf = self.config['env_config']['signal_length'], self.config['env_config']['signal_range']

        actions, self.to_kill_list = revenge_target(self.action_space, observations, self.to_kill_list, signal_conf)

        return actions, [], {}

    def get_weights(self):
        return None

    def set_weights(self, weights):
        pass

    def compute_log_likelihoods(self, actions: Union[List[TensorType], TensorType],
                                obs_batch: Union[List[TensorType], TensorType],
                                state_batches: Optional[List[TensorType]] = None,
                                prev_action_batch: Optional[Union[List[TensorType],
                                                                  TensorType]] = None,
                                prev_reward_batch: Optional[Union[List[TensorType],
                                                                  TensorType]] = None,
                                actions_normalized: bool = True) -> TensorType:
        pass

    def compute_gradients(self, postprocessed_batch: SampleBatch) -> \
            Tuple[ModelGradients, Dict[str, TensorType]]:
        pass

    def apply_gradients(self, gradients: ModelGradients) -> None:
        pass

    def load_batch_into_buffer(self, batch: SampleBatch, buffer_index: int = 0) -> int:
        pass

    def get_num_samples_loaded_into_buffer(self, buffer_index: int = 0) -> int:
        pass

    def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0):
        pass

    def export_model(self, export_dir: str, onnx: Optional[int] = None) -> None:
        pass

    def import_model_from_h5(self, import_file: str) -> None:
        pass

    def export_checkpoint(self, export_dir: str) -> None:
        pass

    def learn_on_batch(self, samples):
        """No learning."""
        return {}
