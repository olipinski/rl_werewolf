from src.policies.RevengeTarget import RevengeTarget

import logging

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

from gym_ww.callbacks import CustomCallbacks
from gym_ww.envs import CONFIGS
from src.models import ParametricActionsModel
from src.other.custom_utils import trial_name_creator
from gym_ww.wrappers import EvaluationWrapper

from src.utils import Params

_ = Params()


def mapping_static(agent_id):
    if "wolf" in agent_id:
        return "wolf_p_static"
    elif "vil" in agent_id:
        return "vill_p"
    else:
        raise NotImplementedError(f"Policy for role {agent_id} not implemented")


if __name__ == '__main__':
    _ = ParametricActionsModel
    ray.init(local_mode=Params.debug, logging_level=logging.DEBUG)

    env_configs = CONFIGS

    env = EvaluationWrapper(env_configs)

    # define policies
    vill_p = (PPOTFPolicy, env.observation_space, env.action_space, {})
    ww_p = (RevengeTarget, env.observation_space, env.action_space, {})

    policies = dict(
        wolf_p_static=ww_p,
        wolf_p=vill_p,
        vill_p=vill_p,
    )

    configs = {
        "env": EvaluationWrapper,
        "env_config": env_configs,
        "framework": "tfe",
        "eager_tracing": False,
        "num_workers": Params.n_workers,
        "num_gpus": Params.n_gpus,
        "batch_mode": "complete_episodes",
        "train_batch_size": 400,
        "rollout_fragment_length": 300,

        # PPO parameter taken from OpenAi paper
        "lr": 3e-4,
        "lambda": .95,
        "gamma": .998,
        "entropy_coeff": 0.01,
        "clip_param": 0.2,
        "use_critic": True,
        "use_gae": True,
        "grad_clip": 5,
        "num_sgd_iter": 10,

        "callbacks": CustomCallbacks,

        "model": {
            "use_lstm": False,
            "custom_model": "pa_model",  # using custom parametric action model
        },
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": mapping_static,
            "policies_to_train": "vill_p"

        },

    }

    analysis = tune.run(
        PPOTrainer,
        local_dir=Params.RAY_DIR,
        config=configs,
        trial_name_creator=trial_name_creator,
        checkpoint_freq=Params.checkpoint_freq,
        keep_checkpoints_num=Params.max_checkpoint_keep,
        resume=Params.resume_training
    )
