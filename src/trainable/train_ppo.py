from policies.RandomTargetUnite import RandomTargetUnite
from policies.RevengeTarget import RevengeTarget
from utils import Params

from gym_ww import ww, vil

_ = Params()

import logging

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

from callbacks import on_episode_end
from models import ParametricActionsModel
from other.custom_utils import trial_name_creator
from policies.RandomTarget import RandomTarget
from wrappers import EvaluationWrapper


def mapping_static(agent_id):
    if "wolf" in agent_id:
        return "wolf_p_static"
    elif "vil" in agent_id:
        return "vill_p"
    else:
        raise NotImplementedError(f"Policy for role {agent_id} not implemented")


if __name__ == '__main__':

    _ = ParametricActionsModel
    ray.init(local_mode=Params.debug, logging_level=logging.INFO, num_gpus=Params.n_gpus)

    CONFIGS = dict(

        existing_roles=[ww, vil],  # list of existing roles [werewolf, villanger]
        num_players=Params.num_player,
        penalties=dict(
            # penalty dictionary
            # penalty to give for each day that has passed
            day=0,
            # when a player dies
            death=-5,
            # victory
            victory=+25,
            # lost
            lost=-25,
            # penalty used for punishing votes that are not chosen during execution/kill.
            # If agent1 outputs [4,2,3,1,0] as a target list and agent2 get executed then agent1 get
            # a penalty equal to index_of(agent2,targets)*penalty
            trg_accord=-1,

            # Penalty for a wasted voting round, i.e. the agents didn't reach the required threshold of X%
            wasted_round=-5,

        ),
        max_days=Params.max_days,

        # signal is used in the communication phase to signal other agents about intentions
        # the length concerns the dimension of the signal while the components is the range of values it can fall into
        # a range value of 2 is equal to binary variable
        signal_length=Params.signal_length,
        signal_range=Params.signal_range,

        # Multi communication round config
        num_rounds=Params.num_rounds,
        req_threshold=Params.req_threshold,

        # Callback and file configs for logging
        log_match_file=Params.log_match_file,
        episode_file=Params.episode_file,
        log_step=Params.log_step,
        alternating=Params.alternating,

    )

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
        "eager": True,
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

        # todo: remove this [here](https://github.com/ray-project/ray/issues/7991)
        #"simple_optimizer": True,

        "callbacks": {
            "on_episode_end": on_episode_end,
        },

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
