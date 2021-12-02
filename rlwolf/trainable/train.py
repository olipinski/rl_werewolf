import logging
import os.path
import sys
import uuid

import ray
# noinspection PyPackageRequirements
from absl import flags
from ray import tune
from ray.rllib.agents.ppo import APPOTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.appo_torch_policy import AsyncPPOTorchPolicy
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_torch

from rlwolf.gym_environment import ww, vil
from rlwolf.gym_environment.callbacks import CustomCallbacks
from rlwolf.gym_environment.wrappers import EvaluationWrapper
from rlwolf.models.PaModel import ParametricActionsModel
from rlwolf.other.custom_utils import trial_name_creator
from rlwolf.policies.RandomTarget import RandomTarget
from rlwolf.policies.RandomTargetUnite import RandomTargetUnite
from rlwolf.policies.RevengeTarget import RevengeTarget
from rlwolf.utils.dir_manage import initialize_dirs, empty_dirs

# Register our custom model
ModelCatalog.register_custom_model("pa_model", ParametricActionsModel)

# Import torch through Ray
torch, nn = try_import_torch()

FLAGS = flags.FLAGS


# Static Methods
def mapping_static(agent_id):
    if "wolf" in agent_id:
        return "wolf_p_static"
    elif "vil" in agent_id:
        return "vill_p"
    else:
        raise NotImplementedError(f"Policy for role {agent_id} not implemented")


def mapping_dynamic(agent_id):
    if "wolf" in agent_id:
        return "wolf_p"
    elif "vil" in agent_id:
        return "vill_p"
    else:
        raise NotImplementedError(f"Policy for role {agent_id} not implemented")


# Default dirs are computed here
unid = str(uuid.uuid4())[:8]

wdir = os.getcwd().split("rlwolf")[0]
srcdir = os.path.join(wdir, "rlwolf")
logdir = os.path.join(wdir, "log_dir")
raydir = os.path.join(logdir, "ray_results")
gamelogdir = os.path.join(logdir, "match_log")
evaldir = os.path.join(logdir, "eval")
episodefile = os.path.join(evaldir, "episode.pkl")
logmatchfile = os.path.join(gamelogdir, f"{unid}_log.log")

# FLAG Definitions

# Directory flags
flags.DEFINE_string("unique_id", unid, "Unique ID for this run."
                                       "Default will be generated with UUID4.")
flags.DEFINE_string("ww_working_dir", wdir, "Working directory")
flags.DEFINE_string("ww_src_dir", srcdir, "Source directory for the wolf")
flags.DEFINE_string("ww_log_dir", logdir, "Log directory")
flags.DEFINE_string("ww_ray_dir", raydir, "Ray Directory")
flags.DEFINE_string("ww_game_log_dir", gamelogdir, "Game log directory")
flags.DEFINE_string("ww_eval_dir", evaldir, "Evaluation files directory")
flags.DEFINE_string("ww_episode_file", episodefile, "Episode file location")
flags.DEFINE_string("ww_log_match_file", logmatchfile, "Location of the match log file")

# Performance flags
flags.DEFINE_boolean("debug", False, "Enable debug mode.")
flags.DEFINE_integer("n_cpus", torch.multiprocessing.cpu_count(), "How many CPUs to use. Default is all detected.")
flags.DEFINE_integer("n_gpus", torch.cuda.device_count(), "How many GPUs to use. Default is all detected.")
flags.DEFINE_integer("n_workers", 10, "How many workers to use.")

# Evaluation flags
flags.DEFINE_integer("checkpoint_freq", 50, "Checkpointing frequency.")
flags.DEFINE_integer("log_step", 500, "Log step size.")
flags.DEFINE_integer("max_checkpoint_keep", 10, "Number of checkpoints to keep.")
flags.DEFINE_boolean("resume_training", False, "Whether to resum training.")
flags.DEFINE_boolean("alternating", False, "???")

# Environmental flags
flags.DEFINE_integer("n_players", 9, "How many players to setup in the game.")
flags.DEFINE_integer("max_days", 10, "The maximum number of in-game days that the game can have."
                                     "The match will end forcefully if this is exceeded.")
flags.DEFINE_integer("message_length", 0, "The maximum length of the message that the agents can send to each other.")
flags.DEFINE_integer("vocab_size", 9, "The size of the vocabulary that the agents can use.")
flags.DEFINE_integer("n_voting_rounds", 3, "The maximum number of voting rounds before the agents agree")

# Policy flags
flags.DEFINE_enum("vil_policy", "APPO", ["APPO", "PPO"], "Which policy to use for training the villagers."
                                                         "Valid options are APPO and PPO.")
flags.DEFINE_enum("ww_policy", "random", ["random", "revenge", "unite"], "Which policy to use for the werewolves."
                                                                         "Valid options are random, revenge and unite")

# Ray Params
flags.DEFINE_integer("batch_size", 500, "The batch size for Ray Tune to use.")
flags.DEFINE_integer("rollout_length", 100, "The rollout fragment length for Ray Tune to use.")

# QOL Flags
flags.DEFINE_boolean("help", False, "Display the help message.")

# Main method
if __name__ == '__main__':

    FLAGS(sys.argv)

    flag_dict = FLAGS.flag_values_dict()

    if FLAGS.help:
        print(FLAGS)
        exit(0)

    print(flag_dict)

    if not FLAGS.resume_training:
        empty_dirs([FLAGS.ww_log_dir])

    initialize_dirs([FLAGS.ww_working_dir, FLAGS.ww_src_dir, FLAGS.ww_log_dir,
                     FLAGS.ww_ray_dir, FLAGS.ww_game_log_dir, FLAGS.ww_eval_dir])

    ray.init(local_mode=FLAGS.debug,
             logging_level=logging.DEBUG if FLAGS.debug else logging.INFO,
             num_gpus=FLAGS.n_gpus)

    env_config = dict(
        existing_roles=[ww, vil],  # list of existing roles [werewolf, villager]
        num_players=FLAGS.n_players,
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
        ),

        max_days=FLAGS.max_days,

        # signal is used in the communication phase to signal other agents about intentions
        # the length concerns the dimension of the signal while the components is the range of values it can fall into
        # a range value of 2 is equal to binary variable
        signal_length=FLAGS.message_length,
        signal_range=FLAGS.vocab_size,

        # Add voting round count parameter
        n_voting_rounds=FLAGS.n_voting_rounds,

        # Callback and file configs for logging
        ww_log_match_file=FLAGS.ww_log_match_file,
        episode_file=FLAGS.ww_episode_file,
        log_step=FLAGS.log_step,
        alternating=FLAGS.alternating,
    )

    env = EvaluationWrapper(env_config)

    # This will set the specific params for both PPO and APPO
    # which are then updated with a call to the temp_conf_dict dictionary
    if FLAGS.vil_policy == "APPO":
        policy = AsyncPPOTorchPolicy
        reuse_actors = True
        trainer = APPOTrainer
        temp_conf_dict = {"lr": 3e-4, "lambda": 0.95, "gamma": 0.998, "num_sgd_iter": 1, "replay_proportion": 0.05}
    elif FLAGS.vil_policy == "PPO":
        policy = PPOTorchPolicy
        reuse_actors = False
        trainer = PPOTrainer
        temp_conf_dict = {"lr": 3e-4, "lambda": .95, "gamma": .998, "entropy_coeff": 0.01, "clip_param": 0.2,
                          "use_critic": True, "use_gae": True, "grad_clip": 5, "num_sgd_iter": 10, }
    else:
        # Default case, though this should never run
        policy = None
        reuse_actors = None
        trainer = None
        temp_conf_dict = None

    if FLAGS.ww_policy == "random":
        # Default for APPO
        ww_pl = RandomTarget
    elif FLAGS.ww_policy == "revenge":
        # Default for PPO
        ww_pl = RevengeTarget
    elif FLAGS.ww_policy == "unite":
        ww_pl = RandomTargetUnite
    else:
        # Default case, though this should never run
        ww_pl = None

    # Set the policies as per the if statements above
    vill_p = (policy, env.observation_space, env.action_space, {})
    ww_p = (ww_pl, env.observation_space, env.action_space, {})

    policies = dict(
        wolf_p_static=ww_p,
        wolf_p=vill_p,
        vill_p=vill_p,
    )

    # Config for the Ray Tune run
    configs = {
        "env": EvaluationWrapper,
        "env_config": env_config,
        "framework": "torch",
        "num_workers": FLAGS.n_workers,
        "num_gpus": FLAGS.n_gpus,
        "batch_mode": "complete_episodes",
        "train_batch_size": FLAGS.batch_size,
        "rollout_fragment_length": FLAGS.rollout_length,

        "callbacks": CustomCallbacks,

        # model configs
        "model": {
            "use_lstm": True,
            "custom_model": "pa_model",  # using custom parametric action model
        },
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": mapping_static,
            "policies_to_train": "vill_p"

        },
    }

    # Update config with params for APPO or PPO
    configs.update(temp_conf_dict)

    analysis = tune.run(
        trainer,
        local_dir=FLAGS.ww_ray_dir,
        config=configs,
        trial_name_creator=trial_name_creator,
        checkpoint_freq=FLAGS.checkpoint_freq,
        keep_checkpoints_num=FLAGS.max_checkpoint_keep,
        resume=FLAGS.resume_training,
        reuse_actors=reuse_actors
    )
