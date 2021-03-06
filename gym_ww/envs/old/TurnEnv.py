import math
import random

import gym
import numpy as np
from gym import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.env import EnvContext

from envs import CONFIGS
from gym_ww import logger, ww, vil
from src.other.custom_utils import most_frequent, suicide_num, pprint
####################
# global vars
####################
# penalty fro breaking a rule
from utils import Params

####################
# names for roles
####################

rule_break_penalty = -50


class TurnEnvWw(MultiAgentEnv):
    """


    """
    metadata = {'players': ['human']}

    def __init__(self, configs, roles=None, flex=0):
        """

        :param num_players: int, number of player, must be grater than 4
        :param roles: list of str, list of roles for each agent
        :param flex: float [0,1), percentage of targets to consider when voting, 0 is just one, depend on the number of player.
            EG:  if num_players=10 -> targets are list of 10 elements, 10*0.5=5 -> first 5 player are considered when voting
        """

        # if config is dict
        if isinstance(configs, EnvContext) or isinstance(configs, dict):
            # get num player
            try:
                num_players = configs['num_players']
            except KeyError:
                raise AttributeError(f"Attribute 'num_players' should be present in the EnvContext")


        elif isinstance(configs, int):
            # used for back compatibility
            num_players = configs
        else:
            raise AttributeError(f"Type {type(configs)} is invalid for config")

        # number of player should be more than 5
        assert num_players >= 5, "Number of player should be >= 5"

        if roles is None:
            # number of wolves should be less than villagers
            num_wolves = math.floor(math.sqrt(num_players))
            num_villagers = num_players - num_wolves
            roles = [ww] * num_wolves + [vil] * num_villagers
            # random.shuffle(roles)
            logger.info(f"Starting game with {num_players} players: {num_villagers} {vil} and {num_wolves} {ww}")
        else:
            assert len(
                roles) == num_players, f"Length of role list ({len(roles)}) should be equal to number of players ({num_players})"
            num_wolves = len([elem for elem in roles if elem == ww])

        self.num_players = num_players
        self.num_wolves = num_wolves
        self.roles = roles
        self.penalties = CONFIGS['penalties']
        self.max_days = CONFIGS['max_days']
        self.signal_length = CONFIGS['signal_length']
        self.signal_range = CONFIGS['signal_range']

        # used for logging game
        self.ep_step = 0

        if flex == 0:
            self.flex = 1
        else:
            self.flex = math.floor(num_players * flex)

        # define empty attributes, refer to initialize method for more info
        self.status_map = None
        self.shuffle_map = None
        self.unshuffle_map = None
        self.is_night = True
        self.is_comm = True
        self.day_count = 0
        self.phase = 0
        self.is_done = False
        self.custom_metrics = None
        self.role_map = None
        self.initialize()

    #######################################
    #       INITALIZATION
    #######################################

    def initialize_info(self):

        self.custom_metrics = dict(
            suicide=0,  # number of times a player vote for itself
            win_wolf=0,  # number of times wolves win
            win_vil=0,  # number of times villagers win
            tot_days=0,  # total number of days before a match is over
        )

    def initialize(self):
        """
        Initialize attributes for new run
        :return:
        """

        self.role_map = {idx: self.roles[idx] for idx in range(self.num_players)}

        # map to shuffle player ids at the start of each game, check the readme under PolicyWw for more info
        sh = sorted(range(self.num_players), key=lambda k: random.random())
        self.shuffle_map = {idx: sh[idx] for idx in range(self.num_players)}
        self.unshuffle_map = {sh[idx]: idx for idx in range(self.num_players)}

        # list for agent status (dead=0, alive=1)
        self.status_map = [1 for _ in range(self.num_players)]

        # bool flag to keep track of turns
        self.is_night = True

        # first phase is communication night phase
        self.is_comm = True

        # reset is done
        self.is_done = False

        # reset day
        self.day_count = 0

        # reset info dict
        self.initialize_info()

        # step used for logging matches
        if self.ep_step == Params.log_step:
            self.ep_step = 0
        else:
            self.ep_step += 1

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

            Returns:
                observation (object): the initial observation.
            """
        if Params.log_step == self.ep_step:
            logger.info("Reset called")
        self.initialize()
        init_signal = {p: [-1] * self.signal_length for p in range(self.num_players)}
        obs = self.observe(phase=0, signal=init_signal, targets={k: -1 for k in range(self.num_players)})
        obs, _, _, _ = self.convert(obs, {}, {}, {}, 0)
        return obs

    #######################################
    #       MAIN CORE
    #######################################

    def day(self, actions, rewards):
        """
        Run the day phase, that is execute target based on votes and reward accordingly or the voting
        :param actions: dict, map id_ to vote
        :param rewards: dict, maps agent id_ to curr reward
        :return: updated rewards
        """

        def execution(actions, rewards):
            """
            To be called when is execution phase
            :return:
            """

            self.custom_metrics["suicide"] += suicide_num(actions)

            # get the agent to be executed
            target = most_frequent(actions)

            # penalize for non divergent target
            rewards = self.target_accord(target, rewards, actions)

            # if target is alive
            if self.status_map[target]:
                # log
                if Params.log_step == self.ep_step:
                    logger.debug(f"Player {target} ({self.role_map[target]}) has been executed")

                # for every agent alive, [to be executed agent too]
                for id_ in self.get_ids('all', alive=True):
                    # add/subtract penalty
                    if id_ == target:
                        rewards[id_] += self.penalties.get("death")
                    else:
                        rewards[id_] += self.penalties.get("execution")

                # kill target
                self.status_map[target] = 0
            else:
                # penalize agents for executing a dead one
                for id_ in self.get_ids("all", alive=True):
                    rewards[id_] += self.penalties.get('execute_dead')
                if Params.log_step == self.ep_step:
                    logger.debug(f"Players tried to execute dead agent {target}")

            # update day
            self.day_count += 1

            return rewards

        # call the appropriate method depending on the phase
        if self.is_comm:
            if Params.log_step == self.ep_step:
                logger.debug("Day Time| Voting")
            return rewards
        else:
            if Params.log_step == self.ep_step:
                logger.debug("Day Time| Executing")
            rewards = {id_: val + self.penalties.get('day') for id_, val in rewards.items()}
            return execution(actions, rewards)

    def night(self, actions, rewards):
        """
        Is night, time to perform actions!
        During this phase, villagers action are not considered
        :param actions: dict, map id_ to vote
        :param rewards: dict, maps agent id_ to curr reward
        :return: return updated rewards
        """

        if self.is_comm:
            if Params.log_step == self.ep_step:
                logger.debug("Night Time| Voting")
        else:
            if Params.log_step == self.ep_step:
                logger.debug("Night Time| Eating")

            # execute wolf actions
            rewards = self.wolf_action(actions, rewards)

        return rewards

    def step(self, actions_dict):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            actions_dict (dict): a list of action provided by the agents

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # remove roles from ids
        actions_dict = {int(k.split("_")[1]): v for k, v in actions_dict.items()}

        signals, targets = self.split_target_signal(actions_dict)

        # rewards start from zero
        rewards = {id_: 0 for id_ in self.get_ids("all", alive=False)}

        # execute night action
        if self.is_night:
            rewards = self.night(targets, rewards)
        else:  # else go with day
            # apply action by day
            rewards = self.day(targets, rewards)

        # prepare for phase shifting
        is_night, is_comm, phase = self.update_phase()

        # print actions
        if Params.log_step == self.ep_step:
            filter_ids = self.get_ids(ww, alive=True) if phase in [0, 1] else self.get_ids('all', alive=True)
            pprint(targets, signals, self.roles, signal_length=self.signal_length, logger=logger,
                   filtered_ids=filter_ids)

        # get dones
        dones, rewards = self.check_done(rewards)
        # get observation
        obs = self.observe(phase, signals, targets)

        # initialize infos with dict
        infos = {idx: {'role': self.roles[idx]} for idx in self.get_ids("all", alive=False)}

        # convert to return in correct format, do not modify anything except for dones
        obs, rewards, dones, info = self.convert(obs, rewards, dones, infos, phase)

        # if game over reset
        if self.is_done:
            self.custom_metrics["tot_days"] = self.day_count

            dones["__all__"] = True
            # normalize infos
            self.normalize_metrics()
        else:
            dones["__all__"] = False

        # shift phase
        self.is_night = is_night
        self.is_comm = is_comm

        return obs, rewards, dones, info

    def wolf_action(self, actions, rewards):
        """
        Perform wolf action, that is kill agent based on votes and reward
        :param actions: dict, map id_ to vote
        :param rewards: dict, maps agent id_ to curr reward
        :return: updated rewards
        """

        def kill(actions, rewards):

            # upvote suicide info
            self.custom_metrics["suicide"] += suicide_num(actions)

            if not len(wolves_ids):
                raise Exception("Game not done but wolves are dead, have reset been called?")

            # get agent to be eaten
            target = most_frequent(actions)

            # penalize for different ids
            rewards = self.target_accord(target, rewards, actions)

            # if target is alive
            if self.status_map[target]:
                # kill him
                self.status_map[target] = 0
                # penalize dead player
                rewards[target] += self.penalties.get("death")
                # reward wolves
                for id_ in wolves_ids:
                    rewards[id_] += self.penalties.get("kill")
                if Params.log_step == self.ep_step:
                    logger.debug(f"Wolves killed {target} ({self.role_map[target]})")



            else:
                if Params.log_step == self.ep_step:
                    logger.debug(f"Wolves tried to kill dead agent {target}")
                # penalize the wolves for eating a dead player
                for id_ in wolves_ids:
                    rewards[id_] += self.penalties.get('execute_dead')

            if target in wolves_ids:
                # penalize the agent for eating one of their kind
                for id_ in wolves_ids:
                    rewards[id_] += self.penalties.get('kill_wolf')

            return rewards

        wolves_ids = self.get_ids(ww, alive=True)
        # filter action to get only wolves
        actions = {k: v for k, v in actions.items() if k in wolves_ids}

        # call the appropriate method depending on the phase
        if self.is_comm:
            return rewards
        else:
            return kill(actions, rewards)

    #######################################
    #       UPDATER
    #######################################

    def update_phase(self):
        """
        Shift the phase to the next one, keep elif explicit for readability
        :return:
            night: bool, value of is_night
            comm: bool, value of is_com
            phase: int, value of phase
        """

        if self.is_night and self.is_comm:
            comm = False
            phase = 0
            night = True

        elif self.is_night and not self.is_comm:
            night = False
            comm = True
            phase = 1

        elif not self.is_night and self.is_comm:
            comm = False
            phase = 2
            night = False

        elif not self.is_night and not self.is_comm:
            night = True
            comm = True
            phase = 3

        else:
            raise ValueError("Something wrong when shifting phase")

        self.phase = phase
        return night, comm, phase

    #######################################
    #       UTILS
    #######################################

    def split_target_signal(self, actions_dict):
        """
        Split signal and target from the action dictionary
        :param actions_dict: dict[int->obj], map agent id to action
        :return: signals and target
        """
        # split signals from targets
        signals = {k: v[1:] for k, v in actions_dict.items()}

        # remove signals from action dict
        targets = {k: v[0] for k, v in actions_dict.items()}

        # apply unshuffle
        targets = {k: self.unshuffle_map[v] for k, v in targets.items()}

        return signals, targets

    def normalize_metrics(self):
        """
        In here normalization for custom metrics should be executed.
        Notice that this method needs to be called before the reset.
        :return: None
        """

        self.custom_metrics["suicide"] /= (self.day_count + 1)
        self.custom_metrics["suicide"] /= self.num_players

    def convert(self, obs, rewards, dones, info, phase):
        """
        Convert everything in correct format
        :param obs:
        :param rewards:
        :param dones:
        :param info:
        :return:
        """

        # remove villagers from night phase
        if phase in [0, 1] and False:
            rewards = {id_: rw for id_, rw in rewards.items() if self.get_ids(ww, alive=False)}
            obs = {id_: rw for id_, rw in obs.items() if self.get_ids(ww, alive=False)}
            dones = {id_: rw for id_, rw in dones.items() if self.get_ids(ww, alive=False)}
            info = {id_: rw for id_, rw in info.items() if self.get_ids(ww, alive=False)}

        # if the match is not done yet remove dead agents
        if not self.is_done:
            # filter out dead agents from rewards
            rewards = {id_: rw for id_, rw in rewards.items() if self.status_map[id_]}
            obs = {id_: rw for id_, rw in obs.items() if self.status_map[id_]}
            dones = {id_: rw for id_, rw in dones.items() if self.status_map[id_]}
            info = {id_: rw for id_, rw in info.items() if self.status_map[id_]}

        # add roles to ids for policy choosing
        rewards = {f"{self.roles[k]}_{k}": v for k, v in rewards.items()}
        obs = {f"{self.roles[k]}_{k}": v for k, v in obs.items()}
        dones = {f"{self.roles[k]}_{k}": v for k, v in dones.items()}
        info = {f"{self.roles[k]}_{k}": v for k, v in info.items()}

        # convert to float
        rewards = {k: float(v) for k, v in rewards.items()}

        return obs, rewards, dones, info

    def check_done(self, rewards):
        """
        Check if the game is over, moreover return true for dead agent in done
        :param rewards: dict, maps agent id_ to curr reward
        :return:
            dones: list of bool statement
            rewards: update rewards
        """
        dones = {id_: 0 for id_ in rewards.keys()}

        for idx in range(self.num_players):
            # done if the player is not alive
            done = not self.status_map[idx]
            dones[idx] = done

        # get list of alive agents
        alives = self.get_ids('all', alive=True)

        # if there are more wolves than villagers than they won
        wolf_won = len(self.get_ids(ww)) >= len(self.get_ids(vil))
        # if there are no more wolves than the villager won
        village_won = all([role == vil for id_, role in self.role_map.items() if id_ in alives])

        if wolf_won:  # if wolves won
            # set flag to true (for reset)
            self.is_done = True
            # reward
            for idx in self.get_ids(ww, alive=False):
                rewards[idx] += self.penalties.get('victory')
            for idx in self.get_ids(vil, alive=False):
                rewards[idx] += self.penalties.get('lost')
            if Params.log_step == self.ep_step:
                logger.info(f"{CONFIGS['win_log_str']}Wolves won{CONFIGS['win_log_str']}")
            self.custom_metrics['win_wolf'] += 1

        if village_won:
            self.is_done = True
            for idx in self.get_ids(vil, alive=False):
                rewards[idx] += self.penalties.get('victory')
            for idx in self.get_ids(ww, alive=False):
                rewards[idx] += self.penalties.get('lost')
            if Params.log_step == self.ep_step:
                logger.info(f"{CONFIGS['win_log_str']} Villagers won {CONFIGS['win_log_str']}")
            self.custom_metrics['win_vil'] += 1

        if self.day_count >= self.max_days - 1:
            self.is_done = True

        return dones, rewards

    def get_ids(self, role, alive=True):
        """
        Return a list of ids given a role
        :param role: str, the role of the wanted ids
        :param alive: bool, if to get just alive players or everyone
        :return: list of ints
        """

        if role == "all":
            ids = list(self.role_map.keys())
        else:
            # get all the ids for a given role
            ids = [id_ for id_, rl in self.role_map.items() if rl == role]

        # filter out dead ones
        if alive:
            ids = [id_ for id_ in ids if self.status_map[id_]]

        return ids

    def target_accord(self, chosen_target, rewards, targets):
        """
        Reward/penalize agent based on the target chose for execution/kill depending on the choices it made.
        This kind of reward shaping is done in order for agents to output targets which are more likely to be chosen
        :param targets: dict[int->int], maps an agent to its target
        :param chosen_target: int, agent id_ chosen for execution/kill
        :param rewards: dict[int->int], map agent id_ to reward
        :return: updated rewards
        """

        for id_, vote in targets.items():
            # if the agent hasn't voted for the executed agent then it takes a penalty
            if vote != chosen_target:
                penalty = self.penalties["trg_accord"]
                rewards[id_] += penalty

        return rewards

    #######################################
    #       SPACES
    #######################################

    @property
    def action_space(self):
        """
        :return:
        """

        # the action space is made of two parts: the first element is the actual target they want to be executed
        # and the other ones are the signal space
        space = gym.spaces.MultiDiscrete([self.num_players] * (self.signal_length + 1))
        # high=[self.num_players]+[self.signal_range-1]*self.signal_length
        # low=[-1]+[0]*self.signal_length
        # space = gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.int32)

        # should be a list of targets
        return space

    @property
    def observation_space(self):
        """
        Return observation space in gym box
        :return:
        """

        obs = dict(
            # number of days passed
            day=spaces.Discrete(self.max_days),
            # idx is agent id_, value is boll for agent alive
            status_map=spaces.MultiBinary(self.num_players),
            # number in range number of phases [com night, night, com day, day]
            phase=spaces.Discrete(4),
            # targets is now a vector, having an element outputted from each agent
            targets=gym.spaces.Box(low=-1, high=self.num_players, shape=(self.num_players,), dtype=np.int32),
            # signal is a matrix of dimension [num_player, signal_range]
            signal=gym.spaces.Box(low=-1, high=self.signal_range - 1, shape=(self.num_players, self.signal_length),
                                  dtype=np.int32)

        )
        obs = gym.spaces.Dict(obs)

        return obs

    def observe(self, phase, signal, targets):
        """
        Return observation object
        :return:
        """

        def add_missing(signal, targets):
            """
            Add missing values (for dead agents) to targets and signal
            :param signal: ndarray, signal of size [num_player, signal_lenght]
            :param targets: dict[int->int], mapping agent ids to targets
            :return: tuple
                1: signal
                2: targets
            """

            # if the list of outputs is full then do nothing
            if len(targets) == self.num_players:
                return signal, targets

            # get indices to add
            to_add = set(range(self.num_players)) - set(targets.keys())

            # add a list of -1 of length signal_length to the signal
            sg = [-1] * self.signal_length

            # update dict with -1
            targets.update({elem: -1 for elem in to_add})
            signal.update({elem: sg for elem in to_add})

            return signal, targets

        observations = {}

        signal, targets = add_missing(signal, targets)
        # make matrix out of signals of size [num_player,signal_length]
        signal = np.stack(list(signal.values()))
        # apply shuffle to status map
        st = [self.status_map[self.shuffle_map[idx]] for idx in range(self.num_players)]
        # generate shuffle function to be applied to numpy matrix
        shuffler = np.vectorize(lambda x: self.shuffle_map[x] if x in self.shuffle_map.keys() else x)
        tg = list(targets.values())
        tg = shuffler(tg)

        for idx in self.get_ids("all", alive=False):
            # build obs dict
            obs = dict(
                day=self.day_count,  # day passed
                status_map=np.array(st),  # agent_id:alive?
                phase=phase,
                targets=tg,
                signal=signal
            )

            observations[idx] = obs

        return observations
